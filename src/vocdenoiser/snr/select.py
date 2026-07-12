"""Apply a threshold to a scan CSV and emit the clean-subset manifest."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def select_clean(
    csv_path: str | Path,
    src_dir: str | Path,
    out_manifest: str | Path,
    snr_threshold: float | None = None,
    keep_percentile: float | None = None,
    broadband_floor: float | None = None,
    exclude_multi_source: bool = True,
    link_dir: str | Path | None = None,
) -> dict:
    """Select clips at/above a cutoff and write a manifest CSV.

    Provide either ``snr_threshold`` (absolute dB) or ``keep_percentile`` (keep
    the top X%). With ``exclude_multi_source`` set, clips with ``n_segments>1``
    (likely target-call + overlapping transient) are dropped. If ``link_dir`` is
    given, selected files are symlinked there for convenient downstream loading.

    ``broadband_floor`` additionally drops clips whose secondary
    ``snr_broadband_db`` is below the given dB — the primary ``snr_db``
    under-penalizes broadband hiss for narrowband/tonal calls, so a floor on the
    broadband channel removes hissy clips the primary cutoff lets through. A row
    with a missing/unparseable ``snr_broadband_db`` fails the floor (conservative).
    """
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_bb = reader.fieldnames is not None and "snr_broadband_db" in reader.fieldnames
        if broadband_floor is not None and not has_bb:
            raise ValueError(
                f"--broadband-floor given but {csv_path} has no 'snr_broadband_db' "
                f"column (got {reader.fieldnames}). Re-run 'snr scan' to produce it."
            )
        for row in reader:
            if row.get("error"):
                continue
            try:
                row["_snr"] = float(row["snr_db"])
                row["_nseg"] = float(row.get("n_segments") or 1)
            except (TypeError, ValueError):
                continue
            try:
                row["_bb"] = float(row["snr_broadband_db"]) if has_bb else float("nan")
            except (TypeError, ValueError):
                row["_bb"] = float("nan")
            rows.append(row)

    if snr_threshold is None:
        if keep_percentile is None:
            raise ValueError("Provide snr_threshold or keep_percentile")
        vals = sorted(r["_snr"] for r in rows)
        idx = int((1.0 - keep_percentile / 100.0) * len(vals))
        snr_threshold = vals[min(idx, len(vals) - 1)]

    selected = [r for r in rows if r["_snr"] >= snr_threshold]
    if exclude_multi_source:
        selected = [r for r in selected if r["_nseg"] <= 1]
    n_before_bb = len(selected)
    if broadband_floor is not None:
        # NaN >= floor is False, so missing-broadband rows are dropped too.
        selected = [r for r in selected if r["_bb"] >= broadband_floor]
    n_dropped_bb = n_before_bb - len(selected)

    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "snr_db", "snr_broadband_db", "n_segments", "path"])
        for r in sorted(selected, key=lambda r: -r["_snr"]):
            bb = "" if r["_bb"] != r["_bb"] else f"{r['_bb']:.3f}"
            writer.writerow([r["filename"], f"{r['_snr']:.3f}", bb, int(r["_nseg"]),
                             str(Path(src_dir) / r["filename"])])

    if link_dir:
        link_dir = Path(link_dir)
        link_dir.mkdir(parents=True, exist_ok=True)
        for r in selected:
            dst = link_dir / r["filename"]
            src = Path(src_dir) / r["filename"]
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass

    summary = {
        "threshold_db": snr_threshold,
        "broadband_floor_db": broadband_floor,
        "n_total": len(rows),
        "n_selected": len(selected),
        "n_dropped_broadband": n_dropped_bb,
        "frac_selected": len(selected) / max(len(rows), 1),
        "manifest": str(out_manifest),
    }
    msg = (
        f"Selected {summary['n_selected']}/{summary['n_total']} clips "
        f"({100 * summary['frac_selected']:.1f}%) at >= {snr_threshold:.2f} dB"
    )
    if broadband_floor is not None:
        msg += (
            f", broadband >= {broadband_floor:.2f} dB "
            f"(floor dropped {n_dropped_bb} more)"
        )
    print(f"{msg} -> {out_manifest}")
    return summary
