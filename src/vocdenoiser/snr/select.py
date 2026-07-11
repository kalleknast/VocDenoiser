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
    exclude_multi_source: bool = True,
    link_dir: str | Path | None = None,
) -> dict:
    """Select clips at/above a cutoff and write a manifest CSV.

    Provide either ``snr_threshold`` (absolute dB) or ``keep_percentile`` (keep
    the top X%). With ``exclude_multi_source`` set, clips with ``n_segments>1``
    (likely target-call + overlapping transient) are dropped. If ``link_dir`` is
    given, selected files are symlinked there for convenient downstream loading.
    """
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            try:
                row["_snr"] = float(row["snr_db"])
                row["_nseg"] = float(row.get("n_segments") or 1)
            except (TypeError, ValueError):
                continue
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

    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "snr_db", "n_segments", "path"])
        for r in sorted(selected, key=lambda r: -r["_snr"]):
            writer.writerow([r["filename"], f"{r['_snr']:.3f}", int(r["_nseg"]),
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
        "n_total": len(rows),
        "n_selected": len(selected),
        "frac_selected": len(selected) / max(len(rows), 1),
        "manifest": str(out_manifest),
    }
    print(
        f"Selected {summary['n_selected']}/{summary['n_total']} clips "
        f"({100 * summary['frac_selected']:.1f}%) at >= {snr_threshold:.2f} dB -> {out_manifest}"
    )
    return summary
