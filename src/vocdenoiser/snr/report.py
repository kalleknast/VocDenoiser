"""Turn a scan CSV into a human-readable distribution report.

Always writes a markdown report with a text histogram, the label-free threshold
candidates, a kept-count-vs-threshold table, and representative filenames per
decile (so you can *listen* across the SNR range before choosing a cutoff). If
matplotlib is importable it also writes a PNG; otherwise it degrades gracefully.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vocdenoiser.snr.threshold import fit_gmm_1d, kept_count_curve, otsu_threshold


def load_scores(csv_path: str | Path) -> tuple[np.ndarray, list[str], dict]:
    """Load a scan CSV -> (snr_db array, filenames, aux columns dict).

    Rows with a non-empty ``error`` column are dropped from the score array but
    counted in ``aux['n_errors']``.
    """
    filenames: list[str] = []
    snr: list[float] = []
    n_segments: list[float] = []
    dom_freq: list[float] = []
    bandwidth: list[float] = []
    flatness: list[float] = []
    n_errors = 0
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                n_errors += 1
                continue
            try:
                s = float(row["snr_db"])
            except (TypeError, ValueError):
                n_errors += 1
                continue
            snr.append(s)
            filenames.append(row["filename"])
            n_segments.append(float(row.get("n_segments") or "nan"))
            dom_freq.append(float(row.get("dom_freq_hz") or "nan"))
            bandwidth.append(float(row.get("bandwidth_hz") or "nan"))
            flatness.append(float(row.get("flatness") or "nan"))
    aux = {
        "n_errors": n_errors,
        "n_segments": np.array(n_segments),
        "dom_freq_hz": np.array(dom_freq),
        "bandwidth_hz": np.array(bandwidth),
        "flatness": np.array(flatness),
    }
    return np.array(snr), filenames, aux


def _text_histogram(x: np.ndarray, bins: int = 40, width: int = 60) -> str:
    hist, edges = np.histogram(x, bins=bins)
    top = hist.max() or 1
    lines = []
    for i in range(bins):
        bar = "#" * int(round(width * hist[i] / top))
        lines.append(f"  {edges[i]:7.2f} | {bar} {hist[i]}")
    return "\n".join(lines)


def build_report(csv_path: str | Path, out_dir: str | Path) -> str:
    snr, filenames, aux = load_scores(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(snr)

    gmm = fit_gmm_1d(snr, k=2)
    gmm_cross = gmm.crossover()
    otsu = otsu_threshold(snr)
    thr, kept = kept_count_curve(snr, n=25)

    order = np.argsort(snr)
    deciles = {}
    for d in range(10):
        lo = int(d / 10 * n)
        hi = int((d + 1) / 10 * n)
        idx = order[lo:hi]
        pick = idx[:: max(1, len(idx) // 4)][:4]
        deciles[d] = [(filenames[i], float(snr[i])) for i in pick]

    multi = int((aux["n_segments"] > 1).sum()) if aux["n_segments"].size else 0

    lines = []
    lines.append("# SNR distribution report\n")
    lines.append(f"- clips scored: **{n}**  (errors skipped: {aux['n_errors']})")
    lines.append(
        f"- snr_db: min={snr.min():.2f}  p10={np.percentile(snr, 10):.2f}  "
        f"median={np.median(snr):.2f}  p90={np.percentile(snr, 90):.2f}  max={snr.max():.2f}"
    )
    means = np.sort(gmm.means)
    lines.append(
        f"- GMM(2) component means: {means[0]:.2f} dB (noisy) / {means[1]:.2f} dB (clean); "
        f"weights≈{np.round(gmm.weights, 2).tolist()}"
    )
    lines.append(f"- **GMM crossover threshold: {gmm_cross:.2f} dB**  (Otsu cross-check: {otsu:.2f} dB)")
    lines.append(f"- clips flagged multi-source (n_segments>1): {multi} ({100 * multi / max(n,1):.1f}%)\n")

    lines.append("## snr_db histogram\n```")
    lines.append(_text_histogram(snr))
    lines.append("```\n")

    lines.append("## kept-count vs threshold\n")
    lines.append("| threshold (dB) | kept | kept % |")
    lines.append("|---:|---:|---:|")
    for t, k in zip(thr, kept):
        lines.append(f"| {t:.2f} | {k} | {100 * k / max(n,1):.1f}% |")
    lines.append("")

    for cut, name in [(gmm_cross, "GMM crossover"), (otsu, "Otsu")]:
        k = int((snr >= cut).sum())
        lines.append(f"- keep at **{name}** ({cut:.2f} dB) -> **{k}** clips ({100 * k / max(n,1):.1f}%)")
    lines.append("")

    lines.append("## representative clips per decile (for listening)\n")
    for d in range(9, -1, -1):
        samples = ", ".join(f"{fn} ({s:.1f}dB)" for fn, s in deciles[d])
        lines.append(f"- decile {d + 1} (cleanest→noisiest as d falls): {samples}")
    lines.append("")

    report_md = out_dir / "snr_report.md"
    report_md.write_text("\n".join(lines))

    png = _maybe_plot(snr, gmm_cross, otsu, thr, kept, out_dir)
    if png:
        lines.insert(1, f"\n![snr distribution]({Path(png).name})\n")
        report_md.write_text("\n".join(lines))
    print(f"Wrote {report_md}" + (f" and {png}" if png else " (no matplotlib -> text only)"))
    return str(report_md)


def _maybe_plot(snr, gmm_cross, otsu, thr, kept, out_dir) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(snr, bins=60, color="#4C72B0", alpha=0.85)
    ax1.axvline(gmm_cross, color="#C44E52", lw=2, label=f"GMM crossover {gmm_cross:.1f} dB")
    ax1.axvline(otsu, color="#55A868", lw=2, ls="--", label=f"Otsu {otsu:.1f} dB")
    ax1.set_xlabel("snr_db (masked per-bin)")
    ax1.set_ylabel("count")
    ax1.set_title("SNR distribution")
    ax1.legend()
    ax2.plot(thr, kept, color="#4C72B0", marker="o", ms=3)
    ax2.axvline(gmm_cross, color="#C44E52", lw=2)
    ax2.set_xlabel("threshold (dB)")
    ax2.set_ylabel("clips kept")
    ax2.set_title("kept-count vs threshold")
    fig.tight_layout()
    png = Path(out_dir) / "snr_report.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return str(png)
