"""Validate that the SNR score is call-type AGNOSTIC.

Two checks, neither of which needs ground-truth call-type labels:

1. Morphology-correlation guard (Spearman). Over the whole scanned set, the SNR
   score should be only weakly correlated with morphology proxies (dominant
   frequency, bandwidth, spectral flatness). A strong correlation would mean the
   score is secretly ranking "phee-ness" rather than cleanliness.

2. Bias-injection test. Take the cleanest clips as near-clean references, mix in
   real background noise (``data/Noise`` + ``data/Cigarra``) at a sweep of known
   SNRs, and confirm (a) the measured score rises monotonically with injected
   SNR and (b) the response curves *overlap* across morphology strata (low vs
   high dominant frequency, narrow vs broad band). Overlapping curves = the
   metric responds to noise the same way regardless of call morphology.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vocdenoiser.audio import read_wav, read_wav_segment, wav_num_frames
from vocdenoiser.snr.metric import DEFAULT_PARAMS, SNRParams, clip_features


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_x = np.mean(x**2) + 1e-12
    p_n = np.mean(noise**2) + 1e-12
    scale = np.sqrt(p_x / (p_n * 10 ** (snr_db / 10.0)))
    return x + scale * noise


def _noise_segment(path: str, length: int, rng: np.random.RandomState) -> np.ndarray:
    """A random ``length``-sample slice of a (possibly very long) noise file.

    Reads only the needed frames via ``read_wav_segment`` so validating against
    60 s / 11 MB colony-noise recordings on the pCloud mount stays cheap.
    """
    total = wav_num_frames(path)
    if total < length:
        y, _ = read_wav(path)
        reps = int(np.ceil(length / max(len(y), 1)))
        return np.tile(y, reps)[:length]
    start = rng.randint(0, total - length + 1)
    seg, _ = read_wav_segment(path, start, length)
    return seg


def _load_scan(csv_path: str | Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            try:
                row["_snr"] = float(row["snr_db"])
                row["_dom"] = float(row["dom_freq_hz"])
                row["_bw"] = float(row["bandwidth_hz"])
                row["_flat"] = float(row["flatness"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def run_validation(
    scan_csv: str | Path,
    src_dir: str | Path,
    noise_dirs: list[str | Path],
    out_dir: str | Path,
    n_clean: int = 120,
    snr_levels=(-5, 0, 5, 10, 15, 20),
    params: SNRParams = DEFAULT_PARAMS,
    seed: int = 0,
) -> str:
    rows = _load_scan(scan_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # --- Check 1: morphology-correlation guard ------------------------------
    # Reported both on the full set and on the CLEAN subset (top-tercile SNR).
    # On the full set, morphology proxies are noise-contaminated (a noisy clip is
    # genuinely flatter AND lower-SNR), so |rho| is inflated by that confound. On
    # already-clean clips the proxies reflect true morphology, so the clean-subset
    # rho is the more honest test of whether the *metric* is call-type biased.
    snr = np.array([r["_snr"] for r in rows])
    dom = np.array([r["_dom"] for r in rows])
    bw = np.array([r["_bw"] for r in rows])
    flat = np.array([r["_flat"] for r in rows])
    clean_mask = snr >= np.percentile(snr, 66)
    guard = {
        "dom_freq_hz": (spearman(snr, dom), spearman(snr[clean_mask], dom[clean_mask])),
        "bandwidth_hz": (spearman(snr, bw), spearman(snr[clean_mask], bw[clean_mask])),
        "flatness": (spearman(snr, flat), spearman(snr[clean_mask], flat[clean_mask])),
    }

    # --- Check 2: bias-injection across morphology strata -------------------
    clean = sorted(rows, key=lambda r: -r["_snr"])[:n_clean]
    doms = np.array([r["_dom"] for r in clean])
    bws = np.array([r["_bw"] for r in clean])
    dom_t1, dom_t2 = np.percentile(doms, [33, 66])
    bw_med = np.median(bws)

    def stratum(r: dict) -> str:
        d = "domLo" if r["_dom"] < dom_t1 else ("domHi" if r["_dom"] >= dom_t2 else "domMid")
        b = "bwNarrow" if r["_bw"] < bw_med else "bwBroad"
        return f"{d}|{b}"

    noise_paths = []
    for nd in noise_dirs:
        noise_paths += [str(p) for p in Path(nd).rglob("*") if p.suffix.lower() == ".wav"]
    if not noise_paths:
        raise FileNotFoundError(f"No noise WAVs under {noise_dirs}")

    # measured[stratum][level] -> list of measured snr_db
    from collections import defaultdict

    measured = defaultdict(lambda: defaultdict(list))
    per_clip_curves = []  # (injected -> measured) monotonicity per clip
    for r in clean:
        y, sr = read_wav(str(Path(src_dir) / r["filename"]))
        if len(y) < params.n_fft // 2:
            continue
        npath = noise_paths[rng.randint(len(noise_paths))]
        noise = _noise_segment(npath, len(y), rng)
        row_curve = []
        for lvl in snr_levels:
            mixed = _mix_at_snr(y, noise, float(lvl))
            m = clip_features(mixed, sr, params)["snr_db"]
            measured[stratum(r)][lvl].append(m)
            row_curve.append(m)
        per_clip_curves.append(row_curve)

    curves = np.array(per_clip_curves)  # (clips, levels)
    mono_frac = float(np.mean([np.all(np.diff(c) > -0.5) for c in curves]))  # allow tiny noise
    # spread across strata at each level: max-min of per-stratum means
    strata = sorted(measured.keys())
    level_spreads = []
    stratum_curves = {}
    for s in strata:
        stratum_curves[s] = [float(np.mean(measured[s][lvl])) for lvl in snr_levels]
    for j, lvl in enumerate(snr_levels):
        means = [stratum_curves[s][j] for s in strata if not np.isnan(stratum_curves[s][j])]
        level_spreads.append(max(means) - min(means) if len(means) > 1 else 0.0)
    max_spread = float(np.max(level_spreads)) if level_spreads else float("nan")

    # --- verdict -------------------------------------------------------------
    # The controlled injection test (same call, varied noise+morphology) is the
    # primary arbiter of metric bias; the clean-subset Spearman backs it up.
    # "Substantial overlap" = cross-stratum spread under 20% of the injected SNR
    # sweep (curves for different call morphologies track each other closely).
    injected_range = max(snr_levels) - min(snr_levels)
    spread_tol = 0.20 * injected_range
    guard_ok = all(abs(clean) < 0.45 for (_full, clean) in guard.values() if not np.isnan(clean))
    mono_ok = mono_frac > 0.9
    spread_ok = max_spread < spread_tol
    verdict = "PASS" if (mono_ok and spread_ok and guard_ok) else "REVIEW"

    lines = ["# SNR call-agnosticism validation\n", f"**Verdict: {verdict}**\n"]
    lines.append(
        "The **injection test (Check 2)** is the primary evidence — it holds the call "
        "fixed and varies only noise + morphology stratum, so a small cross-stratum "
        "spread directly means the metric is call-agnostic. Check 1's full-set ρ is "
        "confounded by noise (see note); the clean-subset ρ is the fairer guard.\n"
    )
    lines.append("## Check 1 — morphology-correlation guard (want clean-subset |ρ| small)\n")
    lines.append("| morphology proxy | ρ (full set) | ρ (clean subset) | ok? |")
    lines.append("|---|---:|---:|:--:|")
    for k, (full, clean) in guard.items():
        ok = "✅" if (np.isnan(clean) or abs(clean) < 0.45) else "⚠️"
        lines.append(f"| {k} | {full:+.3f} | {clean:+.3f} | {ok} |")
    lines.append(
        "\nFull-set |ρ| is inflated because noisy clips are genuinely both flatter/"
        "broader AND lower-SNR — the proxy is contaminated by the very thing we "
        "measure. On clean clips the proxy reflects true call shape, so a small "
        "clean-subset |ρ| indicates the metric is not ranking call morphology.\n"
    )
    lines.append("## Check 2 — bias-injection across morphology strata\n")
    lines.append(f"- reference clips: {len(curves)} (cleanest by snr_db)")
    lines.append(f"- injected SNR sweep: {list(snr_levels)} dB")
    lines.append(f"- monotonic response (measured rises with injected): **{100*mono_frac:.0f}%** of clips {'✅' if mono_ok else '⚠️'}")
    lines.append(
        f"- max cross-stratum spread of the mean response: **{max_spread:.2f} dB** "
        f"(tolerance {spread_tol:.1f} dB = 20% of the {injected_range:.0f} dB injected sweep) "
        f"{'✅' if spread_ok else '⚠️'}\n"
    )
    lines.append("Per-stratum mean measured snr_db vs injected level:\n")
    header = "| stratum | " + " | ".join(f"{l}dB" for l in snr_levels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(snr_levels) + 1))
    for s in strata:
        cells = " | ".join(f"{v:.1f}" for v in stratum_curves[s])
        lines.append(f"| {s} | {cells} |")
    lines.append(
        "\nCurves that overlap (small spread) across low/high dominant-frequency and "
        "narrow/broad-band strata are the direct evidence the metric is call-agnostic.\n"
    )

    report = out_dir / "snr_validation.md"
    report.write_text("\n".join(lines))
    print(f"Validation verdict: {verdict}. Wrote {report}")
    return str(report)
