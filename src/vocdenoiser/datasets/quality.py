"""Per-clip call-quality scoring for the external benchmark datasets.

External corpora (InfantMarmosetsVox, MarmAudio) come from different colonies and
recording chains than our own ``data/Vocalizations``, so some clips are noisy,
over-recorded (**clipped**), or mostly **silence** (loose annotation boundaries).
Feeding those into the identity/call-type eval biases the result, so we score
each prepared clip and optionally drop the bad ones.

The score reuses the call-agnostic SNR metric (:func:`vocdenoiser.snr.metric.clip_features`
— band-free SNR, active fraction, temporal entropy, co-occurring-source count) and
adds two waveform-domain checks that the spectral metric can't see:

* ``clip_frac`` — fraction of samples at digital full scale (over-recording).
* ``peak_dbfs`` / ``rms_dbfs`` — level, to catch near-silent "calls".
"""

from __future__ import annotations

import numpy as np

from vocdenoiser.snr.metric import clip_features

# Quality metrics appended to the loader label CSVs (in this order).
QUALITY_COLS = ["snr_db", "snr_broadband_db", "active_frac", "n_segments",
                "peak_dbfs", "rms_dbfs", "clip_frac", "duration_s"]

# Report-only "concern" thresholds for the printed summary (not used to filter).
_CONCERN_SNR_DB = 10.0
_CONCERN_CLIP_FRAC = 0.01
_CONCERN_PEAK_DBFS = -40.0


def clip_quality(sig: np.ndarray, sr: int) -> dict:
    """Call-agnostic SNR features plus waveform level/clipping for one clip."""
    sig = np.asarray(sig, dtype=np.float64)
    q = clip_features(sig, sr)
    a = np.abs(sig)
    peak = float(a.max()) if a.size else 0.0
    rms = float(np.sqrt(np.mean(sig ** 2))) if a.size else 0.0
    q["peak_dbfs"] = 20.0 * np.log10(peak + 1e-10)
    q["rms_dbfs"] = 20.0 * np.log10(rms + 1e-10)
    q["clip_frac"] = float(np.mean(a >= 0.999)) if a.size else 0.0
    return q


def quality_fail_reasons(
    q: dict,
    *,
    min_snr: float | None = None,
    min_active_frac: float | None = None,
    max_clip_frac: float | None = None,
    min_peak_dbfs: float | None = None,
    min_dur: float | None = None,
    max_segments: int | None = None,
) -> list[str]:
    """Which (if any) supplied quality thresholds this clip fails."""
    reasons = []
    if min_snr is not None and q["snr_db"] < min_snr:
        reasons.append("low_snr")
    if min_active_frac is not None and q["active_frac"] < min_active_frac:
        reasons.append("low_active")
    if max_clip_frac is not None and q["clip_frac"] > max_clip_frac:
        reasons.append("clipped")
    if min_peak_dbfs is not None and q["peak_dbfs"] < min_peak_dbfs:
        reasons.append("near_silent")
    if min_dur is not None and q["duration_s"] < min_dur:
        reasons.append("too_short")
    if max_segments is not None and q["n_segments"] > max_segments:
        reasons.append("multi_source")
    return reasons


def summarize(qualities: list[dict]) -> str:
    """Human-readable quality distribution + concern-flag counts."""
    n = len(qualities)
    if n == 0:
        return "quality: no clips scored"

    def pct(key: str, p: float) -> float:
        return float(np.percentile([q[key] for q in qualities], p))

    low_snr = sum(q["snr_db"] < _CONCERN_SNR_DB for q in qualities)
    clipped = sum(q["clip_frac"] > _CONCERN_CLIP_FRAC for q in qualities)
    silent = sum(q["peak_dbfs"] < _CONCERN_PEAK_DBFS for q in qualities)
    multi = sum(q["n_segments"] > 1 for q in qualities)

    def pctreport(count: int) -> str:
        return f"{count} ({100 * count / n:.1f}%)"

    return "\n".join([
        f"quality of {n} clips:",
        f"  snr_db      p10/50/90 = {pct('snr_db', 10):.1f} / {pct('snr_db', 50):.1f} / {pct('snr_db', 90):.1f} dB",
        f"  active_frac p10/50/90 = {pct('active_frac', 10):.3f} / {pct('active_frac', 50):.3f} / {pct('active_frac', 90):.3f}",
        f"  peak_dbfs   p10/50    = {pct('peak_dbfs', 10):.1f} / {pct('peak_dbfs', 50):.1f} dBFS",
        f"  concerns: snr<{_CONCERN_SNR_DB:g}dB={pctreport(low_snr)} | "
        f"clipped>{_CONCERN_CLIP_FRAC:g}={pctreport(clipped)} | "
        f"near-silent<{_CONCERN_PEAK_DBFS:g}dBFS={pctreport(silent)} | "
        f"multi-source={pctreport(multi)}",
    ])
