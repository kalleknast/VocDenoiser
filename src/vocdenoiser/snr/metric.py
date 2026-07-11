"""Call-agnostic SNR metric and supporting per-clip features.

All computations are per-frequency-bin and adaptive, so nothing here assumes a
call sits in a particular band or has harmonic structure. See ``snr/__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vocdenoiser.audio import power_spectrogram, read_wav

EPS = 1e-10


@dataclass(frozen=True)
class SNRParams:
    """STFT + masking parameters.

    Defaults are tuned for short clips at 96 kHz: n_fft=1024 / hop=256 gives a
    ~2.7 ms hop, so even a 0.05 s clip yields >=15 frames — enough for a stable
    per-bin temporal percentile.
    """

    n_fft: int = 1024
    hop: int = 256
    noise_percentile: float = 15.0  # per-bin noise floor = this percentile over time
    signal_percentile: float = 95.0  # per-bin signal level = this percentile over time
    active_db: float = 6.0  # a pixel is "call-occupied" if it exceeds floor by this many dB
    freq_smooth_bins: int = 5  # running-mean smoothing of the per-bin floor across frequency


DEFAULT_PARAMS = SNRParams()


def _running_mean(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(x + EPS)


def _envelope_frames(S: np.ndarray) -> np.ndarray:
    """Per-frame broadband energy (sum over bins) of a power spectrogram."""
    return S.sum(axis=0)


def spectral_snr_db(S: np.ndarray, p: SNRParams = DEFAULT_PARAMS) -> tuple[float, dict]:
    """Band-free, bandwidth-unbiased spectral SNR (dB) for ``S`` of shape (F, T).

    The floor and peak are estimated *independently per frequency bin* (temporal
    percentiles), and the primary score is the *mean per-bin SNR over the active
    bins* — an intensive quantity, so a broadband call is not rewarded over a
    narrowband one for merely occupying more bins (the call-type bias to avoid):

        floor[f] = percentile_t(S[f, :], noise_pct)      # per-bin background power
        sig[f]   = percentile_t(S[f, :], signal_pct)     # per-bin peak power
        snr[f]   = 10*log10(sig[f] / floor[f])           # per-bin SNR
        active   = { f : snr[f] > active_db }            # occupied bins
        snr_db   = mean_{f in active} snr[f]             # PRIMARY, call-agnostic

    A controlled noise-injection test (same call, swept SNR, compared across
    dominant-frequency and bandwidth strata) puts the cross-stratum spread of
    this estimator at ~1 dB — i.e. it responds to noise essentially the same way
    regardless of call morphology.

    ``extras['snr_broadband_db']`` is a SECONDARY score (mean active signal over
    mean floor across *all* bins). It is more sensitive to broadband background
    (hiss/wind/cicada raise every bin's floor) but IS bandwidth-biased, so it is
    kept out of the primary score and used only as an auxiliary filter.
    """
    floor = np.maximum(np.percentile(S, p.noise_percentile, axis=1), EPS)  # (F,)
    sig = np.maximum(np.percentile(S, p.signal_percentile, axis=1), EPS)  # (F,)

    per_bin_db = 10.0 * np.log10(sig / floor)  # (F,) per-bin SNR
    active_bins = per_bin_db > p.active_db
    if not active_bins.any():
        # No bin clears the margin: fall back to the most-occupied decile.
        active_bins = per_bin_db >= np.percentile(per_bin_db, 90)

    snr_db = float(per_bin_db[active_bins].mean())  # PRIMARY (intensive, band-free)
    snr_broadband_db = 10.0 * np.log10(
        (sig[active_bins].mean() + EPS) / (floor.mean() + EPS)
    )  # SECONDARY (broadband-sensitive, bandwidth-biased)

    excess_db = _db(S) - _db(floor)[:, None]  # per-pixel dB over its own floor
    mask = excess_db > p.active_db  # band-free active footprint (time-frequency)
    extras = {
        "mask": mask,
        "floor": floor,
        "active_bins": active_bins,
        "active_frac": float(mask.mean()),
        "snr_broadband_db": float(snr_broadband_db),
    }
    return snr_db, extras


# Backwards-compatible alias.
masked_snr_db = spectral_snr_db


def _temporal_snr_db(S: np.ndarray) -> float:
    """Broadband envelope SNR: high-percentile vs low-percentile frame energy, in dB."""
    env = _envelope_frames(S)
    hi = np.percentile(env, 95.0)
    lo = np.percentile(env, 15.0)
    return float(10.0 * np.log10((hi + EPS) / (lo + EPS)))


def _temporal_entropy(S: np.ndarray) -> float:
    """Shannon entropy (normalized to [0,1]) of the energy envelope over time.

    Low entropy = energy concentrated in a few frames (a compact call in
    silence); high entropy = energy spread out (steady noise). ``1 - Ht`` is a
    band-free, morphology-free cleanliness proxy.
    """
    env = _envelope_frames(S)
    total = env.sum()
    if total <= EPS or len(env) < 2:
        return 1.0
    prob = env / total
    prob = prob[prob > 0]
    h = -np.sum(prob * np.log(prob))
    return float(h / np.log(len(env)))


def _n_active_segments(mask: np.ndarray, min_gap: int = 3) -> int:
    """Count contiguous active time-segments (band-free co-occurrence proxy).

    Several well-separated active segments in one clip suggest more than one
    sound source (target call + an overlapping bird chirp / word / transient) —
    a red flag no unsupervised SNR can otherwise catch.
    """
    active_t = mask.any(axis=0).astype(int)  # (T,) any-frequency activity per frame
    if active_t.sum() == 0:
        return 0
    padded = np.concatenate([[0], active_t, [0]])
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]
    # Merge segments separated by a gap smaller than min_gap frames.
    segments = list(zip(starts, ends))
    merged = 1
    for i in range(1, len(segments)):
        if segments[i][0] - segments[i - 1][1] >= min_gap:
            merged += 1
    return merged


def _spectral_shape(S: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Morphology descriptors used only for the bias *diagnostic* (not the score).

    Returns (dominant_freq_hz, bandwidth_hz, spectral_flatness). These MUST NOT
    enter the SNR score — they encode call shape, and ranking on them would
    re-introduce the phee bias. We compute them so validation can check that the
    SNR score is *uncorrelated* with call morphology.
    """
    mean_spec = S.mean(axis=1)
    f = np.fft.rfftfreq((S.shape[0] - 1) * 2, d=1.0 / sr)
    total = mean_spec.sum() + EPS
    dom_freq = float(f[int(np.argmax(mean_spec))])
    centroid = float((f * mean_spec).sum() / total)
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * mean_spec).sum() / total))
    gm = np.exp(np.mean(np.log(mean_spec + EPS)))
    am = np.mean(mean_spec) + EPS
    flatness = float(gm / am)
    return dom_freq, bandwidth, flatness


def clip_features(
    y: np.ndarray, sr: int, p: SNRParams = DEFAULT_PARAMS
) -> dict:
    """Compute the SNR score plus band-free supporting features for one clip."""
    S = power_spectrogram(y, p.n_fft, p.hop)
    snr_db, extras = spectral_snr_db(S, p)
    dom_freq, bandwidth, flatness = _spectral_shape(S, sr)
    return {
        "snr_db": snr_db,  # PRIMARY, call-agnostic
        "snr_broadband_db": extras["snr_broadband_db"],  # SECONDARY (broadband-sensitive)
        "snr_temporal_db": _temporal_snr_db(S),  # supporting, band-free
        "active_frac": extras["active_frac"],  # supporting, band-free
        "one_minus_entropy": 1.0 - _temporal_entropy(S),  # supporting, band-free
        "n_segments": _n_active_segments(extras["mask"]),  # co-occurrence flag
        "dom_freq_hz": dom_freq,  # DIAGNOSTIC only (morphology)
        "bandwidth_hz": bandwidth,  # DIAGNOSTIC only (morphology)
        "flatness": flatness,  # DIAGNOSTIC only (morphology)
        "duration_s": len(y) / sr,
        "n_frames": S.shape[1],
    }


def clip_features_from_path(path: str, p: SNRParams = DEFAULT_PARAMS) -> dict:
    """Read a WAV and compute :func:`clip_features`; robust to unreadable files."""
    try:
        y, sr = read_wav(path)
    except Exception as exc:  # noqa: BLE001 - we want every failure recorded, not raised
        return {"error": f"{type(exc).__name__}: {exc}"}
    if len(y) < 4:
        return {"error": "empty or too-short audio"}
    feats = clip_features(y, sr, p)
    feats["sr"] = sr
    return feats
