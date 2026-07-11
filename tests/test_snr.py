"""Tests for the call-agnostic SNR pipeline (numpy-only; no torch needed)."""

import wave
from pathlib import Path

import numpy as np

from vocdenoiser.audio import read_wav, read_wav_segment
from vocdenoiser.snr.metric import DEFAULT_PARAMS, clip_features, spectral_snr_db
from vocdenoiser.snr.threshold import fit_gmm_1d, otsu_threshold
from vocdenoiser.snr.validate import spearman
from vocdenoiser.audio import power_spectrogram


def _write_wav(path, y, sr=96000):
    y16 = np.clip(y, -1, 1)
    y16 = (y16 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y16.tobytes())


def _tone(freq=8000.0, dur=0.5, sr=96000):
    t = np.arange(int(dur * sr)) / sr
    env = np.hanning(len(t))
    return env * np.sin(2 * np.pi * freq * t)


def test_wav_roundtrip(tmp_path: Path):
    y = _tone()
    p = tmp_path / "t.wav"
    _write_wav(p, y)
    y2, sr = read_wav(p)
    assert sr == 96000
    assert len(y2) == len(y)
    assert np.corrcoef(y, y2)[0, 1] > 0.999


def test_read_segment_matches_full(tmp_path: Path):
    y = _tone(dur=1.0)
    p = tmp_path / "t.wav"
    _write_wav(p, y)
    seg, sr = read_wav_segment(p, 1000, 500)
    full, _ = read_wav(p)
    assert np.allclose(seg, full[1000:1500], atol=1e-4)


def test_snr_decreases_with_noise():
    """The core property: adding noise must lower the SNR score."""
    rng = np.random.RandomState(0)
    y = _tone()
    scores = []
    for noise_amp in [0.0, 0.05, 0.2, 0.5]:
        mixed = y + noise_amp * rng.randn(len(y))
        S = power_spectrogram(mixed, DEFAULT_PARAMS.n_fft, DEFAULT_PARAMS.hop)
        scores.append(spectral_snr_db(S)[0])
    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_snr_agnostic_across_frequency():
    """A clean tone at 3 kHz and at 9 kHz should score similarly (no band bias)."""
    s_lo = spectral_snr_db(power_spectrogram(_tone(3000.0)))[0]
    s_hi = spectral_snr_db(power_spectrogram(_tone(9000.0)))[0]
    assert abs(s_lo - s_hi) < 4.0


def test_clip_features_keys():
    feats = clip_features(_tone(), 96000)
    for k in ["snr_db", "snr_broadband_db", "active_frac", "n_segments", "dom_freq_hz"]:
        assert k in feats
    assert 6000 < feats["dom_freq_hz"] < 10000  # 8 kHz tone


def test_gmm_and_otsu_separate_two_modes():
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.normal(10, 1, 500), rng.normal(20, 1, 500)])
    gmm = fit_gmm_1d(x, k=2)
    cross = gmm.crossover()
    assert 12 < cross < 18
    assert 12 < otsu_threshold(x) < 18


def test_spearman_signs():
    x = np.arange(100.0)
    assert spearman(x, x) > 0.99
    assert spearman(x, -x) < -0.99
