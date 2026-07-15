"""Smoke tests for the β-VAE denoiser: shapes, 16-dim latent, finite loss.

The heavy ML stack is optional (the ``ml`` extra), so these skip cleanly when
torch/lightning are absent — e.g. on a numpy-only install of the SNR pipeline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from vocdenoiser.denoise.beta_vae import BetaVAE  # noqa: E402
from vocdenoiser.denoise.config import Config  # noqa: E402


def _tiny_cfg(**overrides) -> Config:
    base = dict(n_mels=32, n_frames=32, base_channels=8, latent_dim=16, batch_size=2)
    base.update(overrides)
    return Config(**base)


def test_forward_pass_shapes_roundtrip():
    cfg = _tiny_cfg()
    model = BetaVAE(cfg).eval()
    x = torch.randn(cfg.batch_size, *cfg.spec_shape)

    recon, mu, logvar = model(x)

    assert recon.shape == x.shape, "decoder must reconstruct the exact input shape"
    assert mu.shape == (cfg.batch_size, 16)
    assert logvar.shape == (cfg.batch_size, 16)


def test_latent_dim_is_16():
    model = BetaVAE(_tiny_cfg())
    assert model.cfg.latent_dim == 16
    mu, logvar = model.encode(torch.randn(3, *model.cfg.spec_shape))
    assert mu.shape[1] == 16 and logvar.shape[1] == 16


def test_loss_is_finite():
    cfg = _tiny_cfg()
    model = BetaVAE(cfg)
    x = torch.randn(cfg.batch_size, *cfg.spec_shape)
    recon, mu, logvar = model(x)
    total, mse, kl = model.loss(recon, x, mu, logvar)
    assert torch.isfinite(total) and torch.isfinite(mse) and torch.isfinite(kl)
    assert kl.item() >= 0.0


def test_training_step_finite():
    cfg = _tiny_cfg()
    model = BetaVAE(cfg)
    noisy = torch.randn(cfg.batch_size, *cfg.spec_shape)
    clean = torch.randn(cfg.batch_size, *cfg.spec_shape)
    loss = model.training_step((noisy, clean), 0)
    assert torch.isfinite(loss)


def test_config_rejects_indivisible_geometry():
    with pytest.raises(ValueError):
        Config(n_mels=30, n_frames=32)  # 30 not divisible by the 2**4 downsample


def test_search_model_residual_l2_stays_finite():
    """Every residual=True search candidate crashed (NaN blowup) because the config model
    lacked the hand model's guards. Residual makes recon start at ≈ the large-magnitude
    noisy input, so the first-step loss/grads are large; with L2 + no logvar clamp / no
    grad clip it NaN'd. Check the worst case (residual, l2) stays finite through a few
    aggressive optimizer steps — the harness supplies the grad-clip the Trainer would."""
    from vocdenoiser.search.model_factory import build_search_model
    from vocdenoiser.search.space import Candidate

    cfg = _tiny_cfg(base_channels=16)
    cand = Candidate(
        n_conv_layers=4, base_channels=16, channel_mult=2.0, kernel_size=4,
        norm="none", act="silu", latent_dim=8, residual=True, recon_loss="l2",
        beta=6.0, beta_schedule="warmup", optimizer="adam", lr=3e-3, batch_size=2,
    )
    model = build_search_model(cand, cfg)

    # The logvar clamp is present (the deterministic guard against exp() overflow).
    _, logvar = model.encode(torch.randn(2, *cfg.spec_shape))
    assert logvar.abs().max().item() <= 10.0 + 1e-4

    opt = model.configure_optimizers()
    for _ in range(5):
        opt.zero_grad()
        noisy = 20.0 * torch.randn(2, *cfg.spec_shape)  # large magnitude, like raw log-mel
        clean = 20.0 * torch.randn(2, *cfg.spec_shape)
        recon, mu, logvar = model(noisy)
        assert recon.shape == noisy.shape  # residual add requires matching shapes
        total, _, _ = model._loss(recon, clean, mu, logvar)
        assert torch.isfinite(total)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # what the harness Trainer does
        opt.step()

    mu, logvar = model.encode(20.0 * torch.randn(2, *cfg.spec_shape))
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()


def test_estimate_params_matches_factory():
    """``estimate_params`` mirrors the factory's construction arithmetic in pure Python so
    the (torch-free) operators can reject oversized candidates before paying to build them.
    That duplication is only safe while the two agree exactly — this is the guard. It
    reproduces every num_params in both shipped ledgers; here it is checked against the
    real module across the families that stress it (norm affine params, odd/even kernels,
    both depths, channel_mult rounding)."""
    from vocdenoiser.search.model_factory import build_search_model
    from vocdenoiser.search.space import Candidate, estimate_params

    for n_conv_layers in (3, 4):
        for norm in ("batch", "group", "none"):
            for kernel_size in (3, 4, 5):
                for channel_mult in (1.5, 2.0):
                    cand = Candidate(
                        n_conv_layers=n_conv_layers, base_channels=16,
                        channel_mult=channel_mult, kernel_size=kernel_size, norm=norm,
                        latent_dim=8,
                    )
                    cfg = _tiny_cfg(n_mels=64, n_frames=64, base_channels=16, latent_dim=8)
                    model = build_search_model(cand, cfg)
                    actual = sum(p.numel() for p in model.parameters())
                    predicted = estimate_params(cand, n_mels=64, n_frames=64)
                    assert predicted == actual, (
                        f"estimate_params drifted from the factory for {cand}: "
                        f"predicted {predicted}, built {actual}"
                    )


def test_diverged_candidate_aborts_instead_of_burning_its_budget():
    """A candidate that NaNs used to skip every remaining step and still be scored -inf at
    the end — 4 such candidates burned 2.7h of a 16.8h run in search_ledger_v2. Once the
    non-finite streak passes the patience it must raise, so the harness can record the same
    crash immediately. A transient bad batch must NOT trip it."""
    from vocdenoiser.search.model_factory import CandidateDiverged, build_search_model
    from vocdenoiser.search.space import Candidate

    cfg = _tiny_cfg()
    model = build_search_model(Candidate(base_channels=8, latent_dim=16), cfg)
    model.nonfinite_patience = 3
    batch = (torch.randn(2, *cfg.spec_shape), torch.randn(2, *cfg.spec_shape))

    # A finite step resets the streak: two non-finite steps either side of a good one
    # must not abort (patience is about persistent divergence, not one bad batch).
    nan_batch = (torch.full((2, *cfg.spec_shape), float("nan")), batch[1])
    assert model.training_step(nan_batch, 0) is None  # skipped, not raised
    assert model.training_step(nan_batch, 0) is None
    assert torch.isfinite(model.training_step(batch, 0))
    assert model._nonfinite_streak == 0

    with pytest.raises(CandidateDiverged):
        for _ in range(model.nonfinite_patience):
            model.training_step(nan_batch, 0)


# --- real-noise mixing (extension beyond SPECS.md) ------------------------

def test_noise_bed_weight_one_is_the_real_background():
    from vocdenoiser.denoise import augment

    cfg = _tiny_cfg(real_noise_weight=1.0)
    g = torch.Generator().manual_seed(0)
    real = torch.randn(4096)
    bed = augment.noise_bed(4096, [], cfg, g, real_bg=real)
    assert torch.allclose(bed, augment._normalise_rms(real), atol=1e-5)


def test_noise_bed_weight_zero_ignores_real():
    from vocdenoiser.denoise import augment

    cfg = _tiny_cfg(real_noise_weight=0.0)
    real = torch.randn(4096)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    with_real = augment.noise_bed(4096, [], cfg, g1, real_bg=real)
    without = augment.noise_bed(4096, [], cfg, g2, real_bg=None)
    assert torch.allclose(with_real, without)  # weight 0 == original synthetic recipe


def test_pitch_shift_no_kernel_blowup_at_96k():
    """The reimplemented pitch shift must not build a giant resample kernel (OOM)."""
    from vocdenoiser.denoise import augment

    cfg = _tiny_cfg(sr=96_000, pitch_pct=5.0)
    g = torch.Generator().manual_seed(3)
    wav = torch.randn(48_000)  # ~0.5 s at 96 kHz
    out = augment.pitch_shift(wav, cfg, g)
    assert torch.isfinite(out).all() and out.numel() > 0


def _write_pcm_wav(path, y, sr):
    """16-bit PCM mono WAV via the stdlib (matches the real colony-noise files)."""
    import wave

    import numpy as np

    i16 = (np.clip(np.asarray(y), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(i16.tobytes())


def test_dataset_scans_and_mixes_real_noise(tmp_path):
    import torchaudio

    from vocdenoiser.denoise.dataset import PheeDenoiseDataset

    sr = 96_000
    calls = []
    for i in range(3):
        p = tmp_path / f"call{i}.wav"
        torchaudio.save(str(p), 0.3 * torch.randn(1, sr // 2), sr)
        calls.append(p)
    ndir = tmp_path / "Noise"
    ndir.mkdir()
    _write_pcm_wav(ndir / "bg.WAV", 0.3 * torch.randn(sr).numpy(), sr)  # uppercase .WAV, 16-bit PCM

    cfg = _tiny_cfg(sr=sr, noise_dirs=(str(ndir),), real_noise_weight=0.5)
    ds = PheeDenoiseDataset(cfg, calls)
    assert len(ds._noise_files) == 1  # found the uppercase .WAV via the stdlib header read

    bg = ds._real_background(cfg.waveform_len, torch.Generator().manual_seed(0))
    assert bg is not None and bg.numel() == cfg.waveform_len

    noisy, clean = ds[0]
    assert noisy.shape == tuple(cfg.spec_shape)
    assert clean.shape == tuple(cfg.spec_shape)


def test_dataset_warns_when_noise_dirs_empty(tmp_path, capsys):
    from vocdenoiser.denoise.dataset import PheeDenoiseDataset

    p = tmp_path / "call.wav"
    import torchaudio

    torchaudio.save(str(p), 0.3 * torch.randn(1, 48_000), 96_000)
    cfg = _tiny_cfg(noise_dirs=(str(tmp_path / "does_not_exist"),), real_noise_weight=0.5)
    PheeDenoiseDataset(cfg, [p])
    assert "synthetic noise only" in capsys.readouterr().out
