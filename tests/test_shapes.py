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
    torchaudio.save(str(ndir / "bg.wav"), 0.3 * torch.randn(1, sr), sr)
    (ndir / "bg.wav").rename(ndir / "bg.WAV")  # real data uses uppercase .WAV

    cfg = _tiny_cfg(sr=sr, noise_dirs=(str(ndir),), real_noise_weight=0.5)
    ds = PheeDenoiseDataset(cfg, calls)
    assert len(ds._noise_files) == 1  # found the .WAV via case-insensitive scan

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
