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
