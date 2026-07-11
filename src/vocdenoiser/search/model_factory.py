"""Build a configurable β-VAE LightningModule from a :class:`Candidate`.

This is the search's *object level*: a family of conv β-VAEs parameterised by the
candidate grammar (depth, width, kernel, norm, activation, residual, dropout,
loss, optimiser). It deliberately mirrors the interface of the hand-designed
reference model in ``vocdenoiser.denoise.beta_vae`` (same ``(noisy, clean)``
batches, same ``val_loss`` logging) so it drops into the existing
``build_dataloaders`` / Lightning ``Trainer`` plumbing unchanged.

Requires torch + lightning; imported lazily by the harness so the rest of the
search package stays torch-free.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from vocdenoiser.denoise.config import Config
from vocdenoiser.search.space import Candidate

_ACT = {"leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True), "gelu": nn.GELU, "silu": nn.SiLU}


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    return nn.Identity()


class ConfigurableBetaVAE(L.LightningModule):
    """A β-VAE whose depth/width/kernel/norm/act/residual come from a Candidate."""

    def __init__(self, cand: Candidate, cfg: Config) -> None:
        super().__init__()
        self.cand = cand
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cand", "cfg"])

        n = cand.n_conv_layers
        ds = 2**n
        if cfg.n_mels % ds or cfg.n_frames % ds:
            raise ValueError(
                f"n_mels/n_frames ({cfg.n_mels}x{cfg.n_frames}) must be divisible by "
                f"2**n_conv_layers ({ds}) for candidate {cand.id}"
            )
        chans = [1] + [int(round(cand.base_channels * cand.channel_mult**i)) for i in range(n)]
        pad = cand.kernel_size // 2
        act = _ACT[cand.act]

        enc: list[nn.Module] = []
        for i in range(n):
            enc.append(
                nn.Conv2d(chans[i], chans[i + 1], cand.kernel_size, stride=2, padding=pad)
            )
            enc.append(_norm(cand.norm, chans[i + 1]))
            enc.append(act())
            if cand.dropout:
                enc.append(nn.Dropout2d(cand.dropout))
        self.encoder = nn.Sequential(*enc)

        self._enc_c = chans[-1]
        self._enc_h = cfg.n_mels // ds
        self._enc_w = cfg.n_frames // ds
        self._flat = self._enc_c * self._enc_h * self._enc_w
        self.fc_mu = nn.Linear(self._flat, cand.latent_dim)
        self.fc_logvar = nn.Linear(self._flat, cand.latent_dim)
        self.fc_dec = nn.Linear(cand.latent_dim, self._flat)

        dec: list[nn.Module] = []
        for i in range(n, 0, -1):
            out_ch = chans[i - 1]
            dec.append(
                nn.ConvTranspose2d(
                    chans[i], out_ch, cand.kernel_size, stride=2, padding=pad,
                    output_padding=1,
                )
            )
            if i > 1:
                dec.append(_norm(cand.norm, out_ch))
                dec.append(act())
        self.decoder = nn.Sequential(*dec)
        self.residual = cand.residual

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self._enc_c, self._enc_h, self._enc_w)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        recon = self.decode(self.reparameterize(mu, logvar))
        if self.residual:  # predict a residual correction to the noisy input
            recon = recon + x
        return recon, mu, logvar

    def _beta(self) -> float:
        if self.cand.beta_schedule == "warmup" and self.trainer is not None:
            frac = min(1.0, (self.global_step + 1) / max(1, self.trainer.max_steps or 1))
            return self.cand.beta * frac
        return self.cand.beta

    def _loss(self, recon, target, mu, logvar):
        recon_loss = (
            F.l1_loss(recon, target) if self.cand.recon_loss == "l1"
            else F.mse_loss(recon, target)
        )
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + self._beta() * kl, recon_loss, kl

    def _step(self, batch, stage):
        noisy, clean = batch
        recon, mu, logvar = self(noisy)
        total, rec, kl = self._loss(recon, clean, mu, logvar)
        self.log_dict(
            {f"{stage}_loss": total, f"{stage}_recon": rec, f"{stage}_kl": kl},
            prog_bar=(stage == "val"), batch_size=noisy.size(0),
        )
        return total

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW if self.cand.optimizer == "adamw" else torch.optim.Adam
        return opt(self.parameters(), lr=self.cand.lr, weight_decay=self.cand.weight_decay)


def build_search_model(cand: Candidate, cfg: Config) -> ConfigurableBetaVAE:
    cand.validate()
    return ConfigurableBetaVAE(cand, cfg)
