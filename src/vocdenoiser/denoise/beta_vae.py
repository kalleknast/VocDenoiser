"""2D convolutional β-VAE for spectrogram denoising (SPECS.md architecture).

Encoder: 4 stride-2 conv layers (channels 32→64→128→256) reduce the input
log-mel to a flat vector, then two linear heads produce the 16-dim ``mu`` and
``logvar``. Decoder: a symmetric mirror (linear → 4 stride-2 transposed convs)
reconstructing the cleaned spectrogram at the exact input shape.

Objective (supervised noisy→clean):

    L = recon(x̂, x_clean) + β · D_KL( q(z|x_noisy) ‖ N(0, I) )

where ``recon`` is L1 (default) or MSE and ``β`` may be linearly warmed up over
the first few epochs (see :class:`Config`). The encoder consumes the *noisy*
spectrogram; the reconstruction target is the *clean* one. With a constant
``β = 1`` and L2, this reduces to a vanilla VAE.
"""

from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from vocdenoiser.denoise.config import DOWNSAMPLE, Config


class BetaVAE(L.LightningModule):
    """Convolutional β-VAE LightningModule."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        c = cfg.base_channels
        chans = [1, c, c * 2, c * 4, c * 8]
        self._enc_h = cfg.n_mels // DOWNSAMPLE
        self._enc_w = cfg.n_frames // DOWNSAMPLE
        self._enc_c = chans[-1]
        self._flat = self._enc_c * self._enc_h * self._enc_w

        enc = []
        for i in range(4):
            enc += [
                nn.Conv2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(self._flat, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self._flat, cfg.latent_dim)

        self.fc_dec = nn.Linear(cfg.latent_dim, self._flat)
        dec = []
        for i in range(4, 0, -1):
            out_ch = chans[i - 1] if i > 1 else chans[0]
            dec += [
                nn.ConvTranspose2d(chans[i], out_ch, kernel_size=4, stride=2, padding=1),
                # No BN/activation after the final layer -> linear log-mel output.
                *(
                    [nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)]
                    if i > 1
                    else []
                ),
            ]
        self.decoder = nn.Sequential(*dec)

    # --- core VAE ---------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        # Clamp logvar so exp(logvar) / exp(0.5·logvar) can't overflow to inf and
        # NaN out the run (a single overshoot past logvar≈88 kills float32). The
        # bounds are wide enough to never bind in a healthy run.
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return self.fc_mu(h), logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self._enc_c, self._enc_h, self._enc_w)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # --- loss / steps -----------------------------------------------------
    def _beta(self) -> float:
        """Current KL weight. ``warmup`` ramps 0 → ``cfg.beta`` linearly over the
        first ``cfg.beta_warmup_epochs`` epochs so the model learns to reconstruct
        before KL pressure engages (mitigates early posterior collapse); ``const``
        holds it fixed."""
        if self.cfg.beta_schedule != "warmup" or self.trainer is None:
            return self.cfg.beta
        warm_epochs = max(1, self.cfg.beta_warmup_epochs)
        # Prefer a smooth per-step ramp, but trainer.num_training_batches is +inf
        # until the train loop is set up (e.g. during the pre-training validation
        # sanity check, where int(inf) would overflow), so fall back to a coarse
        # per-epoch ramp whenever it isn't a finite positive count yet.
        spe = getattr(self.trainer, "num_training_batches", float("inf"))
        if math.isfinite(spe) and spe > 0:
            frac = (self.global_step + 1) / (warm_epochs * spe)
        else:
            frac = self.current_epoch / warm_epochs
        return self.cfg.beta * min(1.0, frac)

    def loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # L1 (default) gives sharper, more outlier-robust spectrogram reconstructions
        # than MSE; both reduce over all bins. The search preferred L1 decisively.
        if self.cfg.recon_loss == "l1":
            recon_err = F.l1_loss(recon, target, reduction="mean")
        else:
            recon_err = F.mse_loss(recon, target, reduction="mean")
        # KL of q(z|x) ‖ N(0, I): sum over latent dims, mean over the batch.
        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        # Weight the reconstruction by the number of spectrogram bins so it is on
        # the same scale as the latent-summed KL (equivalently: sum the per-bin recon
        # log-likelihood). Without this, a per-bin-mean recon (~1/n_bins) is dwarfed
        # by β·KL, so the objective is >99.9% KL and the model collapses the posterior
        # instead of learning to denoise. `recon_err`/`kl` are returned unweighted for
        # readable logging.
        n_bins = target[0].numel()
        total = n_bins * recon_err + self._beta() * kl
        return total, recon_err, kl

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        noisy, clean = batch
        recon, mu, logvar = self(noisy)
        total, recon_err, kl = self.loss(recon, clean, mu, logvar)
        bs = noisy.size(0)
        self.log_dict(
            {f"{stage}_loss": total, f"{stage}_recon": recon_err, f"{stage}_kl": kl},
            prog_bar=(stage == "val"),
            batch_size=bs,
        )
        return total

    def training_step(self, batch, _batch_idx):  # noqa: D102
        loss = self._step(batch, "train")
        # A single degenerate/NaN batch must not poison the weights: gradient
        # clipping caps magnitude but does NOT sanitize a NaN, so skip the optimizer
        # step (return None) whenever the loss is non-finite.
        if not torch.isfinite(loss):
            return None
        return loss

    def validation_step(self, batch, _batch_idx):  # noqa: D102
        return self._step(batch, "val")

    def configure_optimizers(self):  # noqa: D102
        opt_cls = torch.optim.AdamW if self.cfg.optimizer == "adamw" else torch.optim.Adam
        return opt_cls(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
