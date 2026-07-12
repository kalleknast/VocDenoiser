"""2D convolutional β-VAE for spectrogram denoising (SPECS.md architecture).

Encoder: 4 stride-2 conv layers (channels 32→64→128→256) reduce the input
log-mel to a flat vector, then two linear heads produce the 16-dim ``mu`` and
``logvar``. Decoder: a symmetric mirror (linear → 4 stride-2 transposed convs)
reconstructing the cleaned spectrogram at the exact input shape.

Objective (supervised noisy→clean):

    L = MSE(x̂, x_clean) + β · D_KL( q(z|x_noisy) ‖ N(0, I) )

The encoder consumes the *noisy* spectrogram; the reconstruction target is the
*clean* one. With ``β`` configurable, ``β = 1`` recovers a vanilla VAE.
"""

from __future__ import annotations

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
    def loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse = F.mse_loss(recon, target, reduction="mean")
        # KL of q(z|x) ‖ N(0, I): sum over latent dims, mean over the batch.
        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        # Weight the reconstruction by the number of spectrogram bins so it is on
        # the same scale as the latent-summed KL (equivalently: sum the Gaussian
        # log-likelihood over bins). Without this, MSE averaged over ~32k bins is
        # ~n_bins smaller than β·KL, so the objective is >99.9% KL and the model
        # collapses the posterior instead of learning to denoise. `mse`/`kl` are
        # still returned unweighted for readable logging.
        n_bins = target[0].numel()
        total = n_bins * mse + self.cfg.beta * kl
        return total, mse, kl

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        noisy, clean = batch
        recon, mu, logvar = self(noisy)
        total, mse, kl = self.loss(recon, clean, mu, logvar)
        bs = noisy.size(0)
        self.log_dict(
            {f"{stage}_loss": total, f"{stage}_mse": mse, f"{stage}_kl": kl},
            prog_bar=(stage == "val"),
            batch_size=bs,
        )
        return total

    def training_step(self, batch, _batch_idx):  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch, _batch_idx):  # noqa: D102
        return self._step(batch, "val")

    def configure_optimizers(self):  # noqa: D102
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
