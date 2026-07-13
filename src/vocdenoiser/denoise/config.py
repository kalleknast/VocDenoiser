"""Central configuration for the β-VAE denoiser.

Every tunable knob (data root, STFT/mel geometry, model width, optimiser, and
the colony-noise recipe constants from ``SPECS.md``) lives here so nothing is
scattered as a literal across the codebase. ``train.py`` / ``eval.py`` build a
:class:`Config` from CLI flags with an env-var fallback for the data root, which
keeps the pipeline environment-agnostic (Colab Drive *or* a local box) — no path
is ever hardcoded.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

# Downsampling factor of the 4-layer, stride-2 encoder (2**4). The spectrogram's
# mel and frame dimensions must be divisible by this so the symmetric decoder
# reconstructs the exact input shape without interpolation.
DOWNSAMPLE = 16

DATA_ROOT_ENV = "VOCDENOISER_DATA_ROOT"
# In-repo clean phee-call set (50k isolated calls). Used as the final fallback
# when neither --data-root nor $VOCDENOISER_DATA_ROOT is given. Relative to CWD,
# so it stays portable across Colab / local box.
DEFAULT_DATA_ROOT = "data/Vocalizations"

# Env-var fallback for the durable-OUTPUT root (checkpoints, search ledger, SNR
# scans). Set it to a mounted Drive folder on Colab so progress survives a runtime
# reset; unset -> outputs stay repo-relative and the local box is unaffected.
OUTPUT_ROOT_ENV = "VOCDENOISER_OUTPUT_ROOT"


@dataclass
class Config:
    """All hyperparameters for data synthesis, model, and training."""

    # --- Data -------------------------------------------------------------
    # Root holding the *clean* isolated phee-call WAVs (the off-repo training
    # set — see CLAUDE.local.md). NOT data/Noise/, which is colony noise only.
    data_root: str | None = None
    # Real recorded colony-noise dirs, blended into the noise bed alongside the
    # synthetic recipe (see ``real_noise_weight``). ``data/Noise`` = ventilation /
    # colony background, ``data/Cigarra`` = cicada. Override on Colab, e.g.
    # ``--noise-dirs /content/Noise /content/Cigarra``. WAV/.WAV, read per item.
    noise_dirs: tuple[str, ...] = ("data/Noise", "data/Cigarra")
    # Glob used to discover clean calls under ``data_root`` (recursive).
    clean_glob: str = "**/*.wav"

    # --- Audio / spectrogram ---------------------------------------------
    # Working rate. The archive is 96 kHz 16-bit PCM (SPECS.md), but phee energy is
    # essentially all below ~22 kHz, so we operate at 44.1 kHz: 2.2x less data/IO,
    # finer in-band mel resolution, and a native rate match to InfantMarmosetsVox
    # (no upsample artifact in the cross-dataset benchmark). DEVIATION from SPECS.md's
    # 96 kHz — flagged deliberately.
    sr: int = 44_100
    # Safety net: any WAV whose header rate != this is resampled to it before the
    # STFT, so a stray 96 kHz file (e.g. the un-downsampled original archive) is
    # auto-converted; a no-op on audio already at 44.1 kHz. Pre-downsample with
    # `python -m vocdenoiser.datasets.resample` to skip the per-item resample cost.
    resample_sr: int | None = 44_100
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    n_frames: int = 256  # fixed time dimension (crop/pad); ~1.5 s at 44.1 kHz / hop 256
    f_min: float = 1000.0  # phee energy is >~5 kHz; also avoids empty low-freq mel bands
    f_max: float | None = None  # None -> effective_sr / 2
    # Log-mel dB normalisation window: (db - db_min) / (db_ref - db_min), clamped.
    db_ref: float = 0.0
    db_min: float = -80.0

    # --- Model ------------------------------------------------------------
    latent_dim: int = 16  # scientifically validated for marmoset similarity (SPECS)
    base_channels: int = 32  # encoder ch = [32, 64, 128, 256]
    beta: float = 4.0  # KL weight in L = recon + beta * D_KL
    # Reconstruction term: "l1" (sharper spectrograms, robust to outliers — the
    # architecture search preferred it over MSE) or "l2" (Gaussian NLL / MSE).
    recon_loss: str = "l1"
    # KL schedule: "warmup" ramps beta 0 -> beta over the first `beta_warmup_epochs`
    # epochs so reconstruction is learned before KL pressure engages (guards against
    # early posterior collapse); "const" holds beta fixed.
    beta_schedule: str = "warmup"
    beta_warmup_epochs: int = 5

    # --- Optimisation -----------------------------------------------------
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "adamw"  # "adam" or "adamw" (decoupled weight decay)
    weight_decay: float = 1e-4  # AdamW decoupled L2; 0 makes AdamW behave like Adam
    max_epochs: int = 100
    # Early stopping on val_loss: halt after this many epochs with no improvement
    # (0 disables — train the full max_epochs). val_loss plateaus by ~epoch 5-10 on
    # this data, so this typically saves most of the 100-epoch budget.
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.0  # min val_loss decrease that counts as improvement
    num_workers: int = 4
    val_frac: float = 0.1
    ckpt_dir: str = "checkpoints"
    # Root for durable OUTPUTS. Resolved --output-root > $VOCDENOISER_OUTPUT_ROOT >
    # "." (repo-relative). Point at a Drive mount on Colab so checkpoints survive a
    # reset; never hardcoded, so the same command runs locally unchanged.
    output_root: str | None = None
    seed: int = 42

    # --- Colony-noise recipe (SPECS.md / noise-recipe skill) --------------
    # Composite SYNTHETIC noise-bed mixing weights: 40% pink, 50% babble, 10% transients.
    weight_pink: float = 0.40
    weight_babble: float = 0.50
    weight_transient: float = 0.10
    # Blend of the REAL recorded background (from ``noise_dirs``) into the bed:
    # 0 = synthetic only (the SPECS recipe), 1 = real recordings only, 0.5 = both
    # equally. Extension beyond SPECS.md — real-noise mixing is a deliberate
    # addition (natural cicada/colony backgrounds alongside the synthetic bed).
    real_noise_weight: float = 0.50
    snr_db_min: float = -5.0  # dynamic SNR sampled uniformly in [-5, +15] dB
    snr_db_max: float = 15.0
    max_offset_ms: float = 500.0  # temporal offsets 0–500 ms between sources
    babble_min_calls: int = 5  # babble mixes 5–10 attenuated shifted calls
    babble_max_calls: int = 10
    babble_atten_db: tuple[float, float] = (-20.0, -6.0)  # per-call attenuation range
    transient_ms_min: float = 5.0  # white-noise bursts 5–20 ms
    transient_ms_max: float = 20.0
    n_transients: tuple[int, int] = (2, 6)  # bursts per clip
    # Bioacoustic perturbations applied to the call (shared by input & target so
    # they stay time-aligned — noise is only added to the input).
    pitch_pct: float = 5.0  # pitch shift ±5%
    stretch_pct: float = 10.0  # time stretch ±10%
    augment: bool = True  # master switch for noise + perturbations (off for eval)

    _extras: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.n_mels % DOWNSAMPLE or self.n_frames % DOWNSAMPLE:
            raise ValueError(
                f"n_mels ({self.n_mels}) and n_frames ({self.n_frames}) must be "
                f"divisible by {DOWNSAMPLE} for the symmetric encoder/decoder."
            )

    # --- Derived quantities ----------------------------------------------
    @property
    def effective_sr(self) -> int:
        """Sample rate the STFT actually sees (after optional resampling)."""
        return self.resample_sr or self.sr

    @property
    def effective_f_max(self) -> float:
        return self.f_max if self.f_max is not None else self.effective_sr / 2

    @property
    def waveform_len(self) -> int:
        """Waveform length (samples) that yields exactly ``n_frames`` mel frames.

        torchaudio's MelSpectrogram uses ``center=True`` (reflection pad of
        ``n_fft//2`` each side), so ``frames ≈ 1 + len // hop``.
        """
        return (self.n_frames - 1) * self.hop

    @property
    def spec_shape(self) -> tuple[int, int, int]:
        """(channels, mels, frames) of a single spectrogram sample."""
        return (1, self.n_mels, self.n_frames)

    def resolved_data_root(self) -> Path:
        """Clean-call root, resolved as --data-root > $VOCDENOISER_DATA_ROOT > default.

        Points at the isolated phee-call WAVs (``data/Vocalizations`` by default),
        NOT ``data/Noise`` — that folder is colony noise for augmentation only. For
        training speed, copy the calls to local disk first rather than reading off
        the pCloud FUSE mount.
        """
        root = self.data_root or os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT
        return Path(root).expanduser()

    def resolved_noise_dirs(self) -> list[Path]:
        """Real-noise dirs as expanded Paths (may not all exist; caller checks)."""
        return [Path(d).expanduser() for d in self.noise_dirs]

    def resolved_output_root(self) -> Path:
        """Durable-output root: --output-root > $VOCDENOISER_OUTPUT_ROOT > CWD."""
        root = self.output_root or os.environ.get(OUTPUT_ROOT_ENV)
        return Path(root).expanduser() if root else Path(".")

    def resolved_ckpt_dir(self) -> Path:
        """Checkpoint dir under the output root (an absolute ``ckpt_dir`` overrides)."""
        return self.resolved_output_root() / self.ckpt_dir

    # --- CLI plumbing -----------------------------------------------------
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Register a --flag for every scalar config field (dashes for underscores)."""
        defaults = Config()
        for f in fields(Config):
            if f.name.startswith("_") or f.name in {"babble_atten_db", "n_transients"}:
                continue
            default = getattr(defaults, f.name)
            flag = "--" + f.name.replace("_", "-")
            if f.type == "bool" or isinstance(default, bool):
                parser.add_argument(
                    flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None
                )
            elif isinstance(default, (list, tuple)):
                parser.add_argument(flag, dest=f.name, nargs="+", default=None)
            else:
                parser.add_argument(flag, dest=f.name, default=None)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Config:
        """Build a Config, coercing provided CLI strings to each field's type."""
        defaults = cls()
        kwargs: dict = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            val = getattr(args, f.name, None)
            if val is None:
                continue
            default = getattr(defaults, f.name)
            if isinstance(default, bool):
                kwargs[f.name] = bool(val)
            elif isinstance(default, (list, tuple)):
                kwargs[f.name] = tuple(val)
            elif isinstance(default, int) and not isinstance(default, bool):
                kwargs[f.name] = int(val)
            elif isinstance(default, float):
                kwargs[f.name] = float(val)
            else:
                kwargs[f.name] = val
        return cls(**kwargs)
