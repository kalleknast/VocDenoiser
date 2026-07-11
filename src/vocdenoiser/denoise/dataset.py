"""Dataset yielding ``(noisy_spec, clean_spec)`` log-mel pairs.

Each item:
  1. load a clean isolated phee call (WAV) and, if configured, resample it;
  2. optionally apply a shared bioacoustic perturbation (pitch/stretch) — this
     becomes the aligned *call*;
  3. the perturbed call → clean log-mel (**target**);
  4. call + synthetic colony-noise bed at a sampled SNR → noisy log-mel (**input**).

Augmentation RNG is derived per (base seed, epoch, index) so a given epoch is
exactly replayable, yet every epoch re-randomises. Set ``cfg.augment = False``
(or use :meth:`eval_view`) to emit clean→clean pairs for latent extraction.

IO note: WAVs are read from ``cfg.data_root`` on ``__getitem__``. Point that at a
**local-disk** copy of the clean set — reading off the pCloud FUSE mount in a
training loop is the slow path this project explicitly avoids.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from vocdenoiser.denoise import augment
from vocdenoiser.denoise.config import Config


def list_clean_calls(cfg: Config) -> list[Path]:
    """Discover clean-call WAVs under the resolved data root (sorted, recursive)."""
    root = cfg.resolved_data_root()
    files = sorted(root.glob(cfg.clean_glob))
    if not files:
        raise FileNotFoundError(
            f"No WAVs matching '{cfg.clean_glob}' under {root}. "
            "Is --data-root pointing at the clean phee-call set?"
        )
    return files


class PheeDenoiseDataset(Dataset):
    """(noisy_spec, clean_spec) pairs synthesised from clean phee calls."""

    def __init__(
        self,
        cfg: Config,
        files: list[Path],
        *,
        epoch: int = 0,
        babble_pool_size: int = 12,
    ) -> None:
        import torchaudio

        self.cfg = cfg
        self.files = files
        self.epoch = epoch
        self.babble_pool_size = babble_pool_size
        self._ta = torchaudio
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.effective_sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.effective_f_max,
            power=2.0,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self) -> int:
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation RNG stream (call from a training callback)."""
        self.epoch = epoch

    # --- audio helpers ----------------------------------------------------
    def _load(self, path: Path) -> torch.Tensor:
        wav, sr = self._ta.load(str(path))
        wav = wav.mean(dim=0)  # downmix to mono
        if self.cfg.resample_sr and sr != self.cfg.resample_sr:
            wav = self._ta.functional.resample(wav, sr, self.cfg.resample_sr)
        return wav.float()

    def _generator(self, index: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.cfg.seed * 1_000_003 + self.epoch * 9973 + index)
        return g

    def _log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform → normalised log-mel of shape (1, n_mels, n_frames)."""
        wav = augment._fit_length(wav, self.cfg.waveform_len)
        db = self._to_db(self._mel(wav))  # (n_mels, ~n_frames)
        db = db[:, : self.cfg.n_frames]
        if db.shape[1] < self.cfg.n_frames:
            db = torch.nn.functional.pad(db, (0, self.cfg.n_frames - db.shape[1]))
        norm = (db - self.cfg.db_min) / (self.cfg.db_ref - self.cfg.db_min)
        return norm.clamp(0.0, 1.0).unsqueeze(0)

    def _babble_pool(self, index: int, generator: torch.Generator) -> list[torch.Tensor]:
        pool: list[torch.Tensor] = []
        n = min(self.babble_pool_size, len(self.files))
        for _ in range(n):
            j = augment._randint(0, len(self.files) - 1, generator)
            if j == index:
                continue
            try:
                pool.append(self._load(self.files[j]))
            except Exception:  # noqa: BLE001 - a bad file must not kill a batch
                continue
        return pool

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        g = self._generator(index)
        call = self._load(self.files[index])

        if cfg.augment:
            if cfg.stretch_pct:
                call = augment.time_stretch(call, cfg, g)
            if cfg.pitch_pct:
                call = augment.pitch_shift(call, cfg, g)

        call = augment._fit_length(call, cfg.waveform_len)
        clean_spec = self._log_mel(call)

        if not cfg.augment:
            return clean_spec, clean_spec  # clean→clean for eval / latent extraction

        pool = self._babble_pool(index, g)
        bed = augment.noise_bed(call.numel(), pool, cfg, g)
        snr_db = augment._uniform(cfg.snr_db_min, cfg.snr_db_max, g)
        noisy = augment.mix_at_snr(call, bed, snr_db)
        noisy_spec = self._log_mel(noisy)
        return noisy_spec, clean_spec
