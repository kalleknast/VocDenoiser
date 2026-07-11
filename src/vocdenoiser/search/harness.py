"""The frozen evaluation harness.

A harness scores a :class:`Candidate` under a *fixed compute budget* and returns
one scalar metric (higher = better), seed-averaged. Nothing the search proposes
can change the harness — the data, the budget, and the metric are the invariant
that makes heterogeneous candidates comparable (the central autoresearch idea).

- :class:`MockHarness` — deterministic synthetic landscape; no torch. Used by the
  tests and for dry-running the loop locally without a GPU.
- :class:`TorchHarness` — real training of a candidate for a fixed budget and a
  spectrogram-domain SI-SDR metric on a pinned validation set (torch + GPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vocdenoiser.search.space import Candidate


@dataclass
class EvalResult:
    metric: float
    metric_std: float = 0.0
    num_params: int = 0
    peak_vram_mb: float = 0.0
    train_seconds: float = 0.0
    status: str = "keep"
    seeds: list[int] = field(default_factory=list)


class MockHarness:
    """Deterministic synthetic fitness landscape (no torch).

    The metric rewards proximity to a hidden optimum in knob space plus
    seed-dependent Gaussian noise, so the search visibly climbs and the
    noise-aware accept rule is genuinely exercised. Purely for tests / dry runs.
    """

    OPT = {
        "base_channels": 48,
        "latent_dim": 16,
        "beta": 3.0,
        "lr": 1e-3,
        "kernel_size": 4,
        "n_conv_layers": 4,
        "act": "silu",
        "norm": "group",
        "residual": True,
        "recon_loss": "l1",
    }

    def __init__(self, noise_std: float = 0.15) -> None:
        self.noise_std = noise_std

    def _clean_metric(self, c: Candidate) -> float:
        s = 0.0
        s -= abs(np.log(c.base_channels) - np.log(self.OPT["base_channels"]))
        s -= abs(np.log(c.latent_dim) - np.log(self.OPT["latent_dim"]))
        s -= abs(np.log(c.beta) - np.log(self.OPT["beta"]))
        s -= abs(np.log(c.lr) - np.log(self.OPT["lr"]))
        s -= 0.3 * abs(c.kernel_size - self.OPT["kernel_size"])
        s -= 0.3 * abs(c.n_conv_layers - self.OPT["n_conv_layers"])
        s += 0.4 * (c.act == self.OPT["act"])
        s += 0.4 * (c.norm == self.OPT["norm"])
        s += 0.3 * (c.residual == self.OPT["residual"])
        s += 0.3 * (c.recon_loss == self.OPT["recon_loss"])
        return 10.0 + s  # offset so metrics are comfortably positive

    def evaluate(self, c: Candidate, seeds: list[int]) -> EvalResult:
        vals = []
        for seed in seeds:
            rng = np.random.RandomState(abs(hash((c.id, seed))) % (2**32))
            vals.append(self._clean_metric(c) + rng.normal(0, self.noise_std))
        vals = np.array(vals)
        # crude param proxy for the simplicity tie-break
        nparams = c.base_channels * c.n_conv_layers * c.kernel_size**2 * 1000
        return EvalResult(
            metric=float(vals.mean()),
            metric_std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            num_params=int(nparams),
            train_seconds=0.0,
            status="keep",
            seeds=list(seeds),
        )


class TorchHarness:
    """Train a candidate for a fixed budget; metric = spectrogram SI-SDR (dB).

    Frozen invariants: the clean-call data + colony-noise recipe (from the
    denoise DataModule), the canonical spectrogram geometry (from the denoise
    Config — NOT searchable), the compute budget (``max_steps`` or ``max_time``),
    and the validation set. Only the model family + optimisation vary.
    """

    def __init__(
        self,
        base_config_overrides: dict | None = None,
        max_steps: int = 400,
        max_time: str | None = None,
        val_batches: int = 20,
        vram_cap_mb: float | None = None,
    ) -> None:
        self.base_overrides = base_config_overrides or {}
        self.max_steps = max_steps
        self.max_time = max_time
        self.val_batches = val_batches
        self.vram_cap_mb = vram_cap_mb

    def evaluate(self, c: Candidate, seeds: list[int]) -> EvalResult:
        import time

        import torch

        from vocdenoiser.denoise.config import Config
        from vocdenoiser.denoise.train import build_dataloaders
        from vocdenoiser.search.metric import spectrogram_si_sdr
        from vocdenoiser.search.model_factory import build_search_model

        metrics: list[float] = []
        nparams = 0
        peak_vram = 0.0
        t0 = time.time()
        for seed in seeds:
            torch.manual_seed(seed)
            overrides = {**self.base_overrides, **c.to_config_overrides(), "seed": seed}
            cfg = Config(**overrides)
            model = build_search_model(c, cfg)
            nparams = sum(p.numel() for p in model.parameters())
            _train_ds, train_dl, val_dl = build_dataloaders(cfg)
            try:
                import lightning as L

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                trainer = L.Trainer(
                    max_steps=self.max_steps,
                    max_time=self.max_time,
                    enable_checkpointing=False,
                    logger=False,
                    enable_progress_bar=False,
                    detect_anomaly=False,
                )
                trainer.fit(model, train_dl, val_dl)
                m = spectrogram_si_sdr(model, val_dl, max_batches=self.val_batches)
                if torch.cuda.is_available():
                    peak_vram = torch.cuda.max_memory_allocated() / 1e6
            except Exception as exc:  # noqa: BLE001 - a crashed candidate is logged, not fatal
                return EvalResult(
                    metric=float("-inf"), status="crash", seeds=list(seeds),
                    num_params=nparams, train_seconds=time.time() - t0,
                )
            if not np.isfinite(m):
                return EvalResult(metric=float("-inf"), status="crash", seeds=list(seeds),
                                  num_params=nparams, train_seconds=time.time() - t0)
            metrics.append(float(m))

        vals = np.array(metrics)
        status = "keep"
        if self.vram_cap_mb and peak_vram > self.vram_cap_mb:
            status = "discard"
        return EvalResult(
            metric=float(vals.mean()),
            metric_std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            num_params=int(nparams),
            peak_vram_mb=float(peak_vram),
            train_seconds=time.time() - t0,
            status=status,
            seeds=list(seeds),
        )
