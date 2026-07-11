"""Typed candidate grammar: the searchable architecture + hyperparameters.

A :class:`Candidate` is a validated, serializable description of one denoiser
configuration — the autoresearch "object" the search mutates. The spectrogram
geometry (n_mels / n_frames / n_fft / hop) is deliberately NOT here: it belongs
to the frozen harness so every candidate is scored in the same canonical domain
(otherwise metrics across candidates aren't comparable). What the search *does*
control is the model family and optimisation.

Everything is numpy/stdlib so the grammar, mutation and crossover are testable
without torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

# Discrete choice sets and continuous (log-uniform) ranges for each knob.
CHOICES = {
    "n_conv_layers": [3, 4, 5],
    "base_channels": [16, 24, 32, 48, 64],
    "channel_mult": [1.5, 2.0, 2.5],
    "kernel_size": [3, 4, 5],
    "norm": ["batch", "group", "none"],
    "act": ["leaky_relu", "gelu", "silu"],
    "latent_dim": [8, 16, 24, 32],  # 16 is the SPECS default; searchable
    "residual": [False, True],
    "optimizer": ["adam", "adamw"],
    "batch_size": [16, 32, 64],
    "recon_loss": ["l1", "l2"],
    "beta_schedule": ["const", "warmup"],
}
LOG_RANGES = {
    "lr": (1e-4, 3e-3),
    "beta": (0.5, 8.0),
    "weight_decay": (1e-6, 1e-2),
    "dropout": (0.0, 0.3),  # linear, not log; handled specially
}


@dataclass(frozen=True)
class Candidate:
    """One denoiser configuration. Frozen + hashable for ledger de-duplication."""

    # architecture
    n_conv_layers: int = 4
    base_channels: int = 32
    channel_mult: float = 2.0
    kernel_size: int = 4
    norm: str = "batch"
    act: str = "leaky_relu"
    latent_dim: int = 16
    residual: bool = False
    dropout: float = 0.0
    # loss
    beta: float = 4.0
    beta_schedule: str = "const"
    recon_loss: str = "l2"
    # optimisation
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    # bookkeeping (not part of the identity hash)
    parent_id: str | None = field(default=None, compare=False)
    origin: str = field(default="seed", compare=False)  # seed|random|mutate|crossover|llm

    def validate(self) -> None:
        assert self.n_conv_layers in CHOICES["n_conv_layers"], self.n_conv_layers
        assert self.base_channels > 0 and self.latent_dim > 0
        assert self.kernel_size in CHOICES["kernel_size"]
        assert self.norm in CHOICES["norm"] and self.act in CHOICES["act"]
        assert self.optimizer in CHOICES["optimizer"]
        assert self.recon_loss in CHOICES["recon_loss"]
        assert 0.0 <= self.dropout < 1.0
        assert self.lr > 0 and self.beta >= 0 and self.weight_decay >= 0

    @property
    def id(self) -> str:
        """Stable short hash of the *identity* fields (excludes bookkeeping)."""
        d = {k: v for k, v in asdict(self).items() if k not in {"parent_id", "origin"}}
        blob = json.dumps(d, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def to_config_overrides(self) -> dict:
        """Map to :class:`vocdenoiser.denoise.config.Config` field overrides.

        Only fields the denoise Config understands are emitted; the extended
        architecture knobs (n_conv_layers, norm, act, ...) are consumed by the
        search model factory, which builds a configurable model from the full
        candidate.
        """
        return {
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


# --- samplers / operators (all take an explicit RandomState for reproducibility) ---


def _log_uniform(lo: float, hi: float, rng: np.random.RandomState) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_candidate(rng: np.random.RandomState, origin: str = "random") -> Candidate:
    c = Candidate(
        n_conv_layers=int(rng.choice(CHOICES["n_conv_layers"])),
        base_channels=int(rng.choice(CHOICES["base_channels"])),
        channel_mult=float(rng.choice(CHOICES["channel_mult"])),
        kernel_size=int(rng.choice(CHOICES["kernel_size"])),
        norm=str(rng.choice(CHOICES["norm"])),
        act=str(rng.choice(CHOICES["act"])),
        latent_dim=int(rng.choice(CHOICES["latent_dim"])),
        residual=bool(rng.choice(CHOICES["residual"])),
        dropout=float(rng.uniform(*LOG_RANGES["dropout"])),
        beta=_log_uniform(*LOG_RANGES["beta"], rng),
        beta_schedule=str(rng.choice(CHOICES["beta_schedule"])),
        recon_loss=str(rng.choice(CHOICES["recon_loss"])),
        optimizer=str(rng.choice(CHOICES["optimizer"])),
        lr=_log_uniform(*LOG_RANGES["lr"], rng),
        weight_decay=_log_uniform(*LOG_RANGES["weight_decay"], rng),
        batch_size=int(rng.choice(CHOICES["batch_size"])),
        origin=origin,
    )
    return c


def mutate(parent: Candidate, rng: np.random.RandomState, n_edits: int = 1) -> Candidate:
    """Change ``n_edits`` knob(s) of ``parent`` — a local move in the search space."""
    fields = list(CHOICES) + list(LOG_RANGES)
    picks = rng.choice(fields, size=min(n_edits, len(fields)), replace=False)
    changes: dict = {}
    for f in picks:
        if f in CHOICES:
            changes[f] = type(getattr(parent, f))(rng.choice(CHOICES[f]))
        elif f == "dropout":
            changes[f] = float(rng.uniform(*LOG_RANGES["dropout"]))
        else:
            changes[f] = _log_uniform(*LOG_RANGES[f], rng)
    return replace(parent, parent_id=parent.id, origin="mutate", **changes)


def crossover(a: Candidate, b: Candidate, rng: np.random.RandomState) -> Candidate:
    """Uniform crossover: each knob taken from ``a`` or ``b`` at random."""
    changes = {}
    for f in list(CHOICES) + list(LOG_RANGES):
        changes[f] = getattr(a if rng.random() < 0.5 else b, f)
    return replace(a, parent_id=f"{a.id}+{b.id}", origin="crossover", **changes)
