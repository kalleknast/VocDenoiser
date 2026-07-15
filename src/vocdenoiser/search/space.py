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
# The upper bounds here drop n_conv_layers=5 / base_channels=64, which removed two observed
# crash modes. They do NOT bound model size: 20% of this space still exceeds 4M params and
# the largest member is 10.7M (n_conv_layers=3, base_channels=48, channel_mult=2.0,
# kernel_size=5, latent_dim=32). Size is enforced separately, against a computed param count
# — see MAX_PARAMS / estimate_params.
CHOICES = {
    "n_conv_layers": [3, 4],
    "base_channels": [16, 24, 32, 48],
    "channel_mult": [1.5, 2.0],
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

# Size ceiling, in parameters (the hand model's scale). A fixed per-candidate budget cannot
# train a 10M-param net to convergence, so oversized models score *undertrained* — the metric
# degrades into a proxy for convergence speed (params↔metric anti-correlated at ≈-0.54 in the
# 70-record ledger) rather than architecture quality. Raise this only alongside a much larger
# training budget.
MAX_PARAMS = 4_000_000

# Frozen spectrogram geometry, mirrored from denoise.config.Config solely to *size* a
# candidate without importing torch (this module stays dependency-light so the grammar and
# operators are testable on CPU). The search does not control the geometry — the harness
# does — so duplicating the two frozen values here cannot desync the scoring domain; it only
# needs to stay in step with Config if that geometry is ever re-frozen.
GEOM_N_MELS = 128
GEOM_N_FRAMES = 256

# Rejection-sampling attempts before an operator gives up. ~80% of the space fits under the
# default MAX_PARAMS, so this exhausts only if max_params is set below the space minimum.
_MAX_TRIES = 100


def estimate_params(
    c: Candidate, n_mels: int = GEOM_N_MELS, n_frames: int = GEOM_N_FRAMES
) -> int:
    """Exact trainable-parameter count of the model ``build_search_model(c)`` would build.

    Mirrors :class:`~vocdenoiser.search.model_factory.ConfigurableBetaVAE`'s construction
    arithmetic in pure Python — no torch, so the operators can reject oversized candidates
    *before* paying to build and train them. Verified to reproduce the ledger's recorded
    ``num_params`` exactly across every architecture family in the space; if the factory's
    layer layout changes, this must change with it (``test_estimate_params_matches_factory``
    is the torch-gated guard).

    Note the dominant term is the dense bottleneck (fc_mu/fc_logvar/fc_dec over the flattened
    encoder output), which shrinks 4x per extra conv layer. So *fewer* layers means a *bigger*
    model — which is why bounding n_conv_layers from above cannot bound size.
    """
    n = c.n_conv_layers
    chans = [1] + [int(round(c.base_channels * c.channel_mult**i)) for i in range(n)]
    k, affine = c.kernel_size, c.norm in ("batch", "group")
    p = 0
    for i in range(n):  # encoder: Conv2d + optional norm (act/dropout are parameter-free)
        p += k * k * chans[i] * chans[i + 1] + chans[i + 1]
        if affine:
            p += 2 * chans[i + 1]
    ds = 2**n
    flat = chans[-1] * (n_mels // ds) * (n_frames // ds)
    p += 2 * (flat * c.latent_dim + c.latent_dim)  # fc_mu, fc_logvar
    p += c.latent_dim * flat + flat  # fc_dec
    for i in range(n, 0, -1):  # decoder: ConvTranspose2d, norm+act on all but the last
        p += k * k * chans[i] * chans[i - 1] + chans[i - 1]
        if i > 1 and affine:
            p += 2 * chans[i - 1]
    return p


def fits(c: Candidate, max_params: int | None = MAX_PARAMS) -> bool:
    """Whether ``c`` is within the size ceiling (``max_params=None`` disables the check)."""
    return max_params is None or estimate_params(c) <= max_params


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


def random_candidate(
    rng: np.random.RandomState, origin: str = "random", max_params: int | None = MAX_PARAMS
) -> Candidate:
    """Sample uniformly from the space, rejecting draws over ``max_params``."""
    for _ in range(_MAX_TRIES):
        c = _sample(rng, origin)
        if fits(c, max_params):
            return c
    raise ValueError(
        f"no candidate under max_params={max_params} in {_MAX_TRIES} draws; the smallest "
        f"model in CHOICES is ~{230_767:,} params — raise the cap"
    )


def _sample(rng: np.random.RandomState, origin: str) -> Candidate:
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


def mutate(
    parent: Candidate,
    rng: np.random.RandomState,
    n_edits: int = 1,
    max_params: int | None = MAX_PARAMS,
) -> Candidate:
    """Change ``n_edits`` knob(s) of ``parent`` — a local move in the search space.

    A single edit can blow the size ceiling even from a compact parent (dropping
    n_conv_layers 4->3 quadruples the dense bottleneck), so oversized edits are re-drawn.
    """
    fields = list(CHOICES) + list(LOG_RANGES)
    for _ in range(_MAX_TRIES):
        picks = rng.choice(fields, size=min(n_edits, len(fields)), replace=False)
        changes: dict = {}
        for f in picks:
            if f in CHOICES:
                changes[f] = type(getattr(parent, f))(rng.choice(CHOICES[f]))
            elif f == "dropout":
                changes[f] = float(rng.uniform(*LOG_RANGES["dropout"]))
            else:
                changes[f] = _log_uniform(*LOG_RANGES[f], rng)
        child = replace(parent, parent_id=parent.id, origin="mutate", **changes)
        if fits(child, max_params):
            return child
    # Every edit overflowed: the parent sits in a corner where the local neighbourhood is
    # all oversized. Restart rather than return the parent unchanged (which the loop would
    # just skip as a duplicate, wasting the proposal).
    return random_candidate(rng, origin="random", max_params=max_params)


def crossover(
    a: Candidate,
    b: Candidate,
    rng: np.random.RandomState,
    max_params: int | None = MAX_PARAMS,
) -> Candidate:
    """Uniform crossover: each knob taken from ``a`` or ``b`` at random.

    Two in-budget parents can still produce an oversized child (a's wide channels with b's
    shallow depth), so overflowing draws are re-rolled.
    """
    fields = list(CHOICES) + list(LOG_RANGES)
    for _ in range(_MAX_TRIES):
        changes = {f: getattr(a if rng.random() < 0.5 else b, f) for f in fields}
        child = replace(a, parent_id=f"{a.id}+{b.id}", origin="crossover", **changes)
        if fits(child, max_params):
            return child
    return random_candidate(rng, origin="random", max_params=max_params)
