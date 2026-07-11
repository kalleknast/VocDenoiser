"""Noise-aware accept rule.

Karpathy's autoresearch accepts any improvement because bits-per-byte on a fixed
seed is essentially deterministic. Audio reconstruction metrics (SI-SDR / MSE)
are noisier: a tiny gain can be seed noise. So a challenger is accepted only if
its seed-averaged metric beats the incumbent by more than a multiple of the
combined seed spread — plus a simplicity tie-break that prefers the smaller model
when the metrics are statistically indistinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AcceptDecision:
    accept: bool
    reason: str


def noise_aware_accept(
    challenger_metric: float,
    challenger_std: float,
    incumbent_metric: float,
    incumbent_std: float,
    k_sigma: float = 1.0,
    challenger_params: int | None = None,
    incumbent_params: int | None = None,
) -> AcceptDecision:
    """Decide whether ``challenger`` replaces ``incumbent`` (both higher=better).

    Accept if ``challenger - incumbent > k_sigma * sqrt(std_c^2 + std_i^2)``.
    Within the noise band, accept only if the challenger is a *simplification*
    (fewer params) — mirroring autoresearch's "equal-or-better but simpler is a
    win" criterion, which counteracts metric-chasing bloat.
    """
    band = k_sigma * ((challenger_std**2 + incumbent_std**2) ** 0.5)
    delta = challenger_metric - incumbent_metric
    if delta > band:
        return AcceptDecision(True, f"+{delta:.3f} > noise band {band:.3f}")
    if abs(delta) <= band and challenger_params is not None and incumbent_params is not None:
        if challenger_params < incumbent_params:
            return AcceptDecision(
                True,
                f"within noise band ({delta:+.3f}) but simpler "
                f"({challenger_params} < {incumbent_params} params)",
            )
    return AcceptDecision(False, f"{delta:+.3f} <= noise band {band:.3f}")
