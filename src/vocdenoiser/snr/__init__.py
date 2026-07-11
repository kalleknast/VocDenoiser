"""Call-agnostic SNR estimation and clean-subset selection.

The primary metric is a *masked per-bin noise-floor SNR*: the noise floor is
estimated independently in every frequency bin (temporal low-percentile), a
band-free active mask marks the call's time-frequency footprint, and the score
is the median excess-over-floor inside that footprint. Because the floor and the
mask are per-bin and adaptive, the score makes no band / harmonic / morphology
assumption, so it does not bias toward tonal "phee" calls over broadband calls
(trill, twitter, ek, ...).
"""

from vocdenoiser.snr.metric import (
    DEFAULT_PARAMS,
    SNRParams,
    clip_features,
    spectral_snr_db,
)

__all__ = ["SNRParams", "DEFAULT_PARAMS", "clip_features", "spectral_snr_db"]
