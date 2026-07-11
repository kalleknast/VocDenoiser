"""Label-free threshold selection on the 1-D SNR distribution.

We have no ground-truth clean/noisy labels, so the cutoff is derived from the
shape of the score histogram: a 2-component Gaussian mixture (EM) locates the
clean/noisy crossover, Otsu gives a non-parametric cross-check, and a percentile
lets the caller control the kept-count directly. All numpy, no sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GMM1D:
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray

    def _pdf_components(self, x: np.ndarray) -> np.ndarray:
        var = self.stds[None, :] ** 2
        norm = 1.0 / np.sqrt(2.0 * np.pi * var)
        return self.weights[None, :] * norm * np.exp(-((x[:, None] - self.means[None, :]) ** 2) / (2 * var))

    def crossover(self) -> float:
        """SNR value where the two components have equal responsibility.

        Searched on a dense grid between the two component means (the clean/noisy
        decision boundary). Falls back to the midpoint if no sign change exists.
        """
        order = np.argsort(self.means)
        lo, hi = self.means[order[0]], self.means[order[-1]]
        if hi - lo < 1e-9:
            return float(lo)
        grid = np.linspace(lo, hi, 1024)
        comp = self._pdf_components(grid)  # (grid, 2 or k)
        c_lo, c_hi = comp[:, order[0]], comp[:, order[-1]]
        diff = c_hi - c_lo
        sign = np.sign(diff)
        change = np.where(np.diff(sign) != 0)[0]
        if len(change):
            return float(grid[change[0]])
        return float(0.5 * (lo + hi))


def fit_gmm_1d(x: np.ndarray, k: int = 2, iters: int = 200, seed: int = 0) -> GMM1D:
    """Fit a 1-D Gaussian mixture with EM. Deterministic init from quantiles."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    q = np.linspace(0.1, 0.9, k)
    means = np.quantile(x, q)
    stds = np.full(k, x.std() / k + 1e-6)
    weights = np.full(k, 1.0 / k)
    for _ in range(iters):
        var = stds[None, :] ** 2
        norm = 1.0 / np.sqrt(2.0 * np.pi * var)
        resp = weights[None, :] * norm * np.exp(-((x[:, None] - means[None, :]) ** 2) / (2 * var))
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum == 0] = 1e-300
        resp = resp / resp_sum
        nk = resp.sum(axis=0) + 1e-12
        new_means = (resp * x[:, None]).sum(axis=0) / nk
        new_var = (resp * (x[:, None] - new_means[None, :]) ** 2).sum(axis=0) / nk
        new_stds = np.sqrt(np.maximum(new_var, 1e-8))
        new_weights = nk / len(x)
        if np.allclose(new_means, means, atol=1e-6) and np.allclose(new_stds, stds, atol=1e-6):
            means, stds, weights = new_means, new_stds, new_weights
            break
        means, stds, weights = new_means, new_stds, new_weights
    return GMM1D(means=means, stds=stds, weights=weights)


def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    """Otsu's method: the histogram split maximizing between-class variance."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    if total == 0:
        return float(np.median(x))
    p = w / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.argmax(sigma_b2))])


def kept_count_curve(scores: np.ndarray, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, kept_counts): how many clips survive each cutoff."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    thresholds = np.linspace(scores.min(), scores.max(), n)
    kept = np.array([(scores >= t).sum() for t in thresholds])
    return thresholds, kept
