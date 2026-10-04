"""Forecast scores. All errors are divided by a per-series scale (mean absolute day-over-day change in training data), so
series of different size and unit can be averaged and compared (the MASE idea)."""
from __future__ import annotations

import numpy as np


def series_scale(train: np.ndarray, period: int) -> float:
    return float(np.mean(np.abs(train[period:] - train[:-period]))) + 1e-9


def mase(y: np.ndarray, point: np.ndarray, scale: float) -> np.ndarray:
    """Per-window scaled mean absolute error over the horizon. y, point: [N, H]."""
    return np.abs(y - point).mean(axis=1) / scale


def pinball(y: np.ndarray, q: dict[float, np.ndarray], scale: float) -> np.ndarray:
    """Scaled mean pinball (quantile) loss over levels and steps: a proper score for the whole predictive distribution."""
    tot = 0.0
    for tau, f in q.items():
        d = y - f
        tot = tot + np.maximum(tau * d, (tau - 1) * d).mean(axis=1)
    return tot / len(q) / scale


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float, scale: float) -> np.ndarray:
    """Winkler interval score: width plus a penalty 2/alpha times how far the truth falls outside. Lower is better."""
    s = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return s.mean(axis=1) / scale


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return ((y >= lo) & (y <= hi)).mean(axis=1)


def width(lo: np.ndarray, hi: np.ndarray, scale: float) -> np.ndarray:
    return (hi - lo).mean(axis=1) / scale


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval resampling whole clusters (series), because windows from one series are correlated."""
    ids = np.unique(clusters)
    sums = np.array([values[clusters == c].sum() for c in ids])
    cnts = np.array([(clusters == c).sum() for c in ids], dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    est = sums[draw].sum(1) / cnts[draw].sum(1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))
