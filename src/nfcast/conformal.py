"""Split conformal prediction. Calibration windows (a period after training, before test) turn any forecaster's output into
an interval with finite-sample coverage guarantees *if* calibration and test data are exchangeable. Shift breaks that, and
the benchmark measures by how much."""
from __future__ import annotations

import numpy as np


def conf_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    """The ceil((n+1)(1-alpha))-th smallest score along axis 0 (the finite-sample-corrected quantile). With too few scores
    for the requested level it returns the largest."""
    n = scores.shape[0]
    k = int(np.ceil(round((n + 1) * (1 - alpha), 9)))            # round: 10 * 0.8 is 8.000000000000002
    return np.sort(scores, axis=0)[min(k, n) - 1]


def interval_from_point(point_cal, y_cal, point_test, scale: float, alpha: float):
    """Absolute-residual conformal: point ± q, with q fitted per forecast step. Inputs [N, H]; returns lo, hi [N, H]."""
    q = conf_quantile(np.abs(y_cal - point_cal) / scale, alpha) * scale
    return point_test - q, point_test + q


def interval_from_quantiles(lo_cal, hi_cal, y_cal, lo_test, hi_test, scale: float, alpha: float):
    """Conformalised quantile regression (Romano et al.): widen or shrink the model's own [lo, hi] by the calibrated amount."""
    q = conf_quantile(np.maximum(lo_cal - y_cal, y_cal - hi_cal) / scale, alpha) * scale
    return lo_test - q, hi_test + q
