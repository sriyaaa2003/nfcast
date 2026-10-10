"""Global gradient-boosting forecaster: the strongest classical machine-learning baseline.

One histogram gradient-boosting regressor per forecast knot (1, 2, 6, 12, 24, 36 steps ahead, configurable), trained on windows from
the training period of every series. Inputs are the last `lags` steps, longer rolling means, time of day and which cell type and
target the series is; everything is divided by the recent level so one model serves series of very different size."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import Config, TARGETS
from .base import Forecast, Forecaster, Meta

class GBM(Forecaster):
    name = "gbm"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        m = cfg.models
        self.knots = list(m.gbm_knots)
        self.lags = m.gbm_lags
        self.day = cfg.eval.seasonal_period
        self.types = [f"type{i}" for i in range(cfg.data.n_types)]
        self.history_length = max(self.day, self.lags)
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.series_meta: list[tuple[str, str]] = []

    def _features(self, hist: np.ndarray, origin_step: np.ndarray, nf: list[str], target: list[str]) -> tuple[np.ndarray, np.ndarray]:
        level = hist[:, -self.lags:].mean(axis=1, keepdims=True) + 1e-6
        lag = hist[:, -self.lags:] / level
        roll = np.stack([hist[:, -(self.day // 4):].mean(1), hist[:, -self.day:].mean(1)], axis=1) / level
        tod = (origin_step % self.day) / self.day
        tf = np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)], axis=1)
        onehot = np.array([[float(a == c) for c in self.types] + [float(b == t) for t in TARGETS] for a, b in zip(nf, target)])
        return np.hstack([lag, roll, tf, onehot]), level

    def fit(self, train: dict[tuple[str, int, str], np.ndarray]) -> None:
        stride = self.cfg.models.gbm_train_stride_steps
        Hmax = max(self.knots)
        X, Y = [], {k: [] for k in self.knots}
        for (nf, _inst, target), s in train.items():
            origins = np.arange(self.history_length, len(s) - Hmax, stride)
            hist = np.stack([s[o - self.history_length:o] for o in origins])
            feats, level = self._features(hist, origins, [nf] * len(origins), [target] * len(origins))
            X.append(feats)
            for k in self.knots:
                Y[k].append(s[origins + k - 1][:, None] / level)
        Xall = np.vstack(X)
        for k in self.knots:
            self.models[k] = HistGradientBoostingRegressor(max_iter=self.cfg.models.gbm_max_iter, learning_rate=0.08,
                                                           random_state=self.cfg.seed).fit(Xall, np.vstack(Y[k]).ravel())

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        if meta is None:
            raise ValueError("GBM needs the series metadata (origin, cell type, target)")
        feats, level = self._features(histories[:, -self.history_length:], meta.origins, meta.nf, meta.target)
        knot_pred = np.stack([self.models[k].predict(feats) for k in self.knots], axis=1) * level      # [N, K]
        steps = np.arange(1, horizon + 1)
        out = np.stack([np.interp(steps, self.knots, row) for row in knot_pred])
        return Forecast(out)
