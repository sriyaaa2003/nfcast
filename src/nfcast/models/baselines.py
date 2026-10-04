"""Classical baselines: naive, seasonal naive, seasonal average, Holt-Winters (ETS)."""
from __future__ import annotations

import warnings

import numpy as np

from ..config import Config
from .base import Forecast, Forecaster, Meta


class Naive(Forecaster):
    name = "naive"
    history_length = 1

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        return Forecast(np.repeat(histories[:, -1:], horizon, axis=1))


class SeasonalNaive(Forecaster):
    """The value one season ago at the same time of day."""
    name = "seasonal_naive"

    def __init__(self, period: int):
        self.period = period
        self.history_length = period

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        # forecast for step h (1-based) is the value at origin + h - 1 - period
        idx = np.arange(horizon) - self.period
        return Forecast(histories[:, idx])


class SeasonalAverage(Forecaster):
    """Mean of the same time of day over the last k days: a stronger, smoother seasonal baseline."""
    name = "seasonal_average"

    def __init__(self, period: int, days: int):
        self.period, self.days = period, days
        self.history_length = period * days

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        idx = np.arange(horizon)
        stack = [histories[:, idx - self.period * k] for k in range(1, self.days + 1)]
        return Forecast(np.mean(stack, axis=0))


class ETS(Forecaster):
    """Additive Holt-Winters on a downsampled recent history, interpolated back to the minute grid."""
    name = "ets"

    def __init__(self, cfg: Config):
        self.step = cfg.models.ets_downsample_minutes
        self.season = cfg.eval.seasonal_period // self.step
        self.history_length = cfg.models.ets_context_days * cfg.eval.seasonal_period

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        n, L = histories.shape
        k = L // self.step
        coarse = histories[:, L - k * self.step:].reshape(n, k, self.step).mean(axis=2)
        steps = int(np.ceil(horizon / self.step)) + 1
        grid = np.arange(steps) * self.step + (self.step - 1) / 2.0 + 1.0   # centre of each coarse step, in minutes ahead
        out = np.empty((n, horizon))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n):
                try:
                    fc = ExponentialSmoothing(coarse[i], trend=None, seasonal="add", seasonal_periods=self.season,
                                              initialization_method="estimated").fit().forecast(steps)
                except Exception:
                    fc = np.full(steps, coarse[i, -1])
                out[i] = np.interp(np.arange(1, horizon + 1), grid, fc)
        return Forecast(out)
