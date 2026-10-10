"""Classical baselines: naive, seasonal naive, seasonal average, Holt-Winters (ETS)."""
from __future__ import annotations

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
    """Additive Holt-Winters (level + additive seasonality, no trend), written in numpy.

    The smoothing weights are chosen per history by minimising the one-step squared error over a small grid, evaluated for
    the whole grid at once. (statsmodels' implementation is blocked by this machine's application-control policy.)"""
    name = "ets"
    ALPHAS = np.array([0.05, 0.1, 0.2, 0.4, 0.7])
    GAMMAS = np.array([0.0, 0.05, 0.15, 0.4])

    def __init__(self, cfg: Config):
        self.step = cfg.models.ets_downsample_steps
        self.season = cfg.eval.seasonal_period // self.step
        self.history_length = cfg.models.ets_context_days * cfg.eval.seasonal_period

    def _fit_forecast(self, y: np.ndarray, horizon: int) -> np.ndarray:
        m = self.season
        a, g = np.meshgrid(self.ALPHAS, self.GAMMAS)
        a, g = a.ravel(), g.ravel()
        level = np.full(len(a), y[:m].mean())
        seas = np.repeat((y[:m] - y[:m].mean())[:, None], len(a), axis=1)        # [m, grid]
        sse = np.zeros(len(a))
        for t in range(m, len(y)):
            s = seas[t % m]
            err = y[t] - (level + s)
            sse += err ** 2
            new_level = level + a * err
            seas[t % m] = s + g * (1 - a) * err
            level = new_level
        best = int(np.argmin(sse))
        t0 = len(y)
        return level[best] + np.array([seas[(t0 + h) % m, best] for h in range(horizon)])

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        n, L = histories.shape
        k = L // self.step
        coarse = histories[:, L - k * self.step:].reshape(n, k, self.step).mean(axis=2)
        steps = int(np.ceil(horizon / self.step)) + 1
        grid = np.arange(steps) * self.step + (self.step - 1) / 2.0 + 1.0
        out = np.empty((n, horizon))
        for i in range(n):
            fc = self._fit_forecast(coarse[i], steps)
            out[i] = np.interp(np.arange(1, horizon + 1), grid, fc)
        return Forecast(out)
