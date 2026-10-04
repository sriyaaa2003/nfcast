"""Model registry: builds the forecasters named in the configuration."""
from __future__ import annotations

from ..config import Config
from .base import Forecast, Forecaster, Meta
from .baselines import ETS, Naive, SeasonalAverage, SeasonalNaive
from .gbm import GBM

__all__ = ["Forecast", "Forecaster", "Meta", "build_model", "build_models"]


def build_model(name: str, cfg: Config) -> Forecaster:
    p = cfg.eval.seasonal_period
    if name == "naive":
        return Naive()
    if name == "seasonal_naive":
        return SeasonalNaive(p)
    if name == "seasonal_average":
        return SeasonalAverage(p, cfg.models.seasonal_average_days)
    if name == "ets":
        return ETS(cfg)
    if name == "gbm":
        return GBM(cfg)
    from .foundation import TTM, Chronos            # imported lazily: heavy, and needs the model weights
    f = cfg.models.foundation
    if name == "chronos_tiny":
        return Chronos(f.chronos_tiny, "chronos_tiny", cfg)
    if name == "chronos_small":
        return Chronos(f.chronos_small, "chronos_small", cfg)
    if name == "ttm":
        return TTM(cfg)
    raise ValueError(f"unknown model {name!r}")


def build_models(cfg: Config) -> list[Forecaster]:
    return [build_model(n, cfg) for n in cfg.models.enabled]
