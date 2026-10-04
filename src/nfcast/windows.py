"""Rolling-origin forecast windows. An origin is the minute the forecast is issued; the model sees only minutes before it."""
from __future__ import annotations

import numpy as np

from .config import Config
from .sim import split_bounds

MIN_HISTORY = 4320      # longest history any model asks for (3 days), so every model is scored on the same origins


def origins_for(cfg: Config, split: str, horizon: int) -> np.ndarray:
    a, b = split_bounds(cfg)[split]
    return np.arange(max(a, MIN_HISTORY), b - horizon + 1, cfg.eval.origin_stride_minutes)


def gather(series: np.ndarray, origins: np.ndarray, length: int) -> np.ndarray:
    """[N, length] histories ending just before each origin (series[o-length:o]); the future is never included."""
    if origins.min() < length:
        raise ValueError("origin earlier than the required history")
    return np.stack([series[o - length:o] for o in origins])


def future(series: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([series[o:o + horizon] for o in origins])
