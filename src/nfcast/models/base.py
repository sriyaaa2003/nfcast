"""Forecaster interface shared by classical baselines and foundation models."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Meta:
    """Which series each history belongs to and where its forecast origin is (used by global models)."""
    origins: np.ndarray            # [N] step index of the forecast origin
    nf: list[str]
    target: list[str]


@dataclass
class Forecast:
    point: np.ndarray                                  # [N, H] median / mean forecast
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)   # level -> [N, H], empty if the model gives none


class Forecaster:
    """A model that forecasts the next `horizon` steps from the history before the forecast origin.

    `history_length` is how much past it needs; the harness hands it exactly the last `history_length` points, so a
    model can never see the future by construction (the harness, not the model, enforces that)."""

    name: str = "model"
    history_length: int = 1
    native_quantiles: bool = False

    def fit(self, train: dict[tuple[str, int, str], np.ndarray]) -> None:   # optional
        return None

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        raise NotImplementedError
