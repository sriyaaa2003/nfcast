"""Time-series foundation models: Chronos-Bolt (probabilistic) and Tiny Time Mixer (TTM, point forecasts).

Both run zero-shot on CPU. TTM can also be fine-tuned (see finetune.py) and then wrapped again with `TTM(model=...)`.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import torch

from ..config import Config
from .base import Forecast, Forecaster, Meta


class Chronos(Forecaster):
    """Chronos-Bolt (Amazon): a T5 encoder-decoder trained on a large time-series corpus; predicts quantiles directly."""
    native_quantiles = True

    def __init__(self, model_id: str, label: str, cfg: Config):
        from chronos import BaseChronosPipeline
        self.name = label
        self.history_length = cfg.eval.context_length
        self.levels = list(cfg.eval.quantile_levels)
        self.batch = cfg.models.foundation.batch_size
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)

    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        if horizon > 64:
            raise ValueError("Chronos-Bolt forecasts up to 64 steps in one pass")
        qs, point = [], []
        for i in range(0, len(histories), self.batch):
            ctx = [torch.tensor(h, dtype=torch.float32) for h in histories[i:i + self.batch]]
            q, mean = self.pipe.predict_quantiles(inputs=ctx, prediction_length=horizon, quantile_levels=self.levels)
            qs.append(q.numpy())
            point.append(mean.numpy())
        q = np.concatenate(qs)
        return Forecast(np.concatenate(point), {lv: q[:, :, j] for j, lv in enumerate(self.levels)})


def load_ttm_model(model_id: str):
    """Load TTM through its model class only.

    The `tsfm_public` package's top-level import also pulls in data-loading tooling (pyarrow dataset) that this project
    does not use and that some locked-down Windows policies block. Registering the package path without running its
    __init__ imports exactly the model code and nothing else."""
    if "tsfm_public" not in sys.modules or not hasattr(sys.modules["tsfm_public"], "__path__"):
        spec = importlib.util.find_spec("tsfm_public")
        if spec is None or spec.origin is None:
            raise ImportError("granite-tsfm is not installed (pip install granite-tsfm)")
        stub = types.ModuleType("tsfm_public")
        stub.__path__ = [os.path.dirname(spec.origin)]
        sys.modules["tsfm_public"] = stub
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained(model_id)
    model.eval()
    return model


class TTM(Forecaster):
    """IBM Granite Tiny Time Mixer r2: under 1M parameters, context 512, horizon up to 96, point forecasts."""

    def __init__(self, cfg: Config, label: str = "ttm", model=None):
        self.name = label
        self.model = model if model is not None else load_ttm_model(cfg.models.foundation.ttm)
        self.history_length = int(self.model.config.context_length)
        self.max_horizon = int(self.model.config.prediction_length)
        self.batch = cfg.models.foundation.batch_size

    @torch.no_grad()
    def predict(self, histories: np.ndarray, horizon: int, meta: Meta | None = None) -> Forecast:
        if horizon > self.max_horizon:
            raise ValueError(f"TTM predicts up to {self.max_horizon} steps")
        self.model.eval()
        out = []
        for i in range(0, len(histories), self.batch):
            x = torch.tensor(histories[i:i + self.batch, -self.history_length:], dtype=torch.float32).unsqueeze(-1)
            out.append(self.model(past_values=x).prediction_outputs[:, :horizon, 0].numpy())
        return Forecast(np.concatenate(out))
