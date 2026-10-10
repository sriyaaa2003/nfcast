"""Typed, validated configuration. YAML is the single source of truth; there are no code-side defaults."""
from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Paths:
    data_dir: str
    results_dir: str


@dataclass(frozen=True)
class Data:
    raw_dir: str                   # daily CSV files of the Telecom Italia Milan data
    start_date: str                # date of the first file (ISO)
    days: int
    n_types: int                   # cell types found by clustering the weekly profile of the busiest squares
    cells_per_type: int
    pool_top: int                  # candidate squares: the busiest this many by mean internet activity in the training days


@dataclass(frozen=True)
class Split:
    train_days: int
    calib_days: int                # the remaining days are the test period


@dataclass(frozen=True)
class Eval:
    context_length: int
    horizons: list[int]
    origin_stride_steps: int
    quantile_levels: list[float]
    interval_alpha: float
    series_per_cell: int           # instances per cell type used in the benchmark (the rest stay out)
    n_boot: int
    seasonal_period: int           # steps per day: the seasonal-naive lag and the MASE scale


@dataclass(frozen=True)
class Foundation:
    chronos_tiny: str
    chronos_small: str
    ttm: str
    timesfm: str
    batch_size: int


@dataclass(frozen=True)
class Models:
    enabled: list[str]
    foundation: Foundation
    seasonal_average_days: int
    ets_downsample_steps: int
    ets_context_days: int
    gbm_knots: list[int]
    gbm_lags: int
    gbm_train_stride_steps: int
    gbm_max_iter: int


@dataclass(frozen=True)
class Finetune:
    fractions: list[float]         # share of the training windows used (the data regimes)
    epochs: int
    batch_size: int
    learning_rate: float
    window_stride_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    methods: list[str]             # full | lora
    max_windows: int               # cap on training windows per run, for CPU time
    seeds: int                     # independent repeats of every adaptation run (data sampling and training)
    source_type: str       # cross-type experiment: adapt on this cell type only
    source_target: str     # cross-target experiment: adapt on this target only


@dataclass(frozen=True)
class Orchestrate:
    decision_steps: int
    lookahead_steps: int
    headrooms: list[float]
    scale_down_delay_steps: int
    policies: list[str]
    cell_type: str
    target: str
    series: int
    base_util: float               # target utilisation of the baseline reactive rule
    plan_util: float               # utilisation the proactive planner provisions for (the reactive rule keeps its own, lower target)
    reactive_targets: list[float]  # target utilisations swept as the reactive frontier
    violation_util: float          # a step is a violation when demand exceeds this share of capacity


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    data: Data
    split: Split
    eval: Eval
    models: Models
    finetune: Finetune
    orchestrate: Orchestrate


TARGETS = ("internet", "calls", "sms")


def _build(tp: typing.Any, data: typing.Any, path: str) -> typing.Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping")
        hints = typing.get_type_hints(tp)
        unknown = set(data) - set(hints)
        if unknown:
            raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name not in data:
                raise ConfigError(f"{path}.{f.name}: missing")
            kwargs[f.name] = _build(hints[f.name], data[f.name], f"{path}.{f.name}")
        return tp(**kwargs)
    if origin is list:
        (arg,) = typing.get_args(tp)
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected a list")
        return [_build(arg, x, f"{path}[{i}]") for i, x in enumerate(data)]
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ConfigError(f"{path}: expected a number")
        return float(data)
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise ConfigError(f"{path}: expected an integer")
        return data
    if not isinstance(data, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}")
    return data


KNOWN_MODELS = {"naive", "seasonal_naive", "seasonal_average", "ets", "gbm", "chronos_tiny", "chronos_small", "ttm", "timesfm"}


def validate(cfg: Config) -> None:
    d = cfg.data
    if d.n_types < 2 or d.cells_per_type < cfg.eval.series_per_cell + 1 or d.pool_top < d.n_types * d.cells_per_type:
        raise ConfigError("data: need n_types >= 2, cells_per_type > eval.series_per_cell and pool_top >= n_types * cells_per_type")
    if cfg.split.train_days + cfg.split.calib_days >= d.days:
        raise ConfigError("split: train_days + calib_days must leave a test period")
    e = cfg.eval
    if not all(0 < q < 1 for q in e.quantile_levels) or e.quantile_levels != sorted(e.quantile_levels):
        raise ConfigError("eval.quantile_levels must be increasing values in (0,1)")
    if not 0 < e.interval_alpha < 1:
        raise ConfigError("eval.interval_alpha must be in (0,1)")
    if max(e.horizons) > 96:
        raise ConfigError("eval.horizons: the TTM release used here predicts at most 96 steps")
    unknown = set(cfg.models.enabled) - KNOWN_MODELS
    if unknown:
        raise ConfigError(f"models.enabled: unknown models {sorted(unknown)}")
    if not set(cfg.finetune.methods) <= {"full", "lora"}:
        raise ConfigError("finetune.methods: full | lora")
    if not all(0 < f <= 1 for f in cfg.finetune.fractions):
        raise ConfigError("finetune.fractions must be in (0,1]")
    o = cfg.orchestrate
    if o.target not in TARGETS or cfg.finetune.source_target not in TARGETS:
        raise ConfigError("orchestrate.target / finetune.source_target must be a known target")
    if not set(o.policies) <= {"reactive", "naive", "seasonal_naive", "seasonal_average", "gbm", "ttm", "chronos_tiny", "chronos_small", "timesfm", "oracle"}:
        raise ConfigError("orchestrate.policies: unknown policy")


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    chosen = path or os.environ.get("NFCAST_CONFIG")
    if chosen is None:
        chosen = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(chosen, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _build(Config, raw, "config")
    validate(cfg)
    return cfg
