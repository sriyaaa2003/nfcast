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
class Traffic:
    n_sources: int                 # ON/OFF sources whose superposition gives heavy-tailed, bursty traffic
    rate_mbps_mean: float
    rate_sigma: float
    on_alpha: float                # Pareto shape of ON durations (1 < alpha < 2: long-range dependence)
    off_alpha: float
    on_min_minutes: float
    off_min_minutes: float
    base_mbps: float               # deterministic daily-pattern component
    daily_amp1: float              # 24 h harmonic
    daily_amp2: float              # 12 h harmonic (commuter double peak)
    weekend_factor: float
    instance_scale_sigma: float    # log-normal spread of site size between instances
    noise_cv: float
    events_per_instance: int       # demand surges (stadium, outage recovery) per instance
    event_multiplier_min: float
    event_multiplier_max: float
    event_minutes_min: int
    event_minutes_max: int


@dataclass(frozen=True)
class NFProfile:
    name: str                      # upf | amf | smf | cuup (selects the demand model)
    instances: int
    cpu_base_cores: float
    cpu_per_unit: float            # cores per Mbit/s (upf, cuup) or per request/s (amf, smf)
    cpu_noise_cv: float
    pod_cores: float
    pod_mem_gib: float
    mem_base_gib: float
    mem_per_unit_gib: float
    mem_ema_minutes: float         # memory follows load through this smoothing time
    leak_gib_per_min: float        # slow memory leak until the pod restarts (0 disables)
    signalling_exponent: float     # amf/smf: request rate grows with traffic**exponent
    burst_cv: float                # amf/smf: gamma noise on the request rate
    periodic_minutes: int          # cuup: a housekeeping task every N minutes (0 disables)
    periodic_cores: float
    periodic_duration_minutes: int


@dataclass(frozen=True)
class Hpa:
    target_util: float             # reactive autoscaler aims at this pod utilisation
    down_stabilization_minutes: int
    min_replicas: int
    max_replicas: int


@dataclass(frozen=True)
class Power:
    idle_w: float
    max_w: float
    alpha: float                   # power ~ idle + (max - idle) * util ** alpha
    noise_cv: float


@dataclass(frozen=True)
class Sim:
    days: int
    nf_profiles: list[NFProfile]
    traffic: Traffic
    hpa: Hpa
    power: Power


@dataclass(frozen=True)
class Split:
    train_days: int
    calib_days: int                # the remaining days are the test period


@dataclass(frozen=True)
class Eval:
    context_length: int
    horizons: list[int]
    origin_stride_minutes: int
    quantile_levels: list[float]
    interval_alpha: float
    series_per_cell: int           # instances per NF type used in the benchmark (the rest stay out)
    n_boot: int
    seasonal_period: int           # minutes in the seasonal-naive lag and the MASE scale


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
    ets_downsample_minutes: int
    ets_context_days: int
    gbm_knots: list[int]
    gbm_lags: int
    gbm_train_stride_minutes: int
    gbm_max_iter: int


@dataclass(frozen=True)
class Finetune:
    fractions: list[float]         # share of the training windows used (the data regimes)
    epochs: int
    batch_size: int
    learning_rate: float
    window_stride_minutes: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    methods: list[str]             # full | lora
    max_windows: int               # cap on training windows per run, for CPU time
    seeds: int                     # independent repeats of every adaptation run (data sampling and training)
    source_nf: str         # cross-NF experiment: adapt on this NF type only
    source_target: str     # cross-target experiment: adapt on this target only


@dataclass(frozen=True)
class Orchestrate:
    decision_minutes: int
    lookahead_minutes: int
    headrooms: list[float]
    scale_down_delay_minutes: int
    policies: list[str]
    nf: str
    target: str
    series: int
    plan_util: float               # utilisation the proactive planner provisions for (the reactive HPA keeps its own, lower target)
    reactive_targets: list[float]  # HPA target utilisations swept as the reactive frontier
    violation_util: float          # a minute is a violation when demand exceeds this share of capacity


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    sim: Sim
    split: Split
    eval: Eval
    models: Models
    finetune: Finetune
    orchestrate: Orchestrate


TARGETS = ("traffic_mbps", "cpu_cores", "cpu_util_pod", "mem_gib", "energy_w")


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
    names = [p.name for p in cfg.sim.nf_profiles]
    if len(set(names)) != len(names):
        raise ConfigError("sim.nf_profiles: duplicate names")
    if not set(names) <= {"upf", "amf", "smf", "cuup"}:
        raise ConfigError("sim.nf_profiles: names must be upf, amf, smf or cuup")
    for p in cfg.sim.nf_profiles:
        if p.instances < 1 or p.pod_cores <= 0 or p.pod_mem_gib <= 0:
            raise ConfigError(f"nf {p.name}: instances, pod_cores and pod_mem_gib must be positive")
    t = cfg.sim.traffic
    if t.on_alpha <= 1 or t.off_alpha <= 1:
        raise ConfigError("sim.traffic: Pareto shapes must be > 1")
    if cfg.split.train_days + cfg.split.calib_days >= cfg.sim.days:
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
    if o.nf not in names or o.target not in TARGETS:
        raise ConfigError("orchestrate.nf / target must refer to a configured NF and a known target")
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
