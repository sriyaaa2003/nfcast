"""Closed-loop test: does a better forecast save energy at equal service quality?

Each policy controls the replica count of one NF over the test period. Every `decision_minutes` it forecasts the next
`lookahead_minutes` of CPU demand, takes the peak, adds a headroom margin and provisions for it. A reactive floor keeps
capacity at least at what the last observed minute needed, so a bad forecast can hurt energy but cannot starve the pod for a
whole interval. The baseline is the plain reactive autoscaler (HPA with a scale-down stabilisation window) that produced the
telemetry. Energy uses the power model without noise; a violation minute is one where demand exceeds `violation_util`
of the provisioned capacity."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import metrics
from .bench import LOG, forecast_windows
from .config import Config
from .models import build_model
from .sim import hpa_replicas, series_dict, simulate, split_bounds
from .windows import future

EPS_REPLICAS = 1e-9


def power(util: np.ndarray, replicas: np.ndarray, cfg: Config) -> np.ndarray:
    p = cfg.sim.power
    u = np.clip(util, 0.0, 1.0)
    return replicas * (p.idle_w + (p.max_w - p.idle_w) * u ** p.alpha)


def account(demand: np.ndarray, replicas: np.ndarray, pod_cores: float, cfg: Config) -> dict:
    util = demand / (replicas * pod_cores)
    return dict(energy_kwh=float(power(util, replicas, cfg).sum() / 60 / 1000),
                violation_min=int((util > cfg.orchestrate.violation_util).sum()),
                scale_events=int((np.diff(replicas) != 0).sum()),
                mean_replicas=float(replicas.mean()))


def plan_replicas(fc_peak: np.ndarray, origins: np.ndarray, demand: np.ndarray, pod_cores: float, headroom: float, cfg: Config,
                  start: int, end: int) -> np.ndarray:
    """Per-minute replicas for [start, end) from forecast peaks at each decision origin (origins are `decision_minutes` apart)."""
    h, o = cfg.sim.hpa, cfg.orchestrate
    rep = np.empty(end - start)
    per = pod_cores * cfg.orchestrate.plan_util
    for j, t in enumerate(origins):
        plan = np.ceil(fc_peak[j] * (1 + headroom) / per)
        nxt = origins[j + 1] if j + 1 < len(origins) else end
        for m in range(t, nxt):
            floor = np.ceil(demand[m - 1] / per)            # reactive floor: last observed minute
            rep[m - start] = max(plan, floor)
    return np.clip(rep, h.min_replicas, h.max_replicas)


def run_orchestration(cfg: Config) -> pd.DataFrame:
    df = simulate(cfg)
    series = series_dict(df, cfg)
    o = cfg.orchestrate
    profile = next(p for p in cfg.sim.nf_profiles if p.name == o.nf)
    keys = [k for k in series if k[0] == o.nf and k[2] == o.target][:o.series]
    a, b = split_bounds(cfg)["test"]
    la = o.lookahead_minutes
    origins = np.arange(max(a, 4320), b - la + 1, o.decision_minutes)
    start, end = int(origins[0]), int(origins[-1] + o.decision_minutes)
    end = min(end, b)
    actual_rep = {k: df[(df.nf == k[0]) & (df.instance == k[1])].sort_values("minute").replicas.to_numpy() for k in keys}
    rows = []
    for k in keys:                                   # baseline: the reactive autoscaler that produced the telemetry
        r = account(series[k][start:end], actual_rep[k][start:end], profile.pod_cores, cfg)
        rows.append(dict(policy="reactive", headroom=np.nan, instance=k[1], **r))
    for tu in o.reactive_targets:                    # the same reactive autoscaler, tuned to run hotter
        hot = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, hpa=dataclasses.replace(cfg.sim.hpa, target_util=tu)))
        for k in keys:
            rep = hpa_replicas(series[k], profile.pod_cores, hot)[start:end]
            rows.append(dict(policy=f"reactive_{tu}", headroom=np.nan, instance=k[1], **account(series[k][start:end], rep, profile.pod_cores, cfg)))
    for name in [p for p in o.policies if p != "reactive"]:
        if name == "oracle":
            peaks = {k: np.array([series[k][t:t + la].max() for t in origins]) for k in keys}
        else:
            model = build_model(name, cfg)
            if name == "gbm":
                tr = split_bounds(cfg)["train"][1]
                allk = [kk for kk in series if kk[1] < cfg.eval.series_per_cell]
                model.fit({kk: series[kk][:tr] for kk in allk})
            point, _ = forecast_windows(model, series, keys, origins, la)
            peaks = {k: point[i].max(axis=1) for i, k in enumerate(keys)}
        LOG(f"orchestrate: {name} forecasts done")
        for hr in o.headrooms:
            for k in keys:
                rep = plan_replicas(peaks[k], origins, series[k], profile.pod_cores, hr, cfg, start, end)
                rows.append(dict(policy=name, headroom=hr, instance=k[1], **account(series[k][start:end], rep, profile.pod_cores, cfg)))
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, headroom): totals over instances, plus energy saving vs the reactive baseline."""
    g = res.groupby(["policy", "headroom"], dropna=False)[["energy_kwh", "violation_min", "scale_events", "mean_replicas"]].sum().reset_index()
    base = g[g.policy == "reactive"].iloc[0]
    g["energy_saving_pct"] = 100 * (base.energy_kwh - g.energy_kwh) / base.energy_kwh
    g["violation_change_min"] = g.violation_min - base.violation_min
    return g


def best_at_budget(summary: pd.DataFrame, budget_min: int) -> pd.DataFrame:
    """Least-energy configuration of each policy family among those with at most `budget_min` violation minutes in total.
    All tuned reactive settings form one family ('reactive_tuned'), so a forecast has to beat the best reactive tuning."""
    s = summary.copy()
    s["family"] = np.where(s.policy.str.startswith("reactive"), "reactive_tuned", s.policy)
    ok = s[s.violation_min <= budget_min]
    best = ok.sort_values("energy_kwh").groupby("family").head(1).sort_values("energy_kwh")
    return best.reset_index(drop=True)


def dominated_by_reactive(summary: pd.DataFrame) -> pd.Series:
    """True for configurations for which some reactive setting uses no more energy and has no more violations."""
    react = summary[summary.policy.str.startswith("reactive")]
    return summary.apply(lambda r: bool(((react.energy_kwh <= r.energy_kwh) & (react.violation_min <= r.violation_min)
                                          & ~((react.policy == r.policy) & (react.headroom.fillna(-1) == (-1 if np.isnan(r.headroom) else r.headroom)))).any()), axis=1)
