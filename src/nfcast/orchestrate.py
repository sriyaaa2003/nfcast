"""Capacity-planning replay on real cellular traffic: does a better forecast let a cell run leaner at the same service quality?

Each policy sets the capacity of one cell for every 10-minute step of the test period. The demand is the real internet
activity of that cell, so the only assumption is the policy rule, not a model of the system. A *violation step* is one where
demand exceeds `violation_util` of the capacity provisioned.

* reactive: capacity covers the highest demand seen in the last `scale_down_delay_steps + 1` steps at a target utilisation
  (`base_util` for the baseline, `reactive_targets` for the hotter settings that make up the reactive frontier);
* forecast policies: every `decision_steps` the model forecasts the next `lookahead_steps`, the planner provisions for the peak
  plus a headroom at `plan_util`, and a reactive floor keeps capacity at least at what the last observed step needed, so a bad
  forecast can cost capacity but cannot starve the cell for a whole interval;
* oracle: the same planner with the true peak.

Capacity is reported in "cell-days": capacity divided by the cell's mean training demand, summed over steps and divided by the
steps in a day, so cells of different size add up. No power model is used: capacity-steps are the resource, not energy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bench import LOG, forecast_windows
from .config import Config
from .data import load_panel, series_dict, split_bounds
from .models import build_model
from .windows import min_history


def reactive_capacity(demand: np.ndarray, util: float, delay: int, start: int, end: int) -> np.ndarray:
    """Capacity for steps [start, end): the highest demand of the previous `delay + 1` observed steps at utilisation `util`."""
    out = np.empty(end - start)
    for t in range(start, end):
        out[t - start] = demand[max(0, t - 1 - delay):t].max() / util
    return out


def plan_capacity(peaks: np.ndarray, origins: np.ndarray, demand: np.ndarray, headroom: float, plan_util: float, start: int, end: int) -> np.ndarray:
    cap = np.empty(end - start)
    for j, t in enumerate(origins):
        plan = peaks[j] * (1 + headroom) / plan_util
        nxt = origins[j + 1] if j + 1 < len(origins) else end
        for m in range(t, nxt):
            cap[m - start] = max(plan, demand[m - 1] / plan_util)       # reactive floor: the last observed step
    return cap


def account(demand: np.ndarray, capacity: np.ndarray, scale: float, cfg: Config) -> dict:
    day = cfg.eval.seasonal_period
    return dict(capacity_cell_days=float((capacity / scale).sum() / day),
                violation_steps=int((demand > cfg.orchestrate.violation_util * capacity).sum()),
                scale_events=int((np.abs(np.diff(capacity)) > 1e-9).sum()),
                mean_capacity=float(capacity.mean() / scale))


def run_orchestration(cfg: Config) -> pd.DataFrame:
    series = series_dict(load_panel(cfg), cfg)
    o = cfg.orchestrate
    keys = [k for k in series if k[0] == o.cell_type and k[2] == o.target][:o.series]
    bounds = split_bounds(cfg)
    a, b = bounds["test"]
    la = o.lookahead_steps
    origins = np.arange(max(a, min_history(cfg)), b - la + 1, o.decision_steps)
    start, end = int(origins[0]), min(int(origins[-1] + o.decision_steps), b)
    scale = {k: float(series[k][:bounds["train"][1]].mean()) for k in keys}
    rows = []
    for u, name in [(o.base_util, "reactive")] + [(u, f"reactive_{u}") for u in o.reactive_targets]:
        for k in keys:
            cap = reactive_capacity(series[k], u, o.scale_down_delay_steps, start, end)
            rows.append(dict(policy=name, headroom=np.nan, instance=k[1], **account(series[k][start:end], cap, scale[k], cfg)))
    for name in [p for p in o.policies if p != "reactive"]:
        if name == "oracle":
            peaks = {k: np.array([series[k][t:t + la].max() for t in origins]) for k in keys}
        else:
            model = build_model(name, cfg)
            if name == "gbm":
                tr = bounds["train"][1]
                model.fit({kk: series[kk][:tr] for kk in series if kk[1] < cfg.eval.series_per_cell})
            point, _ = forecast_windows(model, series, keys, origins, la)
            peaks = {k: point[i].max(axis=1) for i, k in enumerate(keys)}
        LOG(f"orchestrate: {name} forecasts done")
        for hr in o.headrooms:
            for k in keys:
                cap = plan_capacity(peaks[k], origins, series[k], hr, o.plan_util, start, end)
                rows.append(dict(policy=name, headroom=hr, instance=k[1], **account(series[k][start:end], cap, scale[k], cfg)))
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, headroom): totals over cells, plus the capacity saving against the baseline reactive rule."""
    g = res.groupby(["policy", "headroom"], dropna=False)[["capacity_cell_days", "violation_steps", "scale_events", "mean_capacity"]].sum().reset_index()
    base = g[g.policy == "reactive"].iloc[0]
    g["capacity_saving_pct"] = 100 * (base.capacity_cell_days - g.capacity_cell_days) / base.capacity_cell_days
    g["violation_change"] = g.violation_steps - base.violation_steps
    return g


def best_at_budget(summary: pd.DataFrame, budget_steps: int) -> pd.DataFrame:
    """Least-capacity configuration of each policy family among those with at most `budget_steps` violation steps in total.
    All tuned reactive settings form one family ('reactive_tuned'), so a forecast has to beat the best reactive tuning."""
    s = summary.copy()
    s["family"] = np.where(s.policy.str.startswith("reactive"), "reactive_tuned", s.policy)
    ok = s[s.violation_steps <= budget_steps]
    return ok.sort_values("capacity_cell_days").groupby("family").head(1).sort_values("capacity_cell_days").reset_index(drop=True)


def dominated_by_reactive(summary: pd.DataFrame) -> pd.Series:
    """True for configurations for which some other reactive setting uses no more capacity and has no more violations."""
    react = summary[summary.policy.str.startswith("reactive")]

    def dominated(r):
        same = (react.policy == r.policy) & (react.headroom.fillna(-1) == (-1 if np.isnan(r.headroom) else r.headroom))
        return bool(((react.capacity_cell_days <= r.capacity_cell_days) & (react.violation_steps <= r.violation_steps) & ~same).any())

    return summary.apply(dominated, axis=1)
