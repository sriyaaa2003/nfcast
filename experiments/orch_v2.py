"""nfcast v2, part 2: capacity planning with forecasts of the *peak*.

v1 provisioned forecast_peak x (1 + headroom) with a hand-picked headroom grid. A better forecast should help only if its error in the
peak is known, so here the peak itself is calibrated: on the calibration period the ratio  true_peak / forecast_peak  of every decision
window is recorded, and the plan is  forecast_peak x Q_beta(ratio), where Q_beta is the beta-quantile of those ratios. beta is the
dial that trades capacity for violations; there is no other tuning. Everything is evaluated on the test period with the v1 accounting
(capacity in cell-days, violation = demand above 90% of capacity) and compared with the v1 reactive rule at every utilisation target.
"""
import sys, os, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from nfcast.config import load_config
from nfcast.data import load_panel, series_dict, split_bounds
from nfcast.orchestrate import reactive_capacity, plan_capacity, account

cfg = load_config(None)
series = series_dict(load_panel(cfg), cfg)
o = cfg.orchestrate
avail = [m for m in ("ttm", "chronos_small", "timesfm", "gbm") if os.path.exists(f"data/cache/orch_{m}.pkl")]
C = {m: pickle.load(open(f"data/cache/orch_{m}.pkl", "rb")) for m in avail}
keys = C[avail[0]]["keys"]
bounds = split_bounds(cfg)
train_end = bounds["train"][1]
scale = {k: float(series[k][:train_end].mean()) for k in keys}
orig = {s: C[avail[0]][s]["origins"] for s in ("calib", "test")}
Y = {s: C[avail[0]][s]["y"] for s in ("calib", "test")}                 # [S, O, la]
true_peak = {s: Y[s].max(axis=2) for s in Y}                           # [S, O]
la = o.lookahead_steps

def point(m, s):                  # [S, O, la]
    return C[m][s]["point"]

def peak_of(pt):
    return np.maximum(pt.max(axis=2), 0.0)

forecasts = {m: point(m, "calib") for m in avail}
cands = {m: (point(m, "calib"), point(m, "test")) for m in avail}
if len(avail) >= 2:
    cands["mean_all"] = (np.mean([point(m, "calib") for m in avail], axis=0), np.mean([point(m, "test") for m in avail], axis=0))
    fm = [m for m in avail if m != "gbm"]
    if len(fm) >= 2:
        cands["mean_fm"] = (np.mean([point(m, "calib") for m in fm], axis=0), np.mean([point(m, "test") for m in fm], axis=0))
# upper-quantile path of native-quantile models (0.9) as an alternative peak forecast
for m in avail:
    q = C[m]["test"]["q"]
    if 0.9 in q:
        cands[m + "_q90"] = (C[m]["calib"]["q"][0.9], C[m]["test"]["q"][0.9])

a, b = bounds["test"]
start, end = int(orig["test"][0]), min(int(orig["test"][-1] + o.decision_steps), b)
rows = []
for u, name in [(o.base_util, "reactive")] + [(u, f"reactive_{u}") for u in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)]:
    for k in keys:
        cap = reactive_capacity(series[k], u, o.scale_down_delay_steps, start, end)
        rows.append(dict(policy=name, param=u, family="reactive", instance=k[1], **account(series[k][start:end], cap, scale[k], cfg)))
for name, (cal_pt, te_pt) in cands.items():
    pk_cal, pk_te = peak_of(cal_pt), peak_of(te_pt)
    ratio = (true_peak["calib"] + 1e-6) / (pk_cal + 1e-6)             # pooled over the series: the ratio is scale-free
    for hr in o.headrooms:                                            # v1 policy: fixed headroom
        for i, k in enumerate(keys):
            cap = plan_capacity(pk_te[i], orig["test"], series[k], hr, o.plan_util, start, end)
            rows.append(dict(policy=name, param=hr, family=f"{name}|headroom", instance=k[1], **account(series[k][start:end], cap, scale[k], cfg)))
    for beta in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995):  # v2 policy: calibrated peak
        qb = float(np.quantile(ratio, beta))
        for i, k in enumerate(keys):
            cap = plan_capacity(pk_te[i] * qb, orig["test"], series[k], 0.0, o.plan_util, start, end)
            rows.append(dict(policy=name, param=beta, family=f"{name}|calibrated_peak", instance=k[1], **account(series[k][start:end], cap, scale[k], cfg)))
# oracle (true peak) and the v1 baseline
for hr in (0.0,):
    for i, k in enumerate(keys):
        cap = plan_capacity(true_peak["test"][i], orig["test"], series[k], hr, o.plan_util, start, end)
        rows.append(dict(policy="oracle", param=hr, family="oracle", instance=k[1], **account(series[k][start:end], cap, scale[k], cfg)))
R = pd.DataFrame(rows)
G = R.groupby(["family", "policy", "param"], as_index=False)[["capacity_cell_days", "violation_steps", "scale_events"]].sum()
base = G[(G.policy == "reactive")].iloc[0]
G["saving_pct"] = 100 * (base.capacity_cell_days - G.capacity_cell_days) / base.capacity_cell_days
G.to_csv("results/v2/orchestration.csv", index=False); R.to_csv("results/v2/orchestration_cells.csv", index=False)

def best_at(budget, fam_filter):
    s = G[fam_filter & (G.violation_steps <= budget)]
    return s.sort_values("capacity_cell_days").iloc[0] if len(s) else None
lines = ["### Least capacity (cell-days) with at most N violation steps over the 12 cells\n"]
budgets = [150, 300, 600, 1000, 1500]
tab = []
fams = sorted(set(G.family) - {"reactive"})
for fam in ["reactive"] + fams:
    row = {"policy family": "reactive (tuned)" if fam == "reactive" else fam}
    for bud in budgets:
        r = best_at(bud, G.family == fam)
        row[f"<= {bud}"] = f"{r.capacity_cell_days:.1f}" if r is not None else "-"
    tab.append(row)
lines.append(pd.DataFrame(tab).to_markdown(index=False))
lines.append(f"\nBaseline reactive rule at 60% utilisation: {base.capacity_cell_days:.1f} cell-days, {int(base.violation_steps)} violation steps. "
             f"Oracle (true peak): {G[G.family == 'oracle'].capacity_cell_days.iloc[0]:.1f} cell-days, {int(G[G.family == 'oracle'].violation_steps.iloc[0])} violation steps.")
open("results/v2/orchestration.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
