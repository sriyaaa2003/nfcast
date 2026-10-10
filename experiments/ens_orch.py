"""nfcast v2, part 1: do ensembles and stacking beat every single foundation model? Part 2: does calibrating the *peak* (instead of
adding a fixed headroom) give better capacity plans? Pure post-processing of data/cache/*.pkl."""
import sys, os, json, pickle, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from scipy.optimize import nnls
from nfcast.config import load_config
from nfcast.data import load_panel, series_dict, split_bounds
from nfcast import metrics, conformal
from nfcast.orchestrate import reactive_capacity, plan_capacity, account

cfg = load_config(None)
series = series_dict(load_panel(cfg), cfg)
train_end = split_bounds(cfg)["train"][1]
rng = np.random.default_rng(5)
out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out_lines.append(s)

# ------------------------------------------------------------------------------------------------ part 1: accuracy
avail = [m for m in ("ttm", "chronos_small", "chronos_tiny", "timesfm", "gbm") if os.path.exists(f"data/cache/fc_{m}.pkl")]
C = {m: pickle.load(open(f"data/cache/fc_{m}.pkl", "rb")) for m in avail}
keys = C[avail[0]]["keys"]
scales = np.array([metrics.series_scale(series[k][:train_end], cfg.eval.seasonal_period) for k in keys])
cell_of = {}
for i, k in enumerate(keys):
    cell_of.setdefault((k[0], k[2]), []).append(i)
alpha = cfg.eval.interval_alpha

def mase_win(point, y, sl):               # [S, O, H] -> [S, O]
    return np.abs(y[:, :, sl] - point[:, :, sl]).mean(axis=2) / scales[:, None]

def cluster_ci(M, n=2000):
    per_series = M.mean(axis=1)           # series are the independent units
    idx = rng.integers(0, len(per_series), (n, len(per_series)))
    b = per_series[idx].mean(axis=1)
    return float(per_series.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def combos():
    out = {m: [m] for m in avail}
    fms = [m for m in avail if m != "gbm"]
    for r in (2, 3, 4, 5):
        for c in itertools.combinations(avail, r):
            out["mean(" + ",".join(c) + ")"] = list(c)
    return out

def get(split, models, H, agg="mean"):
    arr = np.stack([C[m][split]["point"] for m in models])
    return arr.mean(axis=0) if agg == "mean" else np.median(arr, axis=0)

y = {s: C[avail[0]][s]["y"] for s in ("calib", "test")}
rows = []
for H in cfg.eval.horizons:
    sl = slice(0, H)
    for name, ms in combos().items():
        p = get("test", ms, H)
        m, lo, hi = cluster_ci(mase_win(p, y["test"], sl))
        rows.append({"horizon": H, "model": name, "mase": m, "lo": lo, "hi": hi})
    # stacking: non-negative weights summing to 1, fitted on calibration windows, one weight vector per horizon
    A = np.stack([ (C[m]["calib"]["point"][:, :, sl] / scales[:, None, None]).ravel() for m in avail], axis=1)
    b = (y["calib"][:, :, sl] / scales[:, None, None]).ravel()
    w, _ = nnls(A, b)
    w = w / w.sum()
    p = sum(wi * C[m]["test"]["point"] for wi, m in zip(w, avail))
    mm, lo, hi = cluster_ci(mase_win(p, y["test"], sl))
    rows.append({"horizon": H, "model": "stack_nnls(" + ",".join(f"{m}:{wi:.2f}" for m, wi in zip(avail, w)) + ")", "mase": mm, "lo": lo, "hi": hi})
R = pd.DataFrame(rows)
for H in cfg.eval.horizons:
    P(f"\n### Accuracy, {H * 10}-minute horizon (test MASE, 95% CI resampling series)\n")
    sub = R[R.horizon == H].sort_values("mase").head(14)
    P(sub[["model", "mase", "lo", "hi"]].round(3).to_markdown(index=False))
    P("\nsingles: " + ", ".join(f"{m} {R[(R.horizon == H) & (R.model == m)].mase.iloc[0]:.3f}" for m in avail))
R.to_csv("results/v2/accuracy.csv", index=False)

# interval quality for the best ensemble and best single (conformal, same protocol as bench.py)
def interval_stats(point_cal, point_test, H):
    sl = slice(0, H)
    cov, iscore = [], []
    for (nf, tgt), idx in cell_of.items():
        sc = scales[idx]
        pc = np.concatenate([point_cal[i][:, sl] for i in idx]); yc = np.concatenate([y["calib"][i][:, sl] for i in idx])
        scc = np.concatenate([np.full(point_cal.shape[1], scales[i]) for i in idx])
        for i in idx:
            q = conformal.conf_quantile(np.abs(yc - pc) / scc[:, None], alpha) * scales[i]
            lo, hi = point_test[i][:, sl] - q, point_test[i][:, sl] + q
            cov.append(metrics.coverage(y["test"][i][:, sl], lo, hi).mean())
            iscore.append(metrics.interval_score(y["test"][i][:, sl], lo, hi, alpha, scales[i]).mean())
    return float(np.mean(cov)), float(np.mean(iscore))
P("\n### Conformal intervals (80%) of single models and ensembles, 360-minute horizon\n")
H = max(cfg.eval.horizons)
rows = []
CB = combos()
for name in list(avail) + ["mean(" + ",".join(c) + ")" for c in [tuple(avail[:2]), tuple(avail[:3]), tuple(avail)]]:
    ms = CB[name]
    cv, isc = interval_stats(get("calib", ms, H), get("test", ms, H), H)
    rows.append({"model": name, "coverage": round(cv, 3), "interval_score": round(isc, 3)})
P(pd.DataFrame(rows).to_markdown(index=False))
open("results/v2/ensembles.md", "w", encoding="utf-8").write("\n".join(out_lines))
