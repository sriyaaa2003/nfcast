"""Bootstrap over cells: is the capacity saving of a calibrated-peak policy over the best-tuned reactive rule distinguishable from zero?"""
import numpy as np, pandas as pd
R = pd.read_csv("results/v2/orchestration_cells.csv")
cells = sorted(R.instance.unique()); rng = np.random.default_rng(1)
fam = {"reactive": R.family == "reactive"}
for f in ("timesfm_q90|calibrated_peak", "chronos_small_q90|calibrated_peak", "mean_fm|calibrated_peak", "mean_all|calibrated_peak", "timesfm|calibrated_peak"):
    fam[f] = R.family == f
piv = {k: R[m].pivot_table(index=["policy", "param"], columns="instance", values=["capacity_cell_days", "violation_steps"], aggfunc="sum") for k, m in fam.items()}
def best(k, draw, budget):
    p = piv[k]; cap = p["capacity_cell_days"][draw].sum(axis=1); vio = p["violation_steps"][draw].sum(axis=1)
    ok = cap[vio <= budget]
    return ok.min() if len(ok) else np.nan
rows = []
for budget in (150, 300, 600, 1000, 1500):
    for k in list(fam)[1:]:
        pt = 100 * (best("reactive", cells, budget) - best(k, cells, budget)) / best("reactive", cells, budget)
        bs = []
        for _ in range(1000):
            d = list(rng.choice(cells, len(cells), replace=True))
            a, b = best("reactive", d, budget), best(k, d, budget)
            if np.isfinite(a) and np.isfinite(b): bs.append(100 * (a - b) / a)
        rows.append({"budget": budget, "policy": k, "saving_pct": round(pt, 1), "lo": round(np.percentile(bs, 2.5), 1), "hi": round(np.percentile(bs, 97.5), 1), "n_valid": len(bs)})
T = pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv("results/v2/orchestration_boot.csv", index=False)
