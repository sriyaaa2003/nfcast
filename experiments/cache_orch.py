"""Forecasts for the capacity-planning replay: every decision origin (every 6 steps) of the calibration and test periods, horizon 6,
for the 12 internet series of cell type 0 that orchestrate.py uses. Point forecasts and native quantiles are stored."""
import sys, os, time, pickle
import numpy as np
sys.path.insert(0, "src")
import torch
torch.set_num_threads(int(os.environ.get("NT", "6")))
from nfcast.config import load_config
from nfcast.data import load_panel, series_dict, split_bounds
from nfcast.models import build_model
from nfcast.bench import forecast_windows
from nfcast.windows import min_history, future

cfg = load_config(None)
series = series_dict(load_panel(cfg), cfg)
o_ = cfg.orchestrate
keys = [k for k in series if k[0] == o_.cell_type and k[2] == o_.target][: o_.series]
bounds = split_bounds(cfg)
la, step = o_.lookahead_steps, o_.decision_steps
os.makedirs("data/cache", exist_ok=True)
train_end = bounds["train"][1]
names = sys.argv[1:] or ["ttm", "chronos_small", "gbm", "timesfm"]
for name in names:
    out = f"data/cache/orch_{name}.pkl"
    if os.path.exists(out):
        continue
    t0 = time.time()
    model = build_model(name, cfg)
    if name == "gbm":
        model.fit({kk: series[kk][:train_end] for kk in series if kk[1] < cfg.eval.series_per_cell})
    res = {"keys": keys}
    for split in ("calib", "test"):
        a, b = bounds[split]
        origins = np.arange(max(a, min_history(cfg)), b - la + 1, step)
        point, q = forecast_windows(model, series, keys, origins, la)
        y = np.stack([future(series[k], origins, la) for k in keys])
        res[split] = {"origins": origins, "point": point, "q": q, "y": y}
    pickle.dump(res, open(out, "wb"), protocol=4)
    print(name, "done", round(time.time() - t0), flush=True)
