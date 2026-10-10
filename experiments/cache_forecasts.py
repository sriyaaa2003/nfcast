"""Compute and cache every model's forecasts on the calibration and test origins (point + native quantiles), once.
Later experiments (ensembles, stacking, peak-calibrated capacity planning) are pure post-processing of this cache."""
import sys, os, time, pickle
import numpy as np
sys.path.insert(0, "src")
import torch
torch.set_num_threads(int(os.environ.get("NT", "6")))
from nfcast.config import load_config
from nfcast.data import load_panel, series_dict, split_bounds
from nfcast.models import build_model
from nfcast.bench import forecast_windows, bench_keys
from nfcast.windows import origins_for, future

cfg = load_config(None)
series = series_dict(load_panel(cfg), cfg)
keys = bench_keys(cfg, series)
H = max(cfg.eval.horizons)
train_end = split_bounds(cfg)["train"][1]
names = sys.argv[1:] or ["ttm", "chronos_small", "chronos_tiny", "gbm", "timesfm"]
os.makedirs("data/cache", exist_ok=True)
for name in names:
    out = f"data/cache/fc_{name}.pkl"
    if os.path.exists(out):
        continue
    t0 = time.time()
    model = build_model(name, cfg)
    if name == "gbm":
        model.fit({k: series[k][:train_end] for k in keys})
    res = {"keys": keys}
    for split in ("calib", "test"):
        o = origins_for(cfg, split, H)
        point, q = forecast_windows(model, series, keys, o, H)
        y = np.stack([future(series[k], o, H) for k in keys])
        res[split] = {"origins": o, "point": point, "q": q, "y": y}
    pickle.dump(res, open(out, "wb"), protocol=4)
    print(name, "done", round(time.time() - t0), flush=True)
