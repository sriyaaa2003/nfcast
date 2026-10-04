"""Benchmark: every model on identical rolling-origin windows of the held-out test period, point accuracy and calibrated intervals."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import conformal, metrics
from .config import Config, TARGETS
from .models import Meta, build_model
from .models.base import Forecaster
from .sim import series_dict, simulate, split_bounds
from .windows import future, gather, origins_for

LOG = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bench_keys(cfg: Config, series: dict, instances: range | None = None) -> list[tuple[str, int, str]]:
    inst = set(instances if instances is not None else range(cfg.eval.series_per_cell))
    return [k for k in series if k[1] in inst]


def forecast_windows(model: Forecaster, series: dict, keys: list, origins: np.ndarray, horizon: int):
    """Forecast every (series, origin) window. Returns point [S, O, H] and quantiles {level: [S, O, H]} (or {})."""
    S, O = len(keys), len(origins)
    L = model.history_length
    hist = np.concatenate([gather(series[k], origins, L) for k in keys])
    meta = Meta(np.tile(origins, S), [k[0] for k in keys for _ in origins], [k[2] for k in keys for _ in origins])
    f = model.predict(hist, horizon, meta)
    point = f.point.reshape(S, O, horizon)
    q = {lv: v.reshape(S, O, horizon) for lv, v in f.quantiles.items()}
    return point, q


def score_model(model: Forecaster, series: dict, cfg: Config, keys: list, scales: dict) -> pd.DataFrame:
    """Per-window scores on the test period. Conformal calibration uses the calib period, pooled over the instances of one
    (NF type, target) cell, so no test window ever informs an interval."""
    H = max(cfg.eval.horizons)
    alpha = cfg.eval.interval_alpha
    lo_l, hi_l = alpha / 2, 1 - alpha / 2
    res = {}
    for split in ("calib", "test"):
        o = origins_for(cfg, split, H)
        point, q = forecast_windows(model, series, keys, o, H)
        y = np.stack([future(series[k], o, H) for k in keys])
        res[split] = (o, point, q, y)
    rows = []
    cells: dict[tuple[str, str], list[int]] = {}
    for i, k in enumerate(keys):
        cells.setdefault((k[0], k[2]), []).append(i)
    o_t, p_t, q_t, y_t = res["test"]
    _, p_c, q_c, y_c = res["calib"]
    native = lo_l in q_t and hi_l in q_t
    for (nf, tgt), idx in cells.items():
        sc = scales[keys[idx[0]]]          # scale is per series; use each series' own below
        for horizon in cfg.eval.horizons:
            sl = slice(0, horizon)
            # calibration pool: all calib windows of this cell, scores per forecast step
            cal_scale = np.concatenate([np.full(p_c.shape[1], scales[keys[i]]) for i in idx])
            pc = np.concatenate([p_c[i][:, sl] for i in idx]) / cal_scale[:, None]
            yc = np.concatenate([y_c[i][:, sl] for i in idx]) / cal_scale[:, None]
            if native:
                loc = np.concatenate([q_c[lo_l][i][:, sl] for i in idx]) / cal_scale[:, None]
                hic = np.concatenate([q_c[hi_l][i][:, sl] for i in idx]) / cal_scale[:, None]
            for i in idx:
                k, sc = keys[i], scales[keys[i]]
                y, p = y_t[i][:, sl], p_t[i][:, sl]
                if native:
                    lo_raw, hi_raw = q_t[lo_l][i][:, sl], q_t[hi_l][i][:, sl]
                    lo, hi = conformal.interval_from_quantiles(loc, hic, yc, lo_raw / sc, hi_raw / sc, 1.0, alpha)
                    lo, hi = lo * sc, hi * sc
                else:
                    lo, hi = conformal.interval_from_point(pc, yc, p / sc, 1.0, alpha)
                    lo, hi = lo * sc, hi * sc
                rec = dict(model=model.name, nf=k[0], instance=k[1], target=k[2], horizon=horizon,
                           mase=metrics.mase(y, p, sc),
                           interval_score=metrics.interval_score(y, lo, hi, alpha, sc),
                           coverage=metrics.coverage(y, lo, hi), width=metrics.width(lo, hi, sc))
                if native:
                    rec["raw_coverage"] = metrics.coverage(y, lo_raw, hi_raw)
                    rec["pinball"] = metrics.pinball(y, {t: v[i][:, sl] for t, v in q_t.items()}, sc)
                n = len(o_t)
                df = pd.DataFrame({**{kk: (vv if isinstance(vv, np.ndarray) else np.repeat(vv, n)) for kk, vv in rec.items()}, "origin": o_t})
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def run_benchmark(cfg: Config, model_names: list[str] | None = None) -> pd.DataFrame:
    df = simulate(cfg)
    series = series_dict(df, cfg)
    keys = bench_keys(cfg, series)
    train_end = split_bounds(cfg)["train"][1]
    scales = {k: metrics.series_scale(series[k][:train_end], cfg.eval.seasonal_period) for k in series}
    out = []
    for name in model_names or cfg.models.enabled:
        t0 = time.time()
        model = build_model(name, cfg)
        if name == "gbm":
            model.fit({k: series[k][:train_end] for k in keys})
        out.append(score_model(model, series, cfg, keys, scales))
        LOG(f"{name}: {time.time() - t0:.0f}s, test MASE {out[-1].query('horizon == 60').mase.mean():.3f}")
    return pd.concat(out, ignore_index=True)
