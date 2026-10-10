"""Real cellular traffic: the Telecom Italia Big Data Challenge (Barlacchi et al., Scientific Data 2015), Milan, Nov-Dec 2013.

Every 10 minutes the operator reports, for each 235 m x 235 m grid square, SMS, call and internet activity (internet is
the number of data connections, scaled by the provider). This module builds a panel of series from the daily files:

1. the `pool_top` busiest squares by mean internet activity in the training days (quiet squares are mostly noise),
2. their weekly activity profile (training days only) clustered into `n_types` cell types with k-means,
3. `cells_per_type` squares drawn at random (seeded) from each type.

Series are in 10-minute steps; one day is `eval.seasonal_period` steps. Targets are internet, calls (in + out) and SMS (in + out).
Missing intervals are read as no activity (the operator reports only squares with activity).
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans

from .config import Config, TARGETS

MIRROR = "https://huggingface.co/datasets/GreatLove/CellularTrafficPrediction/resolve/main/sms-call-internet-mi/{day}.csv"


def day_names(cfg: Config) -> list[str]:
    d0 = date.fromisoformat(cfg.data.start_date)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(cfg.data.days)]


def download(cfg: Config) -> None:
    """Fetch the daily CSVs from the public mirror on the Hugging Face hub (the original is a Dataverse guestbook download)."""
    import urllib.request

    os.makedirs(cfg.data.raw_dir, exist_ok=True)
    for day in day_names(cfg):
        dest = os.path.join(cfg.data.raw_dir, f"{day}.csv")
        if not os.path.exists(dest):
            print("downloading", day, flush=True)
            urllib.request.urlretrieve(MIRROR.format(day=day), dest + ".part")
            os.replace(dest + ".part", dest)


def _read_day(path: str) -> pl.DataFrame:
    d = pl.read_csv(path, columns=["datetime", "squareid", "sms-in", "sms-out", "call-in", "call-out", "internet"],
                    schema_overrides={"squareid": pl.Int32}, null_values=[""])
    return d.select(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("t"), pl.col("squareid"),
        pl.col("internet").fill_null(0.0).cast(pl.Float32),
        (pl.col("call-in").fill_null(0.0) + pl.col("call-out").fill_null(0.0)).cast(pl.Float32).alias("calls"),
        (pl.col("sms-in").fill_null(0.0) + pl.col("sms-out").fill_null(0.0)).cast(pl.Float32).alias("sms"))


def build_pool(cfg: Config) -> pd.DataFrame:
    """Busiest squares, all days. Cached as parquet so the 35 daily files are read once."""
    cache = os.path.join(cfg.paths.data_dir, "milan_pool.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    days = day_names(cfg)
    train_days = days[: cfg.split.train_days]
    sums = pl.concat([_read_day(os.path.join(cfg.data.raw_dir, f"{day}.csv")).group_by("squareid").agg(pl.col("internet").sum())
                      for day in train_days]).group_by("squareid").agg(pl.col("internet").sum())
    pool = sums.sort("internet", descending=True).head(cfg.data.pool_top)["squareid"].to_list()
    frames = [_read_day(os.path.join(cfg.data.raw_dir, f"{day}.csv")).filter(pl.col("squareid").is_in(pool)) for day in days]
    df = pl.concat(frames).sort(["squareid", "t"]).to_pandas()
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    df.to_parquet(cache)
    return df


def to_matrix(pool: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, list[int], dict[str, np.ndarray]]:
    """[cells, steps] arrays per target on a complete 10-minute grid (missing = 0)."""
    t0 = pool.t.min()          # the first interval is 23:00 UTC = local midnight, so days align with local days
    steps = cfg.data.days * cfg.eval.seasonal_period
    idx = ((pool.t - t0).dt.total_seconds() // 600).astype(int).to_numpy()
    cells = sorted(pool.squareid.unique())
    row = pool.squareid.map({c: i for i, c in enumerate(cells)}).to_numpy()
    out = {}
    for tgt in TARGETS:
        m = np.zeros((len(cells), steps), dtype=np.float32)
        ok = idx < steps
        m[row[ok], idx[ok]] = pool[tgt].to_numpy()[ok]
        out[tgt] = m
    present = np.zeros((len(cells), steps), dtype=bool)
    present[row[idx < steps], idx[idx < steps]] = True
    return present, cells, out


def weekly_profile(m: np.ndarray, cfg: Config) -> np.ndarray:
    """Mean activity per (day of week, time of day) over the training days, normalised per cell. Day 0 is the first file."""
    spd = cfg.eval.seasonal_period
    train = m[:, : cfg.split.train_days * spd].reshape(m.shape[0], -1, spd)
    dow = np.arange(train.shape[1]) % 7
    prof = np.stack([train[:, dow == d].mean(1) for d in range(7)], axis=1).reshape(m.shape[0], -1)
    return prof / np.maximum(prof.mean(1, keepdims=True), 1e-6)


def build_panel(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Long frame (nf = cell type, instance, step, targets) and a description of the chosen cells."""
    pool = build_pool(cfg)
    present, cells, mats = to_matrix(pool, cfg)
    internet = mats["internet"]
    keep = present.mean(1) > 0.99
    prof = weekly_profile(internet, cfg)
    km = KMeans(n_clusters=cfg.data.n_types, n_init=10, random_state=cfg.seed).fit(np.log(prof[keep] + 0.05))
    label = np.full(len(cells), -1)
    label[keep] = km.labels_
    spd = cfg.eval.seasonal_period
    rng = np.random.default_rng(cfg.seed)
    # name each type by when its weekday activity peaks and how much quieter weekends are
    info, order = {}, []
    for k in range(cfg.data.n_types):
        members = np.where(label == k)[0]
        p = prof[members].mean(0).reshape(7, spd)
        order.append((int(np.argmax(p[:5].mean(0))), k))
    names = {k: f"type{j}" for j, (_, k) in enumerate(sorted(order))}
    frames = []
    for k in range(cfg.data.n_types):
        members = np.where(label == k)[0]
        chosen = rng.choice(members, size=min(cfg.data.cells_per_type, len(members)), replace=False)
        p = prof[members].mean(0).reshape(7, spd)
        info[names[k]] = {
            "cells_in_type": int(len(members)),
            "weekday_peak_hour": round(int(np.argmax(p[:5].mean(0))) * 10 / 60, 1),
            "weekend_to_weekday": round(float(p[5:].mean() / p[:5].mean()), 2),
            "mean_internet": round(float(internet[members].mean()), 1),
        }
        for inst, c in enumerate(chosen):
            frames.append(pd.DataFrame({
                "nf": names[k], "instance": inst, "cell": cells[c], "step": np.arange(internet.shape[1], dtype=np.int32),
                **{tgt: mats[tgt][c] for tgt in TARGETS}}))
    return pd.concat(frames, ignore_index=True), {"types": info, "pool": int(len(cells)), "kept": int(keep.sum())}


def split_bounds(cfg: Config) -> dict[str, tuple[int, int]]:
    spd = cfg.eval.seasonal_period
    tr = cfg.split.train_days * spd
    ca = tr + cfg.split.calib_days * spd
    return {"train": (0, tr), "calib": (tr, ca), "test": (ca, cfg.data.days * spd)}


def load_panel(cfg: Config) -> pd.DataFrame:
    cache = os.path.join(cfg.paths.data_dir, "panel.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df, _ = build_panel(cfg)
    df.to_parquet(cache)
    return df


def series_dict(df: pd.DataFrame, cfg: Config) -> dict[tuple[str, int, str], np.ndarray]:
    """(cell type, instance, target) -> float32 array of length days * steps per day."""
    out = {}
    for (nf, inst), g in df.groupby(["nf", "instance"], sort=True):
        g = g.sort_values("step")
        for target in TARGETS:
            out[(nf, int(inst), target)] = g[target].to_numpy(dtype=np.float32)
    return out
