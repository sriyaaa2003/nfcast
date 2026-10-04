"""Adapting TTM to NF telemetry: full fine-tuning and LoRA, at several data budgets, and how far the adaptation transfers
across NF types, targets and unseen instances (sites)."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from . import metrics
from .bench import LOG, score_model
from .config import Config
from .models.foundation import TTM, load_ttm_model
from .sim import series_dict, simulate, split_bounds


def make_windows(series: dict, keys: list, end: int, n: int, ctx: int, pred: int, seed: int):
    """n random training windows from the first `end` minutes of the given series. Past and future lie inside the training period."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        k = keys[rng.integers(len(keys))]
        o = int(rng.integers(ctx, end - pred))
        X.append(series[k][o - ctx:o])
        Y.append(series[k][o:o + pred])
    return torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(-1), torch.tensor(np.stack(Y), dtype=torch.float32).unsqueeze(-1)


def adapt(base, X, Y, cfg: Config, method: str, seed: int):
    """Return a fine-tuned copy of `base`. method 'full' trains every weight; 'lora' trains low-rank adapters on the MLP layers."""
    torch.manual_seed(seed)
    ft = cfg.finetune
    model = copy.deepcopy(base)
    if method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=ft.lora_r, lora_alpha=ft.lora_alpha, lora_dropout=ft.lora_dropout,
                                                 target_modules=["fc1", "fc2"]))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=ft.learning_rate)
    n = len(X)
    steps = ft.epochs * int(np.ceil(n / ft.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ft.learning_rate, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(ft.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, ft.batch_size):
            idx = perm[i:i + ft.batch_size]
            loss = model(past_values=X[idx], future_values=Y[idx]).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    if method == "lora":
        model = model.merge_and_unload()
    n_train = sum(p.numel() for p in trainable)
    return model, n_train


def run_finetune(cfg: Config) -> pd.DataFrame:
    """Experiments (all scored on the test period, intervals conformalised on the calib period):
      pooled    adapt on every NF and target, test on the benchmark instances
      cross_nf  adapt on UPF only, test on the other NF types
      cross_tgt adapt on CPU only, test on the other targets
      new_sites adapt on benchmark instances, test on instances never seen in training"""
    df = simulate(cfg)
    series = series_dict(df, cfg)
    train_end = split_bounds(cfg)["train"][1]
    scales = {k: metrics.series_scale(series[k][:train_end], cfg.eval.seasonal_period) for k in series}
    n_inst = cfg.eval.series_per_cell
    bench_keys = [k for k in series if k[1] < n_inst]
    new_keys = [k for k in series if k[1] >= n_inst]
    base = load_ttm_model(cfg.models.foundation.ttm)
    ft = cfg.finetune
    ctx, pred = base.config.context_length, base.config.prediction_length

    pools = {
        "pooled": (bench_keys, bench_keys),
        "cross_nf": ([k for k in bench_keys if k[0] == ft.source_nf], [k for k in bench_keys if k[0] != ft.source_nf]),
        "cross_tgt": ([k for k in bench_keys if k[2] == ft.source_target], [k for k in bench_keys if k[2] != ft.source_target]),
        "new_sites": (bench_keys, new_keys),
    }
    out = []

    def evaluate(model, label, exp, method, frac, keys, n_params, seed):
        d = score_model(TTM(cfg, label=label, model=model), series, cfg, keys, scales)
        d["experiment"], d["method"], d["fraction"], d["trainable_params"], d["seed"] = exp, method, frac, n_params, seed
        out.append(d)

    for exp, (src, tgt) in pools.items():
        evaluate(base, "ttm_zero_shot", exp, "zero_shot", 0.0, tgt, 0, 0)
        for seed in range(cfg.seed, cfg.seed + ft.seeds):
            for frac in ft.fractions:
                n_win = max(8, int(round(frac * ft.max_windows)))
                X, Y = make_windows(series, src, train_end, n_win, ctx, pred, seed)
                for method in ft.methods:
                    t0 = time.time()
                    model, n_params = adapt(base, X, Y, cfg, method, seed)
                    evaluate(model, f"ttm_{method}", exp, method, frac, tgt, n_params, seed)
                    LOG(f"{exp} {method} frac={frac} seed={seed} ({n_win} windows, {n_params} trainable): {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True)
