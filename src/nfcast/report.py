"""Turn result CSVs into the tables and figures in the README."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import orchestrate
from .config import Config
from .metrics import cluster_bootstrap

ORDER = ["naive", "seasonal_naive", "seasonal_average", "ets", "gbm", "chronos_tiny", "chronos_small", "ttm", "timesfm"]


def _series_id(d: pd.DataFrame) -> np.ndarray:
    return (d.nf + "/" + d.instance.astype(str) + "/" + d.target).to_numpy()


def ci_table(d: pd.DataFrame, by: list[str], metric: str, n_boot: int) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(by, sort=False):
        m, lo, hi = cluster_bootstrap(g[metric].to_numpy(), _series_id(g), n_boot)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), metric: m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def bench_tables(bench: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    nb = cfg.eval.n_boot
    out = {m: ci_table(bench, ["model", "horizon"], m, nb) for m in ("mase", "interval_score", "coverage")}
    top = bench[bench.horizon == max(cfg.eval.horizons)]
    out["by_nf"] = ci_table(top, ["model", "nf"], "mase", nb)
    out["by_target"] = ci_table(top, ["model", "target"], "mase", nb)
    native = bench.dropna(subset=["raw_coverage"])
    if len(native):
        out["native_calibration"] = native.groupby(["model", "horizon"])[["raw_coverage", "coverage"]].mean().reset_index()
    return out


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def fmt_ci(t: pd.DataFrame, metric: str, key: list[str]) -> pd.DataFrame:
    t = t.copy()
    t[metric] = [f"{m:.3f} ({lo:.3f} to {hi:.3f})" for m, lo, hi in zip(t[metric], t.lo, t.hi)]
    return t[key + [metric]]


def plot_bench(bench: pd.DataFrame, cfg: Config, path: str) -> None:
    h = max(cfg.eval.horizons)
    d = bench[bench.horizon == h].groupby(["model", "nf"]).mase.mean().unstack()
    d = d.loc[[m for m in ORDER if m in d.index]]
    ax = d.plot.bar(figsize=(9, 4), width=0.8)
    ax.set_ylabel(f"MASE, {h}-min (lower is better)", fontsize=9)
    ax.set_xlabel("")
    ax.legend(title="nf", loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_finetune(ft: pd.DataFrame, cfg: Config, path: str) -> None:
    h = max(cfg.eval.horizons)
    d = ft[ft.horizon == h]
    exps = list(d.experiment.unique())
    fig, axes = plt.subplots(1, len(exps), figsize=(3.6 * len(exps), 3.4))
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, exps):
        g = d[d.experiment == e]
        ax.axhline(g[g.method == "zero_shot"].mase.mean(), color="grey", ls="--", label="zero-shot")
        for m, gg in g[g.method != "zero_shot"].groupby("method"):
            per = gg.groupby(["fraction", "seed"]).mase.mean().reset_index()
            c = per.groupby("fraction").mase.agg(["mean", "std"])
            ax.errorbar(c.index, c["mean"], yerr=c["std"].fillna(0), marker="o", capsize=3, label=m)
        ax.set_xscale("log")
        ax.set_title(e)
        ax.set_xlabel("share of training budget")
    axes[0].set_ylabel(f"MASE, {h}-min")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_orchestration(summary: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    s = summary.copy()
    s["family"] = np.where(s.policy.str.startswith("reactive"), "reactive", s.policy)
    for fam, g in s.groupby("family"):
        g = g.sort_values("violation_min")
        ax.plot(g.violation_min, g.energy_kwh, marker="o", label=fam, lw=2.5 if fam in ("reactive", "oracle") else 1.2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("violation minutes (total, symlog)")
    ax.set_ylabel("energy (kWh, total)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def build(cfg: Config) -> str:
    rd = cfg.paths.results_dir
    parts = []
    bench = pd.read_csv(os.path.join(rd, "bench_windows.csv"))
    t = bench_tables(bench, cfg)
    plot_bench(bench, cfg, os.path.join(rd, "bench_by_nf.png"))
    for h in cfg.eval.horizons:
        a = fmt_ci(t["mase"][t["mase"].horizon == h], "mase", ["model"]).set_index("model")
        b = t["interval_score"][t["interval_score"].horizon == h].set_index("model").interval_score.map("{:.2f}".format)
        c = t["coverage"][t["coverage"].horizon == h].set_index("model").coverage.map("{:.3f}".format)
        tab = pd.concat([a, b, c], axis=1).loc[[m for m in ORDER if m in a.index]].reset_index()
        parts.append(f"### Benchmark, {h}-minute horizon\n\n" + md(tab))
    if "native_calibration" in t:
        parts.append("### Native vs conformalised coverage (nominal 0.80)\n\n" + md(t["native_calibration"]))
    ft_path = os.path.join(rd, "finetune_windows.csv")
    if os.path.exists(ft_path):
        ft = pd.read_csv(ft_path)
        plot_finetune(ft, cfg, os.path.join(rd, "finetune_curves.png"))
        h = max(cfg.eval.horizons)
        per = ft[ft.horizon == h].groupby(["experiment", "method", "fraction", "seed"]).mase.mean().reset_index()
        per.to_csv(os.path.join(rd, "finetune_summary.csv"), index=False)
        g = per.groupby(["experiment", "method", "fraction"]).mase.agg(["mean", "std"]).reset_index()
        g["mase (mean, sd over seeds)"] = [f"{m:.3f} ({0 if np.isnan(sd) else sd:.3f})" for m, sd in zip(g["mean"], g["std"])]
        parts.append(f"### Adaptation, {h}-minute MASE\n\n" + md(g[["experiment", "method", "fraction", "mase (mean, sd over seeds)"]]))
    op = os.path.join(rd, "orchestration.csv")
    if os.path.exists(op):
        s = orchestrate.summarise(pd.read_csv(op))
        plot_orchestration(s, os.path.join(rd, "orchestration_frontier.png"))
        for budget in (0, 20):
            b = orchestrate.best_at_budget(s, budget)[["family", "headroom", "energy_kwh", "violation_min", "energy_saving_pct"]]
            parts.append(f"### Energy at no more than {budget} violation minutes (all instances)\n\n" + md(b, "{:.2f}"))
    text = "\n\n".join(parts) + "\n"
    with open(os.path.join(rd, "REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text
