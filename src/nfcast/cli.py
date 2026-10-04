"""Command line: nfcast sim | bench | finetune | orchestrate | report | all."""
from __future__ import annotations

import argparse
import os
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nfcast", description="Foundation models for 5G network-function telemetry forecasting.")
    ap.add_argument("command", choices=["sim", "bench", "finetune", "orchestrate", "report", "all"])
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml or $NFCAST_CONFIG)")
    ap.add_argument("--models", nargs="*", help="bench only: restrict to these models")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out = lambda n: os.path.join(cfg.paths.results_dir, n)
    cmds = ["bench", "finetune", "orchestrate", "report"] if a.command == "all" else [a.command]
    for c in cmds:
        if c == "sim":
            from .sim import simulate
            df = simulate(cfg)
            print(df.groupby("nf")[["traffic_mbps", "cpu_cores", "cpu_util_pod", "mem_gib", "energy_w", "replicas"]].mean().round(2))
        elif c == "bench":
            from .bench import run_benchmark
            run_benchmark(cfg, a.models).to_csv(out("bench_windows.csv"), index=False)
        elif c == "finetune":
            from .finetune import run_finetune
            run_finetune(cfg).to_csv(out("finetune_windows.csv"), index=False)
        elif c == "orchestrate":
            from .orchestrate import run_orchestration
            run_orchestration(cfg).to_csv(out("orchestration.csv"), index=False)
        elif c == "report":
            from .report import build
            print(build(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
