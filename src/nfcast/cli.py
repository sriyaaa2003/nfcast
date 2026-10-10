"""Command line: nfcast download | panel | bench | finetune | orchestrate | report | all."""
from __future__ import annotations

import argparse
import os
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nfcast", description="Time-series foundation models on real cellular traffic.")
    ap.add_argument("command", choices=["download", "panel", "bench", "finetune", "orchestrate", "report", "all"])
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml or $NFCAST_CONFIG)")
    ap.add_argument("--models", nargs="*", help="bench only: restrict to these models")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out = lambda n: os.path.join(cfg.paths.results_dir, n)
    cmds = ["download", "panel", "bench", "finetune", "orchestrate", "report"] if a.command == "all" else [a.command]
    for c in cmds:
        if c == "download":
            from .data import download
            download(cfg)
        elif c == "panel":
            import json
            from .data import build_panel
            df, info = build_panel(cfg)
            df.to_parquet(os.path.join(cfg.paths.data_dir, "panel.parquet"))
            with open(out("panel_summary.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            print(json.dumps(info, indent=2))
        elif c == "bench":
            from .bench import run_benchmark
            new = run_benchmark(cfg, a.models)
            if a.models and os.path.exists(out("bench_windows.csv")):          # partial run: replace just those models
                import pandas as pd
                old = pd.read_csv(out("bench_windows.csv"))
                new = pd.concat([old[~old.model.isin(a.models)], new], ignore_index=True)
            new.to_csv(out("bench_windows.csv"), index=False)
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
