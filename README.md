# nfcast: do time-series foundation models forecast real cellular traffic better?

Pretrained time-series foundation models (Chronos-Bolt, Tiny Time Mixer, TimesFM) against classical baselines on **real operator
traffic**, with calibrated intervals, cheap adaptation, and a capacity-planning replay that asks whether a better forecast actually saves
anything. There is no simulator and no synthetic telemetry.

**Data.** The [Telecom Italia Big Data Challenge](https://www.nature.com/articles/sdata201555) (Barlacchi et al., *Scientific Data* 2015;
ODbL): every 10 minutes, SMS, call and internet activity per 235 m grid square of Milan, 1 Nov to 5 Dec 2013 (35 days, 144 steps a day;
daily files from a public Hugging Face mirror). From the 1,500 busiest squares, four **cell types** are found by clustering the weekly
activity profile of the first 21 days (weekday peaks at 13:00, 13:30, 14:00 and 21:00); 12 squares are drawn at random from each. The
benchmark uses 4 cells per type and 3 targets (internet, calls, SMS): **48 series**, trained on 21 days, calibrated on 5, **tested on 9**.
The other 8 cells per type are never used in the benchmark and serve as "new cells" for adaptation.

## Benchmark (test period; intervals resample whole series)

MASE is the error divided by each series' day-over-day change in training data (lower is better); coverage is of 80% intervals.

| model | 1 hour ahead | 6 hours ahead | coverage at 6 h |
|---|---|---|---|
| naive | 0.650 | 1.547 | 0.784 |
| seasonal naive | 0.915 | 0.915 | 0.859 |
| Holt-Winters (own numpy implementation) | 0.939 | 1.244 | 0.820 |
| gradient boosting | 0.581 (0.537 to 0.626) | 0.965 (0.900 to 1.031) | 0.809 |
| Chronos-Bolt tiny | 0.541 | 0.901 | 0.820 |
| **Chronos-Bolt small** | **0.518** (0.471 to 0.565) | 0.769 (0.721 to 0.817) | 0.837 |
| TimesFM 2.0 | 0.543 | 0.856 (0.803 to 0.910) | 0.837 |
| **Tiny Time Mixer** (zero-shot) | 0.532 | **0.684** (0.641 to 0.732) | 0.833 |

- **Zero-shot foundation models beat the classical baselines at both horizons**, with no training on this data: at 1 hour by 7% to 11%
  against gradient boosting, and at 6 hours the best, TTM (0.684), is 29% better than gradient boosting and 25% better than seasonal naive.
  The 1-hour differences among the four foundation models are inside the intervals; at 6 hours TTM and Chronos-Bolt small lead.
- **Seasonal baselines are hard to beat far ahead** because Milan's daily cycle is strong (seasonal naive 0.915 at both horizons), which
  is why the 6-hour gaps are large for naive and Holt-Winters.
- **Intervals are close to calibrated out of the box** (native coverage 0.78 to 0.83 against 0.80); conformal calibration changes little
  here, partly because calibration and test weeks share a regime, which a real shift would break.
- *Caveat:* the models' pretraining corpora are not fully public, so overlap with Milan data cannot be ruled out.

## Adaptation (TTM, 6-hour MASE, 3 seeds)

| experiment | zero-shot | LoRA, 1% of budget | LoRA, full budget | full fine-tuning, 1% | full fine-tuning, full budget |
|---|---|---|---|---|---|
| pooled (all types and targets) | 0.684 | 0.684 | 0.667 | 0.729 | 0.683 |
| train on one cell type, test on the others | 0.678 | 0.677 | 0.670 | 0.714 | 0.699 |
| train on internet only, test on calls and SMS | 0.725 | 0.724 | 0.702 | 0.777 | 0.694 |
| new cells never seen | 0.619 | 0.619 | 0.607 | 0.654 | 0.621 |

LoRA (6% of the parameters) is stable and gives a small gain only at the full budget (about 0.01 to 0.02 MASE). Full fine-tuning on a small
budget is **worse than zero-shot**. Adapting does not change the ranking of the models.

## Does a better forecast save capacity? (replay on the real traffic, no power model)

Policies set the capacity of 12 cells every hour and are scored on capacity provisioned and on steps where demand exceeded 90% of capacity.
The baseline is a reactive rule (capacity covers the last 40 minutes' peak at 60% utilisation: 155 violation steps in all, 145.5 cell-days).

- The same forecasts with a headroom margin trace a frontier: for example TTM with 20% headroom uses **19.9% less capacity** than the
  baseline at 820 violation steps, TimesFM 18.7% less at 606.
- The right comparison is the reactive rule tuned hotter, which also trades capacity for violations (at 70% utilisation: 14.3% less capacity,
  434 violation steps). Forecast policies sit on or slightly beyond that frontier, not far beyond it: at about 1,300 violation steps the
  tuned reactive rule (109.1 cell-days) and TimesFM or TTM with 10% headroom (107.8 to 109.3) are within 2%.
- **A perfect forecast saves 25.9% with zero violations**, so the ceiling is large and today's models capture only a small part of it.

![capacity against violations](results/orchestration_frontier.png)

## Run

```bash
pip install -e .
python -m nfcast.cli all     # download (about 5 GB), panel, benchmark, adaptation, replay, report; several hours on a CPU
pytest                       # 25 tests (foundation-model smoke tests are marked slow)
```

Settings are in `configs/default.yaml`. Details of the evaluation: models see exactly the history they need and nothing after the forecast
origin (tested); split-conformal intervals (CQR for models with native quantiles) are calibrated on a separate period; the Holt-Winters
baseline is reimplemented in numpy because statsmodels is blocked on this machine. Limits: one city, five weeks, 10-minute resolution,
cell-level traffic rather than network-function telemetry, and the capacity replay compares policy rules, not a deployed system.
