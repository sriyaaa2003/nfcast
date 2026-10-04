# nfcast

Do time-series foundation models forecast cloud-native 5G network-function (NF) telemetry better than classical methods, can they be adapted cheaply, and does a better forecast actually save energy?

`nfcast` answers this on a simulator with known dynamics (no operator data is used or claimed). It simulates one-minute telemetry (traffic, CPU, pod utilisation, memory, power) for UPF, AMF, SMF and CU-UP instances running under a reactive Kubernetes-style autoscaler, then benchmarks forecasters, adapts a foundation model, and closes the loop with an autoscaling experiment.

## What is in it

- **Simulator** (`sim.py`): bursty heavy-tailed traffic, diurnal and weekend cycles, surge events, registration storms (AMF), memory-leak sawtooth restarts (UPF, SMF), a periodic housekeeping task (CU-UP), reactive HPA with a scale-down stabilisation window, nonlinear power model. The four NF types behave differently on purpose, so cross-NF transfer is a real test.
- **Models**: naive, seasonal naive, seasonal average, Holt-Winters ETS, global gradient boosting, and the foundation models Chronos-Bolt (tiny, small), Tiny Time Mixer (TTM r2) and TimesFM 2.0 (500M). Every model gets exactly the history it needs and nothing after the forecast origin (tested).
- **Calibrated intervals**: split conformal for every model (CQR for the models with native quantiles), calibrated on a separate period, so intervals are comparable. Nominal coverage 0.80.
- **Adaptation**: TTM fine-tuned fully and with LoRA at 1, 5, 20 and 100 percent of a 4000-window budget, three seeds each; transfer to unseen NF types, unseen targets and unseen instances.
- **Closed loop**: proactive scaling from each forecast versus the reactive autoscaler tuned at several targets, scored on energy and SLA-violation minutes.
- 28 tests (simulator properties, metric correctness, conformal coverage and its failure under shift, no look-ahead, accounting).

## Run it

```bash
pip install -e .
python -m nfcast.cli all        # about 1.5 hours on a CPU (TimesFM needs a ~2 GB download); writes results/
pytest                          # 28 tests; foundation-model smoke tests are marked slow
```

Config is in `configs/default.yaml` (seed 11, 21 days, 6 instances per NF).

## Results

Test period, 80 series. Intervals in brackets come from resampling whole series (windows of one series are correlated). MASE is error divided by each series' day-over-day change in training data; lower is better.

| model | MASE 15 min | MASE 60 min | coverage 60 min |
|---|---|---|---|
| naive | 0.752 | 0.828 | 0.798 |
| seasonal naive | 0.982 | 0.974 | 0.803 |
| ETS | 0.640 | 0.683 | 0.806 |
| gradient boosting | 0.597 (0.542 to 0.657) | 0.651 (0.594 to 0.709) | 0.793 |
| Chronos-Bolt tiny | 0.599 | 0.688 | 0.801 |
| Chronos-Bolt small | 0.606 | 0.695 | 0.800 |
| TTM, zero-shot | 0.613 | 0.707 | 0.802 |
| **TimesFM 2.0, zero-shot** | **0.568** (0.516 to 0.617) | **0.649** (0.599 to 0.693) | 0.805 |

Full tables are in [`results/REPORT.md`](results/REPORT.md).

![MASE by NF type](results/bench_by_nf.png)

1. **Zero-shot TimesFM is the best model at 15 minutes and ties gradient boosting at 60.** Its 15-minute interval (0.516 to 0.617) overlaps gradient boosting's (0.542 to 0.657), so the gap is suggestive, not established. The smaller foundation models match the best classical models at 15 minutes and trail slightly at 60; all of them beat naive, seasonal and ETS only at the short horizon. A tuned gradient-boosting model is hard to beat.
2. **CU-UP is the hardest NF type for every model** (periodic housekeeping spikes). At 60 minutes TimesFM scores 0.753 there, against 0.851 for gradient boosting and 0.89 to 0.92 for Chronos-Bolt and TTM.
3. **The foundation models' native intervals are already well calibrated** (raw coverage 0.78 to 0.80 against a nominal 0.80), so conformal calibration changes little here. That is partly because the simulator is stationary between calibration and test; this will not hold under real shift.
4. **Adaptation helps modestly and needs data.** Three seeds, 60-minute MASE, TTM:

![Adaptation curves](results/finetune_curves.png)

   Zero-shot is 0.707 (pooled). Full fine-tuning reaches 0.673 and LoRA 0.671 at the full budget, with LoRA training 6 percent of the parameters (50,688 of 805,280). With 1 percent of the budget full fine-tuning is no better than zero-shot and noisy (cross-NF: 0.739, worse than zero-shot 0.729); LoRA is stable but also gains little at that size. Adaptation transfers to unseen instances (0.710 to 0.674 and 0.678) and unseen targets (0.677 to 0.629 and 0.646); adapting on UPF only gives a smaller gain on the other NF types (0.729 to 0.702 and 0.693). Seed variation is small for LoRA (sd 0.002 or less) and larger for full fine-tuning at small budgets (up to 0.019). Adapted TTM still does not reach zero-shot TimesFM (0.649).
5. **Better forecasts save little energy.** Provisioning for the forecast peak over the next hour, with a reactive floor, uses 5.5 to 6.1 percent less energy than the deployed autoscaler at 20 or fewer violation minutes (TimesFM 6.0 percent with 9 violation minutes). But the same reactive autoscaler tuned to a hotter target does as well (6.8 percent, 18 violation minutes), and a perfect forecast saves 6.0 percent with zero violations against 3.8 percent for the best zero-violation reactive setting. So the ceiling for forecasting here is about 2 percent of energy.

![Energy against violations](results/orchestration_frontier.png)

## What is not measured

- No real operator or testbed telemetry. Everything describes this generator; model rankings may differ on real traffic.
- Calibration is evaluated without distribution shift between calibration and test.
- Fine-tuning is only done for TTM, not TimesFM or Chronos, and no model had its hyperparameters tuned beyond defaults (the gradient-boosting baseline is trained on this simulator's training period).
- Energy is a modelled power curve, not measured; the closed loop replays recorded demand and does not model how scaling changes demand or pod start-up delays.
- Not tested: a real Kubernetes cluster, other NF types, targets beyond the five here, forecast horizons beyond 60 minutes.
