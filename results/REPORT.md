### Benchmark, 15-minute horizon

| model | mase | interval_score | coverage |
|---|---|---|---|
| naive | 0.752 (0.675 to 0.834) | 3.47 | 0.792 |
| seasonal_naive | 0.982 (0.951 to 1.012) | 4.51 | 0.814 |
| seasonal_average | 0.853 (0.816 to 0.887) | 3.92 | 0.789 |
| ets | 0.640 (0.583 to 0.695) | 2.99 | 0.804 |
| gbm | 0.597 (0.542 to 0.657) | 2.74 | 0.790 |
| chronos_tiny | 0.599 (0.541 to 0.660) | 2.71 | 0.793 |
| chronos_small | 0.606 (0.546 to 0.667) | 2.75 | 0.791 |
| ttm | 0.613 (0.554 to 0.672) | 2.86 | 0.799 |
| timesfm | 0.568 (0.516 to 0.617) | 2.59 | 0.794 |

### Benchmark, 60-minute horizon

| model | mase | interval_score | coverage |
|---|---|---|---|
| naive | 0.828 (0.752 to 0.904) | 3.88 | 0.798 |
| seasonal_naive | 0.974 (0.951 to 0.997) | 4.43 | 0.803 |
| seasonal_average | 0.840 (0.809 to 0.866) | 3.84 | 0.783 |
| ets | 0.683 (0.626 to 0.737) | 3.28 | 0.806 |
| gbm | 0.651 (0.594 to 0.709) | 3.01 | 0.793 |
| chronos_tiny | 0.688 (0.630 to 0.741) | 3.17 | 0.801 |
| chronos_small | 0.695 (0.635 to 0.749) | 3.22 | 0.800 |
| ttm | 0.707 (0.647 to 0.762) | 3.30 | 0.802 |
| timesfm | 0.649 (0.599 to 0.693) | 3.00 | 0.805 |

### Native vs conformalised coverage (nominal 0.80)

| model | horizon | raw_coverage | coverage |
|---|---|---|---|
| chronos_small | 15 | 0.785 | 0.791 |
| chronos_small | 60 | 0.796 | 0.800 |
| chronos_tiny | 15 | 0.785 | 0.793 |
| chronos_tiny | 60 | 0.793 | 0.801 |
| timesfm | 15 | 0.782 | 0.794 |
| timesfm | 60 | 0.786 | 0.805 |

### Adaptation, 60-minute MASE

| experiment | method | fraction | mase (mean, sd over seeds) |
|---|---|---|---|
| cross_nf | full | 0.010 | 0.739 (0.010) |
| cross_nf | full | 0.050 | 0.716 (0.008) |
| cross_nf | full | 0.200 | 0.704 (0.002) |
| cross_nf | full | 1.000 | 0.702 (0.007) |
| cross_nf | lora | 0.010 | 0.728 (0.000) |
| cross_nf | lora | 0.050 | 0.724 (0.001) |
| cross_nf | lora | 0.200 | 0.714 (0.001) |
| cross_nf | lora | 1.000 | 0.693 (0.001) |
| cross_nf | zero_shot | 0.000 | 0.729 (0.000) |
| cross_tgt | full | 0.010 | 0.669 (0.017) |
| cross_tgt | full | 0.050 | 0.670 (0.011) |
| cross_tgt | full | 0.200 | 0.649 (0.001) |
| cross_tgt | full | 1.000 | 0.629 (0.003) |
| cross_tgt | lora | 0.010 | 0.675 (0.000) |
| cross_tgt | lora | 0.050 | 0.672 (0.000) |
| cross_tgt | lora | 0.200 | 0.661 (0.001) |
| cross_tgt | lora | 1.000 | 0.646 (0.001) |
| cross_tgt | zero_shot | 0.000 | 0.677 (0.000) |
| new_sites | full | 0.010 | 0.705 (0.016) |
| new_sites | full | 0.050 | 0.690 (0.010) |
| new_sites | full | 0.200 | 0.687 (0.002) |
| new_sites | full | 1.000 | 0.674 (0.009) |
| new_sites | lora | 0.010 | 0.709 (0.000) |
| new_sites | lora | 0.050 | 0.706 (0.000) |
| new_sites | lora | 0.200 | 0.695 (0.001) |
| new_sites | lora | 1.000 | 0.678 (0.001) |
| new_sites | zero_shot | 0.000 | 0.710 (0.000) |
| pooled | full | 0.010 | 0.705 (0.019) |
| pooled | full | 0.050 | 0.686 (0.009) |
| pooled | full | 0.200 | 0.682 (0.001) |
| pooled | full | 1.000 | 0.673 (0.006) |
| pooled | lora | 0.010 | 0.705 (0.000) |
| pooled | lora | 0.050 | 0.702 (0.000) |
| pooled | lora | 0.200 | 0.693 (0.001) |
| pooled | lora | 1.000 | 0.671 (0.002) |
| pooled | zero_shot | 0.000 | 0.707 (0.000) |

### Energy at no more than 0 violation minutes (all instances)

| family | headroom | energy_kwh | violation_min | energy_saving_pct |
|---|---|---|---|---|
| oracle | 0.00 | 232.56 | 0 | 6.04 |
| reactive_tuned | nan | 238.18 | 0 | 3.77 |

### Energy at no more than 20 violation minutes (all instances)

| family | headroom | energy_kwh | violation_min | energy_saving_pct |
|---|---|---|---|---|
| reactive_tuned | nan | 230.76 | 18 | 6.77 |
| ttm | 0.20 | 232.51 | 12 | 6.06 |
| oracle | 0.00 | 232.56 | 0 | 6.04 |
| timesfm | 0.20 | 232.63 | 9 | 6.01 |
| gbm | 0.20 | 233.02 | 6 | 5.85 |
| chronos_small | 0.20 | 233.76 | 7 | 5.55 |
| naive | 0.30 | 235.71 | 9 | 4.76 |
| seasonal_naive | 0.10 | 239.23 | 17 | 3.34 |
