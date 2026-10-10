### Benchmark, 60-minute horizon

| model | mase | interval_score | coverage |
|---|---|---|---|
| naive | 0.650 (0.592 to 0.705) | 3.77 | 0.800 |
| seasonal_naive | 0.915 (0.860 to 0.970) | 5.95 | 0.857 |
| seasonal_average | 0.955 (0.884 to 1.026) | 6.00 | 0.868 |
| ets | 0.939 (0.878 to 0.997) | 5.80 | 0.789 |
| gbm | 0.581 (0.537 to 0.626) | 3.31 | 0.804 |
| chronos_tiny | 0.541 (0.494 to 0.588) | 2.52 | 0.794 |
| chronos_small | 0.518 (0.471 to 0.565) | 2.43 | 0.804 |
| ttm | 0.532 (0.484 to 0.579) | 2.96 | 0.818 |
| timesfm | 0.543 (0.495 to 0.590) | 2.50 | 0.828 |

### Benchmark, 360-minute horizon

| model | mase | interval_score | coverage |
|---|---|---|---|
| naive | 1.547 (1.428 to 1.664) | 7.98 | 0.784 |
| seasonal_naive | 0.915 (0.857 to 0.978) | 5.87 | 0.859 |
| seasonal_average | 0.954 (0.883 to 1.030) | 5.94 | 0.867 |
| ets | 1.244 (1.155 to 1.335) | 7.12 | 0.820 |
| gbm | 0.965 (0.900 to 1.031) | 5.03 | 0.809 |
| chronos_tiny | 0.901 (0.835 to 0.975) | 4.44 | 0.820 |
| chronos_small | 0.769 (0.721 to 0.817) | 3.80 | 0.837 |
| ttm | 0.684 (0.641 to 0.732) | 3.83 | 0.833 |
| timesfm | 0.856 (0.803 to 0.910) | 4.05 | 0.837 |

### Native vs conformalised coverage (nominal 0.80)

| model | horizon | raw_coverage | coverage |
|---|---|---|---|
| chronos_small | 6 | 0.828 | 0.804 |
| chronos_small | 36 | 0.829 | 0.837 |
| chronos_tiny | 6 | 0.808 | 0.794 |
| chronos_tiny | 36 | 0.790 | 0.820 |
| timesfm | 6 | 0.813 | 0.828 |
| timesfm | 36 | 0.782 | 0.837 |

### Adaptation, 360-minute MASE

| experiment | method | fraction | mase (mean, sd over seeds) |
|---|---|---|---|
| cross_tgt | full | 0.010 | 0.777 (0.023) |
| cross_tgt | full | 0.050 | 0.751 (0.012) |
| cross_tgt | full | 0.200 | 0.742 (0.014) |
| cross_tgt | full | 1.000 | 0.694 (0.012) |
| cross_tgt | lora | 0.010 | 0.724 (0.001) |
| cross_tgt | lora | 0.050 | 0.722 (0.002) |
| cross_tgt | lora | 0.200 | 0.715 (0.001) |
| cross_tgt | lora | 1.000 | 0.702 (0.009) |
| cross_tgt | zero_shot | 0.000 | 0.725 (0.000) |
| cross_type | full | 0.010 | 0.714 (0.013) |
| cross_type | full | 0.050 | 0.694 (0.021) |
| cross_type | full | 0.200 | 0.730 (0.033) |
| cross_type | full | 1.000 | 0.699 (0.009) |
| cross_type | lora | 0.010 | 0.677 (0.000) |
| cross_type | lora | 0.050 | 0.676 (0.000) |
| cross_type | lora | 0.200 | 0.670 (0.002) |
| cross_type | lora | 1.000 | 0.670 (0.007) |
| cross_type | zero_shot | 0.000 | 0.678 (0.000) |
| new_cells | full | 0.010 | 0.654 (0.018) |
| new_cells | full | 0.050 | 0.638 (0.010) |
| new_cells | full | 0.200 | 0.633 (0.019) |
| new_cells | full | 1.000 | 0.621 (0.012) |
| new_cells | lora | 0.010 | 0.619 (0.001) |
| new_cells | lora | 0.050 | 0.617 (0.000) |
| new_cells | lora | 0.200 | 0.614 (0.001) |
| new_cells | lora | 1.000 | 0.607 (0.009) |
| new_cells | zero_shot | 0.000 | 0.619 (0.000) |
| pooled | full | 0.010 | 0.729 (0.023) |
| pooled | full | 0.050 | 0.711 (0.011) |
| pooled | full | 0.200 | 0.702 (0.024) |
| pooled | full | 1.000 | 0.683 (0.012) |
| pooled | lora | 0.010 | 0.684 (0.001) |
| pooled | lora | 0.050 | 0.682 (0.000) |
| pooled | lora | 0.200 | 0.678 (0.001) |
| pooled | lora | 1.000 | 0.667 (0.009) |
| pooled | zero_shot | 0.000 | 0.684 (0.000) |

### Capacity at no more than 200 violation steps (all cells)

| family | headroom | capacity_cell_days | violation_steps | capacity_saving_pct |
|---|---|---|---|---|
| oracle | 0.00 | 107.81 | 0 | 25.89 |
| reactive_tuned | nan | 145.46 | 155 | 0.00 |

### Capacity at no more than 1000 violation steps (all cells)

| family | headroom | capacity_cell_days | violation_steps | capacity_saving_pct |
|---|---|---|---|---|
| oracle | 0.00 | 107.81 | 0 | 25.89 |
| ttm | 0.20 | 116.50 | 820 | 19.91 |
| timesfm | 0.20 | 118.27 | 606 | 18.70 |
| gbm | 0.20 | 119.75 | 802 | 17.67 |
| chronos_small | 0.30 | 123.62 | 736 | 15.01 |
| reactive_tuned | nan | 124.68 | 434 | 14.29 |
| seasonal_naive | 0.20 | 136.73 | 781 | 6.00 |
