### Least capacity (cell-days) with at most N violation steps over the 12 cells

| policy family                     | <= 150   | <= 300   | <= 600   |   <= 1000 |   <= 1500 |
|:----------------------------------|:---------|:---------|:---------|----------:|----------:|
| reactive (tuned)                  | 158.7    | 134.3    | 124.7    |     116.4 |     109.1 |
| chronos_small_q90|calibrated_peak | 160.4    | 127.3    | 121.6    |     113.2 |     108.1 |
| chronos_small_q90|headroom        | 145.2    | 133.2    | 121.5    |     121.5 |     121.5 |
| chronos_small|calibrated_peak     | 194.9    | 159.9    | 134.3    |     124.8 |     112.3 |
| chronos_small|headroom            | -        | -        | -        |     123.6 |     114.7 |
| gbm|calibrated_peak               | 176.2    | 150.4    | 131.8    |     122.7 |     111.6 |
| gbm|headroom                      | -        | -        | 129.2    |     119.8 |     110.8 |
| mean_all|calibrated_peak          | 164.3    | 133.1    | 126.3    |     112.1 |     106.7 |
| mean_all|headroom                 | -        | -        | 125.3    |     116.1 |     107.4 |
| mean_fm|calibrated_peak           | 167.2    | 132.7    | 126.2    |     111.7 |     106.9 |
| mean_fm|headroom                  | -        | -        | 125.0    |     115.8 |     107.1 |
| oracle                            | 107.8    | 107.8    | 107.8    |     107.8 |     107.8 |
| timesfm_q90|calibrated_peak       | 152.4    | 135.2    | 120.9    |     112.7 |     107.6 |
| timesfm_q90|headroom              | 146.4    | 134.4    | 122.6    |     122.6 |     122.6 |
| timesfm|calibrated_peak           | 165.9    | 142.3    | 123.5    |     114.9 |     105   |
| timesfm|headroom                  | -        | -        | 127.6    |     118.3 |     109.3 |
| ttm|calibrated_peak               | 171.9    | 145.9    | 127.2    |     117.8 |     112   |
| ttm|headroom                      | -        | -        | 125.7    |     116.5 |     107.8 |

Baseline reactive rule at 60% utilisation: 145.5 cell-days, 155 violation steps. Oracle (true peak): 107.8 cell-days, 0 violation steps.