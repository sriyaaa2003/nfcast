
### Accuracy, 60-minute horizon (test MASE, 95% CI resampling series)

| model                                                                           |   mase |    lo |    hi |
|:--------------------------------------------------------------------------------|-------:|------:|------:|
| mean(ttm,chronos_small,chronos_tiny,timesfm)                                    |  0.501 | 0.456 | 0.546 |
| stack_nnls(ttm:0.23,chronos_small:0.28,chronos_tiny:0.31,timesfm:0.12,gbm:0.06) |  0.502 | 0.461 | 0.548 |
| mean(ttm,chronos_small,chronos_tiny,timesfm,gbm)                                |  0.503 | 0.46  | 0.546 |
| mean(chronos_small,chronos_tiny,timesfm)                                        |  0.503 | 0.458 | 0.545 |
| mean(ttm,chronos_small,timesfm)                                                 |  0.504 | 0.459 | 0.547 |
| mean(chronos_small,timesfm)                                                     |  0.504 | 0.46  | 0.549 |
| mean(ttm,chronos_small)                                                         |  0.505 | 0.458 | 0.552 |
| mean(ttm,chronos_tiny,timesfm)                                                  |  0.505 | 0.462 | 0.547 |
| mean(chronos_small,chronos_tiny,timesfm,gbm)                                    |  0.506 | 0.464 | 0.549 |
| mean(ttm,chronos_small,chronos_tiny)                                            |  0.506 | 0.46  | 0.55  |
| mean(ttm,chronos_small,timesfm,gbm)                                             |  0.507 | 0.463 | 0.549 |
| mean(ttm,chronos_small,chronos_tiny,gbm)                                        |  0.507 | 0.465 | 0.55  |
| mean(chronos_tiny,timesfm)                                                      |  0.508 | 0.465 | 0.553 |
| mean(ttm,chronos_tiny,timesfm,gbm)                                              |  0.509 | 0.464 | 0.552 |

singles: ttm 0.532, chronos_small 0.518, chronos_tiny 0.541, timesfm 0.543, gbm 0.581

### Accuracy, 360-minute horizon (test MASE, 95% CI resampling series)

| model                                                                           |   mase |    lo |    hi |
|:--------------------------------------------------------------------------------|-------:|------:|------:|
| ttm                                                                             |  0.684 | 0.64  | 0.726 |
| stack_nnls(ttm:0.86,chronos_small:0.00,chronos_tiny:0.01,timesfm:0.00,gbm:0.13) |  0.69  | 0.647 | 0.733 |
| mean(ttm,chronos_small)                                                         |  0.693 | 0.651 | 0.735 |
| mean(ttm,chronos_small,timesfm)                                                 |  0.717 | 0.669 | 0.759 |
| mean(ttm,chronos_small,gbm)                                                     |  0.732 | 0.681 | 0.776 |
| mean(ttm,chronos_small,chronos_tiny,timesfm)                                    |  0.733 | 0.69  | 0.78  |
| mean(ttm,chronos_small,timesfm,gbm)                                             |  0.734 | 0.687 | 0.779 |
| mean(ttm,chronos_small,chronos_tiny)                                            |  0.734 | 0.685 | 0.777 |
| mean(ttm,chronos_tiny)                                                          |  0.738 | 0.692 | 0.783 |
| mean(ttm,chronos_small,chronos_tiny,timesfm,gbm)                                |  0.74  | 0.691 | 0.787 |
| mean(ttm,timesfm)                                                               |  0.743 | 0.698 | 0.788 |
| mean(ttm,chronos_small,chronos_tiny,gbm)                                        |  0.743 | 0.694 | 0.79  |
| mean(ttm,chronos_tiny,timesfm)                                                  |  0.744 | 0.696 | 0.79  |
| mean(ttm,chronos_tiny,timesfm,gbm)                                              |  0.752 | 0.703 | 0.801 |

singles: ttm 0.684, chronos_small 0.769, chronos_tiny 0.901, timesfm 0.856, gbm 0.965

### Conformal intervals (80%) of single models and ensembles, 360-minute horizon

| model                                            |   coverage |   interval_score |
|:-------------------------------------------------|-----------:|-----------------:|
| ttm                                              |      0.833 |            3.825 |
| chronos_small                                    |      0.849 |            4.527 |
| chronos_tiny                                     |      0.832 |            5.205 |
| timesfm                                          |      0.829 |            4.735 |
| gbm                                              |      0.809 |            5.032 |
| mean(ttm,chronos_small)                          |      0.842 |            3.945 |
| mean(ttm,chronos_small,chronos_tiny)             |      0.839 |            4.173 |
| mean(ttm,chronos_small,chronos_tiny,timesfm,gbm) |      0.835 |            4.047 |