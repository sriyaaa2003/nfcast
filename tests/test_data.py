import numpy as np
import pandas as pd

from nfcast.data import split_bounds, to_matrix, weekly_profile


def test_split_bounds_cover_the_whole_period_without_overlap(cfg):
    b = split_bounds(cfg)
    day = cfg.eval.seasonal_period
    assert b["train"] == (0, cfg.split.train_days * day)
    assert b["train"][1] == b["calib"][0] and b["calib"][1] == b["test"][0] and b["test"][1] == cfg.data.days * day


def test_matrix_places_values_on_the_ten_minute_grid_and_marks_missing_steps(cfg):
    t0 = pd.Timestamp("2013-10-31 23:00:00")
    pool = pd.DataFrame({"t": [t0, t0 + pd.Timedelta(minutes=10), t0 + pd.Timedelta(minutes=30)], "squareid": [7, 7, 7],
                         "internet": [1.0, 2.0, 4.0], "calls": [0.1, 0.2, 0.4], "sms": [0.0, 0.0, 0.0]})
    present, cells, mats = to_matrix(pool, cfg)
    assert cells == [7]
    assert mats["internet"][0, :4].tolist() == [1.0, 2.0, 0.0, 4.0]     # the 20-minute mark has no record: read as zero
    assert present[0, :4].tolist() == [True, True, False, True]


def test_weekly_profile_is_normalised_and_keeps_the_daily_shape(cfg):
    day = cfg.eval.seasonal_period
    steps = cfg.data.days * day
    shape = 1 + np.sin(2 * np.pi * np.arange(steps) / day)
    m = np.stack([10 * shape, 3 * shape]).astype(np.float32)
    prof = weekly_profile(m, cfg)
    assert prof.shape == (2, 7 * day)
    np.testing.assert_allclose(prof.mean(1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(prof[0], prof[1], rtol=1e-4)                # a cell's size does not change its profile
