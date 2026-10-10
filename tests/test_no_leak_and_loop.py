import numpy as np
import pytest

from nfcast import orchestrate
from nfcast.models import Meta, build_model
from nfcast.windows import future, gather, min_history, origins_for


def test_gather_ends_before_the_origin():
    s = np.arange(10000, dtype=np.float32)
    h = gather(s, np.array([5000, 6000]), 100)
    assert h[0, -1] == 4999 and h[1, 0] == 5900


def test_gather_rejects_origins_without_enough_history():
    with pytest.raises(ValueError):
        gather(np.zeros(100), np.array([10]), 50)


def test_origins_stay_inside_their_split_and_leave_room_for_the_horizon(cfg):
    from nfcast.data import split_bounds
    for split in ("calib", "test"):
        o = origins_for(cfg, split, 36)
        a, b = split_bounds(cfg)[split]
        assert o.min() >= max(a, min_history(cfg)) and o.max() + 36 <= b


@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "seasonal_average"])
def test_forecast_does_not_depend_on_the_future(cfg, series, name):
    """Overwrite everything from the origin onward: a leak-free forecast cannot change."""
    k = next(iter(series))
    s = series[k].copy()
    o = np.array([cfg.data.days * cfg.eval.seasonal_period - 300])
    m = build_model(name, cfg)
    before = m.predict(gather(s, o, m.history_length), 36, Meta(o, [k[0]], [k[2]])).point
    s2 = s.copy()
    s2[o[0]:] = 1e6
    after = m.predict(gather(s2, o, m.history_length), 36, Meta(o, [k[0]], [k[2]])).point
    assert np.array_equal(before, after)


def test_future_slices_start_at_the_origin():
    s = np.arange(1000, dtype=np.float32)
    assert future(s, np.array([100]), 5)[0].tolist() == [100, 101, 102, 103, 104]


def test_seasonal_naive_returns_the_value_a_day_earlier(cfg):
    day = cfg.eval.seasonal_period
    s = np.arange(5000, dtype=np.float32)
    o = np.array([3000])
    m = build_model("seasonal_naive", cfg)
    f = m.predict(gather(s, o, m.history_length), 3).point[0]
    assert f.tolist() == [3000 - day, 3001 - day, 3002 - day]


def test_reactive_capacity_covers_the_recent_peak_at_the_target_utilisation():
    demand = np.array([1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    cap = orchestrate.reactive_capacity(demand, util=0.5, delay=2, start=3, end=8)
    # at step 3 the last 3 observed steps (0..2) include the peak of 4; the peak leaves the window at step 6
    assert cap.tolist() == [8.0, 8.0, 8.0, 2.0, 2.0]


def test_plan_capacity_never_drops_below_the_last_observed_step():
    demand = np.array([2.0] * 3 + [9.0] + [2.0] * 4)
    cap = orchestrate.plan_capacity(np.array([1.0, 1.0]), np.array([3, 6]), demand, headroom=0.0, plan_util=0.5, start=3, end=8)
    # forecast plan = 1 / 0.5 = 2, but the floor is demand[m - 1] / 0.5: step 4 follows the observed 9
    assert cap.tolist() == [4.0, 18.0, 4.0, 4.0, 4.0]


def test_account_counts_violations_and_scale_events(cfg):
    demand = np.array([1.0, 1.0, 3.9, 3.9])
    cap = np.array([4.0, 4.0, 4.0, 8.0])
    r = orchestrate.account(demand, cap, scale=2.0, cfg=cfg)
    assert r["violation_steps"] == 1                # step 2: 3.9 > 0.9 * 4; step 3 has capacity 8
    assert r["scale_events"] == 1
    assert r["capacity_cell_days"] == pytest.approx((4 + 4 + 4 + 8) / 2.0 / cfg.eval.seasonal_period)


def test_best_at_budget_compares_forecasts_with_the_best_reactive_setting():
    import pandas as pd
    s = pd.DataFrame({"policy": ["reactive", "reactive_0.8", "ttm"], "headroom": [np.nan, np.nan, 0.1],
                      "capacity_cell_days": [10.0, 8.0, 7.0], "violation_steps": [0, 5, 0],
                      "scale_events": [1, 1, 1], "mean_capacity": [1, 1, 1],
                      "capacity_saving_pct": [0, 20, 30], "violation_change": [0, 5, 0]})
    best = orchestrate.best_at_budget(s, 0)
    assert set(best.family) == {"reactive_tuned", "ttm"} and best.iloc[0].family == "ttm"
    assert orchestrate.dominated_by_reactive(s).tolist() == [False, False, False]
