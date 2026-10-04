import numpy as np
import pytest

from nfcast import orchestrate
from nfcast.models import Meta, build_model
from nfcast.windows import MIN_HISTORY, future, gather, origins_for


def test_gather_ends_before_the_origin():
    s = np.arange(10000, dtype=np.float32)
    h = gather(s, np.array([5000, 6000]), 100)
    assert h[0, -1] == 4999 and h[1, 0] == 5900


def test_gather_rejects_origins_without_enough_history():
    with pytest.raises(ValueError):
        gather(np.zeros(100), np.array([10]), 50)


def test_origins_stay_inside_their_split_and_leave_room_for_the_horizon(cfg):
    from nfcast.sim import split_bounds
    for split in ("calib", "test"):
        o = origins_for(cfg, split, 60)
        a, b = split_bounds(cfg)[split]
        assert o.min() >= max(a, MIN_HISTORY) and o.max() + 60 <= b


@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "seasonal_average"])
def test_forecast_does_not_depend_on_the_future(cfg, series, name):
    """Overwrite everything from the origin onward: a leak-free forecast cannot change."""
    k = next(iter(series))
    s = series[k].copy()
    o = np.array([cfg.sim.days * 1440 - 3000])
    m = build_model(name, cfg)
    before = m.predict(gather(s, o, m.history_length), 60, Meta(o, [k[0]], [k[2]])).point
    s2 = s.copy()
    s2[o[0]:] = 1e6
    after = m.predict(gather(s2, o, m.history_length), 60, Meta(o, [k[0]], [k[2]])).point
    assert np.array_equal(before, after)


def test_future_slices_start_at_the_origin():
    s = np.arange(1000, dtype=np.float32)
    assert future(s, np.array([100]), 5)[0].tolist() == [100, 101, 102, 103, 104]


def test_seasonal_naive_returns_the_value_a_day_earlier(cfg):
    s = np.arange(5000, dtype=np.float32)
    o = np.array([3000])
    m = build_model("seasonal_naive", cfg)
    f = m.predict(gather(s, o, m.history_length), 3).point[0]
    assert f.tolist() == [3000 - 1440, 3001 - 1440, 3002 - 1440]


def test_power_rises_with_load_and_with_replicas(cfg):
    p = orchestrate.power
    assert p(np.array([0.9]), np.array([2.0]), cfg) > p(np.array([0.3]), np.array([2.0]), cfg)
    assert p(np.array([0.5]), np.array([4.0]), cfg) > p(np.array([0.5]), np.array([2.0]), cfg)


def test_account_counts_violations_and_scale_events(cfg):
    demand = np.array([1.0, 1.0, 3.9, 3.9])
    rep = np.array([1.0, 1.0, 1.0, 2.0])
    r = orchestrate.account(demand, rep, 4.0, cfg)
    assert r["violation_min"] == 1                 # minute 2: 3.9 / (1*4) = 0.975 > 0.9; minute 3 has two replicas
    assert r["scale_events"] == 1
