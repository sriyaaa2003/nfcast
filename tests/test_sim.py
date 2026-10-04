import dataclasses

import numpy as np

from nfcast.config import TARGETS
from nfcast.sim import hpa_replicas, simulate, split_bounds


def test_shape_and_determinism(cfg, sim_df):
    n_series = sum(p.instances for p in cfg.sim.nf_profiles)
    assert len(sim_df) == n_series * cfg.sim.days * 1440
    again = simulate(cfg)
    assert np.array_equal(sim_df[list(TARGETS)].to_numpy(), again[list(TARGETS)].to_numpy())


def test_different_seed_changes_data(cfg, sim_df):
    other = simulate(dataclasses.replace(cfg, seed=cfg.seed + 1))
    assert not np.array_equal(sim_df.cpu_cores.to_numpy(), other.cpu_cores.to_numpy())


def test_values_are_physical(sim_df, cfg):
    assert (sim_df.cpu_cores > 0).all() and (sim_df.mem_gib > 0).all() and (sim_df.energy_w > 0).all()
    h = cfg.sim.hpa
    assert sim_df.replicas.between(h.min_replicas, h.max_replicas).all()
    assert not sim_df[list(TARGETS)].isna().any().any()


def test_traffic_has_a_daily_cycle(sim_df):
    t = sim_df[(sim_df.nf == "upf") & (sim_df.instance == 0)].traffic_mbps.to_numpy()
    days = t[: (len(t) // 1440) * 1440].reshape(-1, 1440)
    prof = days.mean(0)
    # the average day is clearly non-flat: afternoon/evening vs early morning
    assert prof.max() / prof.min() > 1.5


def test_hpa_scales_up_at_once_and_down_only_after_the_window(cfg):
    demand = np.concatenate([np.full(30, 4.0), np.full(10, 12.0), np.full(60, 4.0)])
    rep = hpa_replicas(demand, 4.0, cfg)
    w = cfg.sim.hpa.down_stabilization_minutes
    assert rep[30] <= rep[31]                                  # up: the very next minute
    assert rep[31] > rep[29]
    assert rep[40 + w - 2] == rep[41]                          # still held high inside the stabilisation window
    assert rep[-1] < rep[41]                                   # eventually scaled down


def test_splits_do_not_overlap(cfg):
    b = split_bounds(cfg)
    assert b["train"][1] == b["calib"][0] and b["calib"][1] == b["test"][0]
    assert b["test"][1] == cfg.sim.days * 1440


def test_nf_types_behave_differently(sim_df):
    """Cross-NF generalisation is only meaningful if the NFs are not copies of each other."""
    g = sim_df.groupby("nf").cpu_cores
    cv = (g.std() / g.mean())
    assert cv.max() / cv.min() > 1.2
