import numpy as np
import pytest

from nfcast import conformal, metrics


def test_mase_is_zero_for_a_perfect_forecast_and_scales():
    y = np.arange(12.0).reshape(2, 6)
    assert metrics.mase(y, y, 1.0).max() == 0
    assert np.allclose(metrics.mase(y, y + 2, 4.0), 0.5)


def test_pinball_matches_hand_calculation():
    y = np.array([[10.0]])
    q = {0.1: np.array([[8.0]]), 0.9: np.array([[11.0]])}
    # tau=0.1, y above forecast: 0.1*2 = 0.2; tau=0.9, y below forecast: (0.9-1)*(-1) = 0.1; mean 0.15
    assert metrics.pinball(y, q, 1.0)[0] == pytest.approx(0.15)


def test_interval_score_penalises_misses():
    y = np.array([[5.0]])
    inside = metrics.interval_score(y, np.array([[4.0]]), np.array([[6.0]]), 0.2, 1.0)[0]
    miss = metrics.interval_score(y, np.array([[1.0]]), np.array([[3.0]]), 0.2, 1.0)[0]
    assert inside == pytest.approx(2.0)
    assert miss == pytest.approx(2.0 + (2 / 0.2) * 2.0)


def test_coverage_and_width():
    y = np.array([[1.0, 5.0]])
    lo, hi = np.array([[0.0, 0.0]]), np.array([[2.0, 2.0]])
    assert metrics.coverage(y, lo, hi)[0] == 0.5
    assert metrics.width(lo, hi, 2.0)[0] == 1.0


def test_cluster_bootstrap_interval_contains_the_mean():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(20), 15)
    vals = rng.normal(1.0, 0.3, 300) + np.repeat(rng.normal(0, 0.2, 20), 15)
    m, lo, hi = metrics.cluster_bootstrap(vals, clusters, 500)
    assert lo < m < hi


def test_conformal_quantile_has_the_finite_sample_correction():
    scores = np.arange(1, 10, dtype=float).reshape(-1, 1)      # n = 9
    # level = ceil(10 * 0.8) / 9 = 8/9 -> the 8th smallest score
    assert conformal.conf_quantile(scores, 0.2)[0] == 8.0


def test_split_conformal_covers_when_data_are_exchangeable():
    rng = np.random.default_rng(1)
    n, h = 2000, 5
    truth_cal, truth_test = rng.normal(0, 1, (n, h)), rng.normal(0, 1, (n, h))
    pred = np.zeros((n, h))
    lo, hi = conformal.interval_from_point(pred, truth_cal, pred, 1.0, 0.2)
    cov = ((truth_test >= lo) & (truth_test <= hi)).mean()
    assert 0.77 < cov < 0.83


def test_conformal_undercovers_under_shift():
    """The failure the benchmark is built to expose: calibration noise smaller than test noise."""
    rng = np.random.default_rng(2)
    n, h = 2000, 3
    lo, hi = conformal.interval_from_point(np.zeros((n, h)), rng.normal(0, 1, (n, h)), np.zeros((n, h)), 1.0, 0.2)
    shifted = rng.normal(0, 2, (n, h))
    assert ((shifted >= lo) & (shifted <= hi)).mean() < 0.6


def test_cqr_widens_an_overconfident_band():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, (1000, 2))
    narrow_lo, narrow_hi = np.full_like(y, -0.2), np.full_like(y, 0.2)
    lo, hi = conformal.interval_from_quantiles(narrow_lo, narrow_hi, y, narrow_lo, narrow_hi, 1.0, 0.2)
    assert (hi - lo).mean() > 1.5
