import numpy as np
import pytest

from nfcast.models import Meta, build_model

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", ["chronos_tiny", "ttm"])
def test_foundation_model_forecasts_a_seasonal_series(cfg, name):
    t = np.arange(5000)
    s = (10 + 3 * np.sin(2 * np.pi * t / 1440)).astype(np.float32)
    m = build_model(name, cfg)
    o = np.array([4500])
    f = m.predict(np.stack([s[o[0] - m.history_length:o[0]]]), 60, Meta(o, ["upf"], ["cpu_cores"]))
    assert f.point.shape == (1, 60) and np.isfinite(f.point).all()
    assert np.abs(f.point[0] - s[4500:4560]).mean() < 1.5       # far better than a flat-zero forecast
    if m.native_quantiles:
        lo, hi = f.quantiles[0.1], f.quantiles[0.9]
        assert (lo <= hi + 1e-6).all()
