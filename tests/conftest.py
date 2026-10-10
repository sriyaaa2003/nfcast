import numpy as np
import pytest

from nfcast.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def series(cfg):
    """A small hand-built daily-cycle series per (cell type, target): a unit-test input, not study data."""
    rng = np.random.default_rng(0)
    n = cfg.data.days * cfg.eval.seasonal_period
    t = np.arange(n)
    day = cfg.eval.seasonal_period
    base = 50 + 20 * np.sin(2 * np.pi * t / day) + rng.normal(0, 1.5, n)
    return {("type0", 0, "internet"): base.astype(np.float32), ("type1", 0, "calls"): (0.2 * base).astype(np.float32)}
