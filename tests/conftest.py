import pytest

from nfcast.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sim_df(cfg):
    from nfcast.sim import simulate
    return simulate(cfg)


@pytest.fixture(scope="session")
def series(sim_df, cfg):
    from nfcast.sim import series_dict
    return series_dict(sim_df, cfg)
