import pytest

from learned_judgment.env import load_config


def pytest_configure(config):
    config.addinivalue_line("markers", "system: runs the real Go2 simulator and walking policy")


@pytest.fixture
def config():
    return load_config()
