import pytest

from learned_judgment.env import Go2World, load_config


def pytest_configure(config):
    config.addinivalue_line("markers", "system: runs the real Go2 simulator and walking policy")


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config["world"]["trace_dir"] = str(tmp_path / "traces")  # test episodes never write into artifacts/env/traces
    return config


@pytest.fixture
def schema(config):
    return config["robot"]["walking_policy"]["command_schema"]


@pytest.fixture
def zero(schema):
    return {name: 0.0 for name in schema["fields"]}


@pytest.fixture
def forward(zero):
    return zero | {"vel_x": 0.6}


@pytest.fixture
def period(schema):
    return schema["update_period_s"]


@pytest.fixture
def world(config):
    return Go2World(config, config["runtime"]["seed"])
