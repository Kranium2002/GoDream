"""Pure utility test for the config loader."""
from learned_judgment.env import load_config


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("robot:\n  model: unitree_go2\n")
    assert load_config(path) == {"robot": {"model": "unitree_go2"}}
    assert load_config()["robot"]["model"] == "unitree_go2"  # default: the project's config.yaml
