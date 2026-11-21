"""Pure utility tests for the config loader and the config's own consistency."""
from learned_judgment.env import load_config


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("robot:\n  model: unitree_go2\n")
    assert load_config(path) == {"robot": {"model": "unitree_go2"}}
    assert load_config()["robot"]["model"] == "unitree_go2"  # default: the project's config.yaml


def test_dodge_declares_every_floor_type_a_preset_can_have():
    config = load_config()
    floor_types = set(config["world"]["hidden_effects"]["surface_friction_by_type"])
    assert {p["floor_type"] for p in config["world"]["presets"].values()} <= floor_types
    assert set(config["skills"]["dodge"]["declared_surfaces"]) == floor_types
