"""Pure utility tests for the config loader and the config's own consistency."""
from learned_judgment.env import load_config


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("robot:\n  model: unitree_go2\n")
    assert load_config(path) == {"robot": {"model": "unitree_go2"}}
    assert load_config()["robot"]["model"] == "unitree_go2"  # default: the project's config.yaml


def test_presets_hold_the_values_agreed_with_person_b():
    # MESSAGES.md, 2026-11-21 17:37 UTC: each preset 0.08 s of delay with one control period of jitter; default, then
    # one effect each: 3 kg off-centre, foot friction 0.1 against 0.4, strength 0.8, -5 degrees.
    config = load_config()
    world, he = config["world"], config["world"]["hidden_effects"]
    assert he["command_latency_jitter_s"] == config["robot"]["walking_policy"]["command_schema"]["update_period_s"]
    assert he["surface_friction_by_type"] == {"default": 0.4, "slippery": 0.1}
    default = {"payload_mass_kg": 0.0, "payload_offset_m": [0.0, 0.0], "floor_type": "default", "slope_deg": 0.0,
               "motor_strength_fraction": 1.0, "command_latency_s": 0.08}
    presets = world["presets"]
    assert presets["default"] == default
    assert presets["heavy_payload"]["payload_mass_kg"] == 3.0 and presets["heavy_payload"]["payload_offset_m"] != [0, 0]
    assert presets["slippery_floor"] == default | {"floor_type": "slippery"}
    assert presets["tired_motors"] == default | {"motor_strength_fraction": 0.8}
    assert presets["downhill"] == default | {"slope_deg": -5.0}
    assert {k: v for k, v in presets["heavy_payload"].items() if not k.startswith("payload")} == {
        k: v for k, v in default.items() if not k.startswith("payload")}
    assert set(presets) == {"default", "heavy_payload", "slippery_floor", "tired_motors", "downhill"}


def test_dodge_declares_every_floor_type_a_preset_can_have():
    config = load_config()
    floor_types = set(config["world"]["hidden_effects"]["surface_friction_by_type"])
    assert {p["floor_type"] for p in config["world"]["presets"].values()} <= floor_types
    assert set(config["skills"]["dodge"]["declared_surfaces"]) == floor_types
