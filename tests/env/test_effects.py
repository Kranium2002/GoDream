"""Hidden effects and presets on the real simulator and walking policy: each effect switches on its own and is
measured, the presets load and stand, and the hidden values never reach Person B."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from learned_judgment.env import Go2World, load_config
from learned_judgment.env.effects import measure_effects

pytestmark = pytest.mark.system
EFFECTS = ["payload", "friction", "slope", "motor_strength", "latency"]
PRESETS = ["default", "heavy_payload", "slippery_floor", "tired_motors", "downhill"]


@pytest.fixture(scope="module")
def e(tmp_path_factory):
    """One measure_effects run (eleven episodes) shared by the tests that read it."""
    return json.loads(measure_effects(load_config(), seed=5, out_dir=tmp_path_factory.mktemp("effects")).read_text())


@pytest.fixture
def cfg(config, tmp_path):
    return config | {"world": config["world"] | {"trace_dir": str(tmp_path)}}


def test_all_effects_off_equals_baseline(e):
    assert e["all_off"] == e["baseline"]


@pytest.mark.parametrize("effect", EFFECTS)
def test_one_effect_changes_only_its_measurement(e, effect):
    assert e[effect]["measured"] != e["baseline"][e[effect]["quantity"]]
    assert e[effect]["hidden"].keys() == {effect}


def short_episode(world, attempt_seed=None):
    """Spawn, one zero-command control period, finish: the first observation, the outcome and the hidden sidecar."""
    obs = world.reset({"settle_s": 0.0}, attempt_seed=attempt_seed)
    world.step({name: 0.0 for name in world.command_schema["fields"]}, world.policy.control_dt)
    out = world.finish_episode()
    return obs, out, json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_loads_and_stands(config, zero, preset):
    w = Go2World(config, seed=5, preset=preset); w.reset(); w.step(zero, 1.0)
    assert w.finish_episode()["fall"] == 0


def test_hidden_values_never_reach_person_b(config, zero):
    w = Go2World(config, seed=5, preset="heavy_payload"); obs = w.reset(); out = w.finish_episode()
    leaked = {"payload_offset_m", "motor_strength_fraction", "command_latency_s", "friction", "slope_deg"}
    assert not leaked & (set(obs) | set(out))


def test_dead_motors_fall_is_measured_not_raised(config, zero):
    # 0.0 is outside the 0.8-1.0 experiment range on purpose: presets are not range-checked, so a real fall can be
    # produced
    cfg = copy.deepcopy(config); cfg["world"]["presets"]["default"]["motor_strength_fraction"] = 0.0
    w = Go2World(cfg, seed=5); w.reset({"settle_s": 0.0}); w.step(zero, 3.0)
    out = w.finish_episode()
    assert out["fall"] == 1 and out["fall_time_s"] is not None


def test_hidden_values_go_to_the_sidecar_with_their_decimals(cfg):
    preset = cfg["world"]["presets"]["tired_motors"]
    he = cfg["world"]["hidden_effects"]
    _, _, hidden = short_episode(Go2World(cfg, seed=5, preset="tired_motors"))
    assert set(hidden["hidden_effects"]) == set(EFFECTS)  # config.yaml switches all five on
    assert hidden["hidden_effects"]["motor_strength"] == preset["motor_strength_fraction"] == 0.8
    base = he["surface_friction_by_type"][preset["floor_type"]]
    assert abs(hidden["hidden_effects"]["friction"] / base - 1) <= he["friction_per_run_variation"]
    assert hidden["hidden_effects"]["friction"] != base  # the per-run variation is drawn, not zero


def test_seed_fixes_the_hidden_world_and_the_declared_payload(cfg):
    runs = [short_episode(Go2World(cfg, seed=s, preset="heavy_payload")) for s in (5, 5, 6)]
    (obs_a, out_a, hidden_a), (obs_b, _, hidden_b), (obs_c, _, hidden_c) = runs
    assert hidden_a["hidden_effects"] == hidden_b["hidden_effects"] != hidden_c["hidden_effects"]
    declared = obs_a["declared_payload_kg"]
    assert declared == obs_b["declared_payload_kg"] == out_a["declared_payload_kg"] != obs_c["declared_payload_kg"]
    true_kg = hidden_a["hidden_effects"]["payload"]["mass_kg"]
    assert true_kg == cfg["world"]["presets"]["heavy_payload"]["payload_mass_kg"]
    assert declared != true_kg and abs(declared / true_kg - 1) <= 0.2  # declared_mass_error_fraction
    assert obs_a["declared_surface"] == cfg["world"]["presets"]["heavy_payload"]["floor_type"]


def test_declared_values_are_fixed_attributes_equal_to_every_observation_and_the_outcome(cfg, zero, period):
    w = Go2World(cfg, seed=5, preset="heavy_payload")
    declared = (w.declared_payload_kg, w.declared_surface)
    for k in range(2):
        seen = [w.reset({"settle_s": 0.0}, attempt_seed=k), w.step(zero, period), w.finish_episode()]
        assert all((d["declared_payload_kg"], d["declared_surface"]) == declared for d in seen)
    with pytest.raises(AttributeError):
        w.declared_payload_kg = 3.0


def test_latency_is_a_delay_line_whatever_the_step_lengths(cfg, zero, forward, period):
    # A recorder re-sends the same command once per control period: that must not restart the delay or redraw it.
    def run(calls):
        w = Go2World(cfg, seed=5)
        w.reset({"settle_s": 0.1}, attempt_seed=1)
        w.step(zero, 0.2)
        for _ in range(calls):
            w.step(forward, 50 * period / calls)
        out = w.finish_episode()
        sidecar = json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())
        return dict(np.load(out["trace_path"])), sidecar["attempt_latency_s"]
    (split, latency), (whole, _) = run(50), run(1)
    assert latency > 0 and split.keys() == whole.keys() and all(np.array_equal(split[k], whole[k]) for k in split)


def test_latency_jitter_is_drawn_per_attempt_and_repeats_with_the_attempt_seed(cfg):
    a, b = Go2World(cfg, seed=5), Go2World(cfg, seed=5)
    latencies = [short_episode(a, k)[2]["attempt_latency_s"] for k in range(6)]
    assert len(set(latencies)) > 1
    assert short_episode(b, 3)[2]["attempt_latency_s"] == latencies[3]
    nominal = cfg["world"]["presets"]["default"]["command_latency_s"]
    assert all(abs(t - nominal) <= cfg["world"]["hidden_effects"]["command_latency_jitter_s"] + 1e-9 for t in latencies)


def test_command_active_lags_commands_received_by_the_latency(cfg, zero, forward, period):
    w = Go2World(cfg, seed=5)
    w.reset({"settle_s": 0.0}); w.step(zero, 0.2); w.step(forward, 0.5)
    out = w.finish_episode()
    trace = np.load(out["trace_path"])
    latency = json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())["attempt_latency_s"]
    issued = out["commands_received"][1][0]
    first_active = trace["t_s"][trace["command_active"][:, 0] == forward["vel_x"]][0]  # vel_x is column 0
    assert latency > 0 and first_active - issued == pytest.approx(latency + period)


def test_on_a_slope_the_policy_senses_true_gravity(config):
    # Gravity is tilted instead of the floor, so the trunk's orientation is relative to the floor; the policy must
    # still sense "down" as an IMU on a real slope would. Its projected_gravity is policy_obs[3:6].
    w = Go2World(config, seed=5, preset="downhill")
    obs = w.reset({"settle_s": 0.1})
    g = w.model.opt.gravity / np.linalg.norm(w.model.opt.gravity)
    inverse, true_down = np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inverse, obs["base_quat_wxyz"])
    mujoco.mju_rotVecQuat(true_down, g, inverse)
    assert abs(g[0]) > 0.08  # 5 degrees of slope
    # policy_obs is read one control period before base_quat_wxyz; the trunk turns by far less than 0.01 rad in it
    assert np.allclose(obs["policy_obs"][3:6], true_down, atol=0.01)
