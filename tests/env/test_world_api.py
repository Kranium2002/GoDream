"""Go2World, the API Person B drives the dog through, on the real simulator and walking policy."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from learned_judgment.env import CommandError, Go2World

pytestmark = pytest.mark.system
OBS_KEYS = {"t_s", "base_pos_m", "base_quat_wxyz", "base_lin_vel_m_s", "base_ang_vel_rad_s", "joint_pos_rad",
            "joint_vel_rad_s", "policy_obs", "foot_contact", "fallen", "ball", "command_active", "declared_payload_kg",
            "declared_surface"}
OUTCOME_KEYS = {"elapsed_s", "fall", "fall_time_s", "sim_unstable", "commands_received", "final_base_pos_m",
                "declared_payload_kg", "declared_surface", "collision", "first_collision_time_s", "min_separation_m",
                "recovery_time_s", "recovered", "trace_path"}


def yaw_of(quats_wxyz):
    w, x, y, z = np.asarray(quats_wxyz).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test_reset_returns_documented_observation(world):
    obs = world.reset()
    assert set(obs) == OBS_KEYS and obs["t_s"] == 0.0 and obs["joint_pos_rad"].shape == (12,)


def test_step_rounds_to_control_periods(world, zero, period):
    world.reset()
    assert world.step(zero, 25 * period)["t_s"] == pytest.approx(25 * period)
    assert world.step(zero, 0.6 * period)["t_s"] == pytest.approx(26 * period)


def test_calls_out_of_order_raise(world, zero):
    with pytest.raises(RuntimeError):
        world.step(zero, 0.1)
    world.reset(); world.finish_episode()
    with pytest.raises(RuntimeError):
        world.finish_episode()


def test_unknown_names_raise(world, zero):
    with pytest.raises(ValueError, match="wind"):
        world.reset({"wind": 1})
    world.reset()
    with pytest.raises(CommandError, match="jump"):
        world.step(zero | {"jump": 1.0}, 0.1)


def test_unknown_preset_raises(config):
    with pytest.raises(ValueError, match="moon"):
        Go2World(config, 1, preset="moon")


def test_bad_values_raise(world, zero):
    with pytest.raises(CommandError, match="vel_x"):
        world.reset({"dog_forward_speed_m_s": 1.5})
    with pytest.raises(ValueError, match="settle_s"):
        world.reset({"settle_s": -0.1})
    world.reset({"settle_s": 0.0})
    for duration in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="duration"):
            world.step(zero, duration)


def test_reset_trots_at_the_situation_speed_until_the_decision_instant(world):
    obs = world.reset({"dog_forward_speed_m_s": 0.6})
    yaw = yaw_of(obs["base_quat_wxyz"])
    assert obs["command_active"] == {"vel_x": 0.6, "vel_y": 0.0, "yaw_rate": 0.0}
    # Gate 1's trot_0.6 trace stays between 0.417 and 0.652 m/s after 1.5 s; a standing dog would be near 0.
    assert obs["base_lin_vel_m_s"][:2] @ [np.cos(yaw), np.sin(yaw)] > 0.3


def world_rate(q0, q1, dt):
    """The constant world-frame angular velocity that turns orientation q0 into q1 in dt."""
    inverse, change, rate = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inverse, q0)
    mujoco.mju_mulQuat(change, q1, inverse)
    mujoco.mju_quat2Vel(rate, change, dt)
    return rate


def test_velocities_are_in_the_world_frame(world, zero, period):
    # Trot and turn left for 5 s. Once the heading has turned well past 90 degrees, body-frame velocities would no
    # longer match the world-frame change of position and orientation.
    world.reset()
    obs = [world.step(zero | {"vel_x": 0.6, "yaw_rate": 0.8}, period) for _ in range(250)]
    pos, vel, ang, quat = (np.array([o[k] for o in obs]) for k in
                           ("base_pos_m", "base_lin_vel_m_s", "base_ang_vel_rad_s", "base_quat_wxyz"))
    yaw = np.unwrap(yaw_of(quat))
    assert yaw[-1] - yaw[0] > np.pi / 2  # positive yaw_rate turns anticlockwise seen from above
    assert np.allclose(pos[-1] - pos[0], np.trapezoid(vel, dx=period, axis=0), atol=0.02)
    rates = np.array([world_rate(a, b, period) for a, b in zip(quat, quat[1:])])
    # Mean error per axis was 0.04, 0.04, 0.01 rad/s; the body-frame value would be off by about 0.5 in x and y.
    assert np.abs((ang[1:] + ang[:-1]) / 2 - rates).mean(axis=0).max() < 0.1
    forward = (vel[:, :2] * np.stack([np.cos(yaw), np.sin(yaw)], axis=1)).sum(axis=1)
    assert forward.mean() > 0  # positive vel_x: positive speed along the heading


def test_finish_episode_reports_outcome_and_saves_trace(config, world, forward):
    world.reset()
    world.step(forward, 0.5)
    world.step(forward | {"yaw_rate": 0.5}, 0.5)
    out = world.finish_episode()
    assert set(out) == OUTCOME_KEYS
    assert out["elapsed_s"] == pytest.approx(1.0)
    assert (out["fall"], out["fall_time_s"], out["sim_unstable"]) == (0, None, 0)
    assert [t for t, _ in out["commands_received"]] == pytest.approx([0.0, 0.5])
    assert out["commands_received"][1][1] == forward | {"yaw_rate": 0.5}
    assert (out["declared_payload_kg"], out["declared_surface"]) == (0.0, "default")
    trace = np.load(out["trace_path"])
    assert trace["t_s"].shape == (51,) and trace["joint_pos_rad"].shape == (51, 12)  # decision instant + 50 periods
    assert np.array_equal(trace["base_pos_m"][-1], out["final_base_pos_m"])
    hidden = json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())
    assert {k: hidden[k] for k in ("preset", "seed", "attempt_seed")} == {
        "preset": "default", "seed": config["runtime"]["seed"], "attempt_seed": 0}
    # config.yaml switches all five effects on; their values are tested in test_effects.py
    assert set(hidden["hidden_effects"]) == {"payload", "friction", "slope", "motor_strength", "latency"}


def test_trace_files_are_unique_and_optional(config, tmp_path, zero, period):
    world = Go2World(config | {"world": config["world"] | {"trace_dir": str(tmp_path)}}, seed=1)
    paths = []
    for _ in range(2):  # the same seed and attempt seed twice
        world.reset(attempt_seed=0)
        world.step(zero, period)
        paths.append(world.finish_episode()["trace_path"])
    assert paths[0] != paths[1] and all(Path(p).exists() for p in paths)
    quiet = Go2World(config | {"world": config["world"] | {"trace_dir": str(tmp_path / "quiet"), "save_traces": False}},
                     seed=1)
    quiet.reset()
    quiet.step(zero, period)
    assert quiet.finish_episode()["trace_path"] is None and not (tmp_path / "quiet").exists()


def test_fall_is_latched_with_its_first_time(world, zero, period):
    world.reset()
    data = world.policy.data
    standing = data.qpos.copy(), data.qvel.copy()
    data.qpos[2] += 0.3  # lifted clear of the floor and rolled 70 degrees, beyond robot.fall.max_tilt_deg
    data.qpos[3:7] = [np.cos(np.radians(35)), np.sin(np.radians(35)), 0.0, 0.0]
    assert world.step(zero, period)["fallen"]
    data.qpos[:], data.qvel[:] = standing  # stood back up: the fall stays recorded
    assert world.step(zero, 1.0)["fallen"]
    out = world.finish_episode()
    assert out["fall"] == 1 and out["fall_time_s"] == pytest.approx(world.model.opt.timestep)  # first physics step


def test_fall_before_the_decision_instant_has_a_negative_time(config, zero, period):
    # Spawned with the trunk box (half height 0.057 m) through the floor: the fall is caught during reset's settling.
    config["robot"]["walking_policy"]["spawn"] |= {"base_pos": [0.0, 0.0, 0.05], "auto_base_height": False}
    world = Go2World(config, seed=1)
    world.reset()
    world.step(zero, period)
    out = world.finish_episode()
    assert out["fall"] == 1 and out["fall_time_s"] < 0


def test_reset_rejects_bad_seeds_and_values_without_abandoning_the_episode(world, zero, period):
    world.reset()
    for situation, seed in (({}, -1), ({"settle_s": True}, 0), ({"settle_s": "2"}, 0)):
        with pytest.raises(ValueError):
            world.reset(situation, attempt_seed=seed)
    assert world.step(zero, period)["t_s"] == pytest.approx(period)  # the open episode goes on


def test_returned_observation_is_a_copy(world, zero):
    obs = world.reset()
    obs["base_pos_m"] += 10.0
    obs = world.step(zero, 0.1)
    obs["base_pos_m"] += 10.0
    trace = np.load(world.finish_episode()["trace_path"])
    assert np.all(np.abs(trace["base_pos_m"]) < 1.0)


def test_bad_simulator_state_is_reported(world, zero, period, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MuJoCo appends its warning to MUJOCO_LOG.TXT in the working directory
    world.reset()
    world.policy.data.qvel[0] = np.nan  # as if the simulation had blown up; MuJoCo warns and resets the state
    world.step(zero, period)
    assert world.finish_episode()["sim_unstable"] == 1


def test_render_frame_shows_the_dog(config, world):
    world.reset()
    frame = world.render_frame()
    size = config["world"]["render"]
    assert frame.shape == (size["height"], size["width"], 3) and frame.dtype == np.uint8
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1  # not a single flat colour
