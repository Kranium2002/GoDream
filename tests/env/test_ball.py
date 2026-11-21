"""The thrown ball on the real simulator and walking policy: placement, DODGE measurements and the throw protocol."""
import itertools
import json

import mujoco
import numpy as np
import pytest

from learned_judgment.env import BALL_KEYS, Go2World, load_config
from learned_judgment.env.calibrate import calibrate_throws
from learned_judgment.env.policy import WalkingPolicy
from learned_judgment.env.scene import build_model

pytestmark = pytest.mark.system


def throw_at(world, lateral_offset_m):
    """Situation dict (the six ball keys) for a straight throw from 3 m to the dog's left, aimed lateral_offset_m ahead
    of the trunk centre: a 0.6 s ballistic flight that is back at the trunk origin's height when it crosses y = 0."""
    flight_s = 0.6
    return {"ball_rel_x_m": lateral_offset_m, "ball_rel_y_m": 3.0, "ball_rel_z_m": 0.0, "ball_vel_x_m_s": 0.0,
            "ball_vel_y_m_s": -3.0 / flight_s, "ball_vel_z_m_s": world.config["ball"]["gravity_m_s2"] * flight_s / 2}


def test_centre_throw_hits_standing_dog(world, zero):
    world.reset(throw_at(world, 0.0)); world.step(zero, 3.0)
    out = world.finish_episode()
    assert out["collision"] == 1 and out["min_separation_m"] == 0 and out["first_collision_time_s"] > 0


def test_wide_throw_misses_even_after_bouncing(world, zero):
    world.reset(throw_at(world, 1.5)); world.step(zero, 3.0)
    out = world.finish_episode()
    assert out["collision"] == 0 and out["min_separation_m"] > 0 and out["recovery_time_s"] == 0.0
    # It did bounce: near the floor its vertical velocity turned from down to up, which gravity alone never does.
    trace = np.load(out["trace_path"])
    z, vz = trace["ball_pos_m"][:, 2], trace["ball_vel_m_s"][:, 2]
    assert np.any((vz[:-1] < 0) & (vz[1:] > 0) & (z[1:] < world.config["ball"]["radius_m"] + 0.15))


def test_ball_ignores_the_chosen_command(config, zero, forward):
    def flight(cmd):
        w = Go2World(config, seed=11); throw = w.sample_dodge_situation()
        w.reset(throw, attempt_seed=0); w.step(cmd, 0.1)  # every sampled throw is still more than 0.1 s from the dog
        return throw, np.load(w.finish_episode()["trace_path"])["ball_pos_m"]
    (t1, p1), (t2, p2) = flight(zero), flight(forward)
    assert t1 == t2 and np.array_equal(p1, p2)


@pytest.fixture(scope="module")
def dev_throws(tmp_path_factory):
    """calibrate_throws on the dev seed 11, shared by the two tests below (40 episodes)."""
    config = load_config()
    config["world"]["trace_dir"] = str(tmp_path_factory.mktemp("traces"))
    out = tmp_path_factory.mktemp("calibration")
    return config, json.loads(calibrate_throws(config, seed=11, n=40, out_dir=out).read_text())


def test_throw_distribution_has_hits_and_misses(dev_throws):
    _, throws = dev_throws
    assert 0.3 <= sum(t["collision"] for t in throws) / 40 <= 0.7


def avoided_count(config, throws):
    """Of the first 10 dev throws that hit a dog holding its initial command, how many some command of a small grid
    of schema-legal commands avoids. The grid is the two range ends of each command field, held for the episode."""
    fields = config["robot"]["walking_policy"]["command_schema"]["fields"]
    grid = [dict(zip(fields, ends)) for ends in itertools.product(*[(f["min"], f["max"]) for f in fields.values()])]
    world, duration = Go2World(config, seed=11), config["skills"]["dodge"]["episode_duration_s"]

    def avoided(throw):
        for command in grid:
            world.reset(throw["situation"], attempt_seed=throw["attempt_seed"])
            world.step(command, duration)
            if world.finish_episode()["collision"] == 0:
                return True
        return False
    return sum(avoided(t) for t in [t for t in throws if t["collision"]][:10])


def test_some_supported_command_avoids_a_hit(dev_throws):
    config, throws = dev_throws
    assert avoided_count(config, throws) >= 1


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test_ball_starts_in_the_decision_instant_heading_frame(world):
    # Trotting, so the heading has drifted away from +x by the decision instant; the frame turns with yaw only.
    situation = {"dog_forward_speed_m_s": 0.6, "ball_rel_x_m": 2.0, "ball_rel_y_m": 1.0, "ball_rel_z_m": 0.5,
                 "ball_vel_x_m_s": -4.0, "ball_vel_y_m_s": 0.5, "ball_vel_z_m_s": 2.0}
    obs = world.reset(situation)
    yaw = yaw_of(obs["base_quat_wxyz"])
    assert abs(yaw) > 0.01
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    ball = obs["ball"]
    assert set(ball) == {"pos_m", "vel_m_s", "radius_m"} and ball["radius_m"] == world.config["ball"]["radius_m"]
    assert np.allclose(ball["pos_m"], obs["base_pos_m"] + turn @ [2.0, 1.0, 0.5], atol=1e-12)
    assert np.allclose(ball["vel_m_s"], turn @ [-4.0, 0.5, 2.0], atol=1e-12)  # over the ground, not relative to the dog


def test_ball_keys_are_all_or_none(world):
    situation = throw_at(world, 0.0)
    del situation["ball_vel_y_m_s"], situation["ball_rel_z_m"]
    with pytest.raises(ValueError, match="ball_rel_z_m.*ball_vel_y_m_s"):
        world.reset(situation)
    with pytest.raises(ValueError, match="ball_vel_x_m_s"):
        world.reset(throw_at(world, 0.0) | {"ball_vel_x_m_s": "4"})


def test_episode_without_ball_has_no_ball_body(config, world, zero):
    obs = world.reset()
    world.step(zero, 0.1)
    out = world.finish_episode()
    assert obs["ball"] is None and world.model.nbody == build_model(config).nbody
    assert "ball_pos_m" not in np.load(out["trace_path"])
    assert (out["collision"], out["first_collision_time_s"], out["min_separation_m"]) == (0, None, None)


def test_ball_scene_keeps_the_robot_first(config):
    plain, ball = build_model(config), build_model(config, ball=True)
    assert ball.nq == plain.nq + 7 and ball.nbody == plain.nbody + 1
    assert ball.jnt_qposadr[ball.body("ball").jntadr[0]] == plain.nq  # the robot's free joint stays the first
    ball_geom = ball.geom("ball")
    assert ball_geom.size[0] == config["ball"]["radius_m"] and ball.body("ball").mass[0] == pytest.approx(
        config["ball"]["mass_kg"])
    assert -ball.opt.gravity[2] == config["ball"]["gravity_m_s2"]


def test_ball_scene_spawns_the_dog_at_the_same_pose(config):
    # Efferent's spawn lowers the base onto the lowest non-world geom (mujoco_backend.py:104-107), the ball included.
    config["robot"]["walking_policy"]["spawn"]["settle_s"] = 0.0  # the spawn alone, before any physics step
    spawned = []
    for ball in (False, True):
        policy = WalkingPolicy(config, build_model(config, ball=ball))
        policy.backend.connect(policy.binding.descriptor)
        spawned.append(policy.data.qpos[:19].copy())
    assert np.array_equal(*spawned)


def test_dog_is_undisturbed_until_the_ball_arrives(config, zero):
    # The same seed and attempt seed with and without a ball, while the ball is still far away (6 m off for 0.3 s after
    # the decision instant). Not bit-identical: the ball's 6 degrees of freedom change how MuJoCo groups its sums over
    # all of them (1e-13 after 1 s of plain stepping), and 6 s of the float32 policy grow that to about 2e-5. A ball
    # taken for the spawn's lowest geom would lift the dog by centimetres; a ball joint before the robot's would scramble
    # every robot slice.
    def dog(situation):
        w = Go2World(config, seed=4); w.reset(situation, attempt_seed=2); w.step(zero, 0.3)
        trace = np.load(w.finish_episode()["trace_path"])
        return {k: trace[k] for k in ("base_pos_m", "base_quat_wxyz", "base_lin_vel_m_s", "joint_pos_rad")}
    world = Go2World(config, seed=4)
    with_ball, without = dog(throw_at(world, 0.0) | {"ball_rel_y_m": 6.0}), dog({})
    assert all(np.allclose(with_ball[k], without[k], rtol=0, atol=1e-4) for k in with_ball)


def test_ball_rebounds_like_a_handball(config):
    # IHF Rule 3:2: dropped from 2 m, a size 3 ball rebounds to 90-110 cm (E6). Bottom of the ball at 2.0 m, far from the
    # dog; measure the height of its bottom at the top of the first rebound.
    model = build_model(config, ball=True)
    data, r = mujoco.MjData(model), config["ball"]["radius_m"]
    q = model.jnt_qposadr[model.body("ball").jntadr[0]]
    data.qpos[q:q + 3] = [10.0, 10.0, 2.0 + r]
    bounced, top = False, 0.0
    for _ in range(2000):  # 4 s
        mujoco.mj_step(model, data)
        z, vz = data.qpos[q + 2], data.qvel[model.body("ball").dofadr[0] + 2]
        bounced |= z < r + 0.005
        if bounced and vz > 0:
            top = max(top, z)
        if bounced and vz < 0 and top > 0:
            break
    assert 0.9 <= top - r <= 1.1


def test_dodge_outcome_and_trace_hold_the_ball(world, zero):
    obs = world.reset(throw_at(world, 0.0)); world.step(zero, 1.0)
    out = world.finish_episode()
    assert {"collision", "first_collision_time_s", "min_separation_m", "recovery_time_s", "recovered"} <= set(out)
    assert out["recovered"] in (0, 1) and isinstance(out["recovery_time_s"], float)
    trace = np.load(out["trace_path"])
    assert trace["ball_pos_m"].shape == trace["ball_vel_m_s"].shape == (51, 3)
    assert np.array_equal(trace["ball_pos_m"][0], obs["ball"]["pos_m"])
