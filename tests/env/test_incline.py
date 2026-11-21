"""The traversable incline (situation key incline_deg) on the real simulator and walking policy."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from learned_judgment.env import Go2World, sample_ramp_situations
from learned_judgment.env.acceptance import trunk_on_floor

pytestmark = pytest.mark.system


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def climb(world, deg, zero, until_m=None, max_s=12.0):
    """Trot up at vel_x 0.6 holding the decision-instant heading with a yaw_rate feedback (test scaffolding only),
    until the trunk is until_m ahead (default: 1 m onto the plateau) or max_s. Returns (start obs, obs per period)."""
    start = world.reset({"dog_forward_speed_m_s": 0.6, "incline_deg": deg})
    inc = world.config["world"]["incline"]
    until_m = until_m or inc["start_x_m"] + inc["length_m"] * math.cos(math.radians(deg)) + 1.0
    yaw0, p0, seen = yaw_of(start["base_quat_wxyz"]), start["base_pos_m"][:2], []
    ahead = np.array([math.cos(yaw0), math.sin(yaw0)])
    for _ in range(round(max_s / world.policy.control_dt)):
        obs = seen[-1] if seen else start
        err = (yaw_of(obs["base_quat_wxyz"]) - yaw0 + math.pi) % (2 * math.pi) - math.pi
        seen.append(world.step(zero | {"vel_x": 0.6, "yaw_rate": max(-1.0, min(1.0, -2.0 * err))},
                               world.policy.control_dt))
        if (seen[-1]["base_pos_m"][:2] - p0) @ ahead >= until_m:
            break
    return start, seen


def along(start, seen):
    yaw0 = yaw_of(start["base_quat_wxyz"])
    return np.array([(o["base_pos_m"][:2] - start["base_pos_m"][:2]) @ [math.cos(yaw0), math.sin(yaw0)] for o in seen])


def test_flat_when_the_key_is_absent_or_zero(config, zero):
    traces = []
    for situation in ({}, {"incline_deg": 0.0}):
        w = Go2World(config, seed=3)
        w.reset(situation, attempt_seed=0); w.step(zero | {"vel_x": 0.6}, 0.5)
        out = w.finish_episode()
        assert (out["ramp_start_m"], out["ramp_end_m"]) == (None, None)
        with pytest.raises(KeyError):
            w.model.body("incline_slope")  # the plain scene: no ramp body at all
        traces.append(dict(np.load(out["trace_path"])))
    assert traces[0].keys() == traces[1].keys() and all(np.array_equal(traces[0][k], traces[1][k]) for k in traces[0])


def test_the_ramp_is_visible_in_the_tracking_frame(config, zero):
    frames = []
    for deg in (0.0, 10.0):
        w = Go2World(config, seed=3)
        frames.append(w.reset({"incline_deg": deg}) and w.render_frame())  # at the decision instant, before a step
        assert w._camera.distance == config["world"]["render"]["tracking_distance_m"]  # not zoomed out by the slab
    differ = np.any(frames[0] != frames[1], axis=2).mean()
    assert differ > 0.05, f"only {differ:.3f} of the pixels changed"


def test_the_ramp_is_placed_ahead_of_the_dog_at_the_decision_instant(config):
    inc = config["world"]["incline"]
    w = Go2World(config, seed=3)
    deg = 10.0
    start = w.reset({"dog_forward_speed_m_s": 0.6, "incline_deg": deg})  # trotted 2 s: the heading has drifted
    yaw0, a = yaw_of(start["base_quat_wxyz"]), math.radians(deg)
    ahead, left = np.array([math.cos(yaw0), math.sin(yaw0), 0]), np.array([-math.sin(yaw0), math.cos(yaw0), 0])
    d, m = w.policy.data, w.model
    slope, plateau = m.body("incline_slope").id, m.body("incline_plateau").id

    def corner(body, local):
        return d.xpos[body] + d.xmat[body].reshape(3, 3) @ local - np.r_[start["base_pos_m"][:2], 0.0]
    foot = corner(slope, [-inc["length_m"] / 2, 0, inc["thickness_m"] / 2])  # where the slope's top face starts
    top = corner(slope, [inc["length_m"] / 2, 0, inc["thickness_m"] / 2])
    level = corner(plateau, [-inc["plateau_length_m"] / 2, 0, inc["thickness_m"] / 2])
    assert foot @ ahead == pytest.approx(inc["start_x_m"]) and foot @ left == pytest.approx(0, abs=1e-9)
    assert foot[2] == pytest.approx(0, abs=1e-9) and top[2] == pytest.approx(inc["length_m"] * math.sin(a))
    assert level == pytest.approx(top)
    assert level @ ahead == pytest.approx(inc["start_x_m"] + inc["length_m"] * math.cos(a))
    w.step({"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}, 0.02)
    out = w.finish_episode()
    assert out["ramp_start_m"] == pytest.approx(inc["start_x_m"])
    assert out["ramp_end_m"] == pytest.approx(inc["start_x_m"] + inc["length_m"] * math.cos(a))


@pytest.mark.parametrize("deg", [5.0, 10.0])
def test_the_dog_climbs_the_ramp_and_reaches_the_plateau(config, zero, deg):
    inc = config["world"]["incline"]
    w = Go2World(config, seed=3)
    start, seen = climb(w, deg, zero)
    out = w.finish_episode()
    s = along(start, seen)
    assert out["fall"] == 0 and out["final_forward_m"] >= out["ramp_end_m"] + 0.9  # 1 m onto the plateau
    rise = inc["length_m"] * math.sin(math.radians(deg))
    assert seen[-1]["base_pos_m"][2] - start["base_pos_m"][2] == pytest.approx(rise, abs=0.06)
    # Every foot is on the slope while the trunk is well inside it: those contacts are ground contacts.
    on_slope = (s > out["ramp_start_m"] + 0.6) & (s < out["ramp_end_m"] - 0.4)
    feet = np.array([o["foot_contact"].sum() for o in seen])[on_slope]
    assert on_slope.sum() > 20 and (feet >= 1).all() and (feet >= 2).mean() > 0.8


def test_a_dog_standing_on_the_ramp_has_not_fallen(config, zero):
    inc = config["world"]["incline"]
    w = Go2World(config, seed=3)
    climb(w, 10.0, zero, until_m=inc["start_x_m"] + 0.5 * inc["length_m"])  # trunk over the middle of the slope
    last = w.step(zero, 2.0)
    out = w.finish_episode()
    assert out["fall"] == 0 and not last["fallen"] and last["foot_contact"].sum() >= 3


def test_the_trunk_on_the_ramp_counts_as_on_the_ground(config):
    inc = config["world"]["incline"]
    w = Go2World(config, seed=3)
    start = w.reset({"incline_deg": 10.0})
    yaw0, d = yaw_of(start["base_quat_wxyz"]), w.policy.data
    s = inc["start_x_m"] + 0.5 * inc["length_m"]  # mid-slope, surface at (s - start) x tan(10 degrees)
    surface = (s - inc["start_x_m"]) * math.tan(math.radians(10.0))
    d.qpos[0:2] = start["base_pos_m"][:2] + s * np.array([math.cos(yaw0), math.sin(yaw0)])
    d.qpos[2] = surface + 0.02  # the trunk box (half height 0.057 m) sunk into the slope
    mujoco.mj_forward(w.model, d)
    assert trunk_on_floor(w.model, d)
    d.qpos[2] = surface + 2.0  # lifted clear
    mujoco.mj_forward(w.model, d)
    assert not trunk_on_floor(w.model, d)


def test_the_angle_is_hidden_and_the_same_seeds_repeat(config, zero):
    runs = []
    for _ in range(2):
        w = Go2World(config, seed=3)
        w.reset({"incline_deg": 7.5}, attempt_seed=2); w.step(zero | {"vel_x": 0.6}, 1.0)
        out = w.finish_episode()
        runs.append((dict(np.load(out["trace_path"])), {k: v for k, v in out.items() if k != "trace_path"},
                     json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())))
    (trace_a, out_a, hidden), (trace_b, out_b, _) = runs
    assert all(np.array_equal(trace_a[k], trace_b[k]) for k in trace_a) and out_a == out_b
    assert hidden["incline_deg"] == 7.5


@pytest.mark.parametrize("bad", [-1.0, 20.5, float("nan"), "5", True])
def test_bad_angles_are_refused(world, bad):
    with pytest.raises(ValueError, match="incline_deg"):
        world.reset({"incline_deg": bad})


def test_foot_slip_is_small_standing_and_larger_on_a_slippery_floor(config, zero):
    # Not zero standing: on the zero command the policy shuffles its feet. Measured (seed 3, 2 s): 0.17 m standing,
    # 0.51 m trotting, 2.25 m trotting on slippery_floor; a foot in contact all 2 s slipped 0.064 m while its centre
    # moved 0.065 m over the floor, so the number is real sliding (scratchpad poc-w3/slip_diag.py).
    def slip(preset, command):
        w = Go2World(config, seed=3, preset=preset)
        w.reset({"dog_forward_speed_m_s": command["vel_x"]}); w.step(command, 2.0)
        return w.finish_episode()["foot_slip_m"]
    trot = zero | {"vel_x": 0.6}
    standing, grippy, slippery = slip("default", zero), slip("default", trot), slip("slippery_floor", trot)
    assert standing < grippy / 2 and slippery > 2 * grippy, (standing, grippy, slippery)


def test_sample_ramp_situations_is_seeded_and_in_range(config):
    a, b = (sample_ramp_situations(config, np.random.default_rng(7), 50) for _ in range(2))
    lo, hi = config["world"]["incline"]["range_deg"]
    assert a == b and len(a) == 50 and all(set(s) == {"incline_deg"} and lo <= s["incline_deg"] <= hi for s in a)
    assert len({s["incline_deg"] for s in a}) == 50 and all(type(s["incline_deg"]) is float for s in a)
