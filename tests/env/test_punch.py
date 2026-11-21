"""The punch (situation key object_kind "punch") on the real simulator and walking policy: a gravity-free body that
flies straight at constant speed and collides physically, measured by the ball's outcome keys."""
import json
from pathlib import Path

import numpy as np
import pytest

from learned_judgment.env import Go2World

pytestmark = pytest.mark.system
HEAD = (0.34, 0.0, 0.017)  # the head camera's mount on base_link (plan, World API)


def at_head(speed, distance=3.0, side=0.0, kind="punch"):
    """Straight at the standing dog's head point from `distance` ahead (shifted `side` m to its left), at `speed`."""
    situation = {"ball_rel_x_m": HEAD[0] + distance, "ball_rel_y_m": side, "ball_rel_z_m": HEAD[2],
                 "ball_vel_x_m_s": -speed, "ball_vel_y_m_s": 0.0, "ball_vel_z_m_s": 0.0}
    return situation | ({} if kind is None else {"object_kind": kind})


def run(world, situation, zero, duration_s=3.0):
    seen = []
    first = world.reset(situation)
    world.step(zero, duration_s, on_period=seen.append)
    return first, seen, world.finish_episode()


def test_a_slow_punch_flies_straight_then_hits_the_head(world, zero):
    first, seen, out = run(world, at_head(1.5), zero)
    assert out["collision"] == 1 and out["min_separation_m"] == 0
    # The head's collision cylinder is centred 0.285 m ahead of the trunk origin, so its front is about at the camera
    # mount (0.34 m): about 3 m of flight less the punch radius at 1.5 m/s, about 2.0 s.
    assert 1.9 < out["first_collision_time_s"] < 2.1
    p0, v0 = first["ball"]["pos_m"], first["ball"]["vel_m_s"]
    before = [o for o in seen if o["t_s"] < out["first_collision_time_s"]]
    assert before[-1]["t_s"] > out["first_collision_time_s"] - 0.03  # the last control period before contact
    for o in before:  # tolerance 1e-9 m and 1e-9 m/s: no gravity, drag or contact acts on it in free flight
        assert np.linalg.norm(o["ball"]["pos_m"] - (p0 + v0 * o["t_s"])) < 1e-9
        assert abs(np.linalg.norm(o["ball"]["vel_m_s"]) - 1.5) < 1e-9
    assert first["ball"]["radius_m"] == world.config["punch"]["radius_m"]
    trace = np.load(out["trace_path"])
    assert trace["ball_pos_m"].shape == (len(seen) + 1, 3)  # the trace arrays hold the punch as they hold the ball


def test_a_punch_half_a_metre_to_the_side_misses(config, world, zero):
    _, _, out = run(world, at_head(1.5, side=0.5), zero)
    # The robot's collision geoms reach at most 0.264 m from the trunk's centre line (their bounding spheres, standing),
    # plus a few cm of standing drift: the closest approach lies between gap - 0.3 m and gap.
    gap = 0.5 - config["punch"]["radius_m"]
    assert out["collision"] == 0 and gap - 0.3 < out["min_separation_m"] < gap


@pytest.mark.parametrize("speed", [1.5, 2.5])
def test_the_dog_does_not_fall_from_a_slow_punch(world, zero, speed):
    _, _, out = run(world, at_head(speed, distance=2.0), zero)
    assert out["collision"] == 1 and out["fall"] == 0, out


def test_object_kind_absent_is_the_ball_exactly(config, zero):
    def episode(situation):
        w = Go2World(config, seed=3)
        _, _, out = run(w, situation, zero, 1.0)
        return dict(np.load(out.pop("trace_path"))), out
    (trace_a, out_a), (trace_b, out_b) = episode(at_head(3.0, kind=None)), episode(at_head(3.0, kind="ball"))
    assert out_a == out_b and all(np.array_equal(trace_a[k], trace_b[k]) for k in trace_a)


def test_an_unknown_object_kind_is_refused(world):
    with pytest.raises(ValueError, match="object_kind"):
        world.reset(at_head(1.5, kind="brick"))


def test_the_kind_is_hidden(world, zero):
    obs_ball, _, out_ball = run(world, at_head(1.5, kind="ball"), zero, 0.1)
    obs_punch, _, out_punch = run(world, at_head(1.5), zero, 0.1)
    hidden = json.loads(Path(out_punch["trace_path"]).with_suffix(".hidden.json").read_text())
    assert hidden["object_kind"] == "punch"
    assert set(obs_punch) == set(obs_ball) and set(obs_punch["ball"]) == set(obs_ball["ball"])
    assert set(out_punch) == set(out_ball) and "punch" not in json.dumps(
        {k: v for k, v in out_punch.items() if k != "trace_path"}, default=str)
