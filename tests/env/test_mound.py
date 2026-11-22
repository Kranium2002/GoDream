"""The mound (situation keys mound_x_m, mound_y_m, mound_radius_m, mound_height_m) on the real simulator and policy."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from learned_judgment.env import Go2World
from learned_judgment.env.mound import MOUND_KEYS, height_for_slope, rim_slope_deg, sphere_radius

pytestmark = pytest.mark.system
GENTLE = {"mound_x_m": 1.3, "mound_y_m": 0.0, "mound_radius_m": 0.8, "mound_height_m": 0.1}  # rim slope 14.3 degrees


def episode(world, situation, seconds=1.0):
    world.reset({"dog_forward_speed_m_s": 0.6, **situation}, attempt_seed=2)
    world.step({"vel_x": 0.6, "vel_y": 0.0, "yaw_rate": 0.0}, seconds)
    out = world.finish_episode()
    trace = dict(np.load(out["trace_path"]))
    hidden = json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())
    return {k: v for k, v in out.items() if k != "trace_path"}, trace, hidden


def test_without_the_keys_the_episode_is_the_plain_one_even_after_a_mound(config):
    fresh = Go2World(config, seed=3)
    out_a, trace_a, hidden_a = episode(fresh, {})
    with pytest.raises(KeyError):
        fresh.model.body("mound")  # the plain scene: no mound body at all
    used = Go2World(config, seed=3)
    episode(used, GENTLE)
    out_b, trace_b, hidden_b = episode(used, {})
    assert out_a == out_b and trace_a.keys() == trace_b.keys()
    assert all(np.array_equal(trace_a[k], trace_b[k]) for k in trace_a)
    assert hidden_a == hidden_b and "mound" not in hidden_a


def test_the_dog_walks_onto_a_gentle_mound_which_is_ground(config):
    world = Go2World(config, seed=3)
    out, trace, hidden = episode(world, GENTLE, seconds=3.0)
    z = trace["base_pos_m"][:, 2]
    assert out["fall"] == 0 and z.max() - z[0] > 0.6 * GENTLE["mound_height_m"]  # the trunk rose with the ground
    on_top = np.linalg.norm(trace["base_pos_m"][:, :2] - hidden["mound"]["centre_xy_m"], axis=1) < 0.5
    assert on_top.sum() > 10 and (trace["foot_contact"][on_top].sum(axis=1) >= 1).all()  # its contacts are ground
    assert {k: hidden["mound"][k] for k in MOUND_KEYS} == GENTLE
    assert hidden["mound"]["rim_slope_deg"] == pytest.approx(rim_slope_deg(0.8, 0.1))
    qw, qx, qy, qz = trace["base_quat_wxyz"][0]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    offset = np.array(hidden["mound"]["centre_xy_m"]) - trace["base_pos_m"][0, :2]
    assert offset == pytest.approx(1.3 * np.array([math.cos(yaw), math.sin(yaw)]))  # 1.3 m ahead at t_s 0
    assert not any("mound" in key for key in out)  # the outcome does not reveal it


@pytest.mark.parametrize("partial", [{"mound_x_m": 2.0}, {k: v for k, v in GENTLE.items() if k != "mound_height_m"}])
def test_partial_keys_raise_naming_the_missing_ones(world, partial):
    missing = [k for k in MOUND_KEYS if k not in partial]
    with pytest.raises(ValueError, match="missing mound key") as error:
        world.reset(partial)
    assert all(k in str(error.value) for k in missing)


def test_limits_raise(world, config):
    limits = config["world"]["mound"]
    too_steep = height_for_slope(1.0, limits["max_rim_slope_deg"] + 2.0)
    for bad, word in (({"mound_radius_m": limits["radius_m"][1] + 0.1}, "mound_radius_m"),
                      ({"mound_height_m": too_steep}, "max_rim_slope_deg"),
                      ({"mound_x_m": 1.0}, "min_gap_m"),
                      ({"mound_height_m": float("nan")}, "mound_height_m")):
        with pytest.raises(ValueError, match=word):
            world.reset({**GENTLE, "mound_radius_m": 1.0, "mound_x_m": 2.0, **bad})


def test_resizing_the_sphere_equals_compiling_it_at_that_radius(config):
    compiled = copy.deepcopy(config)
    compiled["world"]["mound"]["park_radius_m"] = sphere_radius(GENTLE["mound_radius_m"], GENTLE["mound_height_m"])
    (out_a, trace_a, _), (out_b, trace_b, _) = (episode(Go2World(c, seed=3), GENTLE) for c in (compiled, config))
    assert out_a == out_b and all(np.array_equal(trace_a[k], trace_b[k]) for k in trace_a)
