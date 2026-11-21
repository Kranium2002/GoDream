"""The `visual` situation key (object colour and size, floor and backdrop grey, light) on the real simulator."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from learned_judgment.env import Go2World

pytestmark = pytest.mark.system
CHANGES = {"object_rgb": [0.1, 0.8, 0.2], "object_radius_m": 0.15, "floor_grey": 0.2, "backdrop_grey": 0.9,
           "light_dir": [0.6, 0.0, -0.8], "light_strength": 0.35}


def ahead(x, y=0.0, vx=0.0, kind="ball"):
    return {"ball_rel_x_m": x, "ball_rel_y_m": y, "ball_rel_z_m": 0.017, "ball_vel_x_m_s": vx,
            "ball_vel_y_m_s": 0.0, "ball_vel_z_m_s": 0.0, "object_kind": kind}


def head(config, situation, seed=3):
    w = Go2World(config, seed)
    w.reset(situation)
    return w.render_frame(camera="head")


@pytest.mark.parametrize("field", list(CHANGES))
def test_each_setting_changes_the_head_frame(config, field):
    plain = head(config, ahead(1.5))
    changed = head(config, ahead(1.5) | {"visual": {field: CHANGES[field]}})
    assert np.any(plain != changed)


def test_the_backdrop_takes_its_grey(config):
    frame = head(config, {"visual": {"backdrop_grey": 0.9}})
    assert np.abs(frame[0].astype(int) - round(0.9 * 255)).max() <= 2


def test_visual_settings_repeat_and_do_not_leak_into_the_next_episode(config, zero):
    a = head(config, ahead(1.5) | {"visual": CHANGES})
    w = Go2World(config, seed=3)
    w.reset(ahead(1.5) | {"visual": CHANGES}); w.step(zero, 0.02); w.finish_episode()
    assert np.array_equal(head(config, ahead(1.5) | {"visual": CHANGES}), a)
    w.reset(ahead(1.5))  # the same world, now without `visual`: back to the defaults
    assert np.array_equal(w.render_frame(camera="head"), head(config, ahead(1.5)))


@pytest.mark.parametrize("visual, field", [
    ({"colour": 1.0}, "colour"), ({"object_rgb": [1.2, 0.0, 0.0]}, "object_rgb"), ({"object_rgb": [1, 0]}, "object_rgb"),
    ({"floor_grey": -0.1}, "floor_grey"), ({"backdrop_grey": "grey"}, "backdrop_grey"),
    ({"light_dir": [0.0, 0.0, 1.0]}, "light_dir"), ({"light_strength": 2.0}, "light_strength"),
    ({"object_radius_m": 1.0}, "object_radius_m"), ("bright", "visual")])
def test_bad_settings_are_refused_naming_the_field(world, visual, field):
    with pytest.raises(ValueError, match=field):
        world.reset({"visual": visual})


def test_object_size_is_physical_like_a_compiled_radius(config, zero):
    # A punch resized by `visual` gives the same episode, contact included, as a punch compiled at that radius.
    def episode(cfg, visual):
        w = Go2World(cfg, seed=3)
        w.reset(ahead(2.34, vx=-1.5, kind="punch") | visual, attempt_seed=1)
        w.step(zero, 2.0)
        out = w.finish_episode()
        return dict(np.load(out.pop("trace_path"))), out
    compiled = copy.deepcopy(config)
    compiled["punch"]["radius_m"] = 0.1
    (trace_a, out_a), (trace_b, out_b) = episode(compiled, {}), episode(config, {"visual": {"object_radius_m": 0.1}})
    assert out_a["collision"] == 1 and out_a == out_b
    assert all(np.array_equal(trace_a[k], trace_b[k]) for k in trace_a)


def test_a_bigger_object_passes_closer_by_its_extra_radius(config, zero):
    def closest(visual):
        w = Go2World(config, seed=3)
        w.reset(ahead(3.34, y=0.5, vx=-1.5, kind="punch") | visual)
        w.step(zero, 3.0)
        return w.finish_episode()["min_separation_m"]
    assert closest({}) - closest({"visual": {"object_radius_m": 0.1}}) == pytest.approx(0.05, abs=1e-9)


def test_visual_values_stay_hidden(config, forward):
    w = Go2World(config, seed=3)
    seen = [w.reset(ahead(2.0) | {"visual": CHANGES}), w.step(forward, 0.3)]
    out = w.finish_episode()
    hidden = json.loads(Path(out["trace_path"]).with_suffix(".hidden.json").read_text())
    assert hidden["visual"] == CHANGES
    # object_radius_m is left out: the observation's ball.radius_m has reported the object's radius since the ball.
    secrets = [0.1, 0.8, 0.2, 0.9, 0.6, -0.8, 0.35]
    numbers = [float(x) for o in seen + [out] for k, v in o.items()
               if k not in ("ball", "commands_received", "trace_path", "declared_surface", "command_active")
               and v is not None for x in np.ravel(v)]
    assert not [s for s in secrets if any(abs(n - s) < 1e-9 for n in numbers)]
    assert set(seen[0]) == set(Go2World(config, seed=3).reset(ahead(2.0)))
