"""The physics guess must use the thrown object's own radius: the punch body (punch.radius_m) is smaller than the
handball (ball.radius_m), and the formula used the ball's radius for both."""

import copy

import numpy as np
import pytest

from learned_judgment.config import load_config
from learned_judgment.skills import dodge_vision
from learned_judgment.skills.dodge_vision import physics_guess


@pytest.fixture(scope="module")
def config():
    return load_config()


def passing_punch(config, y):
    """A punch flying straight past the trunk at sideways distance y, at trunk height, never touching it."""
    drawn = dodge_vision.sample_threat_scenarios(config, np.random.default_rng(0), 50)
    scenario = copy.deepcopy(next(s for s in drawn if all(k in s["situation"] for k in dodge_vision._ball_keys())))
    scenario["object_kind"] = "punch"
    for key, value in zip(dodge_vision._ball_keys(), (1.0, y, 0.0, -1.0, 0.0, 0.0)):
        scenario["situation"][key] = value
    return scenario


def separation(config, scenario):
    no_op = list(config["skills"]["dodge_vision"]["actions"]).index("NO_OP")
    return physics_guess(config, [scenario], [0.3], np.array([no_op]))[0, 0]


def test_the_clearance_of_a_punch_follows_the_punch_radius(config):
    scenario = passing_punch(config, 0.40)
    base = separation(config, scenario)
    body = config["skills"]["dodge_vision"]["physics"]["body_radius_m"]
    assert base == pytest.approx(0.40 - body - config["punch"]["radius_m"], abs=1e-6)
    bigger = copy.deepcopy(config)
    bigger["punch"]["radius_m"] = config["punch"]["radius_m"] + 0.03
    assert separation(bigger, scenario) == pytest.approx(base - 0.03, abs=1e-9)


def test_the_ball_radius_does_not_move_a_punch(config):
    scenario = passing_punch(config, 0.40)
    other_ball = copy.deepcopy(config)
    other_ball["ball"]["radius_m"] = config["ball"]["radius_m"] + 0.03
    assert separation(other_ball, scenario) == pytest.approx(separation(config, scenario), abs=1e-12)
