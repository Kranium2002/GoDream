"""learned_judgment.dodge_eval: the threat definition and the strict collision view on hand-made records (fixtures
only; nothing here is a result) with the real config.yaml. No episode is run."""

from __future__ import annotations

import copy

import pytest

from learned_judgment.config import load_config
from learned_judgment.dodge_eval import (
    _issue_metrics,
    _method_metrics,
    _strict_collision,
    is_threat,
    threat_deadline,
    threat_definition,
)

CONFIG = load_config()
NEAR = {"mode": "near_approach", "separation_m": 0.08, "outcome": "min_separation_m"}
COLLISION = {"mode": "collision", "outcome": "collision"}


def no_op(separation: float, collision: float) -> dict[str, float]:
    return {"min_separation_m": separation, "collision": collision}


def test_threat_definition_modes() -> None:
    config = copy.deepcopy(CONFIG)
    block = config["skills"]["dodge_vision"]
    block["threat_target"] = dict(NEAR)
    assert threat_definition(config) == NEAR
    block["threat_target"] = "collision_course"
    assert threat_definition(config) == COLLISION
    del block["threat_target"]
    assert threat_definition(config) == COLLISION


def test_is_threat() -> None:
    assert is_threat(NEAR, no_op(0.0, 1.0)) and is_threat(NEAR, no_op(0.05, 0.0))
    assert not is_threat(NEAR, no_op(0.08, 0.0)) and not is_threat(NEAR, no_op(0.3, 0.0))
    assert is_threat(COLLISION, no_op(0.0, 1.0)) and not is_threat(COLLISION, no_op(0.05, 0.0))


def test_threat_deadline() -> None:
    approaching = {"situation": {"ball_rel_x_m": 1.3, "ball_vel_x_m_s": -1.0}}  # punch.aim_x_m 0.3: arrival 1.0 s
    leaving = {"situation": {"ball_rel_x_m": 1.3, "ball_vel_x_m_s": 1.0}}
    assert threat_deadline(CONFIG, approaching, True, 0.9) == 0.9  # contact first
    expected = (1.3 - CONFIG["skills"]["dodge_vision"]["punch"]["aim_x_m"]) / 1.0
    assert threat_deadline(CONFIG, approaching, True, None) == pytest.approx(expected)
    assert threat_deadline(CONFIG, leaving, True, None) is None
    assert threat_deadline(CONFIG, approaching, False, None) is None


def test_issue_metrics_and_strict_view() -> None:
    issues = [("DODGE_LEFT", 0.5), ("DODGE_LEFT", 0.5), None, ("RETREAT", 2.0)]
    near = [True, True, True, False]
    collision = [True, False, True, False]
    deadline = [0.8, 1.0, 0.9, None]
    t_hit = [0.8, None, 0.9, None]
    m = _issue_metrics(issues, near, deadline)
    assert (m["emitted"], m["correct"], m["recalled"], m["false_activations"]) == (3, 2, 2, 1)
    s = _strict_collision(issues, collision, t_hit, 0.95)
    assert (s["n_collisions"], s["emitted"], s["correct"], s["recalled"], s["false_activations"]) == (2, 3, 1, 1, 2)


def test_method_metrics_strict_view() -> None:
    outcome = {"collision": 0.0, "fall": 0.0, "min_separation_m": 0.5, "command_travel_m": 0.0}
    records = [  # a hit dodged in time, a near miss dodged after its arrival, a wide pass with a false dodge
        {"threat": True, "collision_threat": True, "t_hit": 1.0, "t_deadline": 1.0, "issues": {"snn": [[0.6, "DODGE_LEFT"]]}},
        {"threat": True, "collision_threat": False, "t_hit": None, "t_deadline": 1.0, "issues": {"snn": [[1.2, "DODGE_LEFT"]]}},
        {"threat": False, "collision_threat": False, "t_hit": None, "t_deadline": None, "issues": {"snn": [[0.4, "RETREAT"]]}},
    ]
    for r in records:
        r.update(outcomes={"snn": dict(outcome)}, cooldown_conflicts={"snn": 0})
    m = _method_metrics(records, "snn", ("NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT"), 4.0, 0.0, 0.95)
    assert (m["n_threats"], m["issued"], m["issued_correct"], m["recalled"], m["late"]) == (2, 3, 2, 1, 1)
    strict = m["strict_collision"]
    assert (strict["n_threats"], strict["issued_correct"], strict["recalled"], strict["false_activations"]) == (1, 1, 1, 2)
