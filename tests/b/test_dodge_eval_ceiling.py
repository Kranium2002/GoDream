"""Tests of the punch ceiling in learned_judgment.dodge_eval: pure-function fixtures (hand-built outcomes)."""

from __future__ import annotations

import numpy as np
import pytest

from learned_judgment.dodge_eval import ceiling_summary, first_issue_policy, student_against_ceiling

ZERO = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}
LEFT = {"vel_x": 0.0, "vel_y": 0.8, "yaw_rate": 0.0}
NAMES = ["NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT"]
HIT, MISS, FELL = {"collision": 1.0, "fall": 0.0}, {"collision": 0.0, "fall": 0.0}, {"collision": 0.0, "fall": 1.0}


def test_first_issue_policy_issues_once_holds_then_stops():
    policy = first_issue_policy("DODGE_LEFT", LEFT, ZERO, "NO_OP", hold_s=0.6)
    w = np.zeros((8, 12, 12), np.float32)
    first = policy(w, 0.233)
    assert first["action"] == "DODGE_LEFT" and first["issued"] is True and first["command"] == LEFT
    held = policy(w, 0.80)
    assert held["command"] == LEFT and held.get("issued") is not True
    after = policy(w, 0.84)  # 0.233 + 0.6 = 0.833
    assert after["command"] == ZERO and after["action"] == "NO_OP"


def test_ceiling_summary_counts_avoidable_and_unavoidable():
    records = [
        {"scenario_id": 0, "outcomes": {"NO_OP": MISS}},
        {"scenario_id": 1, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": MISS, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
        {"scenario_id": 2, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": HIT, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
        {"scenario_id": 3, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": FELL, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
    ]
    s = ceiling_summary(records, NAMES, "NO_OP")
    assert (s["n_scenarios"], s["n_hit_no_op"], s["n_avoidable"], s["n_unavoidable"]) == (4, 3, 1, 2)
    assert s["avoidable_by_action"] == {"DODGE_LEFT": 1, "DODGE_RIGHT": 0, "RETREAT": 0}
    assert s["unavoidable_ids"] == [2, 3]  # 3: the only clean miss was a fall


def test_ceiling_summary_refuses_an_incomplete_record():
    with pytest.raises(ValueError, match="scenario 5"):
        ceiling_summary([{"scenario_id": 5, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": MISS}}], NAMES, "NO_OP")
    with pytest.raises(ValueError, match="scenario 6"):
        ceiling_summary([{"scenario_id": 6, "outcomes": {"DODGE_LEFT": MISS}}], NAMES, "NO_OP")


def test_student_against_ceiling_uses_the_avoidable_ones_as_denominator():
    ceiling_records = [
        {"scenario_id": 1, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": MISS, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
        {"scenario_id": 2, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": HIT, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
    ]
    student = [{"scenario_id": 1, "outcomes": {"no_op": HIT, "snn": MISS}}, {"scenario_id": 2, "outcomes": {"no_op": HIT, "snn": HIT}}]
    r = student_against_ceiling(ceiling_records, student, "NO_OP")
    assert (r["n_avoidable"], r["student_avoided_of_avoidable"], r["student_hit_unavoidable"]) == (1, 1, 1)
