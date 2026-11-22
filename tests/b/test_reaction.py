"""Tests of learned_judgment.reaction's pure parts: hand-built inputs only (the simulator passes are its commands)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from learned_judgment.reaction import (
    best_action, choose_step, compare_ceilings, eye_times, issue_at_policy, last_ok, reach_times, spread,
)

ZERO = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}
LEFT = {"vel_x": 0.0, "vel_y": 1.0, "yaw_rate": 0.0}
ACTIVE = ["DODGE_LEFT", "DODGE_RIGHT", "RETREAT"]


def test_eye_times_window_then_votes():
    rows = eye_times([15.0, 30.0], [4, 8], votes_needed=3)
    by = {(r["frame_rate_hz"], r["frames"]): r for r in rows}
    assert len(rows) == 4
    assert math.isclose(by[30.0, 8]["first_full_window_s"], 7 / 30) and math.isclose(by[30.0, 8]["earliest_command_s"], 9 / 30)
    assert math.isclose(by[15.0, 4]["first_full_window_s"], 3 / 15) and math.isclose(by[15.0, 4]["earliest_command_s"], 5 / 15)


def test_reach_times_first_sample_at_or_beyond_each_distance():
    t = [0.02, 0.04, 0.06, 0.08]
    along = [0.0, 0.03, 0.07, 0.25]
    assert reach_times(t, along, [0.05, 0.1, 0.2, 0.3]) == [0.06, 0.08, 0.08, None]
    assert reach_times(t, [0.0, -0.3, 0.0, 0.0], [0.05]) == [None]  # the wrong way never counts


def test_spread_quartiles_and_empty():
    s = spread([4.0, 1.0, 3.0, 2.0, 5.0])
    assert (s["n"], s["min"], s["q1"], s["median"], s["q3"], s["max"]) == (5, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert spread([]) == {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}


def test_last_ok_bisects_between_a_known_success_and_a_known_failure():
    probes = []

    def ok(i):
        probes.append(i)
        return i <= 6

    assert last_ok(ok, 0, 10) == 6
    assert len(probes) <= math.ceil(math.log2(10)) and 0 not in probes and 10 not in probes
    assert last_ok(lambda i: pytest.fail("no probe needed"), 3, 4) == 3


def test_choose_step_smallest_step_within_the_budget():
    assert choose_step([8, 8], budget=6) == 1  # 3 + 3 probes
    assert choose_step([8, 8], budget=4) == 2  # grid of 4 points each: 2 + 2
    assert choose_step([1, 1], budget=0) == 1  # nothing to search


def test_best_action_largest_clean_separation_then_readout_order():
    hit, fell = {"collision": 1.0, "fall": 0.0, "min_separation_m": 0.0}, {"collision": 0.0, "fall": 1.0, "min_separation_m": 0.5}
    near, far = {"collision": 0.0, "fall": 0.0, "min_separation_m": 0.05}, {"collision": 0.0, "fall": 0.0, "min_separation_m": 0.12}
    assert best_action({"NO_OP": hit, "DODGE_LEFT": near, "DODGE_RIGHT": far, "RETREAT": fell}, ACTIVE) == "DODGE_RIGHT"
    assert best_action({"NO_OP": hit, "DODGE_LEFT": far, "DODGE_RIGHT": far, "RETREAT": hit}, ACTIVE) == "DODGE_LEFT"
    assert best_action({"NO_OP": hit, "DODGE_LEFT": hit, "DODGE_RIGHT": fell, "RETREAT": hit}, ACTIVE) is None


def test_compare_ceilings_counts_and_kind_changes():
    hit = {"collision": 1.0, "fall": 0.0, "min_separation_m": 0.0, "first_collision_time_s": 0.5}
    miss = {"collision": 0.0, "fall": 0.0, "min_separation_m": 0.1, "first_collision_time_s": None}
    fell = {"collision": 0.0, "fall": 1.0, "min_separation_m": 0.3, "first_collision_time_s": None}
    names = ["NO_OP", *ACTIVE]
    small = [{"scenario_id": 0, "outcomes": {"NO_OP": miss}},
             {"scenario_id": 1, "outcomes": {"NO_OP": hit, "DODGE_LEFT": miss, "DODGE_RIGHT": hit, "RETREAT": hit}},
             {"scenario_id": 2, "outcomes": {"NO_OP": miss}}]
    big = [{"scenario_id": 0, "outcomes": {"NO_OP": hit, "DODGE_LEFT": fell, "DODGE_RIGHT": hit, "RETREAT": hit}},
           {"scenario_id": 1, "outcomes": {"NO_OP": hit, "DODGE_LEFT": miss, "DODGE_RIGHT": miss, "RETREAT": hit}},
           {"scenario_id": 2, "outcomes": {"NO_OP": miss}}]
    kinds = {0: "ambiguous", 1: "incoming_left", 2: "off_path"}
    c = compare_ceilings(small, big, kinds, names, "NO_OP")
    assert (c["a"]["n_hit_no_op"], c["a"]["n_avoidable"], c["b"]["n_hit_no_op"], c["b"]["n_avoidable"]) == (1, 1, 2, 1)
    assert c["b"]["falls"] == {"NO_OP": 0, "DODGE_LEFT": 1, "DODGE_RIGHT": 0, "RETREAT": 0}
    assert c["missed_to_hit_by_kind"] == {"ambiguous": 1} and c["hit_to_missed_by_kind"] == {}
    with pytest.raises(ValueError, match="scenario"):
        compare_ceilings(small, big[:2], kinds, names, "NO_OP")


def test_issue_at_policy_waits_then_issues_once_and_holds():
    policy = issue_at_policy(0.5, "DODGE_LEFT", LEFT, ZERO, "NO_OP", hold_s=0.6)
    w = np.zeros((8, 12, 12), np.float32)
    early = policy(w, 0.3)
    assert early["command"] == ZERO and early["action"] == "NO_OP" and early.get("issued") is not True
    at = policy(w, 0.5)
    assert at["command"] == LEFT and at["issued"] is True
    assert policy(w, 1.0)["command"] == LEFT and policy(w, 1.0).get("issued") is not True
    assert policy(w, 1.1)["command"] == ZERO  # 0.5 + 0.6
