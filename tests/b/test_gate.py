"""Tests of learned_judgment.gate: pure-function fixtures (hand-built readouts) and the real config.yaml."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from learned_judgment.config import ConfigError, load_config
from learned_judgment.gate import UNCERTAIN, GateSettings, ThreatGate

NAMES = ("NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT")
FPS = 30.0

QUIET = (0.1, [0.97, 0.01, 0.01, 0.01])
UNSURE = (0.7, [0.02, 0.9, 0.04, 0.04])  # clear scores, threat between floor 0.5 and threshold 0.9
LEFT = (0.95, [0.05, 0.85, 0.05, 0.05])
RIGHT = (0.95, [0.05, 0.05, 0.85, 0.05])
ALIKE = (0.95, [0.0, 0.7, 0.65, 0.0])  # high threat, action score passes, margin 0.05 does not

SIDESTEP = (("DODGE_LEFT", "DODGE_RIGHT"),)  # config.yaml inference.gate.interchangeable
SPLIT = (0.97, [0.05, 0.45, 0.45, 0.05])  # a punch at the centre: left and right equally good
LEFT_HEAVY = (0.97, [0.05, 0.7, 0.2, 0.05])
RIGHT_HEAVY = (0.97, [0.05, 0.2, 0.7, 0.05])
BACK = (0.97, [0.05, 0.05, 0.05, 0.85])  # a clear RETREAT


def make_settings(**changes) -> GateSettings:
    base = GateSettings(
        action_names=NAMES,
        no_op="NO_OP",
        threat_threshold=0.9,
        action_threshold=0.6,
        margin_threshold=0.3,
        votes_needed=3,
        of_last=4,
        cooldown_s=1.0,
        hold_s=0.6,
        uncertain_floor=0.5,
    )
    return replace(base, **changes) if changes else base


def t(frame: int) -> float:
    return frame / FPS


def feed(gate: ThreatGate, readouts, start: int = 0) -> list:
    return [gate.update(threat, scores, t(start + k)) for k, (threat, scores) in enumerate(readouts)]


def test_quiet_readouts_give_no_op() -> None:
    outs = feed(ThreatGate(make_settings()), [QUIET] * 60)
    for out in outs:
        assert out.valid and out.action == "NO_OP"
        assert not out.issued and not out.holding and not out.passed and not out.cooldown_conflict
        assert out.votes == 0 and out.hold_remaining_s == 0.0 and out.cooldown_remaining_s == 0.0
        assert out.best_action == "DODGE_LEFT"  # first active action on a tie
        assert out.margin == pytest.approx(0.01 - 0.97)


def test_threat_between_floor_and_threshold_is_uncertain() -> None:
    gate = ThreatGate(make_settings())
    for out in feed(gate, [UNSURE] * 30):
        assert out.action == UNCERTAIN and not out.issued and not out.holding
        assert not out.threat_ok and out.action_ok and out.margin_ok and out.votes == 0
    assert gate.update(0.5, UNSURE[1], t(30)).action == UNCERTAIN  # at the floor
    assert gate.update(0.49, UNSURE[1], t(31)).action == "NO_OP"  # below the floor


def test_three_clear_votes_issue() -> None:
    outs = feed(ThreatGate(make_settings()), [LEFT] * 3)
    for out, votes in zip(outs[:2], (1, 2)):
        assert out.action == UNCERTAIN and out.votes == votes and not out.passed and not out.issued
        assert out.hold_remaining_s == 0.0 and out.cooldown_remaining_s == 0.0
    out = outs[2]
    assert out.action == "DODGE_LEFT" and out.issued and out.holding and out.passed and out.votes == 3
    assert out.threat_ok and out.action_ok and out.margin_ok and not out.cooldown_conflict
    assert out.hold_remaining_s == pytest.approx(0.6) and out.cooldown_remaining_s == pytest.approx(1.0)


def test_vote_window() -> None:
    outs = feed(ThreatGate(make_settings()), [LEFT, ALIKE, LEFT, LEFT])
    assert [o.issued for o in outs] == [False, False, False, True]
    assert outs[3].votes == 3 and outs[3].action == "DODGE_LEFT"

    outs = feed(ThreatGate(make_settings()), [LEFT, LEFT, RIGHT])
    assert outs[2].best_action == "DODGE_RIGHT" and outs[2].votes == 1
    assert not outs[2].passed and not outs[2].issued and outs[2].action == UNCERTAIN

    outs = feed(ThreatGate(make_settings()), [LEFT, ALIKE, ALIKE, LEFT, LEFT])
    assert outs[4].votes == 2 and not outs[4].passed
    assert not any(o.issued for o in outs)


def test_small_margin_never_issues() -> None:
    for out in feed(ThreatGate(make_settings()), [ALIKE] * 30):
        assert out.threat_ok and out.action_ok and not out.margin_ok
        assert out.best_action == "DODGE_LEFT" and out.margin == pytest.approx(0.05)
        assert out.votes == 0 and not out.issued and out.action == UNCERTAIN


def test_hold_then_release() -> None:
    gate = ThreatGate(make_settings())
    issue = feed(gate, [LEFT] * 3)[2]
    assert issue.issued
    t_issue = t(2)
    outs = feed(gate, [QUIET] * 27, start=3)  # frames 3..29
    outs += feed(gate, [UNSURE] * 5, start=30)  # frames 30..34
    for frame, out in enumerate(outs, start=3):
        held = t(frame) < t_issue + 0.6
        if frame <= 19:  # up to 17/30 s after the issue
            assert held
        if frame >= 21:
            assert not held
        assert not out.issued
        if held:
            assert out.action == "DODGE_LEFT" and out.holding and out.hold_remaining_s > 0.0
        else:
            assert not out.holding and out.hold_remaining_s == 0.0
            assert out.action == ("NO_OP" if frame < 30 else UNCERTAIN)


def test_hold_ends_at_hold_s() -> None:
    gate = ThreatGate(make_settings())
    feed(gate, [LEFT] * 3)
    t_issue = t(2)
    assert gate.update(*QUIET, t_issue + 0.6 - 1e-6).action == "DODGE_LEFT"
    out = gate.update(*QUIET, t_issue + 0.6)
    assert out.action == "NO_OP" and not out.holding and out.hold_remaining_s == 0.0
    assert out.cooldown_remaining_s == pytest.approx(0.4)


def test_cooldown_refuses_then_allows_a_second_command() -> None:
    gate = ThreatGate(make_settings())
    assert feed(gate, [LEFT] * 3)[2].issued
    t_issue = t(2)
    feed(gate, [QUIET] * 12, start=3)  # frames 3..14
    outs = dict(zip(range(15, 46), feed(gate, [RIGHT] * 31, start=15)))  # frames 15..45

    for frame in (15, 16):  # votes 1 and 2 while holding
        assert not outs[frame].passed and outs[frame].action == "DODGE_LEFT" and not outs[frame].cooldown_conflict
    at_half = outs[17]  # 0.5 s after the issue
    assert at_half.passed and at_half.cooldown_conflict and not at_half.issued
    assert at_half.holding and at_half.action == "DODGE_LEFT"
    at_08 = outs[26]  # 0.8 s after the issue
    assert at_08.passed and at_08.cooldown_conflict and not at_08.issued
    assert not at_08.holding and at_08.action == UNCERTAIN

    second = [frame for frame, out in outs.items() if out.issued]
    expected = next(frame for frame in outs if t(frame) >= t_issue + 1.0)
    assert second == [expected]
    out = outs[expected]
    assert out.action == "DODGE_RIGHT" and out.holding and not out.cooldown_conflict
    assert 1.0 <= t(expected) - t_issue + 1e-9 and t(expected) - t_issue <= 1.0 + 1.0 / FPS + 1e-9
    assert all(outs[frame].cooldown_conflict for frame in outs if 17 <= frame < expected)


def test_after_cooldown_a_new_command_needs_fresh_votes() -> None:
    gate = ThreatGate(make_settings())
    assert feed(gate, [LEFT] * 3)[2].issued
    feed(gate, [QUIET] * 37, start=3)  # frames 3..39, cooldown over at about frame 32
    outs = feed(gate, [RIGHT] * 3, start=40)
    assert [o.votes for o in outs] == [1, 2, 3]
    assert [o.action for o in outs] == [UNCERTAIN, UNCERTAIN, "DODGE_RIGHT"]
    assert [o.issued for o in outs] == [False, False, True]


@pytest.mark.parametrize(
    "threat, scores",
    [
        (0.95, [0.05, math.nan, 0.05, 0.05]),
        (0.95, [0.05, math.inf, 0.05, 0.05]),
        (0.95, [0.05, 0.85, 0.05]),
        (0.95, [0.05, 0.85, 0.05, 0.05, 0.0]),
        (0.95, [0.05, -0.1, 0.05, 0.05]),
        (0.95, None),
        (0.95, "0.1 0.2 0.3 0.4"),
        (0.95, [0.05, "0.85", 0.05, 0.05]),
        (0.95, np.array([[0.05, 0.85, 0.05, 0.05]])),
        (math.nan, [0.05, 0.85, 0.05, 0.05]),
        (1.5, [0.05, 0.85, 0.05, 0.05]),
        (-0.1, [0.05, 0.85, 0.05, 0.05]),
        (None, [0.05, 0.85, 0.05, 0.05]),
    ],
)
def test_invalid_readout(threat, scores) -> None:
    out = ThreatGate(make_settings()).update(threat, scores, 0.0)
    assert not out.valid and out.action == UNCERTAIN
    assert out.votes == 0 and not out.passed and not out.issued
    assert out.threat_confidence == 0.0 and out.action_confidence == 0.0 and out.margin == 0.0
    assert out.best_action == "NO_OP" and not (out.threat_ok or out.action_ok or out.margin_ok)


def test_invalid_readout_takes_a_window_slot_and_numpy_scores_are_valid() -> None:
    bad = (0.95, [0.05, math.nan, 0.05, 0.05])
    outs = feed(ThreatGate(make_settings()), [LEFT, LEFT, bad, LEFT])
    assert [o.issued for o in outs] == [False, False, False, True]
    assert outs[3].votes == 3

    out = ThreatGate(make_settings()).update(np.float64(0.95), np.array(LEFT[1]), 0.0)
    assert out.valid and out.best_action == "DODGE_LEFT" and out.votes == 1


def test_time_must_not_go_backwards() -> None:
    gate = ThreatGate(make_settings())
    gate.update(*LEFT, 1.0)
    gate.update(*LEFT, 1.0)  # the same time is allowed
    with pytest.raises(ValueError):
        gate.update(*LEFT, 0.9)
    for bad in (math.nan, math.inf, None, "1.0"):
        with pytest.raises(ValueError):
            gate.update(*LEFT, bad)
    assert gate.update(*LEFT, 1.1).issued  # the refused updates changed nothing: third vote issues


def test_reset_clears_everything() -> None:
    gate = ThreatGate(make_settings())
    assert feed(gate, [LEFT] * 3, start=100)[2].issued
    gate.reset()
    out = gate.update(*LEFT, 0.0)  # earlier time is fine after reset
    assert out.votes == 1 and not out.holding and not out.issued and out.action == UNCERTAIN
    assert out.hold_remaining_s == 0.0 and out.cooldown_remaining_s == 0.0
    outs = feed(gate, [LEFT] * 2, start=1)
    assert outs[1].issued and not outs[1].cooldown_conflict


@pytest.mark.parametrize(
    "changes",
    [
        {"votes_needed": 5},
        {"votes_needed": 0},
        {"cooldown_s": 0.5},
        {"hold_s": 0.0},
        {"hold_s": math.inf, "cooldown_s": math.inf},
        {"threat_threshold": 1.5},
        {"action_threshold": -0.1},
        {"margin_threshold": math.nan},
        {"uncertain_floor": 1.01},
        {"action_names": ("NO_OP", UNCERTAIN)},
        {"action_names": ("NO_OP",)},
        {"action_names": ("NO_OP", "NO_OP", "DODGE_LEFT")},
        {"no_op": "STAY"},
    ],
)
def test_settings_value_errors(changes) -> None:
    with pytest.raises(ValueError):
        make_settings(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"votes_needed": 3.0},
        {"of_last": True},
        {"threat_threshold": "0.9"},
        {"action_names": list(NAMES)},
    ],
)
def test_settings_type_errors(changes) -> None:
    with pytest.raises(TypeError):
        make_settings(**changes)


def test_from_config() -> None:
    settings = GateSettings.from_config(load_config(), {"threat": 0.9, "action": 0.6, "margin": 0.3})
    assert settings.action_names == ("NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT")
    assert settings.no_op == "NO_OP"
    assert settings.votes_needed == 3 and settings.of_last == 4
    assert settings.cooldown_s == 1.0 and settings.hold_s == 0.6
    assert (settings.threat_threshold, settings.action_threshold, settings.margin_threshold) == (0.9, 0.6, 0.3)
    assert settings.interchangeable == SIDESTEP
    assert settings == make_settings(interchangeable=SIDESTEP)


@pytest.mark.parametrize(
    "thresholds",
    [
        {"threat": 0.9, "action": 0.6},
        {"threat": 0.9, "action": 0.6, "margin": 0.3, "floor": 0.5},
    ],
)
def test_from_config_threshold_keys(thresholds) -> None:
    with pytest.raises(ValueError):
        GateSettings.from_config(load_config(), thresholds)


def test_from_config_needs_exactly_one_zero_command() -> None:
    config = copy.deepcopy(load_config())
    config["skills"]["dodge_vision"]["actions"]["RETREAT"] = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}
    with pytest.raises(ValueError):
        GateSettings.from_config(config, {"threat": 0.9, "action": 0.6, "margin": 0.3})


# --- interchangeable actions: stepping left or right is one decision, the side a detail ---


def sidestep_gate() -> ThreatGate:
    return ThreatGate(make_settings(interchangeable=SIDESTEP))


def test_split_scores_issue_a_sidestep() -> None:
    outs = feed(sidestep_gate(), [SPLIT] * 3)
    for out, votes in zip(outs, (1, 2, 3)):
        assert out.valid and out.threat_ok and out.action_ok and out.margin_ok and out.votes == votes
        assert out.action_confidence == pytest.approx(0.90) and out.margin == pytest.approx(0.85)
        assert out.best_action == "DODGE_LEFT"  # a tie: the first in readout order
    assert [o.action for o in outs] == [UNCERTAIN, UNCERTAIN, "DODGE_LEFT"]
    assert [o.issued for o in outs] == [False, False, True]
    assert outs[2].passed and outs[2].holding and not outs[2].cooldown_conflict


def test_alternating_sides_issue_the_side_with_the_larger_summed_score() -> None:
    for readouts, side in (
        ([LEFT_HEAVY, RIGHT_HEAVY, LEFT_HEAVY], "DODGE_LEFT"),  # summed left 1.6, right 1.1
        ([RIGHT_HEAVY, LEFT_HEAVY, RIGHT_HEAVY], "DODGE_RIGHT"),  # summed left 1.1, right 1.6
        ([LEFT_HEAVY, LEFT_HEAVY, RIGHT_HEAVY], "DODGE_LEFT"),  # summed left 1.6, though the last output prefers right
    ):
        outs = feed(sidestep_gate(), readouts)
        assert [o.votes for o in outs] == [1, 2, 3] and [o.issued for o in outs] == [False, False, True]
        assert outs[1].best_action == "DODGE_LEFT"  # summed 0.9 / 0.9 (a tie) or 1.4 / 0.4
        assert outs[2].action == side and outs[2].best_action == side
        assert outs[2].action_confidence == pytest.approx(0.9) and outs[2].margin == pytest.approx(0.85)

    loud_right = (0.5, [0.0, 0.05, 0.95, 0.0])  # threat below the threshold: no vote
    outs = feed(sidestep_gate(), [loud_right, LEFT_HEAVY, RIGHT_HEAVY, LEFT_HEAVY])
    assert outs[0].votes == 0 and outs[0].best_action == "DODGE_RIGHT"  # its own scores when it casts no vote
    assert outs[3].issued and outs[3].votes == 3
    assert outs[3].action == "DODGE_LEFT"  # the non-voting output does not count (with it: right 2.05, left 1.65)


def test_without_the_group_split_or_alternating_scores_never_issue() -> None:
    outs = feed(ThreatGate(make_settings()), [SPLIT] * 30)
    for out in outs:
        assert out.threat_ok and not out.action_ok and not out.margin_ok and out.votes == 0
        assert out.best_action == "DODGE_LEFT" and out.action_confidence == 0.45 and out.margin == 0.0
        assert not out.issued and out.action == UNCERTAIN
    outs = feed(ThreatGate(make_settings()), [LEFT_HEAVY, RIGHT_HEAVY] * 15)
    assert [o.best_action for o in outs] == ["DODGE_LEFT", "DODGE_RIGHT"] * 15
    assert max(o.votes for o in outs) == 2
    assert not any(o.passed or o.issued for o in outs) and all(o.action == UNCERTAIN for o in outs)


@pytest.mark.parametrize(
    "scores, best, confidence, action_ok",
    [
        ([0.05, 0.25, 0.2, 0.5], "RETREAT", 0.5, False),  # RETREAT 0.5 against the sidestep 0.45
        ([0.05, 0.25, 0.25, 0.45], "DODGE_LEFT", 0.5, False),  # the sidestep 0.5 against RETREAT 0.45
        ([0.03, 0.31, 0.31, 0.35], "DODGE_LEFT", 0.62, True),  # the sidestep passes the action threshold, not the margin
    ],
)
def test_the_margin_is_taken_against_the_other_groups(scores, best, confidence, action_ok) -> None:
    for out in feed(sidestep_gate(), [(0.97, scores)] * 12):
        assert out.threat_ok and out.action_ok == action_ok and not out.margin_ok
        assert out.best_action == best and out.action_confidence == pytest.approx(confidence)
        assert out.margin == pytest.approx(confidence - max(scores[0], sum(scores) - scores[0] - confidence))
        assert out.votes == 0 and not out.passed and not out.issued and out.action == UNCERTAIN


def test_no_op_above_the_sidestep_never_issues() -> None:
    for out in feed(sidestep_gate(), [(0.97, [0.5, 0.25, 0.2, 0.05])] * 12):
        assert out.best_action == "DODGE_LEFT" and out.action_confidence == pytest.approx(0.45)
        assert out.margin == pytest.approx(-0.05) and not out.margin_ok and not out.action_ok
        assert out.votes == 0 and not out.issued and out.action == UNCERTAIN


def test_votes_for_the_sidestep_and_for_retreat_do_not_pool() -> None:
    outs = feed(sidestep_gate(), [SPLIT, BACK] * 6)
    assert [o.best_action for o in outs] == ["DODGE_LEFT", "RETREAT"] * 6
    assert all(o.threat_ok and o.action_ok and o.margin_ok for o in outs)
    assert [o.votes for o in outs] == [1, 1] + [2] * 10
    assert not any(o.passed or o.issued for o in outs)

    outs = feed(sidestep_gate(), [SPLIT, BACK, SPLIT, SPLIT])
    assert [o.votes for o in outs] == [1, 1, 2, 3]
    assert [o.issued for o in outs] == [False, False, False, True] and outs[3].action == "DODGE_LEFT"


@pytest.mark.parametrize(
    "groups",
    [
        (("DODGE_LEFT", "JUMP"),),  # not an action
        (("NO_OP", "DODGE_LEFT"),),  # the zero command
        (("DODGE_LEFT", "DODGE_LEFT"),),  # a name twice in one group
        (("DODGE_LEFT", "DODGE_RIGHT"), ("DODGE_RIGHT", "RETREAT")),  # a name in two groups
        (("DODGE_LEFT",),),  # a group of one
        ((),),  # an empty group
    ],
)
def test_interchangeable_value_errors(groups) -> None:
    with pytest.raises(ValueError):
        make_settings(interchangeable=groups)


@pytest.mark.parametrize(
    "groups",
    [
        [["DODGE_LEFT", "DODGE_RIGHT"]],
        ("DODGE_LEFT", "DODGE_RIGHT"),
        ((("DODGE_LEFT",), "DODGE_RIGHT"),),
    ],
)
def test_interchangeable_type_errors(groups) -> None:
    with pytest.raises(TypeError):
        make_settings(interchangeable=groups)


def test_from_config_interchangeable_key() -> None:
    thresholds = {"threat": 0.9, "action": 0.6, "margin": 0.3}
    config = copy.deepcopy(load_config())
    assert GateSettings.from_config(config, thresholds).interchangeable == (("DODGE_LEFT", "DODGE_RIGHT"),)
    config["inference"]["gate"]["interchangeable"] = []
    assert GateSettings.from_config(config, thresholds) == make_settings()  # empty: every action its own group
    for bad in ("DODGE_LEFT", ["DODGE_LEFT", "DODGE_RIGHT"], [[1, 2]], None):
        config["inference"]["gate"]["interchangeable"] = bad
        with pytest.raises(ConfigError):
            GateSettings.from_config(config, thresholds)
    del config["inference"]["gate"]["interchangeable"]
    with pytest.raises(ConfigError):
        GateSettings.from_config(config, thresholds)
    config["inference"]["gate"]["interchangeable"] = [["DODGE_LEFT", "NO_OP"]]
    with pytest.raises(ValueError):
        GateSettings.from_config(config, thresholds)
