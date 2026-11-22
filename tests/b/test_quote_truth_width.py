"""run_skill.quote_run counts the teacher columns of the configured teacher_features, the truth block included
(Task 4 follow-up). quote_run makes no network call and bills nothing; the widths below come from the spec and the
config, not from the function's output."""

from __future__ import annotations

import copy

import pytest

from learned_judgment.config import load_config, require
from learned_judgment.run_skill import quote_run
from learned_judgment.skills.dodge_vision import TRUTH_NAMES, build_observed_spec
from learned_judgment.teacher import TeacherSettings

DODGE = {"skill": "DODGE", "params": {"threats": ["punch"]}, "source_text": "dodge an incoming punch"}


@pytest.fixture(scope="module")
def config():
    return load_config()


def expected_width(config, situation_columns: int) -> int:
    spec = build_observed_spec(config)
    k = len(spec.outcome_names)
    stacked = TeacherSettings.from_config(config).multi_outcome_mode == "stacked" and k > 1
    return situation_columns + spec.action_features.shape[1] + k + int(stacked)  # stacked adds the outcome-index column


def widths(config) -> set[int]:
    quote = quote_run(DODGE, config)
    assert quote["ok"] and quote["spec"] == "dodge_vision", quote
    return {request["n_features"] for request in quote["estimate"]["tabpfn_tokens"]["requests"]}


def test_quote_counts_the_truth_block(config):
    assert list(require(config, "skills.dodge_vision.teacher_features")) == ["truth", "action", "physics"]
    assert widths(config) == {expected_width(config, len(TRUTH_NAMES))}


def test_quote_counts_the_window_block_as_before(config):
    window = copy.deepcopy(config)
    window["skills"]["dodge_vision"]["teacher_features"] = ["window", "action", "physics"]
    d = require(config, "vision.frames") * require(config, "vision.frame_size") ** 2
    assert widths(window) == {expected_width(config, d)}
