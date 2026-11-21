"""learned_judgment.observed: pure mechanics on small synthetic arrays (fixtures only; nothing
here is a result) with the real config.yaml. No TabPFN request of any kind is made."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from learned_judgment.config import load_config, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.observed import (
    EventLog,
    available_observed_skills,
    dream_request_plan,
    dream_targets,
    encode_inputs,
    encoded_width,
    load_spec,
    make_teacher,
    one_shot_report,
    predict_outcomes,
    save_student,
    student_readout,
    teacher_matrix,
    train_student,
)
from learned_judgment.students.snn_core import SpikingNetwork, load_network

CONFIG = load_config()
LEVELS = tuple(require(CONFIG, "engine.quantile_levels"))
N = require(CONFIG, "vision.frame_size")
NAMES = ("NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT")
OUTCOMES = ("min_separation_m", "collision", "fall", "recovery_time_s", "command_travel_m")


class FakeSpec:
    """A hand-made spec object with the observed-spec attributes (fixture)."""

    name = "fake_observed"
    action_names = NAMES
    action_features = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    action_feature_names = ("vel_x", "vel_y", "yaw_rate")
    no_op_index = 0
    outcome_names = OUTCOMES
    outcome_kinds = ("regression", "binary", "binary", "regression", "regression")
    outcome_bounds = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 1.0], [0.0, 3.0], [0.0, 0.6]])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        return (
            -10.0 * predicted.mean_of("collision")
            - 10.0 * predicted.mean_of("fall")
            + np.minimum(predicted.median_of("min_separation_m"), 0.5)
            - 0.5 * predicted.mean_of("command_travel_m")
            - 0.5 * predicted.mean_of("recovery_time_s")
        )


SPEC = FakeSpec()
# The fixture skill reads the configured dodge_vision threat scale under its own name.
FAKE_CONFIG = copy.deepcopy(CONFIG)
FAKE_CONFIG["skills"][SPEC.name] = {"threat_target_scale": require(CONFIG, "skills.dodge_vision.threat_target_scale")}


def outcome_row(separation=1.0, collision=0.0, fall=0.0, recovery=0.0, travel=0.0) -> list[float]:
    return [separation, collision, fall, recovery, travel]


def scores_of(rows: list[list[float]]) -> np.ndarray:
    values = np.array(rows, dtype=np.float64)
    return SPEC.objective(OutcomeDistribution.point(OUTCOMES, values, np.asarray(LEVELS)))


# ----- teacher matrix ---------------------------------------------------------------------------


@pytest.mark.parametrize("feature_set", [["window", "action", "physics"], ["action", "physics"]])
def test_teacher_matrix_columns_and_layout(feature_set) -> None:
    rng = np.random.default_rng(0)
    n, d = 6, 2 * N * N
    features = rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)
    action = np.array([0, 1, 2, 3, 1, 0])
    physics = rng.normal(size=(n, 5))
    x, layout = teacher_matrix(SPEC, features, action, physics, feature_set)
    window = "window" in feature_set
    width = (d if window else 0) + 3 + 5
    assert x.dtype == np.float64 and x.shape == (n, width) and layout.n_features == width
    assert layout.skill == SPEC.name and layout.categorical == () and layout.n_extra == 0
    assert (layout.n_situation, layout.n_action, layout.n_physics) == ((d if window else 0), 3, 5)
    assert layout.outcome_names == OUTCOMES and layout.outcome_kinds == SPEC.outcome_kinds
    start = d if window else 0
    if window:
        assert np.array_equal(x[:, :d], features.astype(np.float64))
        assert layout.names[0] == "window_0" and layout.names[d - 1] == f"window_{d - 1}"
    assert np.array_equal(x[:, start : start + 3], SPEC.action_features[action])
    assert layout.names[start : start + 3] == ("vel_x", "vel_y", "yaw_rate")
    assert np.array_equal(x[:, start + 3 :], physics)
    assert layout.names[start + 3 :] == tuple(f"physics_{name}" for name in OUTCOMES)


@pytest.mark.parametrize("bad", [["window", "physics"], ["physics", "action"], ["action", "physics", "extra"], "action"])
def test_teacher_matrix_rejects_bad_feature_sets(bad) -> None:
    with pytest.raises(ValueError):
        teacher_matrix(SPEC, np.zeros((1, 4)), np.array([0]), np.zeros((1, 5)), bad)


# ----- targets and encoding -----------------------------------------------------------------------


def test_dream_targets_shapes_and_threat() -> None:
    no_op_best = [outcome_row(travel=0.0)] + [outcome_row(travel=0.6)] * 3  # active commands cost 0.3
    dodge_best = [outcome_row(separation=0.0, collision=1.0)] + [outcome_row(travel=0.6)] * 3
    scores = np.vstack([scores_of(no_op_best), scores_of(dodge_best), scores_of(no_op_best)])
    targets = dream_targets(scores, SPEC, FAKE_CONFIG)
    assert set(targets) == {"action", "threat"}
    assert targets["action"].shape == (3, 4) and targets["threat"].shape == (3, 2)
    for values in targets.values():
        assert np.all(values >= 0.0) and np.allclose(values.sum(axis=1), 1.0, atol=1e-12)
    assert targets["threat"][0, 1] < 0.01 and targets["threat"][2, 1] < 0.01  # logistic(-0.3 / 0.05) = 0.0025
    assert targets["threat"][1, 1] > 0.999
    assert np.argmax(targets["action"][0]) == 0 and np.argmax(targets["action"][1]) == 1


def test_dream_targets_without_no_op_has_no_threat() -> None:
    spec = FakeSpec()
    spec.no_op_index = None
    targets = dream_targets(np.array([[0.0, -1.0, -2.0, -3.0], [1.0, 0.0, 0.5, 0.2]]), spec, FAKE_CONFIG)
    assert set(targets) == {"action"}


def test_encode_inputs() -> None:
    encoded = encode_inputs(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    assert encoded.dtype == np.float64
    assert np.array_equal(encoded, (np.array([[0.0, 0.5, 1.0]]) - 0.5) * math.sqrt(12.0))
    for bad in (np.array([[1.5]]), np.array([[np.nan]]), np.zeros(3)):
        with pytest.raises(ValueError):
            encode_inputs(bad)


# ----- student ------------------------------------------------------------------------------------


def synthetic_dream(m: int = 200, frames: int = 2, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Windows whose first two pixels carry a bit; both heads' targets follow the bit."""
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.0, 1.0, size=(m, frames * N * N)).astype(np.float32)
    bit = rng.integers(0, 2, size=m)
    features[:, 0] = bit
    features[:, 1] = bit
    threat = np.where(bit[:, None] == 1, [0.1, 0.9], [0.9, 0.1])
    action = np.where(bit[:, None] == 1, [0.05, 0.85, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05])
    return features, {"threat": threat, "action": action}, np.arange(m) // 2


@pytest.fixture(scope="module")
def trained():
    features, targets, groups = synthetic_dream()
    seen = []
    result = train_student(
        SPEC, features, targets, CONFIG, hidden_sizes=[16], max_steps=300, seed=7, groups=groups, on_evaluation=seen.append
    )
    return features, targets, groups, result, seen


def test_train_student_parity_and_readout(trained) -> None:
    features, _, _, result, seen = trained
    network = result["network"]
    assert network.layer_sizes == (2 * N * N, 16, 6)
    assert result["heads"] == {"threat": [0, 2], "action": [2, 6]}
    tolerance = require(CONFIG, "students.snn.parity_logit_tolerance")
    for head in ("threat", "action"):
        report = result["parity"][head]
        assert report["decisions_differing"] == 0 and report["max_abs_logit_difference"] <= tolerance
    combined = result["parity"]["combined"]
    assert all(value <= tolerance for value in combined["max_abs_logit_difference"].values())
    assert combined["action_decisions_differing"] == 0
    split = result["split"]
    assert split["by"] == "group" and split["n_groups"] == 100
    assert split["n_train_rows"] + split["n_validation_rows"] == 200 and split["n_validation_rows"] % 2 == 0
    assert seen and seen[0]["step"] == 0 and result["training"]["steps_run"] <= 300
    threat, action = student_readout(network, features)
    assert threat.shape == (200,) and action.shape == (200, 4)
    assert np.all((threat >= 0.0) & (threat <= 1.0)) and np.allclose(action.sum(axis=1), 1.0)
    bit = features[:, 0] == 1.0
    assert threat[bit].mean() > threat[~bit].mean()  # the trained network reads the pixel


def test_train_student_is_deterministic(trained) -> None:
    features, targets, groups, result, _ = trained
    again = train_student(SPEC, features, targets, CONFIG, hidden_sizes=[16], max_steps=300, seed=7, groups=groups)
    assert again["network"].digest() == result["network"].digest()


def test_student_readout_without_threat_head() -> None:
    rng = np.random.default_rng(1)
    network = SpikingNetwork(
        hidden_weights=(rng.normal(size=(3, 5)),), hidden_biases=(np.zeros(3),), betas=(0.9,), thresholds=(1.0,),
        readout_weight=rng.normal(size=(4, 3)), readout_bias=np.zeros(4), readout_beta=0.9, time_steps=4,
    )
    threat, action = student_readout(network, rng.uniform(size=(2, 5)), heads={"action": [0, 4]})
    assert np.array_equal(threat, np.zeros(2)) and np.allclose(action.sum(axis=1), 1.0)


def test_save_student_round_trip(tmp_path) -> None:
    frames = require(CONFIG, "vision.frame_variants")[0]
    window_encoding, diff_floor = require(CONFIG, "vision.encoding"), require(CONFIG, "vision.diff_floor")
    rng = np.random.default_rng(2)
    network = SpikingNetwork(
        hidden_weights=(rng.normal(size=(8, encoded_width(frames, N, window_encoding))),), hidden_biases=(np.zeros(8),), betas=(0.9,),
        thresholds=(1.0,), readout_weight=rng.normal(size=(6, 8)), readout_bias=np.zeros(6), readout_beta=0.9,
        time_steps=5,
    )
    digest = save_student(tmp_path / "student", network, SPEC, frames=frames, config=CONFIG, provenance={"run_id": "t"})
    loaded, meta = load_network(tmp_path / "student" / "network")
    assert loaded.digest() == digest == network.digest()
    assert meta["heads"] == {"threat": [0, 2], "action": [2, 6]}
    student = json.loads((tmp_path / "student" / "student.json").read_text(encoding="utf-8"))
    assert student["format"] == "learned-judgment/observed-student/v1"
    assert student["skill"] == SPEC.name and student["action_names"] == list(NAMES) and student["no_op_index"] == 0
    assert student["heads"] == {"threat": [0, 2], "action": [2, 6]}
    assert student["input"] == {"frames": frames, "frame_size": N, "window_encoding": window_encoding,
                                "diff_floor": diff_floor, "encoding": "(x - 0.5) * sqrt(12)"}
    assert student["vision"]["frames"] == frames and student["thresholds"] is None
    assert student["provenance"] == {"run_id": "t"} and student["network"]["sha256"] == digest
    with pytest.raises(ValueError):
        save_student(tmp_path / "bad", network, SPEC, frames=require(CONFIG, "vision.frames"), config=CONFIG, provenance={})


# ----- events, rows, billing guard ------------------------------------------------------------


def test_event_log_lines_and_errors(tmp_path) -> None:
    fixed = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    log = EventLog(tmp_path / "run" / "events.jsonl", "r1", "fake_observed", clock=lambda: fixed)
    log.emit("teacher_fit", "running")
    log.emit("train", "running", done=100, total=300, data={"loss": np.float64(0.5)})
    lines = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {
        "run_id": "r1", "skill": "fake_observed", "step": "teacher_fit", "state": "running",
        "done": None, "total": None, "time_utc": "2026-11-21T12:00:00+00:00", "data": None,
    }
    assert lines[1]["done"] == 100 and lines[1]["total"] == 300 and lines[1]["data"] == {"loss": 0.5}
    for step, state in (("collect", "running"), ("train", "paused")):
        with pytest.raises(ValueError):
            log.emit(step, state)
    with pytest.raises(ValueError):
        log.emit("train", "running", done=-1)


def hand_rows() -> dict:
    """3 situations x 4 actions, rows deliberately not action-major."""
    measured = {
        0: [outcome_row(), outcome_row(travel=0.6), outcome_row(travel=0.6), outcome_row(travel=0.6)],
        1: [outcome_row(separation=0.0, collision=1.0), outcome_row(separation=0.4, travel=0.6),
            outcome_row(separation=0.0, collision=1.0, travel=0.6), outcome_row(separation=0.2, travel=0.6)],
        2: [outcome_row(separation=0.0, collision=1.0), outcome_row(separation=0.0, collision=1.0, fall=1.0, travel=0.6),
            outcome_row(separation=0.5, travel=0.6), outcome_row(separation=0.1, travel=0.6)],
    }
    order = [(s, a) for a in (3, 1, 0, 2) for s in (2, 0, 1)]
    return {
        "scenario_index": np.array([s for s, _ in order], dtype=np.int64),
        "action": np.array([a for _, a in order], dtype=np.int64),
        "measured": np.array([measured[s][a] for s, a in order], dtype=np.float64),
        "meta": {"scenarios": [{"kind": "static"}, {"kind": "incoming_left"}, {"kind": "incoming_right"}]},
    }


def test_one_shot_report_on_hand_made_rows() -> None:
    rows = hand_rows()
    no_op = one_shot_report(SPEC, rows, np.array([0, 0, 0]), quantile_levels=LEVELS)
    assert no_op["n"] == 3 and no_op["action_counts"] == {"NO_OP": 3, "DODGE_LEFT": 0, "DODGE_RIGHT": 0, "RETREAT": 0}
    assert no_op["outcomes"]["collision"] == {"count": 2, "rate": pytest.approx(2 / 3)}
    assert no_op["outcomes"]["min_separation_m"]["min"] == 0.0
    # best measured: situation 0 NO_OP (0.5), 1 DODGE_LEFT (0.4 - 0.3), 2 DODGE_RIGHT (0.5 - 0.3)
    assert no_op["mean_regret"] == pytest.approx((0.0 + (0.1 + 10.0) + (0.2 + 10.0)) / 3)
    best = one_shot_report(SPEC, rows, np.array([0, 1, 2]), quantile_levels=LEVELS)
    assert best["mean_regret"] == pytest.approx(0.0) and best["outcomes"]["collision"]["count"] == 0
    assert best["outcomes"]["command_travel_m"]["mean"] == pytest.approx(0.4)
    assert best["mean_measured_objective"] == pytest.approx((0.5 + 0.1 + 0.2) / 3)
    assert set(best["by_kind"]) == {"static", "incoming_left", "incoming_right"}
    assert best["by_kind"]["incoming_right"]["action_counts"]["DODGE_RIGHT"] == 1
    wrong = one_shot_report(SPEC, rows, np.array([0, 2, 1]), quantile_levels=LEVELS)
    assert wrong["outcomes"]["fall"]["count"] == 1 and wrong["outcomes"]["collision"]["count"] == 2
    with pytest.raises(ValueError, match="more than one row"):
        one_shot_report(SPEC, {**rows, "action": np.zeros(12, dtype=np.int64)}, np.array([0, 0, 0]), quantile_levels=LEVELS)


def synthetic_context(n: int = 8) -> dict:
    rng = np.random.default_rng(3)
    physics = rng.uniform(0.0, 0.5, size=(n, 5))
    return {
        "features": rng.uniform(size=(n, 2 * N * N)).astype(np.float32),
        "action": np.arange(n) % 4,
        "physics": physics,
        "measured": physics + rng.normal(0.0, 0.1, size=(n, 5)) * np.array([1.0, 1.0, 0.0, 1.0, 0.0]),
    }


def test_predict_outcomes_refuses_without_allow_billing() -> None:
    context = synthetic_context()
    teacher = make_teacher(SPEC, context, CONFIG, ["window", "action", "physics"])

    def forbidden(*_args, **_kwargs):
        pytest.fail("a provider or prediction call was made without allow_billing")

    teacher.predict_residual_distribution = forbidden
    teacher.provider.connect = forbidden
    x, _ = teacher_matrix(SPEC, context["features"], context["action"], context["physics"], ["window", "action", "physics"])
    for allow in (False, None, 1):
        with pytest.raises(RuntimeError, match="allow_billing"):
            predict_outcomes(teacher, SPEC, x, context["physics"], allow_billing=allow)
    with pytest.raises(RuntimeError, match="allow_billing"):
        predict_outcomes(teacher, SPEC, x, context["physics"])


def test_dream_request_plan_follows_the_teacher_rules() -> None:
    teacher = make_teacher(SPEC, synthetic_context(), CONFIG, ["action", "physics"])
    plan = dream_request_plan(teacher, 1000)
    # fall and command_travel_m residuals are constant (0) in the fixture: three outcomes are sent
    assert plan["active_outcomes"] == ["min_separation_m", "collision", "recovery_time_s"]
    assert plan["constant_outcomes"] == ["fall", "command_travel_m"]
    if plan["mode"] == "stacked":
        assert [(r["n_train"], r["n_query"], r["n_features"]) for r in plan["requests"]] == [(24, 3000, 9)]
    else:
        assert [(r["n_train"], r["n_query"], r["n_features"]) for r in plan["requests"]] == [(8, 1000, 8)] * 3


# ----- spec loading -------------------------------------------------------------------------------


def test_available_observed_skills_and_unknown_spec() -> None:
    skills = available_observed_skills()
    assert isinstance(skills, tuple) and all(isinstance(name, str) for name in skills)
    with pytest.raises(ValueError, match="learned_judgment.skills.no_such_skill"):
        load_spec("no_such_skill", CONFIG)
