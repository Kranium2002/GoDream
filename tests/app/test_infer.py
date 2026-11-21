"""Live inference (app/infer.py): the request checks, the frame age, the plumbing to Dev B's runtime and the rule that
the camera-derived window never reaches a log line, an error message or a response. The runtime here is a stand-in
written by the test to exercise this server's own code; it is no evidence that live inference works."""
import copy
import types
import logging

import numpy as np
import pytest

from app import backend, infer
from app.routes import Refusal
from learned_judgment.env import load_config
from tests.app.conftest import write_student

RUNTIME_KEYS = {"action", "command", "threat_confidence", "best_action", "action_confidence", "margin", "gate",
                "cooldown_remaining_s", "hold_remaining_s", "issued", "cooldown_conflict", "valid", "inference_ms"}

WINDOW = {"frames": 2, "height": 3, "width": 3}  # a small window for these tests
MARK = 0.123456789  # a value easy to find if it ever leaks


@pytest.fixture(scope="module")
def config():
    return load_config()


def body(values=None, **extra):
    values = [MARK] * 18 if values is None else values
    return {"window": values, "frames": 2, "captured_ms": 1_000_000.0} | extra


def leaked(text: str) -> bool:
    return "0.1234" in text


# ---- image_window --------------------------------------------------------------------------------------------------

def test_image_window_is_published_from_the_vision_block_in_the_agreed_encoding(config):
    v = config["vision"]
    assert backend.image_window(config) == {
        "frames": v["frames"], "height": v["frame_size"], "width": v["frame_size"], "hz": v["frame_rate_hz"],
        "order": "oldest_first", "normalization": v["pixel_scale"], "crop": "centre_square",
        "luma": v["luma_weights"], "mirrored": False,
        "window_encoding": v["encoding"], "diff_floor": v["diff_floor"],
        "window_encoding_from": "config"}  # no student loaded for this config


def test_image_window_is_null_without_a_vision_block(config):
    assert backend.image_window({k: v for k, v in config.items() if k != "vision"}) is None


def test_inference_names_what_is_missing(config):
    run, reset, window, missing = backend.inference(config)
    if run is not None:
        pytest.skip("a frozen student is configured here")
    assert window is None and missing
    assert config["app"]["infer"]["student_dir"] or "app.infer.student_dir" in " ".join(missing)


# ---- the adapter against Dev B's REAL DodgeRuntime, with a hand-built untrained student (tests/app/conftest.py) ----

def with_student(config, directory):
    own = copy.deepcopy(config)
    own["app"]["infer"]["student_dir"] = str(directory)  # absolute
    return own


def test_the_adapter_loads_a_student_directory_and_returns_dev_bs_answer(config, tmp_path):
    own = with_student(config, write_student(tmp_path / "student", config))
    run, reset, window, missing = backend.inference(own)
    assert missing == [] and window == {"frames": config["vision"]["frames"], "height": config["vision"]["frame_size"],
                                        "width": config["vision"]["frame_size"]}
    frames = np.zeros((window["frames"], window["height"], window["width"]), np.float32)
    out = run(frames, 10.0)
    assert set(out) >= RUNTIME_KEYS and out["valid"] is True
    assert out["action"] == "NO_OP" and out["command"] == config["skills"]["dodge_vision"]["actions"]["NO_OP"]
    run(frames, 9.0)  # earlier than the last call: Dev B's runtime would raise; the adapter holds time still
    reset()
    run(frames, 1.0)  # a new camera session may start its clock anywhere


def test_a_student_without_frozen_thresholds_is_reported_missing(config, tmp_path):
    own = with_student(config, write_student(tmp_path / "student", config, thresholds=None))
    run, _, _, missing = backend.inference(own)
    assert run is None and "thresholds" in " ".join(missing)


def test_a_student_whose_window_differs_from_the_vision_block_is_reported_missing(config, tmp_path):
    own = with_student(config, write_student(tmp_path / "student", config, frames=config["vision"]["frames"] - 1))
    run, _, _, missing = backend.inference(own)
    assert run is None and "vision" in " ".join(missing)


def test_a_directory_without_a_student_is_reported_missing(config, tmp_path):
    run, _, _, missing = backend.inference(with_student(config, tmp_path))
    assert run is None and "student.json" in " ".join(missing)


def test_actions_and_their_hold_come_from_config_unchanged(config):
    dodge = config["skills"]["dodge_vision"]
    assert backend.actions(config) == (dodge["actions"], dodge["hold_s"])
    assert backend.actions({k: v for k, v in config.items() if k != "skills"}) == (None, None)


# ---- the request ---------------------------------------------------------------------------------------------------

def test_a_valid_window_becomes_a_float32_array_oldest_first():
    values = [i / 17 for i in range(18)]
    window, captured_ms, session_start = infer.check_body(body(values), WINDOW)
    assert window.dtype == np.float32 and window.shape == (2, 3, 3)
    assert np.allclose(window.ravel(), values) and captured_ms == 1_000_000.0 and session_start is False
    assert infer.check_body(body(session_start=True), WINDOW)[2] is True


@pytest.mark.parametrize("bad", [
    body([MARK] * 17), body([MARK] * 19), body([MARK] * 17 + [1.5]), body([MARK] * 17 + [-0.1]),
    body([MARK] * 17 + ["0.5"]), body([MARK] * 17 + [True]), body([MARK] * 17 + [None]),
    body(frames=3), body(frames=2.0), {"window": [MARK] * 18, "frames": 2}, body(captured_ms="now"),
    body(captured_ms=float("inf")), body(session_start="yes"), body(extra=1), body("not a list"),
], ids=["short", "long", "above_1", "below_0", "text", "bool", "null", "wrong_frames", "frames_float",
        "no_captured_ms", "captured_text", "captured_inf", "session_text", "extra_key", "window_text"])
def test_a_bad_window_is_refused_without_repeating_its_values(bad):
    with pytest.raises(Refusal) as refused:
        infer.check_body(bad, WINDOW)
    assert refused.value.status == 400 and refused.value.code == "invalid_window"
    assert not leaked(refused.value.message) and not leaked(str(refused.value.body()))


def test_a_non_finite_value_is_refused():
    with pytest.raises(Refusal):
        infer.check_body(body([MARK] * 17 + [float("nan")]), WINDOW)


@pytest.mark.parametrize("captured_ms, received_s, age_s, clamped", [
    (999_900.0, 1000.0, 0.1, False), (1_000_100.0, 1000.0, 0.0, True), (990_000.0, 1000.0, 0.0, True),
    (998_000.0, 1000.0, 2.0, False),
])
def test_the_frame_age_comes_from_wall_clocks_and_skew_is_clamped(captured_ms, received_s, age_s, clamped):
    assert infer.frame_age(captured_ms, received_s, max_age_ms=2000) == (pytest.approx(age_s), clamped)


# ---- the plumbing --------------------------------------------------------------------------------------------------

class StandIn:
    """Dev B's runtime, for this server's plumbing tests only."""

    def __init__(self, fail=False):
        self.calls, self.resets, self.fail = [], 0, fail

    def infer(self, window, t_capture_s):
        if self.fail:
            raise ValueError(f"bad window {window.tolist()}")  # the worst case: an error that repeats the window
        self.calls.append((window.copy(), t_capture_s))
        return {"action": "DODGE_LEFT", "command": {"vel_x": 0.0, "vel_y": 0.5, "yaw_rate": 0.0},
                "threat_confidence": np.float32(0.97), "action_confidence": 0.9, "margin": 0.4,
                "gate": {"passed": True}, "issued": True}

    def reset(self):
        self.resets += 1


def test_a_call_passes_the_window_and_capture_time_and_returns_dev_bs_dict(config):
    runtime = StandIn()
    result = infer.run(body(captured_ms=999_900.0), config, runtime.infer, runtime.reset, WINDOW,
                       received_mono=50.0, received_wall=1000.0)
    window, t_capture_s = runtime.calls[0]
    assert window.shape == (2, 3, 3) and t_capture_s == pytest.approx(49.9)
    assert result["action"] == "DODGE_LEFT" and result["gate"] == {"passed": True}
    assert result["frame_age_ms"] == pytest.approx(100.0) and result["frame_age_clamped"] is False
    assert runtime.resets == 0


def test_session_start_resets_the_runtime_before_inferring(config):
    runtime = StandIn()
    infer.run(body(session_start=True), config, runtime.infer, runtime.reset, WINDOW, 50.0, 1000.0)
    assert runtime.resets == 1 and len(runtime.calls) == 1


def test_a_failed_inference_leaks_the_window_nowhere(config, caplog):
    runtime = StandIn(fail=True)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Refusal) as refused:
            infer.run(body(), config, runtime.infer, runtime.reset, WINDOW, 50.0, 1000.0)
    assert refused.value.status == 502 and refused.value.code == "infer_failed"
    assert not leaked(str(refused.value.body()))
    assert caplog.records and not any(leaked(r.getMessage()) for r in caplog.records)
    assert all(r.exc_info is None for r in caplog.records)  # no traceback


def test_the_infer_event_carries_the_action_confidences_and_thresholds_only():
    result = StandIn().infer(np.zeros((2, 3, 3), np.float32), 0.0)
    assert infer.event(result) == {"action": "DODGE_LEFT", "threat_confidence": result["threat_confidence"],
                                   "action_confidence": 0.9, "thresholds": None}  # the stand-in names none
    assert infer.event({**result, "thresholds": "development"})["thresholds"] == "development"


# ---- development thresholds ----------------------------------------------------------------------------------------

DEV = {"threat": 0.9, "action": 0.6, "margin": 0.3}


def test_frozen_thresholds_are_labelled_frozen(config, tmp_path):
    run, _, window, missing = backend.inference(with_student(config, write_student(tmp_path / "s", config)))
    out = run(np.zeros((window["frames"], window["height"], window["width"]), np.float32), 1.0)
    assert missing == [] and out["thresholds"] == "frozen" and backend.thresholds_mode() == "frozen"


def test_development_thresholds_are_used_only_when_configured_and_say_so(config, tmp_path, caplog):
    own = with_student(config, write_student(tmp_path / "s", config, thresholds=None))
    own["app"]["infer"]["dev_thresholds"] = DEV
    with caplog.at_level(logging.WARNING):
        run, _, window, missing = backend.inference(own)
    out = run(np.zeros((window["frames"], window["height"], window["width"]), np.float32), 1.0)
    assert missing == [] and out["thresholds"] == "development" and backend.thresholds_mode() == "development"
    assert any("development" in r.getMessage() and "not calibrated" in r.getMessage() for r in caplog.records)


def test_development_thresholds_never_override_frozen_ones(config, tmp_path):
    own = with_student(config, write_student(tmp_path / "s", config))
    own["app"]["infer"]["dev_thresholds"] = {"threat": 0.1, "action": 0.1, "margin": 0.0}
    run, _, window, _ = backend.inference(own)
    assert run(np.zeros((window["frames"], window["height"], window["width"]), np.float32), 1.0)["thresholds"] == \
        "frozen"


def test_the_committed_config_has_no_development_thresholds(config):
    assert config["app"]["infer"]["dev_thresholds"] is None


def test_a_calibrated_students_gate_result_is_passed_through(config, tmp_path):
    import json
    directory = write_student(tmp_path / "s", config)
    student = json.loads((directory / "student.json").read_text())
    student["calibration"] = {"gate_passed": False, "precision": 0.97}  # as Dev B's dodge_eval calibrate writes it
    (directory / "student.json").write_text(json.dumps(student))
    run, _, window, missing = backend.inference(with_student(config, directory))
    out = run(np.zeros((window["frames"], window["height"], window["width"]), np.float32), 1.0)
    assert missing == [] and out["gate_passed"] is False and backend.calibration_gate_passed() is False
    assert "did not pass" in infer.infer_capability(types.SimpleNamespace(config=with_student(config, directory)))[
        "reason"]


def test_an_uncalibrated_student_has_no_gate_result(config, tmp_path):
    run, _, window, _ = backend.inference(with_student(config, write_student(tmp_path / "s", config)))
    out = run(np.zeros((window["frames"], window["height"], window["width"]), np.float32), 1.0)
    assert "gate_passed" not in out and backend.calibration_gate_passed() is None


def test_image_window_shows_what_the_loaded_student_reads(config, tmp_path):
    own = with_student(config, write_student(tmp_path / "s", config))  # input without window_encoding: raw
    assert backend.inference(own)[3] == []
    shown = backend.image_window(own)
    assert shown["window_encoding"] == "raw" and shown["diff_floor"] is None and shown["window_encoding_from"] == "student"
    assert backend.image_window(config)["window_encoding"] == config["vision"]["encoding"]  # another config: vision's
