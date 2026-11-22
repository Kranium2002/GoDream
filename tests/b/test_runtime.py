"""Tests of learned_judgment.runtime with the real config.yaml and a hand-built spiking student.

The tiny student is a pure-function fixture for the runtime's plumbing (loading, readout, gate, command):
two 12 x 12 frames, four hidden neurons, six outputs. It is not a trained student and nothing here is a
result of one.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import learned_judgment.runtime as runtime_module
from learned_judgment.config import load_config
from learned_judgment.gate import UNCERTAIN
from learned_judgment.runtime import ENCODING, STUDENT_FORMAT, DodgeRuntime, measure_latency
from learned_judgment.students.snn_core import SpikingNetwork, save_network

FRAMES, N = 2, 12
D = FRAMES * N * N
FPS = 30.0
NAMES = ["NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT"]
THRESHOLDS = {"threat": 0.9, "action": 0.6, "margin": 0.3}
ZERO = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}
KEYS = {
    "action", "command", "threat_confidence", "best_action", "action_confidence", "margin", "gate",
    "cooldown_remaining_s", "hold_remaining_s", "issued", "cooldown_conflict", "valid", "inference_ms",
}
GATE_KEYS = {"threat_ok", "action_ok", "margin_ok", "votes", "votes_needed", "of_last", "passed"}


def t(frame: int) -> float:
    return frame / FPS


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def encode(window: np.ndarray) -> np.ndarray:
    return (window.reshape(-1).astype(np.float64) - 0.5) * math.sqrt(12.0)


def dark() -> np.ndarray:
    return np.zeros((FRAMES, N, N), dtype=np.float32)


def blob() -> np.ndarray:
    """A bright 4 x 4 blob in the left half (columns 0..5) of the newest frame."""
    window = dark()
    window[-1, 4:8, 1:5] = 1.0
    return window


def tiny_network() -> SpikingNetwork:
    """Hidden neuron 0 reads the mean encoded value of the newest frame's left half: a blob lifts its current
    from -0.38 to +0.39, above the 0.25 threshold. It drives threat and DODGE_LEFT; the readout bias favours
    no-threat and NO_OP when nothing spikes. Neurons 1..3 never fire."""
    mask = np.zeros((FRAMES, N, N))
    mask[-1, :, : N // 2] = 1.0
    hidden = np.zeros((4, D))
    hidden[0] = mask.reshape(-1) / mask.sum()
    readout = np.zeros((6, 4))
    readout[:, 0] = [-4.0, 4.0, -4.0, 6.0, 0.0, 0.0]  # [no threat, threat | NO_OP, LEFT, RIGHT, RETREAT]
    return SpikingNetwork(
        hidden_weights=(hidden,),
        hidden_biases=(np.array([1.35, -1.0, -1.0, -1.0]),),
        betas=(0.5,),
        thresholds=(0.25,),
        readout_weight=readout,
        readout_bias=np.array([2.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        readout_beta=0.0,
        time_steps=4,
    )


def write_student(directory, network: SpikingNetwork | None = None, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    save_network(directory / "network", network or tiny_network(), {"note": "hand-built test fixture"})
    student = {
        "format": STUDENT_FORMAT,
        "skill": "dodge_vision",
        "action_names": list(NAMES),
        "heads": {"threat": [0, 2], "action": [2, 6]},
        "no_op_index": 0,
        "input": {"frames": FRAMES, "frame_size": N, "encoding": ENCODING},
        "vision": {"frames": FRAMES, "frame_size": N},
        "thresholds": dict(THRESHOLDS),
        "provenance": {"note": "hand-built test fixture, not a trained student"},
    }
    student.update(changes)
    (directory / "student.json").write_text(json.dumps(student), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def student_dir(tmp_path):
    return write_student(tmp_path / "student")


def command_of(config, name: str) -> dict:
    return {field: float(value) for field, value in config["skills"]["dodge_vision"]["actions"][name].items()}


def assert_plain(value) -> None:
    """Only dict / str / bool / int / float (finite), as json.dumps writes them."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str
            assert_plain(item)
    else:
        assert type(value) in (str, bool, int, float), type(value)
        if type(value) is float:
            assert math.isfinite(value)


# --- the fixture network itself ---


def test_tiny_network_forward_is_as_designed() -> None:
    logits = tiny_network().forward(np.stack([encode(blob()), encode(dark())]))
    threat_blob, actions_blob = softmax(logits[0, :2])[1], softmax(logits[0, 2:])
    threat_dark, actions_dark = softmax(logits[1, :2])[1], softmax(logits[1, 2:])
    assert threat_blob > 0.99 and int(np.argmax(actions_blob)) == 1 and actions_blob[1] > 0.99
    assert threat_dark < 0.2 and int(np.argmax(actions_dark)) == 0


# --- load ---


def test_load_takes_frames_from_the_student_not_config(config, student_dir) -> None:
    assert config["vision"]["frames"] != FRAMES
    runtime = DodgeRuntime.load(student_dir, config)
    assert (runtime.frames, runtime.frame_size) == (FRAMES, N)
    assert runtime.action_names == tuple(NAMES)
    assert runtime.thresholds == THRESHOLDS
    assert runtime.settings.threat_threshold == 0.9 and runtime.settings.no_op == "NO_OP"


def test_load_refuses_a_wrong_format(config, tmp_path) -> None:
    with pytest.raises(ValueError, match="observed-student"):
        DodgeRuntime.load(write_student(tmp_path / "s", format="learned-judgment/observed-student/v0"), config)


def test_load_refuses_other_action_names(config, tmp_path) -> None:
    swapped = ["NO_OP", "DODGE_RIGHT", "DODGE_LEFT", "RETREAT"]
    with pytest.raises(ValueError, match="action_names"):
        DodgeRuntime.load(write_student(tmp_path / "s", action_names=swapped), config)


@pytest.mark.parametrize(
    "changes",
    [
        {"heads": {"threat": [0, 2], "action": [2, 5]}},
        {"heads": {"action": [2, 6], "threat": [0, 1]}},
        {"input": {"frames": 3, "frame_size": N, "encoding": ENCODING}},
        {"input": {"frames": FRAMES, "frame_size": N, "encoding": "x"}},
        {"no_op_index": 1},
    ],
)
def test_load_refuses_a_contract_mismatch(config, tmp_path, changes) -> None:
    with pytest.raises(ValueError):
        DodgeRuntime.load(write_student(tmp_path / "s", **changes), config)


def test_load_checks_the_network_digest(config, student_dir) -> None:
    other = tiny_network()
    np.savez(student_dir / "network.npz", **{**other.to_arrays(), "readout_bias": np.zeros(6)})
    with pytest.raises(ValueError, match="digest"):
        DodgeRuntime.load(student_dir, config)


def test_missing_thresholds_and_the_override(config, tmp_path) -> None:
    unfrozen = write_student(tmp_path / "unfrozen", thresholds=None)
    with pytest.raises(RuntimeError, match="no frozen thresholds: calibrate on simulation dev trajectories first"):
        DodgeRuntime.load(unfrozen, config)
    trial = {"threat": 0.5, "action": 0.4, "margin": 0.1}
    assert DodgeRuntime.load(unfrozen, config, thresholds=trial).thresholds == trial
    frozen = write_student(tmp_path / "frozen")
    runtime = DodgeRuntime.load(frozen, config, thresholds=trial)
    assert runtime.thresholds == trial and runtime.settings.threat_threshold == 0.5


# --- readout ---


def test_readout_is_the_networks_softmax_per_head(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    network = tiny_network()
    rng = np.random.default_rng(0)
    windows = [blob(), dark(), *(rng.random((FRAMES, N, N), dtype=np.float32) for _ in range(5))]
    for window in windows:
        threat, scores = runtime.readout(window)
        logits = network.forward(encode(window)[None, :])[0]
        threat_head = softmax(logits[:2])
        assert type(threat) is float and 0.0 <= threat <= 1.0
        assert threat == pytest.approx(threat_head[1]) and threat_head.sum() == pytest.approx(1.0)
        assert scores.shape == (4,) and scores.dtype == np.float64 and np.all(scores >= 0.0)
        assert scores.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(scores, softmax(logits[2:]))


def test_readout_accepts_rounding_outside_zero_one(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    window = blob()
    window[0, 0, 0], window[0, 0, 1] = 1.0 + 5e-7, -5e-7
    runtime.readout(window)


def bad_windows() -> list:
    nan, high, low = dark(), dark(), dark()
    nan[1, 3, 3] = np.nan
    high[0, 0, 0] = 1.01
    low[1, 0, 0] = -0.01
    inf = dark()
    inf[0, 5, 5] = np.inf
    return [nan, inf, high, low, np.zeros((8, N, N), np.float32), np.zeros((FRAMES, N, N + 1), np.float32)]


@pytest.mark.parametrize("window", bad_windows(), ids=["nan", "inf", "above", "below", "frames", "square"])
def test_readout_refuses_a_bad_window(config, student_dir, window) -> None:
    with pytest.raises(ValueError):
        DodgeRuntime.load(student_dir, config).readout(window)


# --- infer ---


def test_infer_returns_exactly_the_documented_plain_values(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    for frame, window in enumerate([dark(), blob(), blob(), blob(), bad_windows()[0]]):
        out = runtime.infer(window, t(frame))
        assert set(out) == KEYS and set(out["gate"]) == GATE_KEYS
        assert set(out["command"]) == {"vel_x", "vel_y", "yaw_rate"}
        assert_plain(out)
        assert json.loads(json.dumps(out, allow_nan=False)) == out
        assert out["gate"]["votes_needed"] == config["inference"]["gate"]["votes_needed"]
        assert out["gate"]["of_last"] == config["inference"]["gate"]["of_last"]
        assert out["inference_ms"] >= 0.0


def test_three_threat_windows_issue_then_hold_then_no_op_then_cooldown_refusal(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    s = runtime.settings
    assert s.votes_needed == 3  # the frame counts below follow config inference.gate
    left = command_of(config, "DODGE_LEFT")

    outs = [runtime.infer(blob(), t(frame)) for frame in range(3)]
    assert [o["action"] for o in outs] == [UNCERTAIN, UNCERTAIN, "DODGE_LEFT"]
    assert [o["issued"] for o in outs] == [False, False, True]
    assert [o["gate"]["votes"] for o in outs] == [1, 2, 3]
    assert [o["command"] for o in outs] == [ZERO, ZERO, left]
    assert outs[2]["hold_remaining_s"] == pytest.approx(s.hold_s)
    assert outs[2]["cooldown_remaining_s"] == pytest.approx(s.cooldown_s)
    t_issue = t(2)

    frame, held, released = 3, 0, 0
    while t(frame) < t_issue + s.hold_s + 2.0 / FPS:  # quiet windows through the end of the hold
        out = runtime.infer(dark(), t(frame))
        assert not out["issued"] and out["valid"]
        if t(frame) < t_issue + s.hold_s:
            held += 1
            assert out["action"] == "DODGE_LEFT" and out["command"] == left and out["hold_remaining_s"] > 0.0
        else:
            released += 1
            assert out["action"] == "NO_OP" and out["command"] == ZERO and out["hold_remaining_s"] == 0.0
        frame += 1
    assert held > 0 and released > 0

    conflicts = 0
    second = None
    for k in range(frame, frame + 60):  # threat windows again, during and after the cooldown
        out = runtime.infer(blob(), t(k))
        if t(k) < t_issue + s.cooldown_s:
            assert not out["issued"]
            if out["gate"]["passed"]:
                conflicts += 1
                assert out["cooldown_conflict"] and out["action"] == UNCERTAIN and out["command"] == ZERO
        elif out["issued"]:
            second = k
            assert out["action"] == "DODGE_LEFT" and out["command"] == left and not out["cooldown_conflict"]
            break
    assert conflicts > 0
    assert second is not None and t(second) - t_issue <= s.cooldown_s + 1.0 / FPS + 1e-9


def test_quiet_windows_stay_no_op_forever(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    rng = np.random.default_rng(1)
    for frame in range(300):
        window = dark() if frame % 2 else (0.1 * rng.random((FRAMES, N, N))).astype(np.float32)
        out = runtime.infer(window, t(frame))
        assert out["valid"] and out["action"] == "NO_OP" and out["command"] == ZERO
        assert not out["issued"] and not out["gate"]["passed"] and out["gate"]["votes"] == 0
        assert out["threat_confidence"] < config["inference"]["gate"]["uncertain_floor"]


@pytest.mark.parametrize(
    "window",
    [*bad_windows(), None, [[0.0] * N] * N, np.zeros((FRAMES, N, N), dtype=np.int64)],
    ids=["nan", "inf", "above", "below", "frames", "square", "none", "list", "int"],
)
def test_a_bad_window_is_uncertain_and_never_raises(config, student_dir, window) -> None:
    out = DodgeRuntime.load(student_dir, config).infer(window, 0.0)
    assert not out["valid"] and out["action"] == UNCERTAIN and out["command"] == ZERO
    assert out["threat_confidence"] == 0.0 and out["gate"]["votes"] == 0 and not out["issued"]
    json.dumps(out, allow_nan=False)


def test_a_bad_window_during_a_hold_keeps_the_held_command(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    assert [runtime.infer(blob(), t(frame))["issued"] for frame in range(3)] == [False, False, True]
    out = runtime.infer(bad_windows()[0], t(3))
    assert not out["valid"] and out["action"] == "DODGE_LEFT" and out["command"] == command_of(config, "DODGE_LEFT")


def test_reset_starts_a_new_session(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    for frame in range(3):
        runtime.infer(blob(), t(100 + frame))
    with pytest.raises(ValueError):
        runtime.infer(blob(), 0.0)  # time went backwards without a reset
    runtime.reset()
    out = runtime.infer(blob(), 0.0)
    assert out["gate"]["votes"] == 1 and out["hold_remaining_s"] == 0.0 and out["cooldown_remaining_s"] == 0.0


# --- module level and latency ---


def test_module_level_load_infer_reset(student_dir, monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "_current", None)
    with pytest.raises(RuntimeError, match="load"):
        runtime_module.infer(dark(), 0.0)
    with pytest.raises(RuntimeError, match="load"):
        runtime_module.reset()
    loaded = runtime_module.load(student_dir)  # config None: load_config()
    assert isinstance(loaded, DodgeRuntime) and runtime_module._current is loaded
    outs = [runtime_module.infer(blob(), t(frame)) for frame in range(3)]
    assert outs[2]["issued"] and outs[2]["action"] == "DODGE_LEFT"
    runtime_module.reset()
    out = runtime_module.infer(blob(), 0.0)
    assert out["gate"]["votes"] == 1 and not out["issued"] and out["cooldown_remaining_s"] == 0.0
    trial = {"threat": 0.5, "action": 0.4, "margin": 0.1}
    assert runtime_module.load(student_dir, thresholds=trial).thresholds == trial


def test_measure_latency(config, student_dir) -> None:
    runtime = DodgeRuntime.load(student_dir, config)
    result = measure_latency(runtime, n=200, seed=0)
    assert set(result) == {"n", "median_ms", "p95_ms", "max_ms"} and result["n"] == 200
    assert 0.0 < result["median_ms"] <= result["p95_ms"] <= result["max_ms"] < 1000.0
    out = runtime.infer(dark(), 0.0)  # reset afterwards: time 0 is accepted again, nothing is held
    assert out["action"] == "NO_OP" and out["cooldown_remaining_s"] == 0.0
    for bad in (0, True, 2.0):
        with pytest.raises((TypeError, ValueError)):
            measure_latency(runtime, n=bad)
