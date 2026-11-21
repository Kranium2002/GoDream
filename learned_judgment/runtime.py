"""The frozen DODGE student behind the threat gate: one camera window in, one command out (numpy).

Lane contract (Person B, DODGE from the head camera)
----------------------------------------------------
:class:`DodgeRuntime` loads an exported student directory and turns each image window into the
command to send now. It runs on this computer only: it makes no network call, writes nothing and
keeps no frames or features (the gate keeps only its vote window of action names).

Student directory (written by the observed run's export):

    network.npz, network.json   the spiking network (learned_judgment.students.snn_core.save_network),
                                outputs [threat (2) | action (C)]; its digest is checked on load
    student.json                {"format": "learned-judgment/observed-student/v1", "skill",
                                "action_names" (C), "heads": {"threat": [0, 2], "action": [2, 2 + C]},
                                "no_op_index", "input": {"frames", "frame_size", "encoding"}, "vision",
                                "thresholds": {"threat", "action", "margin"} or null, "provenance"}

Per window (F = input.frames and N = input.frame_size, both from student.json, never config.yaml):

    readout   window (F, N, N) float32 in [0, 1], oldest frame first
              -> x = (flatten_window(window) - 0.5) * sqrt(12) in float64, D = F * N * N
              -> one SpikingNetwork forward -> logits (2 + C,)
              -> threat_confidence = softmax(logits[0:2])[1]; action_scores = softmax(logits[2:2 + C])
    gate      learned_judgment.gate.ThreatGate with GateSettings.from_config(config, thresholds)
    command   config.yaml skills.dodge_vision.actions[action] of the emitted action; NO_OP's (zero)
              command for NO_OP and UNCERTAIN

The C actions are the keys of skills.dodge_vision.actions, in config order. Thresholds are the
student's frozen calibration; an explicit ``thresholds`` argument overrides them (calibration only).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from learned_judgment.config import ConfigError, load_config, require
from learned_judgment.gate import UNCERTAIN, GateSettings, ThreatGate
from learned_judgment.students.snn_core import SpikingNetwork, load_network
from learned_judgment.vision import flatten_window

__all__ = ["COMMAND_FIELDS", "ENCODING", "STUDENT_FORMAT", "DodgeRuntime", "infer", "load", "measure_latency", "reset"]

STUDENT_FORMAT = "learned-judgment/observed-student/v1"
ENCODING = "(x - 0.5) * sqrt(12)"
COMMAND_FIELDS = ("vel_x", "vel_y", "yaw_rate")

# A window value may lie this far outside [0, 1] (float32 rounding of the browser's frames).
_RANGE_TOLERANCE = 1e-6
_SCALE = math.sqrt(12.0)
_NO_THRESHOLDS = "the student has no frozen thresholds: calibrate on simulation dev trajectories first"
_NO_RUNTIME = "no runtime is loaded: call learned_judgment.runtime.load(path) first"


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"student.json {name} must be an integer >= 1; got {value!r}")
    return value


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - np.max(logits))
    return shifted / shifted.sum()


class DodgeRuntime:
    """One loaded student and its gate over one camera stream. Build it with :meth:`load`."""

    def __init__(
        self,
        network: SpikingNetwork,
        settings: GateSettings,
        commands: Mapping[str, Mapping[str, float]],
        frames: int,
        frame_size: int,
    ) -> None:
        if network.n_inputs != frames * frame_size * frame_size:
            raise ValueError(f"the network takes {network.n_inputs} inputs, not frames * frame_size**2")
        if network.n_outputs != 2 + len(settings.action_names):
            raise ValueError(f"the network has {network.n_outputs} outputs, not 2 + {len(settings.action_names)}")
        self._network = network
        self._settings = settings
        self._commands = {name: dict(commands[name]) for name in settings.action_names}
        self._frames = frames
        self._frame_size = frame_size
        self._gate = ThreatGate(settings)

    @classmethod
    def load(
        cls, path: str | Path, config: Mapping[str, Any], thresholds: Mapping[str, float] | None = None
    ) -> DodgeRuntime:
        """Load the exported student in directory ``path``.

        Args:
            path:       the student directory (student.json, network.npz, network.json).
            config:     the loaded config.yaml.
            thresholds: {"threat", "action", "margin"} overriding the student's frozen ones; for
                        threshold calibration only.

        Checks: the format string; the network digest (load_network); action_names equal to the
        keys of skills.dodge_vision.actions in order; heads exactly {"threat": [0, 2], "action":
        [2, 2 + C]} and 2 + C network outputs; input frames and frame_size integers >= 1 with
        frames * frame_size**2 network inputs, and the encoding string; no_op_index the index of
        the zero command; every command exactly the fields vel_x, vel_y, yaw_rate.

        Raises:
            FileNotFoundError for a missing file. ValueError for a file that breaks the contract
            (including a digest mismatch). RuntimeError when neither the file nor ``thresholds``
            has thresholds. ConfigError, TypeError, ValueError from GateSettings.from_config.
        """
        directory = Path(path)
        student_path = directory / "student.json"
        with open(student_path, "r", encoding="utf-8") as handle:
            student = json.load(handle)
        if not isinstance(student, dict) or student.get("format") != STUDENT_FORMAT:
            raise ValueError(f"{student_path} is not a {STUDENT_FORMAT} file")
        network, _ = load_network(directory / "network")

        actions = require(config, "skills.dodge_vision.actions")
        if not isinstance(actions, Mapping):
            raise ConfigError(f"config.yaml key 'skills.dodge_vision.actions' must be a mapping; got {actions!r}")
        names = list(actions)
        if student.get("action_names") != names:
            raise ValueError(
                f"{student_path} action_names {student.get('action_names')!r} differ from the keys of "
                f"config.yaml skills.dodge_vision.actions {names!r}"
            )
        heads = {"threat": [0, 2], "action": [2, 2 + len(names)]}
        if student.get("heads") != heads:
            raise ValueError(f"{student_path} heads must be {heads}; got {student.get('heads')!r}")
        if network.n_outputs != 2 + len(names):
            raise ValueError(f"the network has {network.n_outputs} outputs; the heads need {2 + len(names)}")

        spec = student.get("input")
        if not isinstance(spec, Mapping):
            raise ValueError(f"{student_path} input must be an object; got {spec!r}")
        frames = _count(spec.get("frames"), "input.frames")
        frame_size = _count(spec.get("frame_size"), "input.frame_size")
        if spec.get("encoding") != ENCODING:
            raise ValueError(f"{student_path} input.encoding must be {ENCODING!r}; got {spec.get('encoding')!r}")
        if network.n_inputs != frames * frame_size * frame_size:
            raise ValueError(
                f"the network takes {network.n_inputs} inputs; input.frames {frames} x frame_size "
                f"{frame_size}**2 is {frames * frame_size * frame_size}"
            )

        chosen = thresholds if thresholds is not None else student.get("thresholds")
        if chosen is None:
            raise RuntimeError(_NO_THRESHOLDS)
        settings = GateSettings.from_config(config, chosen)
        no_op_index = student.get("no_op_index")
        if isinstance(no_op_index, bool) or no_op_index != names.index(settings.no_op):
            raise ValueError(
                f"{student_path} no_op_index must be {names.index(settings.no_op)} ({settings.no_op}); got {no_op_index!r}"
            )

        commands: dict[str, dict[str, float]] = {}
        for name in names:
            command = actions[name]
            if set(command) != set(COMMAND_FIELDS):
                raise ConfigError(
                    f"config.yaml key 'skills.dodge_vision.actions.{name}' must have exactly the fields "
                    f"{COMMAND_FIELDS}; got {sorted(command)}"
                )
            commands[name] = {field: float(command[field]) for field in COMMAND_FIELDS}
        return cls(network, settings, commands, frames, frame_size)

    # ----- properties ----------------------------------------------------------------------

    @property
    def frames(self) -> int:
        """F: frames per window (student.json input.frames)."""
        return self._frames

    @property
    def frame_size(self) -> int:
        """N: a frame is N x N (student.json input.frame_size)."""
        return self._frame_size

    @property
    def action_names(self) -> tuple[str, ...]:
        """The C actions in readout order."""
        return self._settings.action_names

    @property
    def thresholds(self) -> dict[str, float]:
        """The thresholds in use: {"threat", "action", "margin"} (a copy)."""
        s = self._settings
        return {"threat": s.threat_threshold, "action": s.action_threshold, "margin": s.margin_threshold}

    @property
    def settings(self) -> GateSettings:
        """The gate's settings."""
        return self._settings

    # ----- inference -----------------------------------------------------------------------

    def readout(self, window: np.ndarray) -> tuple[float, np.ndarray]:
        """The student's readout for one window.

        Args:
            window: (frames, frame_size, frame_size) float array (float32 by contract), grey
                    levels in [0, 1] (up to 1e-6 outside is accepted), oldest frame first.

        Returns:
            (threat_confidence, action_scores): softmax(threat logits)[1] as a float in [0, 1],
            and softmax(action logits) as float64 (C,) summing to 1, in action_names order.

        Raises:
            TypeError when ``window`` is not a float numpy array; ValueError for a wrong shape,
            a non-finite value, or a value outside [0, 1] by more than 1e-6.
        """
        if not isinstance(window, np.ndarray):
            raise TypeError(f"window must be a numpy array; got {type(window).__name__}")
        shape = (self._frames, self._frame_size, self._frame_size)
        if window.shape != shape:
            raise ValueError(f"window must have shape {shape}; got {window.shape}")
        if window.dtype.kind != "f":
            raise TypeError(f"window must be a float array (float32); got dtype {window.dtype}")
        flat = flatten_window(window)
        if not np.all(np.isfinite(flat)):
            raise ValueError("window contains non-finite values")
        low, high = float(flat.min()), float(flat.max())
        if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
            raise ValueError(f"window values must lie in [0, 1]; got min {low}, max {high}")
        encoded = (flat.astype(np.float64) - 0.5) * _SCALE
        logits = self._network.forward(encoded[None, :])[0]
        threat = _softmax(logits[0:2])
        scores = _softmax(logits[2:])
        return float(threat[1]), scores

    def infer(self, window: Any, t_capture_s: float) -> dict[str, Any]:
        """Readout and gate for one window captured at ``t_capture_s`` seconds.

        A window that :meth:`readout` refuses does not raise: a NaN readout goes through the
        gate, so the output has valid False and the action is UNCERTAIN (or the held action).

        Returns a JSON-serialisable dict of plain Python values:
            action               the gate's emitted action: an action name or UNCERTAIN
            command              {"vel_x", "vel_y", "yaw_rate"}: that action's command from config
                                 (NO_OP's zero command for NO_OP and UNCERTAIN)
            threat_confidence, best_action, action_confidence, margin   as GateOutput
            gate                 {"threat_ok", "action_ok", "margin_ok", "votes", "votes_needed",
                                 "of_last", "passed"}
            cooldown_remaining_s, hold_remaining_s, issued, cooldown_conflict, valid   as GateOutput
            inference_ms         wall time of readout + gate (time.perf_counter)

        Raises:
            ValueError when t_capture_s is not finite or is earlier than the previous call's
            (call :meth:`reset` for a new camera session).
        """
        start = time.perf_counter()
        try:
            threat, scores = self.readout(window)
        except (TypeError, ValueError):
            threat, scores = math.nan, np.full(len(self._settings.action_names), math.nan)
        out = self._gate.update(threat, scores, t_capture_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        s = self._settings
        command = self._commands[s.no_op if out.action == UNCERTAIN else out.action]
        return {
            "action": str(out.action),
            "command": {field: float(value) for field, value in command.items()},
            "threat_confidence": float(out.threat_confidence),
            "best_action": str(out.best_action),
            "action_confidence": float(out.action_confidence),
            "margin": float(out.margin),
            "gate": {
                "threat_ok": bool(out.threat_ok),
                "action_ok": bool(out.action_ok),
                "margin_ok": bool(out.margin_ok),
                "votes": int(out.votes),
                "votes_needed": int(s.votes_needed),
                "of_last": int(s.of_last),
                "passed": bool(out.passed),
            },
            "cooldown_remaining_s": float(out.cooldown_remaining_s),
            "hold_remaining_s": float(out.hold_remaining_s),
            "issued": bool(out.issued),
            "cooldown_conflict": bool(out.cooldown_conflict),
            "valid": bool(out.valid),
            "inference_ms": float(elapsed_ms),
        }

    def reset(self) -> None:
        """Reset the gate (vote window, hold, cooldown, last time): call when a new camera session starts."""
        self._gate.reset()


# ----- module-level runtime for the server -----------------------------------------------------

_current: DodgeRuntime | None = None


def load(
    path: str | Path, config: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None
) -> DodgeRuntime:
    """:meth:`DodgeRuntime.load` (config None: load_config()), kept as the module's current runtime."""
    global _current
    runtime = DodgeRuntime.load(path, load_config() if config is None else config, thresholds)
    _current = runtime
    return runtime


def infer(window: Any, t_capture_s: float) -> dict[str, Any]:
    """:meth:`DodgeRuntime.infer` of the current runtime. RuntimeError if :func:`load` was not called."""
    if _current is None:
        raise RuntimeError(_NO_RUNTIME)
    return _current.infer(window, t_capture_s)


def reset() -> None:
    """:meth:`DodgeRuntime.reset` of the current runtime. RuntimeError if :func:`load` was not called."""
    if _current is None:
        raise RuntimeError(_NO_RUNTIME)
    _current.reset()


def measure_latency(runtime: DodgeRuntime, n: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Wall time of :meth:`DodgeRuntime.infer` (readout + gate) over ``n`` random valid windows.

    Window k is np.random.default_rng(seed).random((frames, N, N), dtype=float32), drawn outside
    the timed span, and is inferred at t = k seconds. The gate is reset before and after.

    Returns:
        {"n", "median_ms", "p95_ms", "max_ms"} (time.perf_counter, milliseconds).

    Raises:
        TypeError for a runtime that is not a DodgeRuntime or a non-integer n or seed;
        ValueError for n < 1 or seed < 0.
    """
    if not isinstance(runtime, DodgeRuntime):
        raise TypeError(f"runtime must be a DodgeRuntime; got {type(runtime).__name__}")
    for name, value, minimum in (("n", n, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int; got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}; got {value}")
    rng = np.random.default_rng(seed)
    shape = (runtime.frames, runtime.frame_size, runtime.frame_size)
    times_ms = np.empty(n, dtype=np.float64)
    runtime.reset()
    try:
        for k in range(n):
            window = rng.random(shape, dtype=np.float32)
            start = time.perf_counter()
            out = runtime.infer(window, float(k))
            times_ms[k] = (time.perf_counter() - start) * 1000.0
            if not out["valid"]:
                raise RuntimeError(f"random window {k} was not a valid readout")
    finally:
        runtime.reset()
    return {
        "n": n,
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(times_ms.max()),
    }
