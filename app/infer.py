"""Live inference: POST /api/infer.

    POST /api/infer  {window: [frames x height x width numbers in 0..1, oldest frame first], frames, captured_ms,
                     session_start?} -> Dev B's infer() result unchanged, plus latency_ms (measured here, request in to
                     answer out), frame_age_ms, frame_age_clamped, thresholds ("frozen" or "development") and,
                     for a calibrated student, gate_passed; 503 infer_unavailable naming what is missing

captured_ms is the browser's wall-clock time of the newest frame, in epoch milliseconds. Its age is this server's
wall-clock receive time minus captured_ms; an age outside 0..app.infer.max_frame_age_ms is clock skew, taken as 0 and
flagged. t_capture_s for Dev B is the monotonic receive time minus that age. session_start true calls Dev B's reset()
first (a new camera session). The command Dev B returns is checked against the verified command schema and applied
to the stage on the simulation thread; an SSE `infer` event carries the action, confidences and thresholds only.

The window is derived from the laptop camera: it is never logged, stored, put in an event, an error message or a
traceback. Errors are logged by type only.
"""
import logging
import math
import time

import numpy as np

from app import backend
from app.routes import Refusal, capability, route
from learned_judgment.env import CommandError, validate_command

log = logging.getLogger(__name__)


def check_body(body: dict, window: dict):
    """(float32 (frames, height, width) array, captured_ms, session_start) from a request body; Refusal 400
    invalid_window otherwise, whose message repeats none of the values."""
    f, height, width = window["frames"], window["height"], window["width"]
    values = body.get("window")
    ok = (set(body) - {"session_start"} == {"window", "frames", "captured_ms"}
          and type(body["frames"]) is int and body["frames"] == f
          and type(body["captured_ms"]) in (int, float) and math.isfinite(body["captured_ms"])
          and type(body.get("session_start", False)) is bool
          and type(values) is list and len(values) == f * height * width
          and all(type(v) in (int, float) for v in values))
    if ok:
        array = np.asarray(values, np.float64)
        ok = bool(np.isfinite(array).all() and array.min() >= 0.0 and array.max() <= 1.0)
    if not ok:
        raise Refusal(400, "invalid_window", f"An inference request is a window of {f} x {height} x {width} = "
                                             f"{f * height * width} numbers from 0 to 1, oldest frame first, frames "
                                             f"{f}, captured_ms and an optional session_start.")
    return array.astype(np.float32).reshape(f, height, width), body["captured_ms"], body.get("session_start", False)


def frame_age(captured_ms: float, received_wall_s: float, max_age_ms: float):
    """(age in s, clamped): received_wall_s x 1000 - captured_ms, or 0 and True when outside 0..max_age_ms."""
    age_ms = received_wall_s * 1000.0 - captured_ms
    return (age_ms / 1000.0, False) if 0.0 <= age_ms <= max_age_ms else (0.0, True)


def run(body: dict, config: dict, infer_fn, reset_fn, window: dict, received_mono: float, received_wall: float) -> dict:
    """One inference: check the body, reset on a session start, call Dev B with the capture time; Dev B's result plus
    frame_age_ms and frame_age_clamped. Refusal 502 infer_failed if Dev B fails, logged by error type only."""
    frames, captured_ms, session_start = check_body(body, window)
    age_s, clamped = frame_age(captured_ms, received_wall, config["app"]["infer"]["max_frame_age_ms"])
    try:
        if session_start:
            reset_fn()
        result = infer_fn(frames, received_mono - age_s)
    except Exception as e:  # its message could repeat the window: only the type is kept
        log.warning("Inference failed: %s", type(e).__name__)
        raise Refusal(502, "infer_failed", "The inference runtime failed on this window; the server log names the "
                                           "error type.") from None
    if not isinstance(result, dict):
        log.warning("Inference returned a %s, not a dict", type(result).__name__)
        raise Refusal(502, "infer_failed", "The inference runtime returned no result.")
    return {**result, "frame_age_ms": age_s * 1000.0, "frame_age_clamped": clamped}


def event(result: dict) -> dict:
    """The SSE `infer` event: the action, the confidences and which thresholds gave them ("frozen" or
    "development"), so development thresholds are labelled wherever the action shows."""
    return {key: result.get(key) for key in ("action", "threat_confidence", "action_confidence", "thresholds")}


@capability("infer")
def infer_capability(server) -> dict:
    missing = backend.inference(server.config)[3]
    if missing:
        return {"available": False, "reason": f"Live inference needs {'; and '.join(missing)}."}
    notes = [backend.DEV_THRESHOLDS] if backend.thresholds_mode() == "development" else []
    gate = backend.calibration_gate_passed()
    if gate is not None:
        notes.append(f"The student's calibration {'passed' if gate else 'did not pass'} its precision gate.")
    return {"available": True, "reason": " ".join(notes)}


@route("POST", "/api/infer")
def post_infer(h) -> None:
    received_mono, received_wall = time.monotonic(), time.time()
    infer_fn, reset_fn, window, missing = backend.inference(h.server.config)
    if missing:
        raise Refusal(503, "infer_unavailable", f"Live inference needs {'; and '.join(missing)}.")
    from app.duel import stop_for_stage
    stop_for_stage(h.server)
    h.server.player.stop()  # a live-camera session takes the stage back from the episode player
    try:
        result = run(h.read_body(), h.server.config, infer_fn, reset_fn, window, received_mono, received_wall)
    except Refusal:
        raise
    except Exception as e:  # no traceback: its text could carry window values
        log.warning("An inference request failed: %s", type(e).__name__)
        raise Refusal(500, "internal_error", "Something went wrong in the server; its log has the details.") from None
    schema = h.server.sim.schema
    command = result.get("command")
    if command is not None:
        try:
            validate_command(command, schema)
        except CommandError:
            log.warning("The inference runtime returned a command outside the verified command schema")
            raise Refusal(502, "infer_failed", "The inference runtime returned a command outside the verified "
                                               "range, so the stage keeps its command.") from None
        h.submit("drive", {name: float(command[name]) for name in schema["fields"]})
    h.server.sim.announce("infer", event(result), latest_only=True)
    h.send_json(200, {**result, "latency_ms": (time.monotonic() - received_mono) * 1000.0})
