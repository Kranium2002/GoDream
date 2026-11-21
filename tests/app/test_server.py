"""The viewer server (app/server.py) on the real simulator and walking policy, started on a free port by the same
start() the command line uses and driven over HTTP. Waits are in episode time read from the observations, so a slow
or busy machine only makes the tests take longer."""
import copy
import http.client
import json
import logging
import re
import socket
import struct
import threading
import time
import zlib
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import pytest

from app import backend
from app.server import STATIC, start, stop
from learned_judgment.env import load_config

FORWARD = {"vel_x": 0.6, "vel_y": 0.0, "yaw_rate": 0.0}  # the Gate 1 trot speed


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def server(config):
    server = start(config, 0)
    yield server
    stop(server)


def connect(server):
    return http.client.HTTPConnection(*server.server_address[:2], timeout=120)


def request(server, method, path, body=None):
    with closing(connect(server)) as conn:
        conn.request(method, path, body=None if body is None else json.dumps(body))
        response = conn.getresponse()
        assert response.getheader("Content-Type") == "application/json"
        return response.status, json.loads(response.read())


@contextmanager
def event_stream(server):
    """(event name, data) pairs from /api/events as they arrive."""
    with closing(connect(server)) as conn:
        conn.request("GET", "/api/events")
        response = conn.getresponse()
        assert response.status == 200 and response.getheader("Content-Type") == "text/event-stream"

        def events():
            name = None
            for line in response:
                line = line.decode().rstrip("\n")
                if line.startswith("event: "):
                    name = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    yield name, json.loads(line.removeprefix("data: "))
        yield events()


def observe(events, until_s):
    """The observations up to the first one at episode time until_s or later."""
    seen = []
    for name, data in events:
        if name == "observation":
            seen.append(data)
            if data["t_s"] >= until_s:
                return seen


def next_episode_event(events, phase):
    return next(data for name, data in events if name == "episode" and data["phase"] == phase)


def fresh_episode(server, events):
    assert request(server, "POST", "/api/reset", {"preset": "default"}) == (200, {"preset": "default"})
    next_episode_event(events, "start")


def mean_forward(observations):
    return np.mean([o["forward_speed_m_s"] for o in observations])


def assert_plain_sentence(message):
    assert message.endswith(".") and message.count(".") == 1  # one sentence
    assert "\\" not in message and "Traceback" not in message
    assert not re.search(r"\w/\w", message)  # no slash-separated path
    assert not re.search(r"[A-Z]\w*(Error|Exception)\b", message)  # no exception class name


def frame_parts(response):
    """The PNG bodies of a multipart/x-mixed-replace response, as they arrive."""
    while line := response.readline():
        assert line == b"--frame\r\n"
        headers = {}
        while (line := response.readline()) != b"\r\n":
            key, value = line.decode().split(":", 1)
            headers[key.lower()] = value.strip()
        assert headers["content-type"] == "image/png"
        yield response.read(int(headers["content-length"]))
        assert response.read(2) == b"\r\n"


def decode_png(png):
    """H x W x 3 uint8 from an 8-bit RGB PNG whose rows all use filter type 0, checking every chunk's CRC."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, pos = {}, 8
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos:pos + 8])
        data = png[pos + 8:pos + 8 + length]
        assert struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])[0] == zlib.crc32(kind + data)
        chunks.setdefault(kind, []).append(data)
        pos += 12 + length
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", chunks[b"IHDR"][0])
    assert (depth, colour, interlace) == (8, 2, 0) and b"IEND" in chunks
    rows = np.frombuffer(zlib.decompress(b"".join(chunks[b"IDAT"])), np.uint8).reshape(height, 1 + 3 * width)
    assert not rows[:, 0].any()  # filter type 0 on every row: the bytes after it are the pixels
    return rows[:, 1:].reshape(height, width, 3)


def handler_threads():
    return [t for t in threading.enumerate() if t.name.endswith("(process_request_thread)")]


def wait_until(predicate, timeout_s=30.0):
    end = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.05)


CAPABILITY_NAMES = ["drive", "head_camera", "scenarios", "chat", "runs", "results", "infer"]  # HTTP API v0


def test_state_describes_the_controls_from_config(server, config):
    status, state = request(server, "GET", "/api/state")
    assert status == 200
    fields = config["robot"]["walking_policy"]["command_schema"]["fields"]
    assert state["commands"] == {name: {"unit": f["unit"], "min": f["min"], "max": f["max"]}
                                 for name, f in fields.items()}
    assert list(state["commands"]) == list(fields)
    assert state["presets"] == list(config["world"]["presets"])
    assert state["preset"] in state["available_presets"]
    assert state["status"] in ("roaming", "resetting", "fallen")
    assert state["frame"] == {"width": config["world"]["render"]["width"],
                              "height": config["world"]["render"]["height"],
                              "stream_fps": config["app"]["stream_fps"]}
    assert {"skills", "image_window", "actions"} <= set(state)


def test_state_names_every_capability_and_why_it_is_unavailable(server):
    capabilities = request(server, "GET", "/api/state")[1]["capabilities"]
    assert sorted(capabilities) == sorted(CAPABILITY_NAMES)
    for name, capability in capabilities.items():
        assert set(capability) == {"available", "reason"} and isinstance(capability["available"], bool), name
        assert isinstance(capability["reason"], str), name
        if not capability["available"]:  # one sentence; it may name a module or a file, so dots are allowed
            assert capability["reason"].endswith(".") and "Traceback" not in capability["reason"], name
    assert capabilities["drive"]["available"]
    for name in ["head_camera", "scenarios", "runs"]:  # not built in this server yet, or no Dev B runner
        assert not capabilities[name]["available"], name
    assert capabilities["results"]["available"] == (backend.runs_dir(server.config)[0] is not None)
    assert capabilities["infer"]["available"] == (backend.inference(server.config)[0] is not None)


SKILL_KEYS = {"skill", "description", "params", "actions", "outcomes", "available", "missing"}


def test_skills_lists_the_catalogue_and_what_each_unavailable_skill_lacks(server):
    status, body = request(server, "GET", "/api/skills")
    assert status == 200 and isinstance(body, list) and body  # the bare list, in skill_catalog's shape
    for entry in body:
        assert set(entry) >= SKILL_KEYS, entry
        assert entry["available"] or entry["missing"], entry  # never unavailable without saying why
    assert request(server, "GET", "/api/state")[1]["skills"] == body


def test_chat_without_its_pieces_is_unavailable_and_says_what_to_add(server, monkeypatch):
    from app import backend, chat
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    has_key = chat.api_key(server.config["app"]["chat"]) is not None  # a .env at the repository root
    has_backend = backend.declared()[0] is not None
    if has_key and has_backend:
        pytest.skip("the chat has everything it needs here; test_chat.py's live test covers it")
    status, body = request(server, "POST", "/api/chat", {"messages": [{"role": "user", "content": "hello"}]})
    assert status == 503 and body["code"] == "chat_unavailable"
    if not has_key:
        assert "ANTHROPIC_API_KEY" in body["message"]
    if not has_backend:
        assert "learned_judgment.declared" in body["message"]
    capability = request(server, "GET", "/api/state")[1]["capabilities"]["chat"]
    assert capability == {"available": False, "reason": body["message"]}


def test_the_api_key_appears_in_no_response(server, monkeypatch):
    from app import backend
    sentinel = "sk-ant-test-SENTINEL-0123456789"  # not a real key: only to see whether it leaks anywhere
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    calls = [("GET", "/api/state", None), ("GET", "/api/skills", None), ("POST", "/api/chat", {"messages": "bad"})]
    if backend.declared()[0] is None:  # with Dev B's module this would send the fake key to the API
        calls.append(("POST", "/api/chat", {"messages": [{"role": "user", "content": "hello"}]}))
    for method, path, body in calls:
        with closing(connect(server)) as conn:
            conn.request(method, path, body=None if body is None else json.dumps(body))
            response = conn.getresponse()
            raw = response.read().decode()
            assert sentinel not in raw and sentinel not in str(response.getheaders()), path


def test_unknown_api_paths_are_refused_with_a_code(server):
    for method in ["GET", "POST"]:
        status, body = request(server, method, "/api/no_such_thing")
        assert status == 404 and body["code"] == "not_found"


def test_drive_moves_the_dog_and_stop_slows_it(server):
    zero = dict.fromkeys(FORWARD, 0.0)
    with event_stream(server) as events:
        fresh_episode(server, events)
        assert request(server, "POST", "/api/drive", FORWARD) == (200, {"command": FORWARD})
        driving = [o for o in observe(events, until_s=2.5) if o["t_s"] >= 1.5]
        assert all(o["command"] == FORWARD for o in driving)
        assert mean_forward(driving) > 0  # the commanded sign
        assert request(server, "POST", "/api/stop") == (200, {"command": zero})
        stopping = [o for o in observe(events, until_s=driving[-1]["t_s"] + 2.0)
                    if o["t_s"] >= driving[-1]["t_s"] + 1.5]
        assert all(o["command"] == zero for o in stopping)
        assert abs(mean_forward(stopping)) < mean_forward(driving)


@pytest.mark.parametrize("bad", [FORWARD | {"vel_x": 5.0}, {"vel_x": 0.6}, FORWARD | {"vel_y": "left"}],
                         ids=["out_of_range", "incomplete", "not_a_number"])
def test_invalid_drive_is_refused_and_the_dog_keeps_its_command(server, bad):
    with event_stream(server) as events:
        fresh_episode(server, events)
        assert request(server, "POST", "/api/drive", FORWARD)[0] == 200
        before = observe(events, until_s=1.5)[-1]["t_s"]
        status, body = request(server, "POST", "/api/drive", bad)
        assert status == 400 and body["code"] == "invalid_command"
        assert_plain_sentence(body["message"])
        after = [o for o in observe(events, until_s=before + 1.0) if o["t_s"] > before]
        assert all(o["command"] == FORWARD for o in after)
        assert mean_forward(after) > 0


def test_stream_sends_real_png_frames_of_the_configured_size(server, config):
    size = config["world"]["render"]
    with event_stream(server) as events:
        fresh_episode(server, events)
        request(server, "POST", "/api/drive", FORWARD)
        observe(events, until_s=1.0)  # walking
    with closing(connect(server)) as conn:
        conn.request("GET", "/api/stream")
        response = conn.getresponse()
        assert response.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=frame"
        parts = frame_parts(response)
        frames = [decode_png(next(parts)) for _ in range(3)]
    for frame in frames:
        assert frame.shape == (size["height"], size["width"], 3)
        assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1  # not a single flat colour
    assert not np.array_equal(frames[0], frames[-1])


def test_reset_refuses_an_unknown_preset_and_starts_a_default_episode(server, config):
    assert "no_such_world" not in config["world"]["presets"]
    status, body = request(server, "POST", "/api/reset", {"preset": "no_such_world"})
    assert status == 400 and body["code"] == "unknown_preset"
    assert_plain_sentence(body["message"])
    with event_stream(server) as events:
        fresh_episode(server, events)  # roaming, so the next reset ends an episode
        assert request(server, "POST", "/api/reset", {"preset": "default"}) == (200, {"preset": "default"})
        end = next_episode_event(events, "end")
        assert "fall" in end["outcome"] and "commands_received" not in end["outcome"]
        assert "trace_path" not in end["outcome"]  # a server-side file path is none of the browser's business
        assert next_episode_event(events, "start")["preset"] == "default"


def test_static_files_come_only_from_the_static_directory(server):
    with closing(connect(server)) as conn:
        conn.request("GET", "/.gitkeep")
        assert conn.getresponse().status == 200
    page = 200 if (STATIC / "index.html").is_file() else 404
    # app/server.py, config.yaml and win.ini exist: a path that escaped the static directory would find them
    for path, expected in [("/", page), ("/no_such_page.html", 404), ("/../server.py", 404),
                           ("/%2e%2e/server.py", 404), ("/..%5cserver.py", 404), ("/../../config.yaml", 404),
                           ("/%2e%2e%2f%2e%2e%2fconfig.yaml", 404), ("/C:/Windows/win.ini", 404)]:
        with closing(connect(server)) as conn:
            conn.request("GET", path)
            assert conn.getresponse().status == expected, path


@pytest.mark.parametrize("key, value", [("stream_fps", 15), ("observation_hz", 7), ("host", "0.0.0.0")])
def test_server_refuses_to_start_with_an_unusable_app_setting(config, key, value):
    bad = copy.deepcopy(config)
    bad["app"][key] = value
    with pytest.raises(ValueError, match=key):
        start(bad, 0)


@pytest.fixture(scope="module")
def own_server(config, tmp_path_factory):
    """A server of its own, whose config adds a preset that cannot be built (a floor type with no friction) and one
    whose command latency (1 s, presets are not range-checked) is far longer than an observation period."""
    own = copy.deepcopy(config)
    presets = own["world"]["presets"]
    assert own["world"]["hidden_effects"]["enabled"]["latency"]
    presets["unknown_floor"] = presets["default"] | {"floor_type": "no_such_floor"}
    presets["slow_commands"] = presets["default"] | {"command_latency_s": 1.0}
    own["observed"]["runs_dir"] = str(tmp_path_factory.mktemp("runs"))  # absolute: its own, empty runs directory
    server = start(own, 0)
    yield server
    stop(server)  # again after the shutdown test: stop is idempotent


def test_a_preset_that_cannot_be_built_is_offered_as_unavailable(own_server):
    state = request(own_server, "GET", "/api/state")[1]
    assert "unknown_floor" in state["presets"] and "unknown_floor" not in state["available_presets"]
    status, body = request(own_server, "POST", "/api/reset", {"preset": "unknown_floor"})
    assert status == 400 and body["code"] == "preset_unavailable"
    assert_plain_sentence(body["message"])
    assert request(own_server, "GET", "/api/state")[1]["status"] == "roaming"  # the current world goes on


def test_observations_show_the_issued_command_at_once_not_the_delayed_one(own_server):
    """The streamed command is the caller's own; the one the policy receives a hidden latency later would let the
    browser measure that latency."""
    with event_stream(own_server) as events:
        assert request(own_server, "POST", "/api/reset", {"preset": "slow_commands"})[0] == 200
        next_episode_event(events, "start")
        before = observe(events, until_s=0.5)[-1]["t_s"]
        assert request(own_server, "POST", "/api/drive", FORWARD)[0] == 200
        seen = observe(events, until_s=before + 0.5)
        assert seen[-1]["command"] == FORWARD  # 1 s of latency would still show the zero command here


def test_new_run_lines_are_relayed_as_events_and_the_run_can_be_read(own_server):
    """A run folder written while the server runs (a fixture for this plumbing, not a learning run)."""
    root = Path(own_server.config["observed"]["runs_dir"])
    line = {"run_id": "r_test", "skill": "RAMP", "step": "baseline", "state": "done", "done": 4, "total": 4,
            "time_utc": "2026-11-21T22:00:00Z", "data": {"anything": [1, 2]}}
    with event_stream(own_server) as events:
        (root / "r_test").mkdir()
        (root / "r_test" / "events.jsonl").write_text(json.dumps(line) + "\n")
        assert next(data for name, data in events if name == "run") == line
    assert request(own_server, "GET", "/api/runs") == (200, ["r_test"])
    status, run = request(own_server, "GET", "/api/runs/r_test")
    assert status == 200 and run["state"] == "running" and run["baseline"] == {"anything": [1, 2]}
    assert run["steps"] == [{k: line[k] for k in ("step", "state", "done", "total", "data")}]


def test_shutdown_ends_open_streams_and_frees_the_port(own_server):
    port = own_server.server_address[1]
    wait_until(lambda: not handler_threads())  # earlier tests' connections are gone
    with closing(connect(own_server)) as gone:  # a client that leaves mid-stream
        gone.request("GET", "/api/stream")
        next(frame_parts(gone.getresponse()))
    wait_until(lambda: not handler_threads())  # its handler ended without waiting for shutdown
    stream, events = connect(own_server), connect(own_server)
    stream.request("GET", "/api/stream")
    frames = frame_parts(stream.getresponse())
    next(frames)
    events.request("GET", "/api/events")
    events_response = events.getresponse()
    events_response.readline()

    stop(own_server)
    assert not own_server.sim.is_alive()
    for _ in frames:  # the stream ends (a hang would hit the connection's timeout)
        pass
    events_response.read()  # and so does the event stream
    wait_until(lambda: not handler_threads())
    stream.close()
    events.close()
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=30)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # no SO_REUSEADDR: fails while anything still holds the port


# ---- stream cameras, runs and live inference ---------------------------------------------------------------------

def test_stream_refuses_an_unknown_camera_and_the_head_camera_until_it_exists(server):
    for camera, status, code in [("bogus", 400, "unknown_camera"), ("head", 503, "head_camera_unavailable")]:
        got, body = request(server, "GET", f"/api/stream?camera={camera}")
        assert (got, body["code"]) == (status, code)
        assert_plain_sentence(body["message"])


def test_runs_are_listed_unknown_ones_refused_and_starting_one_is_unavailable(server):
    status, found = request(server, "GET", "/api/runs")
    assert status == 200 and isinstance(found, list)
    for path in ["/api/runs/no_such_run", "/api/runs/..", "/api/runs/%2e%2e", "/api/runs/..%5c..%5cconfig.yaml",
                 "/api/runs/"]:
        assert request(server, "GET", path) == (404, {"code": "unknown_run", "message": "There is no run with that id."})
    status, body = request(server, "POST", "/api/runs", {"declared_spec": {"skill": "RAMP", "params": {}},
                                                         "confirm_cost": True})
    assert status == 503 and body["code"] == "runs_unavailable" and "entry point" in body["message"]


MARK = 0.123456789  # a window value easy to find if it ever leaks
SMALL = {"frames": 2, "height": 3, "width": 3}
STEER = {"vel_x": 0.0, "vel_y": 0.5, "yaw_rate": 0.0}


def infer_body(n_values, frames):
    return {"window": [MARK] * n_values, "frames": frames, "captured_ms": time.time() * 1000.0}


def no_leak(text):
    assert "0.1234" not in text


def test_infer_is_unavailable_naming_what_is_missing_and_repeats_no_value(server, caplog):
    if backend.inference(server.config)[0] is not None:
        pytest.skip("Dev B's runtime and a frozen model are present here")
    window = backend.image_window(server.config)
    with caplog.at_level(logging.DEBUG):
        status, body = request(server, "POST", "/api/infer",
                               infer_body(window["frames"] * window["height"] * window["width"], window["frames"]))
    assert status == 503 and body["code"] == "infer_unavailable"
    assert body["message"] == request(server, "GET", "/api/state")[1]["capabilities"]["infer"]["reason"]
    no_leak(json.dumps(body))
    no_leak(" ".join(r.getMessage() for r in caplog.records))


class StandInRuntime:
    """Dev B's runtime, to test this server's plumbing only; no evidence that live inference works."""

    def __init__(self, fail=False):
        self.fail, self.resets = fail, 0

    def infer(self, window, t_capture_s):
        if self.fail:
            raise ValueError(f"bad window {window.tolist()}")
        return {"action": "DODGE_LEFT", "command": STEER, "threat_confidence": np.float32(0.97),
                "action_confidence": 0.9, "margin": 0.4, "gate": {"passed": True}, "issued": True}

    def reset(self):
        self.resets += 1


def test_infer_applies_the_command_and_announces_the_action_only(server, monkeypatch, caplog):
    runtime = StandInRuntime()
    monkeypatch.setattr(backend, "inference", lambda config: (runtime.infer, runtime.reset, SMALL, []))
    with caplog.at_level(logging.DEBUG), event_stream(server) as events:
        fresh_episode(server, events)
        before = observe(events, until_s=0.5)[-1]["t_s"]
        status, body = request(server, "POST", "/api/infer", infer_body(18, 2) | {"session_start": True})
        assert status == 200 and body["action"] == "DODGE_LEFT" and body["command"] == STEER
        assert body["threat_confidence"] == pytest.approx(0.97) and body["latency_ms"] >= 0
        assert body["frame_age_clamped"] in (False, True) and runtime.resets == 1
        assert next(data for name, data in events if name == "infer") == {
            "action": "DODGE_LEFT", "threat_confidence": pytest.approx(0.97), "action_confidence": 0.9}
        assert observe(events, until_s=before + 0.5)[-1]["command"] == STEER  # applied to the stage
    no_leak(json.dumps(body))
    no_leak(" ".join(r.getMessage() for r in caplog.records))


def test_a_failing_inference_leaks_the_window_nowhere(server, monkeypatch, caplog):
    runtime = StandInRuntime(fail=True)
    monkeypatch.setattr(backend, "inference", lambda config: (runtime.infer, runtime.reset, SMALL, []))
    with caplog.at_level(logging.DEBUG):
        status, body = request(server, "POST", "/api/infer", infer_body(18, 2))
        bad_status, bad = request(server, "POST", "/api/infer", infer_body(17, 2))
    assert status == 502 and body["code"] == "infer_failed"
    assert bad_status == 400 and bad["code"] == "invalid_window"
    no_leak(json.dumps(body) + json.dumps(bad))
    no_leak(" ".join(r.getMessage() for r in caplog.records))
    assert all(r.exc_info is None for r in caplog.records if r.name.startswith("app"))
