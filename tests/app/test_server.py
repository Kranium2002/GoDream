"""The viewer server (app/server.py) on the real simulator and walking policy, started on a free port by the same
start() the command line uses and driven over HTTP. Waits are in episode time read from the observations, so a slow
or busy machine only makes the tests take longer."""
import copy
import http.client
import json
import re
import socket
import struct
import threading
import time
import zlib
from contextlib import closing, contextmanager

import numpy as np
import pytest

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


def test_state_describes_the_controls_from_config(server, config):
    status, state = request(server, "GET", "/api/state")
    assert status == 200
    fields = config["robot"]["walking_policy"]["command_schema"]["fields"]
    assert state["command_fields"] == {name: {"unit": f["unit"], "min": f["min"], "max": f["max"]}
                                       for name, f in fields.items()}
    assert list(state["command_fields"]) == list(fields)
    assert state["presets"] == list(config["world"]["presets"])
    assert state["preset"] in state["available_presets"]
    assert state["status"] in ("roaming", "resetting", "fallen")
    assert state["frame"] == {"width": config["world"]["render"]["width"],
                              "height": config["world"]["render"]["height"],
                              "stream_fps": config["app"]["stream_fps"]}
    assert state["capabilities"] == {"drive": True, "throw": False, "brains": False, "teach": False, "metrics": False}


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
def own_server(config):
    """A server of its own, whose config adds a preset that cannot be built (no declared payload)."""
    own = copy.deepcopy(config)
    presets = own["world"]["presets"]
    presets["no_declared_payload"] = {k: v for k, v in presets["default"].items() if k != "declared_payload_kg"}
    server = start(own, 0)
    yield server
    stop(server)  # again after the shutdown test: stop is idempotent


def test_a_preset_that_cannot_be_built_is_offered_as_unavailable(own_server):
    state = request(own_server, "GET", "/api/state")[1]
    assert "no_declared_payload" in state["presets"] and "no_declared_payload" not in state["available_presets"]
    status, body = request(own_server, "POST", "/api/reset", {"preset": "no_declared_payload"})
    assert status == 400 and body["code"] == "preset_unavailable"
    assert_plain_sentence(body["message"])
    assert request(own_server, "GET", "/api/state")[1]["status"] == "roaming"  # the current world goes on


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
