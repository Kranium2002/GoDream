"""The episode player (app/player.py) and the stream settings, against the real simulator, walking policy and Dev B's
real skill code and saved runs. A test that needs a run of results/runs skips when that run is not in this tree."""
import copy
import http.client
import json
import threading
import time
from contextlib import closing

import pytest

from app import backend
from app.player import build_plan
from app.routes import Refusal
from app.server import start, stop
from learned_judgment.env import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def run_or_skip(config, run_id):
    root, _ = backend.runs_dir(config)
    if not (root / run_id / "student" / "student.json").is_file():
        pytest.skip(f"Dev B's {run_id} run is not in this tree")
    return root / run_id


def request(server, method, path, body=None):
    with closing(http.client.HTTPConnection(*server.server_address[:2], timeout=120)) as conn:
        conn.request(method, path, body=None if body is None else json.dumps(body))
        response = conn.getresponse()
        return response.status, json.loads(response.read())


def test_a_plan_replays_the_runs_saved_test_scenarios_baseline_then_student(config):
    folder = run_or_skip(config, "move1")
    plan = build_plan("move1", config)
    records = json.loads((folder / "test_run" / "run.json").read_text(encoding="utf-8"))["records"]
    assert (plan.skill, plan.step, plan.split, plan.student_mode) == ("MOVE", "evaluate", "test", "argmax")
    assert [e["method"] for e in plan.episodes[:4]] == ["physics", "student", "physics", "student"]
    assert len(plan.episodes) == 2 * len(records)
    assert [e["scenario"]["target"] for e in plan.episodes[::2]] == [r["target"] for r in records]
    assert "equal to the sampler's draw" in plan.source  # the saved scenarios are the split seeds' own


def test_a_bare_student_folder_plays_its_student_on_the_dev_split(config):
    run_or_skip(config, "v3-provisional")
    plan = build_plan("v3-provisional", config)
    assert (plan.skill, plan.split, plan.student_mode) == ("DODGE", "dev", "closed_loop")
    assert "no run computes them" in plan.note
    assert all(len(e["seeds"]) == 2 for e in plan.episodes)  # collect_rows' and dodge_eval's per-scenario children


@pytest.mark.parametrize("run_id", ["no_such_run", "../config.yaml", ""])
def test_an_unknown_run_is_refused(config, run_id):
    with pytest.raises(Refusal) as refused:
        build_plan(run_id, config)
    assert refused.value.status == 404 and refused.value.code == "unknown_run"


def test_the_server_refuses_a_stream_size_beyond_the_offscreen_buffer(config):
    bad = copy.deepcopy(config)
    bad["app"]["stream"]["width"] = config["world"]["render"]["width"] + 1
    with pytest.raises(ValueError, match="app.stream.width"):
        start(bad, 0)


@pytest.fixture(scope="module")
def server(config):
    server = start(config, 0)
    yield server
    stop(server)


def test_episode_requests_are_checked(server):
    assert request(server, "POST", "/api/episodes/start", {"run": "move1"})[0] == 400
    assert request(server, "POST", "/api/episodes/start", {"run_id": 3})[0] == 400
    assert request(server, "POST", "/api/episodes/start", {"run_id": "no_such_run"}) == (
        404, {"code": "unknown_run", "message": "There is no run with that id."})
    assert request(server, "POST", "/api/episodes/stop", {}) == (200, {"playing": False})
    assert request(server, "GET", "/api/state")[1]["episodes"] == {"playing": False, "run_id": None, "note": ""}


def test_a_run_plays_on_the_stage_until_the_drive_pad_takes_it_back(server, config):
    """move1's first test episode, the physics rule: the events of the contract, the dog moving toward the target
    marker, the outcome the run saved; then a drive command stops the player and roaming resumes."""
    run_or_skip(config, "move1")
    seen, idle = [], threading.Event()

    def listen():
        with closing(http.client.HTTPConnection(*server.server_address[:2], timeout=300)) as conn:
            conn.request("GET", "/api/events")
            name = None
            for line in conn.getresponse():
                line = line.decode().rstrip("\n")
                if line.startswith("event: "):
                    name = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    seen.append((name, json.loads(line.removeprefix("data: "))))
                    stopped = any(n == "episode" and d.get("state") == "idle" for n, d in seen)
                    if stopped and name == "scene":  # the scene published with the idle event
                        idle.set()
                        return

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    time.sleep(0.5)
    status, body = request(server, "POST", "/api/episodes/start", {"run_id": "move1"})
    assert status == 202 and body["skill"] == "MOVE" and body["step"] == "evaluate" and body["total"] > 1
    assert request(server, "GET", "/api/state")[1]["episodes"]["playing"] is True
    deadline = time.monotonic() + 240
    while not any(n == "episode" and d.get("state") == "end" for n, d in seen):
        assert time.monotonic() < deadline, "no episode ended"
        time.sleep(0.2)
    assert request(server, "POST", "/api/drive", {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0})[0] == 200
    assert idle.wait(60)
    episodes = [d for n, d in seen if n == "episode" and "state" in d]
    start_event, end_event = episodes[0], next(d for d in episodes if d["state"] == "end")
    keys = {"state", "run_id", "skill", "step", "index", "total", "kind", "action", "caption", "outcome", "note"}
    assert all(set(d) == keys for d in episodes)
    assert (start_event["state"], start_event["index"], start_event["action"]) == ("start", 0, "physics rule")
    assert start_event["outcome"] is None and end_event["index"] == 0
    assert {"error", "drift_m", "fall", "error_units"} <= set(end_event["outcome"])
    assert end_event["outcome"]["matches_run_record"] is True  # deterministic: the run's own episode, recomputed
    assert episodes[-1]["state"] == "idle"
    # poses and scenes while it played: the trunk (body 1) moved, the target marker stayed put
    t_start = next(i for i, (n, d) in enumerate(seen) if n == "episode" and d.get("state") == "start")
    t_end = next(i for i, (n, d) in enumerate(seen) if n == "episode" and d.get("state") == "end")
    poses = [d for n, d in seen[t_start:t_end] if n == "pose"]
    scenes = [d for n, d in seen[t_start:t_end] if n == "scene"]
    assert len(poses) > 10 and len(scenes) > 10
    assert poses[-1]["pos"][1] != poses[0]["pos"][1]
    target = scenes[0]["markers"]
    assert {m["shape"] for m in target} == {"line", "disc"} and all(s["markers"] == target for s in scenes)
    assert scenes[0]["objects"] == []  # MOVE has no thrown object or ramp
    assert [d for n, d in seen if n == "scene"][-1] == {"objects": [], "markers": []}  # one empty scene at the stop
    assert request(server, "GET", "/api/state")[1]["episodes"]["playing"] is False
