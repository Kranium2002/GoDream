"""Learning runs (app/runs.py): the event-file reader, the step summary, the run report, the watcher that relays new
lines and the cost confirmation. Every event line, result file and runner here is a fixture written by the test for
these functions alone: none of it is a learning run or evidence of one."""
import json
import types

import pytest

from app import backend, runs
from app.routes import Refusal
from learned_judgment.env import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def line(step="baseline", state="running", done=0, total=10, data=None, **extra):
    event = {"run_id": "r1", "skill": "RAMP", "step": step, "state": state, "done": done, "total": total,
             "time_utc": "2026-11-21T21:00:00Z", "data": {} if data is None else data} | extra
    return (json.dumps(event) + "\n").encode()


# ---- the reader ----------------------------------------------------------------------------------------------------

def test_complete_lines_are_read_and_a_partial_last_line_is_left():
    data = line(done=1) + line(done=2) + line(done=3)[:20]
    events, problems, consumed = runs.parse_lines(data)
    assert [e["done"] for e in events] == [1, 2] and problems == []
    assert consumed == len(line(done=1) + line(done=2))  # the partial line waits for the next read


def test_data_passes_through_untouched():
    data = {"curve": [0.1, 0.2], "nested": {"anything": None}, "unicode": "°"}
    events, _, _ = runs.parse_lines(line(data=data))
    assert events[0]["data"] == data


@pytest.mark.parametrize("bad, code", [
    (b"not json\n", "malformed_line"), (b"[1, 2]\n", "malformed_line"),
    (line(step="warmup"), "unknown_step"), (line(state="paused"), "unknown_state"),
    (line(done="3"), "malformed_line"), (line(done=True), "malformed_line"), (line(done=-1), "malformed_line"),
    (json.dumps({"run_id": "r1"}).encode() + b"\n", "malformed_line"),
], ids=["not_json", "not_an_object", "unknown_step", "unknown_state", "done_text", "done_bool", "done_negative",
        "missing_keys"])
def test_a_bad_line_is_reported_and_skipped_never_fatal(bad, code):
    events, problems, consumed = runs.parse_lines(line(done=1) + bad + line(done=2))
    assert [e["done"] for e in events] == [1, 2]
    assert [p["code"] for p in problems] == [code] and problems[0]["offset"] == len(line(done=1))
    assert consumed == len(line(done=1) + bad + line(done=2))


def test_blank_lines_and_crlf_are_accepted_and_null_progress_is_allowed():
    events, problems, _ = runs.parse_lines(b"\n" + line(done=None, total=None).replace(b"\n", b"\r\n") + b"  \n")
    assert problems == [] and events[0]["done"] is None


def test_an_invalid_utf8_line_is_a_problem():
    events, problems, _ = runs.parse_lines(b"\xff\xfe\n" + line())
    assert len(events) == 1 and [p["code"] for p in problems] == ["malformed_line"]


# ---- the summary ---------------------------------------------------------------------------------------------------

def events_of(*lines):
    return runs.parse_lines(b"".join(lines))[0]


def test_summary_has_the_newest_line_per_step_in_pipeline_order():
    events = events_of(line("attempts", done=1), line("baseline", "done", 10, 10, {"fall_rate": 0.4}),
                       line("attempts", done=7, data={"last": 3}))
    assert runs.summarize(events) == [
        {"step": "baseline", "state": "done", "done": 10, "total": 10, "time_utc": "2026-11-21T21:00:00Z", "data": {"fall_rate": 0.4}},
        {"step": "attempts", "state": "running", "done": 7, "total": 10, "time_utc": "2026-11-21T21:00:00Z", "data": {"last": 3}},
    ]


@pytest.mark.parametrize("lines, state", [
    ([], "starting"), ([line("baseline")], "running"), ([line("evaluate", "done")], "done"),
    ([line("baseline", "done"), line("attempts", "failed")], "failed"),
    ([line("baseline", "refused")], "refused"), ([line("dream", "stopped")], "stopped"),
])
def test_run_state_follows_its_steps(lines, state):
    assert runs.run_state(events_of(*lines)) == state


def test_report_reads_events_and_the_result_file(tmp_path):
    folder = tmp_path / "r1"
    folder.mkdir()
    (folder / runs.EVENTS).write_bytes(line("baseline", "done", 10, 10, {"fall_rate": 0.4}) + b"garbage\n"
                                       + line("evaluate", "running"))
    result = {"before": {"n": 40}, "after": {"n": 40}, "scenario_ids": ["a", "b"]}
    (folder / runs.RESULT).write_text(json.dumps(result))
    report = runs.report(folder)
    assert report["run_id"] == "r1" and report["state"] == "running"
    assert [s["step"] for s in report["steps"]] == ["baseline", "evaluate"]
    assert report["baseline"] == {"fall_rate": 0.4} and report["results"] == result
    assert [p["code"] for p in report["problems"]] == ["malformed_line"]


def test_report_without_a_result_or_with_a_broken_one(tmp_path):
    folder = tmp_path / "r2"
    folder.mkdir()
    (folder / runs.EVENTS).write_bytes(line("attempts"))  # no baseline step yet
    assert runs.report(folder)["results"] is None and runs.report(folder)["baseline"] is None
    (folder / runs.RESULT).write_text("{broken")
    report = runs.report(folder)
    assert report["results"] is None and [p["code"] for p in report["problems"]] == ["malformed_result"]


@pytest.mark.parametrize("run_id, ok", [("r1", True), ("2026-11-21T21-00-00_ramp", True), ("..", False),
                                        ("a/b", False), ("a\\b", False), ("", False), (".hidden", False)])
def test_run_ids_cannot_name_a_path_outside_the_runs_directory(run_id, ok):
    assert bool(runs.RUN_ID.fullmatch(run_id)) is ok


# ---- the watcher ---------------------------------------------------------------------------------------------------

def test_watcher_relays_only_new_complete_lines(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / runs.EVENTS).write_bytes(line(done=1))
    sent, offsets = [], {}
    runs.scan(tmp_path, offsets, sent.append, first=True)
    assert sent == []  # what was there when the server started is history, not news
    with open(old / runs.EVENTS, "ab") as f:
        f.write(line(done=2) + line(done=3)[:15])
    new = tmp_path / "new"
    new.mkdir()
    (new / runs.EVENTS).write_bytes(line(done=10))
    runs.scan(tmp_path, offsets, sent.append)
    assert sorted(e["done"] for e in sent) == [2, 10]
    with open(old / runs.EVENTS, "ab") as f:
        f.write(line(done=3)[15:])
    runs.scan(tmp_path, offsets, sent.append)
    assert sorted(e["done"] for e in sent) == [2, 3, 10]


def test_watcher_starts_over_on_a_rewritten_file(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / runs.EVENTS).write_bytes(line(done=1) + line(done=2))
    sent, offsets = [], {}
    runs.scan(tmp_path, offsets, sent.append, first=True)
    (folder / runs.EVENTS).write_bytes(line(done=5))  # shorter than what was read: a new file
    runs.scan(tmp_path, offsets, sent.append)
    assert [e["done"] for e in sent] == [5]


# ---- starting a run ------------------------------------------------------------------------------------------------

DECLARATION = {"skill": "RAMP", "params": {"incline_deg": 12.0}, "source_text": "cross the slope"}


def stand_ins(problems=()):
    """Dev B's runner and declaration module, for these plumbing tests only."""
    started = []
    runner = types.SimpleNamespace(quote=lambda d, c: {"tabpfn_requests": 3, "estimated_cost_usd": 0.0, "code": "x"},
                                   start=lambda d, c: started.append(d) or "r42")
    declared = types.SimpleNamespace(check_declaration=lambda d, c: list(problems))
    return runner, declared, started


def test_a_run_without_confirm_cost_answers_409_with_the_quote_flat(config):
    runner, declared, started = stand_ins()
    with pytest.raises(Refusal) as refused:
        runs.start_run({"declared_spec": DECLARATION}, config, runner, declared)
    body = refused.value.body()
    assert refused.value.status == 409 and body["code"] == "cost_confirmation_required" and body["message"]
    assert body["tabpfn_requests"] == 3 and body["estimated_cost_usd"] == 0.0  # flat, not nested
    assert started == []


def test_a_confirmed_run_starts_and_answers_its_id(config):
    runner, declared, started = stand_ins()
    assert runs.start_run({"declared_spec": DECLARATION, "confirm_cost": True}, config, runner, declared) == "r42"
    assert started == [DECLARATION]


@pytest.mark.parametrize("body", [{}, {"declared_spec": "RAMP"}, {"declared_spec": DECLARATION, "confirm_cost": 1},
                                  {"declared_spec": DECLARATION, "extra": True}])
def test_a_malformed_run_request_is_refused(config, body):
    runner, declared, started = stand_ins()
    with pytest.raises(Refusal) as refused:
        runs.start_run(body, config, runner, declared)
    assert refused.value.status == 400 and refused.value.code == "invalid_body" and started == []


def test_an_invalid_declaration_is_refused_with_its_problems(config):
    problem = {"code": "out_of_range", "field": "params.incline_deg", "message": "Too steep."}
    runner, declared, started = stand_ins([problem])
    with pytest.raises(Refusal) as refused:
        runs.start_run({"declared_spec": DECLARATION, "confirm_cost": True}, config, runner, declared)
    assert refused.value.status == 400 and refused.value.code == "invalid_declaration"
    assert refused.value.body()["problems"] == [problem] and started == []


def test_the_runner_is_reported_missing_until_dev_b_publishes_it(config):
    runner, missing = backend.runner(config)
    assert runner is None and any("entry point" in m for m in missing)


# ---- run summaries (the skill cards) -------------------------------------------------------------------------------

def test_a_summary_is_built_from_the_run_files(tmp_path):
    folder = tmp_path / "r1"
    (folder / "student").mkdir(parents=True)
    (folder / runs.EVENTS).write_bytes(line("baseline", "done", 4, 4) + line("attempts", "done", 9, 9)
                                       + line("teacher_fit", "running", time_utc="2026-11-21T22:00:00Z"))
    (folder / runs.RESULT).write_text(json.dumps({"skill": "RAMP"}))
    (folder / "student" / "student.json").write_text("{}")
    assert runs.summary(folder) == {"run_id": "r1", "skill": "RAMP", "state": "running",
                                    "updated_utc": "2026-11-21T22:00:00Z", "steps_done": 2, "has_result": True,
                                    "has_student": True}


def test_a_summary_of_a_bare_folder_invents_nothing(tmp_path):
    (tmp_path / "r2").mkdir()
    assert runs.summary(tmp_path / "r2") == {"run_id": "r2", "skill": None, "state": "starting", "updated_utc": None,
                                             "steps_done": 0, "has_result": False, "has_student": False}


def test_a_summary_with_a_broken_result_says_it_has_none(tmp_path):
    folder = tmp_path / "r3"
    folder.mkdir()
    (folder / runs.RESULT).write_text("{broken")
    assert runs.summary(folder)["has_result"] is False


def test_a_run_folder_that_cannot_be_read_is_listed_as_unreadable(tmp_path):
    folder = tmp_path / "r4"
    (folder / runs.EVENTS).mkdir(parents=True)  # a directory where the event file should be: reading it fails
    assert runs.summary(folder) == {"run_id": "r4", "skill": None, "state": "unreadable", "updated_utc": None,
                                    "steps_done": 0, "has_result": False, "has_student": False}


def test_dev_bs_v0_run_summarises_from_its_real_files(config):
    root, _ = backend.runs_dir(config)
    if not (root / "v0").is_dir():
        pytest.skip("Dev B's v0 run is not in this tree")
    card = runs.summary(root / "v0")  # Dev B's real run files, read as they are
    assert card["run_id"] == "v0" and card["skill"] == "dodge_vision" and card["has_student"] is True
    assert card["has_result"] is True and card["state"] != "unreadable"


# ---- Dev B's entry point (learned_judgment.run_skill, agreed 21:48 UTC) -------------------------------------------

def test_the_runner_names_learned_judgment_run_skill_until_it_imports(config):
    import importlib.util
    if importlib.util.find_spec("learned_judgment.run_skill") is not None:
        pytest.skip("Dev B's run_skill is present")
    runner, missing = backend.runner(config)
    assert runner is None and any("learned_judgment.run_skill" in m for m in missing)


def test_the_runner_calls_quote_run_and_start_run_once_the_module_imports(config, monkeypatch):
    calls = []
    stand_in = types.SimpleNamespace(  # Dev B's module, for this plumbing test only
        quote_run=lambda d, c: calls.append(("quote", d)) or {"estimated_cost_usd": 0.5},
        start_run=lambda d, c: calls.append(("start", d)) or "r7")
    monkeypatch.setitem(__import__("sys").modules, "learned_judgment.run_skill", stand_in)
    runner, missing = backend.runner(config)
    assert missing == [] and runner.quote(DECLARATION, config) == {"estimated_cost_usd": 0.5}
    assert runner.start(DECLARATION, config) == "r7" and calls == [("quote", DECLARATION), ("start", DECLARATION)]


# ---- every line of a run (the replay) -----------------------------------------------------------------------------

def test_all_events_are_the_complete_valid_lines_in_file_order(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / runs.EVENTS).write_bytes(line("train", done=1) + b"garbage\n" + line(step="warmup")
                                       + line("train", done=2) + line("train", done=3)[:10])
    assert [e["done"] for e in runs.all_events(folder)] == [1, 2]  # bad lines and the partial last line left out


def test_all_events_of_a_folder_without_an_event_file_is_empty(tmp_path):
    assert runs.all_events(tmp_path) == []


def test_all_events_of_dev_bs_v0_run_are_its_file_lines(config):
    root, _ = backend.runs_dir(config)
    if not (root / "v0").is_dir():
        pytest.skip("Dev B's v0 run is not in this tree")
    lines = [json.loads(raw) for raw in (root / "v0" / runs.EVENTS).read_bytes().splitlines() if raw.strip()]
    events = runs.all_events(root / "v0")  # Dev B's real run file
    assert len(events) == len(lines) and events[0] == lines[0] and events[-1] == lines[-1]
    assert sum(e["step"] == "train" for e in events) > 1  # why the page needs every line, not one per step
