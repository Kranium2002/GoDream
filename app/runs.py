"""Learning runs: GET /api/runs, GET /api/runs/{run_id}, POST /api/runs and the SSE `run` events.

    GET  /api/runs           the run ids found in config.yaml observed.runs_dir (Dev B), sorted
    GET  /api/runs/{run_id}  {run_id, state, steps: [{step, state, done, total, data}], baseline, results, problems}
                             from <runs_dir>/<run_id>/events.jsonl and result.json, read as they are now
    POST /api/runs           {declared_spec, confirm_cost}: 202 {run_id}; without confirm_cost 409
                             cost_confirmation_required with the quote's fields flat in the body; 503 runs_unavailable
                             naming what is missing until Dev B's runner is wired in app/backend.py
Each new complete line of any run's events.jsonl is relayed as an SSE `run` event, unchanged. Dev B's `data` is
passed through untouched everywhere: the page shows what Dev B sends. A bad line is reported and skipped, never fatal.
"""
import json
import logging
import threading
from urllib.parse import unquote, urlsplit
import re

from app import backend
from app.routes import Refusal, capability, route

EVENTS, RESULT = "events.jsonl", "result.json"
STEPS = ("baseline", "attempts", "teacher_fit", "dream", "train", "evaluate")  # pipeline order (Dev B, 20:55 UTC)
STATES = ("running", "done", "failed", "refused", "stopped")
ENDED = ("failed", "refused", "stopped")
KEYS = {"run_id", "skill", "step", "state", "done", "total", "time_utc", "data"}
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")  # a folder name: no separator, no leading dot
MESSAGES = {
    "malformed_line": "A line of the run's event file is not a complete event; it was skipped.",
    "unknown_step": "A line of the run's event file names a step this server does not know; it was skipped.",
    "unknown_state": "A line of the run's event file names a state this server does not know; it was skipped.",
    "malformed_result": "The run's result file is not valid JSON, so no results are shown.",
}
log = logging.getLogger(__name__)


def _count(value) -> bool:
    return value is None or (type(value) is int and value >= 0)


def _event(raw: bytes):
    """(event, None) for a valid line, else (None, problem code)."""
    try:
        event = json.loads(raw.decode("utf-8"))
    except ValueError:  # UnicodeDecodeError is a ValueError
        return None, "malformed_line"
    if not (isinstance(event, dict) and KEYS <= set(event)
            and all(isinstance(event[k], str) for k in ("run_id", "skill", "time_utc"))
            and _count(event["done"]) and _count(event["total"])):
        return None, "malformed_line"
    if event["step"] not in STEPS:
        return None, "unknown_step"
    if event["state"] not in STATES:
        return None, "unknown_state"
    return event, None


def parse_lines(data: bytes):
    """(events, problems, consumed) for the complete lines of data. A last line without its newline is not read:
    consumed stops before it, for the next read. Blank lines are skipped; each bad line is one problem
    {offset, code, message}, offset counted in bytes from the start of data."""
    events, problems, offset = [], [], 0
    while (end := data.find(b"\n", offset)) != -1:
        raw = data[offset:end]
        if raw.strip():
            event, code = _event(raw)
            if code:
                problems.append({"offset": offset, "code": code, "message": MESSAGES[code]})
            else:
                events.append(event)
        offset = end + 1
    return events, problems, offset


def _latest(events) -> dict:
    latest = {}
    for event in events:
        latest[event["step"]] = event
    return latest


def summarize(events) -> list:
    """The newest line of each step, in pipeline order, as {step, state, done, total, data}."""
    latest = _latest(events)
    return [{k: latest[s][k] for k in ("step", "state", "done", "total", "data")} for s in STEPS if s in latest]


def run_state(events) -> str:
    """starting before any line; the first ended state (failed, refused, stopped) of a step in pipeline order; done once
    evaluate is done; else running."""
    if not events:
        return "starting"
    steps = summarize(events)
    ended = [s["state"] for s in steps if s["state"] in ENDED]
    if ended:
        return ended[0]
    return "done" if steps[-1]["step"] == "evaluate" and steps[-1]["state"] == "done" else "running"


def report(folder) -> dict:
    """GET /api/runs/{run_id}'s answer for a run folder, read now."""
    events_file, result_file = folder / EVENTS, folder / RESULT
    events, problems, _ = parse_lines(events_file.read_bytes() if events_file.is_file() else b"")
    steps, results = summarize(events), None
    if result_file.is_file():
        try:
            results = json.loads(result_file.read_bytes().decode("utf-8"))
        except ValueError:
            problems.append({"offset": None, "code": "malformed_result", "message": MESSAGES["malformed_result"]})
    return {"run_id": folder.name, "state": run_state(events), "steps": steps,
            "baseline": next((s["data"] for s in steps if s["step"] == "baseline"), None),
            "results": results, "problems": problems}


def scan(root, offsets: dict, emit, first: bool = False) -> None:
    """Hand emit each new complete line of every <root>/<run_id>/events.jsonl, keeping each file's read position in
    offsets. On the first scan what is already there is history and is skipped; a file seen later is read from its
    start; a file shorter than its position was rewritten and is read again from its start."""
    for path in sorted(root.glob(f"*/{EVENTS}")):
        if not RUN_ID.fullmatch(path.parent.name):
            continue
        with open(path, "rb") as f:
            data = f.read()
        if path not in offsets:
            offsets[path] = data.rfind(b"\n") + 1 if first else 0
        if len(data) < offsets[path]:
            offsets[path] = 0
        start = offsets[path]
        events, problems, consumed = parse_lines(data[start:])
        offsets[path] = start + consumed
        for p in problems:
            log.warning("Run %s: %s at byte %d of its event file", path.parent.name, p["code"], start + p["offset"])
        for event in events:
            emit(event)


def watch(server) -> None:
    """Relay new run lines as SSE `run` events every app.runs.poll_s, in a thread of its own, until server.closing."""
    root, _ = backend.runs_dir(server.config)
    if root is None:
        return

    def loop():
        first = True
        offsets = {}
        while not server.closing.is_set():
            try:
                scan(root, offsets, lambda event: server.sim.announce("run", event), first)
                first = False
            except OSError as e:  # a file being written or removed; the next scan tries again
                log.warning("Reading the run files failed: %s", type(e).__name__)
            server.closing.wait(server.config["app"]["runs"]["poll_s"])

    threading.Thread(target=loop, name="runs", daemon=True).start()


def start_run(body: dict, config: dict, runner, declared) -> str:
    """Check a run request and start it through Dev B's runner; the run id. Refusal 400 for a malformed request or a
    declaration with problems, 409 until the quoted cost is confirmed."""
    if not (set(body) <= {"declared_spec", "confirm_cost"} and isinstance(body.get("declared_spec"), dict)
            and isinstance(body.get("confirm_cost", False), bool)):
        raise Refusal(400, "invalid_body", "A run request is a declared_spec from the chat and confirm_cost, true or "
                                           "false.")
    declaration = body["declared_spec"]
    problems = list(declared.check_declaration(declaration, config))
    if problems:
        raise Refusal(400, "invalid_declaration", "The declaration does not pass its checks; problems lists them.",
                      {"problems": problems})
    quote = runner.quote(declaration, config)
    if not body.get("confirm_cost"):
        raise Refusal(409, "cost_confirmation_required", "This run spends hosted TabPFN usage; confirm the quoted "
                                                         "cost to start it.", quote)
    return runner.start(declaration, config)


def _needs(what: str, missing: list) -> str:
    return f"{what} need {'; and '.join(missing)}."


@capability("runs")
def runs_capability(server) -> dict:
    missing = backend.runner(server.config)[1] + backend.declared()[1]
    return {"available": not missing, "reason": _needs("Learning runs", missing) if missing else ""}


@capability("results")
def results_capability(server) -> dict:
    missing = backend.runs_dir(server.config)[1]
    return {"available": not missing, "reason": _needs("Run results", missing) if missing else ""}


def _root(h):
    root, missing = backend.runs_dir(h.server.config)
    if root is None:
        raise Refusal(503, "runs_unavailable", _needs("Run results", missing))
    return root


@route("GET", "/api/runs")
def list_runs(h) -> None:
    root = _root(h)
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and RUN_ID.fullmatch(p.name)) if root.is_dir() else []
    h.send_json(200, found)


@route("GET", "/api/runs/*")
def get_run(h) -> None:
    run_id = unquote(urlsplit(h.path).path.rsplit("/", 1)[1])
    root = _root(h)
    if not RUN_ID.fullmatch(run_id) or not (root / run_id).is_dir():
        raise Refusal(404, "unknown_run", "There is no run with that id.")
    h.send_json(200, report(root / run_id))


@route("POST", "/api/runs")
def post_run(h) -> None:
    runner, missing = backend.runner(h.server.config)
    declared, declared_missing = backend.declared()
    if missing or declared_missing:
        raise Refusal(503, "runs_unavailable", _needs("Learning runs", missing + declared_missing))
    run_id = start_run(h.read_body(), h.server.config, runner, declared)
    h.send_json(202, {"run_id": run_id})
