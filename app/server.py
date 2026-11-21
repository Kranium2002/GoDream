"""The viewer's local web server: the live Go2 simulation for one browser page.

    uv run python -m app.server [--port N]

Serves on config.yaml app.host and app.port (--port 0 picks a free port) until Ctrl+C. One simulation thread
(app/simulation.py) owns the world; request threads only hand it requests and read what it publishes.

    GET  /api/state    what the page needs to draw its controls: command fields and ranges, presets, status,
                       frame size and rate, the skill catalogue and which capabilities work, each with its reason,
                       and run_modes: per skill, the learning-run modes Dev B's run_skill accepts
    GET  /api/stream   multipart/x-mixed-replace stream of PNG frames rendered by MuJoCo, at app.stream_fps;
                       ?camera=tracking (the default); head is 503 until the head camera exists, anything else 400
    GET  /api/events   Server-Sent Events: observation (app.observation_hz), episode (start, end with the outcome),
                       notice ({code, message})
    POST /api/drive    {vel_x, vel_y, yaw_rate}, validated strictly; applied from the next control period
    POST /api/stop     the zero command
    POST /api/reset    optional {preset}: a fresh episode, on a new world if the preset differs
    GET  /api/skills, POST /api/chat   the skill catalogue and the chat (app/chat.py)
    GET  /api/runs, GET /api/runs/{run_id}, POST /api/runs   learning runs (app/runs.py)
    GET  /api/robot    the stage's bodies, geoms and meshes for the 3D view; SSE pose and robot (app/robot.py)
    POST /api/infer    live inference from the browser's image window (app/infer.py)
    GET  /             and other static files, from app/static only
Every refusal or error is {code, message}: a stable code and one plain sentence; the detail goes to the log on stderr.
Routes live in app/routes.py's table: the ones above are registered here, each later feature in a module of its own,
imported below so that it registers.
"""
import argparse
import json
import logging
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import app.chat  # noqa: F401  registers GET /api/skills, POST /api/chat and the chat capability
import app.infer  # noqa: F401  registers POST /api/infer and the infer capability
import app.robot  # noqa: F401  registers GET /api/robot
import app.runs  # noqa: F401  registers the run routes and the runs and results capabilities
from app import backend
from app.routes import CAPABILITIES, Refusal, capability, find, route, to_json
from app.simulation import Simulation
from learned_judgment.env import CommandError, load_config, validate_command

STATIC = Path(__file__).resolve().parent / "static"
# HTTP API v0's capabilities with what each still lacks; a feature module's own @capability replaces its entry.
NOT_BUILT = {
    "drive": "The simulation is not running; restart the server.",
    "head_camera": "The head camera stream needs the world lane's head camera, and this server does not serve it yet.",
    "scenarios": "Threat scenarios need the world lane's scenario sampler, and this server does not serve them yet.",
    "chat": "The chat module did not register.",  # app/chat.py replaces it
    "runs": "The runs module did not register.",  # app/runs.py replaces it
    "results": "The runs module did not register.",  # app/runs.py replaces it
    "infer": "The inference module did not register.",  # app/infer.py replaces it
}
log = logging.getLogger(__name__)


class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"  # on Windows SO_REUSEADDR would let a second server bind the same port

    def __init__(self, config: dict, port: int):
        self.config, self.sim = config, Simulation(config)
        self.closing = threading.Event()  # set by stop(); background helpers (app/runs.py) end on it
        super().__init__((config["app"]["host"], port), Handler)


class Handler(BaseHTTPRequestHandler):
    """Dispatches to app/routes.py's table; a route handler gets this handler and uses the public helpers below."""
    server: Server
    server_version, sys_version = "viewer", ""  # the Server header names no software or version

    def do_GET(self) -> None:
        self._answer(find("GET", urlsplit(self.path).path) or static)

    def do_POST(self) -> None:
        self._answer(find("POST", urlsplit(self.path).path) or nothing_here)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """http.server's own errors (a malformed request, an unsupported method) as {code, message} too."""
        self.log_error("code %d, message %s", code, message)
        self.send_json(code, {"code": f"http_{code}", "message": "The server cannot handle that request."})

    def _answer(self, handle) -> None:
        try:
            handle(self)
        except Refusal as r:
            self.send_json(r.status, r.body())
        except ConnectionError:
            pass  # the client went away
        except Exception:
            log.exception("%s %s failed", self.command, self.path)
            self.send_json(500, {"code": "internal_error",
                                 "message": "Something went wrong in the server; its log has the details."})

    def send_json(self, status: int, body) -> None:
        self.send_json_text(status, to_json(body))

    def send_json_text(self, status: int, text: str) -> None:
        """An answer already encoded as JSON text (GET /api/robot's, built once per world)."""
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> dict:
        """The request's JSON object; no body is {}."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise Refusal(400, "invalid_body", "The request body must be a JSON object.")
        return body

    def submit(self, kind: str, value) -> None:
        """Hand the simulation thread a request and wait until it has been carried out."""
        try:
            self.server.sim.submit(kind, value).result()
        except Exception:  # the simulation thread has stopped; its log has the reason
            raise Refusal(503, "simulation_stopped", "The simulation is not running; restart the server.") from None

    def start_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def nothing_here(h: Handler) -> None:
    raise Refusal(404, "not_found", "There is nothing at that address.")


def static(h: Handler) -> None:
    """A file of app/static ("/" is index.html); nothing outside that directory is ever read."""
    try:
        file = (STATIC / (unquote(urlsplit(h.path).path).lstrip("/") or "index.html")).resolve()
        found = file.is_relative_to(STATIC) and file.is_file()
    except (OSError, ValueError):  # a name the file system cannot hold
        found = False
    if not found:
        nothing_here(h)
    data = file.read_bytes()
    h.send_response(200)
    h.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def capabilities(server: Server) -> dict:
    return {name: CAPABILITIES[name](server) if name in CAPABILITIES else {"available": False, "reason": reason}
            for name, reason in NOT_BUILT.items()}


@capability("drive")
def drive_capability(server: Server) -> dict:
    running = server.sim.is_alive() and not server.sim.stopped
    return {"available": running, "reason": "" if running else NOT_BUILT["drive"]}


@route("GET", "/api/state")
def state(h: Handler) -> None:
    config, sim = h.server.config, h.server.sim
    render = config["world"]["render"]
    actions, hold_s = backend.actions(config)
    h.send_json(200, {
        "commands": {name: {"unit": f["unit"], "min": f["min"], "max": f["max"]}
                     for name, f in sim.schema["fields"].items()},
        "presets": list(config["world"]["presets"]),
        **sim.state(),
        "frame": {"width": render["width"], "height": render["height"], "stream_fps": config["app"]["stream_fps"]},
        "capabilities": capabilities(h.server),
        "skills": backend.skill_catalog(config),
        "run_modes": backend.run_modes(config),  # {skill: {mode: "" or Dev B's refusal}}, asked of run_skill.plan_run
        "image_window": backend.image_window(config),  # null until config.yaml has Dev B's vision block
        "actions": actions,  # {name: command} of skills.dodge_vision, unchanged
        "action_hold_s": hold_s,  # how long an emitted action's command is held (skills.dodge_vision.hold_s)
    })


@route("GET", "/api/stream")
def stream(h: Handler) -> None:
    camera = parse_qs(urlsplit(h.path).query).get("camera", ["tracking"])[-1]
    if camera == "head":
        raise Refusal(503, "head_camera_unavailable", NOT_BUILT["head_camera"])
    if camera != "tracking":
        raise Refusal(400, "unknown_camera", "The stream offers the cameras tracking and head only.")
    h.start_stream("multipart/x-mixed-replace; boundary=frame")
    for png in h.server.sim.frames():
        h.wfile.write(b"--frame\r\nContent-Type: image/png\r\nContent-Length: %d\r\n\r\n%b\r\n" % (len(png), png))


@route("GET", "/api/events")
def events(h: Handler) -> None:
    h.start_stream("text/event-stream")
    for name, data in h.server.sim.events():
        h.wfile.write(f"event: {name}\ndata: {to_json(data)}\n\n".encode())


@route("POST", "/api/drive")
def drive(h: Handler) -> None:
    command(h, h.read_body())


@route("POST", "/api/stop")
def stop_dog(h: Handler) -> None:
    command(h, h.server.sim.zero)


def command(h: Handler, command: dict) -> None:
    schema = h.server.sim.schema
    try:
        validate_command(command, schema)
    except CommandError as e:
        log.warning("Drive command refused: %s", e)
        raise Refusal(400, "invalid_command", "A drive command needs every value, each a number within its range.")
    command = {name: float(command[name]) for name in schema["fields"]}
    h.submit("drive", command)
    h.send_json(200, {"command": command})


@route("POST", "/api/reset")
def reset(h: Handler) -> None:
    body = h.read_body()
    preset = body.get("preset")
    if set(body) - {"preset"}:
        raise Refusal(400, "invalid_body", "A reset takes nothing but an optional preset name.")
    if preset is not None:
        if not isinstance(preset, str) or preset not in h.server.config["world"]["presets"]:
            raise Refusal(400, "unknown_preset", "There is no world preset with that name.")
        if preset not in h.server.sim.state()["available_presets"]:
            raise Refusal(400, "preset_unavailable", "That world preset cannot be built yet, so the dog stays "
                                                     "where it is.")
    h.submit("reset", preset)
    h.send_json(200, {"preset": h.server.sim.state()["preset"]})


def start(config: dict, port: int) -> Server:
    """Check config.yaml app, bind app.host:port (0 picks a free port), start the simulation thread and, once the dog
    stands in its first episode, serve in a background thread. ValueError for an app setting the server cannot run
    with, RuntimeError if the simulation cannot start (its log has the reason)."""
    if config["app"]["host"] != "127.0.0.1":
        raise ValueError(f"config.yaml app.host must be 127.0.0.1, the viewer serves this machine alone; "
                         f"got {config['app']['host']!r}")
    server = Server(config, port)
    try:
        server.sim.start()
        server.sim.ready.wait()
        if server.sim.stopped:
            raise RuntimeError("the simulation thread could not start; its log above has the reason")
    except BaseException:
        server.sim.stop()
        server.server_close()
        raise
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    app.runs.watch(server)
    return server


def stop(server: Server) -> None:
    """Stop serving, stop the simulation thread (open streams then end) and free the port. Safe to call again."""
    server.closing.set()
    server.shutdown()
    server.sim.stop()
    server.server_close()


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    parser = argparse.ArgumentParser(prog="python -m app.server", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=config["app"]["port"], help="0 picks a free port")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        server = start(config, args.port)
    except ValueError as e:
        parser.exit(2, f"{parser.prog}: {e}\n")
    host, port = server.server_address[:2]
    print(f"Viewer serving at http://{host}:{port}/ (Ctrl+C stops it)", flush=True)
    try:
        while True:
            time.sleep(1)  # until Ctrl+C; time.sleep is interruptible on Windows too
    except KeyboardInterrupt:
        pass
    finally:
        stop(server)


if __name__ == "__main__":
    main()
