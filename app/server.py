"""The viewer's local web server: the live Go2 simulation for one browser page.

    uv run python -m app.server [--port N]

Serves on config.yaml app.host and app.port (--port 0 picks a free port) until Ctrl+C. One simulation thread
(app/simulation.py) owns the world; request threads only hand it requests and read what it publishes.

    GET  /api/state    what the page needs to draw its controls: command fields and ranges, presets, status,
                       frame size and rate, which features this stage offers
    GET  /api/stream   multipart/x-mixed-replace stream of PNG frames rendered by MuJoCo, at app.stream_fps
    GET  /api/events   Server-Sent Events: observation (app.observation_hz), episode (start, end with the outcome),
                       notice ({code, message})
    POST /api/drive    {vel_x, vel_y, yaw_rate}, validated strictly; applied from the next control period
    POST /api/stop     the zero command
    POST /api/reset    optional {preset}: a fresh episode, on a new world if the preset differs
    GET  /             and other static files, from app/static only
Every refusal or error is {code, message}: a stable code and one plain sentence; the detail goes to the log on stderr.
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
from urllib.parse import unquote, urlsplit

from app.simulation import Simulation
from learned_judgment.env import CommandError, load_config, validate_command

STATIC = Path(__file__).resolve().parent / "static"
CAPABILITIES = {"drive": True, "throw": False, "brains": False, "teach": False, "metrics": False}  # of this stage
log = logging.getLogger(__name__)


class Refusal(Exception):
    """A request turned down: its HTTP status, a stable code and one plain sentence for the browser."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"  # on Windows SO_REUSEADDR would let a second server bind the same port

    def __init__(self, config: dict, port: int):
        self.config, self.sim = config, Simulation(config)
        super().__init__((config["app"]["host"], port), Handler)


class Handler(BaseHTTPRequestHandler):
    server: Server
    server_version, sys_version = "viewer", ""  # the Server header names no software or version

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        routes = {"/api/state": self._state, "/api/stream": self._stream, "/api/events": self._events}
        self._answer(routes.get(path, lambda: self._static(path)))

    def do_POST(self) -> None:
        routes = {"/api/drive": self._drive, "/api/stop": self._stop, "/api/reset": self._reset}
        self._answer(routes.get(urlsplit(self.path).path, self._nothing_here))

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """http.server's own errors (a malformed request, an unsupported method) as {code, message} too."""
        self.log_error("code %d, message %s", code, message)
        self._json(code, {"code": f"http_{code}", "message": "The server cannot handle that request."})

    def _answer(self, handle) -> None:
        try:
            handle()
        except Refusal as r:
            self._json(r.status, {"code": r.code, "message": r.message})
        except ConnectionError:
            pass  # the client went away
        except Exception:
            log.exception("%s %s failed", self.command, self.path)
            self._json(500, {"code": "internal_error", "message": "Something went wrong in the server; its log has the details."})

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        """The request's JSON object; no body is {}."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise Refusal(400, "invalid_body", "The request body must be a JSON object.")
        return body

    def _submit(self, kind: str, value) -> None:
        """Hand the simulation thread a request and wait until it has been carried out."""
        try:
            self.server.sim.submit(kind, value).result()
        except Exception:  # the simulation thread has stopped; its log has the reason
            raise Refusal(503, "simulation_stopped", "The simulation is not running; restart the server.") from None

    def _nothing_here(self) -> None:
        raise Refusal(404, "not_found", "There is nothing at that address.")

    def _drive(self) -> None:
        self._command(self._body())

    def _stop(self) -> None:
        self._command(self.server.sim.zero)

    def _command(self, command: dict) -> None:
        schema = self.server.sim.schema
        try:
            validate_command(command, schema)
        except CommandError as e:
            log.warning("Drive command refused: %s", e)
            raise Refusal(400, "invalid_command", "A drive command needs every value, each a number within its range.")
        command = {name: float(command[name]) for name in schema["fields"]}
        self._submit("drive", command)
        self._json(200, {"command": command})

    def _reset(self) -> None:
        body = self._body()
        preset = body.get("preset")
        if set(body) - {"preset"}:
            raise Refusal(400, "invalid_body", "A reset takes nothing but an optional preset name.")
        if preset is not None:
            if not isinstance(preset, str) or preset not in self.server.config["world"]["presets"]:
                raise Refusal(400, "unknown_preset", "There is no world preset with that name.")
            if preset not in self.server.sim.state()["available_presets"]:
                raise Refusal(400, "preset_unavailable", "That world preset cannot be built yet, so the dog stays "
                                                         "where it is.")
        self._submit("reset", preset)
        self._json(200, {"preset": self.server.sim.state()["preset"]})

    def _stream(self) -> None:
        self._start_stream("multipart/x-mixed-replace; boundary=frame")
        for png in self.server.sim.frames():
            self.wfile.write(b"--frame\r\nContent-Type: image/png\r\nContent-Length: %d\r\n\r\n%b\r\n" % (len(png), png))

    def _events(self) -> None:
        self._start_stream("text/event-stream")
        for name, data in self.server.sim.events():
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())

    def _start_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _state(self) -> None:
        config, sim = self.server.config, self.server.sim
        render = config["world"]["render"]
        self._json(200, {
            "command_fields": {name: {"unit": f["unit"], "min": f["min"], "max": f["max"]}
                               for name, f in sim.schema["fields"].items()},
            "presets": list(config["world"]["presets"]),
            **sim.state(),
            "frame": {"width": render["width"], "height": render["height"], "stream_fps": config["app"]["stream_fps"]},
            "capabilities": CAPABILITIES,
        })

    def _static(self, path: str) -> None:
        """A file of app/static ("/" is index.html); nothing outside that directory is ever read."""
        try:
            file = (STATIC / (unquote(path).lstrip("/") or "index.html")).resolve()
            found = file.is_relative_to(STATIC) and file.is_file()
        except (OSError, ValueError):  # a name the file system cannot hold
            found = False
        if not found:
            self._nothing_here()
        data = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
    return server


def stop(server: Server) -> None:
    """Stop serving, stop the simulation thread (open streams then end) and free the port. Safe to call again."""
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
