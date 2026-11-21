"""The viewer server's route and capability tables. A feature is one module of app/ that registers its routes here
and, if the page should know whether it works, its capability; app/server.py imports the modules and dispatches.

    @route("GET", "/api/thing")
    def thing(h): h.send_json(200, {...})          # h is the server's request handler

    @route("GET", "/api/things/*/parts")           # "*" matches any one path segment; the handler reads it from h.path

    @capability("thing")
    def thing_capability(server): return {"available": ..., "reason": "..."}
"""
import json

ROUTES = {}  # (method, path) -> handler(h); a "*" segment in a path matches any one segment
CAPABILITIES = {}  # name -> fn(server) -> {"available": bool, "reason": str}


class Refusal(Exception):
    """A request turned down: its HTTP status, a stable code, one plain sentence for the browser and any extra fields
    for the body (they never replace code or message)."""

    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, dict(extra or {})

    def body(self) -> dict:
        return {**self.extra, "code": self.code, "message": self.message}


def route(method: str, path: str):
    def register(handler):
        assert (method, path) not in ROUTES, f"{method} {path} is registered twice"
        ROUTES[method, path] = handler
        return handler
    return register


def find(method: str, path: str):
    """The handler for a request path, or None: an exact route first, else a pattern whose "*" segments each match
    one segment of the path."""
    if (method, path) in ROUTES:
        return ROUTES[method, path]
    parts = path.split("/")
    for (m, pattern), handler in ROUTES.items():
        segments = pattern.split("/")
        if m == method and "*" in segments and len(segments) == len(parts) and all(
                s in ("*", p) for s, p in zip(segments, parts)):
            return handler
    return None


def capability(name: str):
    def register(fn):
        CAPABILITIES[name] = fn
        return fn
    return register


def to_json(value) -> str:
    """JSON text; numpy scalars and arrays (from Dev B's results) become plain numbers and lists."""
    return json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist") else _refuse(o))


def _refuse(o):
    raise TypeError(f"{type(o).__name__} is not JSON serializable")
