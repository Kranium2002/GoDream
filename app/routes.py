"""The viewer server's route and capability tables. A feature is one module of app/ that registers its routes here
and, if the page should know whether it works, its capability; app/server.py imports the modules and dispatches.

    @route("GET", "/api/thing")
    def thing(h): h.send_json(200, {...})          # h is the server's request handler

    @capability("thing")
    def thing_capability(server): return {"available": ..., "reason": "..."}
"""

ROUTES = {}  # (method, path) -> handler(h)
CAPABILITIES = {}  # name -> fn(server) -> {"available": bool, "reason": str}


class Refusal(Exception):
    """A request turned down: its HTTP status, a stable code and one plain sentence for the browser."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def route(method: str, path: str):
    def register(handler):
        assert (method, path) not in ROUTES, f"{method} {path} is registered twice"
        ROUTES[method, path] = handler
        return handler
    return register


def capability(name: str):
    def register(fn):
        CAPABILITIES[name] = fn
        return fn
    return register
