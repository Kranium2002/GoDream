"""The viewer's simulation thread: the only thread that touches Go2World and its renderer.

Other threads hand it requests through submit() and read what it publishes, all guarded by one condition: its
status, the latest PNG frame, the latest observation and the episode and notice events.
"""
import copy
import logging
import math
import queue
import struct
import threading
import time
import zlib
from concurrent.futures import Future

import numpy as np

from learned_judgment.env import Go2World
from learned_judgment.env.world import FEET

log = logging.getLogger(__name__)


def encode_png(rgb: np.ndarray, level: int) -> bytes:
    """An H x W x 3 uint8 image as an 8-bit RGB PNG, every row with filter type 0 (none), compressed at zlib level."""
    height, width, _ = rgb.shape
    rows = np.hstack([np.zeros((height, 1), np.uint8), rgb.reshape(height, 3 * width)])  # each row: filter, pixels

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), level)) + chunk(b"IEND", b""))


def periods_per(rate_hz: float, period_s: float, key: str) -> int:
    """Control periods between two events at rate_hz. ValueError naming config.yaml app.<key> unless rate_hz divides
    the control rate into whole control periods."""
    every = round(1 / (rate_hz * period_s)) if rate_hz > 0 else 0
    if every < 1 or not math.isclose(every * rate_hz * period_s, 1.0):
        raise ValueError(f"config.yaml app.{key} must divide the {1 / period_s:g} Hz control rate into whole control "
                         f"periods, got {rate_hz}")
    return every


def describe(obs: dict, command: dict) -> dict:
    """What the browser is told about a Go2World observation and the command last issued. policy_obs is left out: on a
    sloped world it reveals a hidden value; so is command_active, which once meant the command the policy receives a
    hidden latency after it was issued."""
    w, x, y, z = obs["base_quat_wxyz"]
    heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # yaw, as acceptance.py computes it
    vx, vy = obs["base_lin_vel_m_s"][:2]
    return {
        "t_s": obs["t_s"],
        "position_m": obs["base_pos_m"].tolist(),
        "heading_rad": heading,
        "forward_speed_m_s": float(vx * math.cos(heading) + vy * math.sin(heading)),
        "sideways_speed_m_s": float(-vx * math.sin(heading) + vy * math.cos(heading)),  # positive to the left
        "yaw_rate_rad_s": float(obs["base_ang_vel_rad_s"][2]),
        "foot_contact": dict(zip(FEET, obs["foot_contact"].tolist())),
        "fallen": obs["fallen"],
        "command": command,
    }


class Simulation(threading.Thread):
    """Roams one Go2World at a time: one continuing episode, advanced one control period per step at the pace of the
    wall clock, with the latest drive command. A fall ends the episode and a fresh one starts after
    app.fall_restart_pause_s. Every episode starts on the zero command, so the dog first stands.

    Requests (submit): ("drive", command) from the next control period on; ("reset", preset or None) ends the episode
    and starts a fresh one, on a new world if the preset differs. Status: resetting, roaming, fallen, or failed after
    an internal error (logged with its traceback; the thread then stops).
    """

    def __init__(self, config: dict):
        super().__init__(name="simulation")
        self.config = copy.deepcopy(config)
        self.config["world"]["save_traces"] = False  # roaming never writes trace files
        self.app = self.config["app"]
        self.seed = self.config["runtime"]["seed"]
        self.schema = self.config["robot"]["walking_policy"]["command_schema"]
        self.period_s = self.schema["update_period_s"]
        self.frame_every = periods_per(self.app["stream_fps"], self.period_s, "stream_fps")
        self.observation_every = periods_per(self.app["observation_hz"], self.period_s, "observation_hz")
        self.zero = dict.fromkeys(self.schema["fields"], 0.0)
        self.command = self.zero
        self.requests = queue.Queue()
        self.ready = threading.Event()  # set once the first episode runs, or once the thread has stopped
        self.changed = threading.Condition()
        # Guarded by self.changed and written by this thread only:
        self.status, self.preset, self.available = "resetting", None, []
        self.frame, self.frame_n = b"", 0
        self.latest = {}  # name -> (count, data) of the events only the newest of which matters: observation, infer
        # shortcut: grows by one entry per episode start, end, notice, run line or inference (a few hundred bytes);
        # trim it if the server ever has to run for days. Also written by announce() from other threads.
        self.event_log = []
        self.stopped = False

    def submit(self, kind: str, value=None) -> Future:
        """Hand this thread a request; the future completes once the request has been carried out."""
        future = Future()
        with self.changed:
            if self.stopped:
                future.set_exception(RuntimeError("the simulation thread has stopped"))
            else:
                self.requests.put((kind, value, future))
        return future

    def stop(self) -> None:
        """Stop the thread and wait for it; open frame and event streams then end. Safe to call again."""
        self.requests.put(("stop", None, Future()))
        self.join()

    def announce(self, name: str, data: dict, latest_only: bool = False) -> None:
        """Add an event from another thread to the event streams: logged for every reader (run), or latest_only, where
        a reader gets the newest one (infer, many a second)."""
        with self.changed:
            self._add(name, data, latest_only)
            self.changed.notify_all()

    def _add(self, name: str, data: dict, latest_only: bool) -> None:  # with self.changed held
        if latest_only:
            self.latest[name] = (self.latest.get(name, (0, None))[0] + 1, data)
        else:
            self.event_log.append((name, data))

    def state(self) -> dict:
        with self.changed:
            return {"status": self.status, "preset": self.preset, "available_presets": list(self.available)}

    def frames(self):
        """Each PNG frame as it is published, starting with the latest; ends when the thread stops."""
        seen = 0
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.frame_n != seen or self.stopped)
                if self.stopped:
                    return
                seen, frame = self.frame_n, self.frame
            yield frame

    def events(self):
        """(name, data) for every logged event (episode, notice, run) from now on and the latest observation and infer
        event after them; a reader too slow for every one of those gets the newest. Ends when the thread stops and all
        has been sent."""
        with self.changed:
            logged, seen = len(self.event_log), {name: n for name, (n, _) in self.latest.items()}

        def fresh():
            return [(name, data) for name, (n, data) in self.latest.items() if seen.get(name) != n]

        while True:
            with self.changed:
                self.changed.wait_for(lambda: len(self.event_log) > logged or fresh() or self.stopped)
                new = self.event_log[logged:] + fresh()
                logged = len(self.event_log)
                seen = {name: n for name, (n, _) in self.latest.items()}
            if not new:
                return
            yield from new

    def run(self) -> None:
        try:
            self._open_worlds()
            self._start_episode()
            self.ready.set()
            self._roam()
        except Exception:
            log.exception("The simulation thread stopped on an error")
            self._publish(("notice", {"code": "simulation_failed",
                                      "message": "The simulation stopped after an internal error; restart the server."}),
                          status="failed")
        finally:
            self.world = None  # its renderer is closed here, on the thread that made it (mujoco.Renderer.__del__)
            with self.changed:
                self.stopped = True
                self.changed.notify_all()
            self.ready.set()
            while not self.requests.empty():  # submit() adds nothing once stopped is set
                self.requests.get()[2].set_exception(RuntimeError("the simulation thread has stopped"))

    def _open_worlds(self) -> None:
        """Build the starting world (Go2World's default preset) and decide, once, which presets can be built."""
        self.world = Go2World(self.config, self.seed)
        available = []
        for name in self.config["world"]["presets"]:
            if name != self.world.preset:
                try:
                    Go2World(self.config, self.seed, name)  # dropped at once: only whether it builds matters
                except Exception:
                    log.warning("Preset %r cannot be built; it is offered as unavailable", name, exc_info=True)
                    continue
            available.append(name)
        with self.changed:
            self.preset, self.available = self.world.preset, available

    def _roam(self) -> None:
        self.due = time.perf_counter()  # wall-clock time of the next control period, or of the end of a fall's pause
        while True:
            try:
                kind, value, future = self.requests.get(timeout=max(0.0, self.due - time.perf_counter()))
            except queue.Empty:  # due: the next control period, or a fresh episode after a fall
                if self.status == "fallen":
                    self._start_episode()
                else:
                    self._step()
                continue
            if kind == "stop":
                future.set_result(None)
                return
            try:
                self._serve(kind, value)
            except BaseException as e:
                future.set_exception(e)
                raise
            future.set_result(None)

    def _serve(self, kind: str, value) -> None:
        if kind == "drive":
            self.command = value
        elif kind == "reset":
            if self.status == "roaming":
                self._end_episode("resetting")
            if value is not None and value != self.world.preset:
                self.world = None  # the old world's renderer is closed here, on its own thread
                self.world = Go2World(self.config, self.seed, value)
                with self.changed:
                    self.preset = value
            self._start_episode()
        else:
            raise ValueError(f"unknown request {kind!r}")

    def _start_episode(self) -> None:
        with self.changed:
            self.status = "resetting"
        obs = self.world.reset()
        self.n = 0  # control periods since the decision instant
        self._publish(("episode", {"phase": "start", "preset": self.world.preset}), obs=obs, frame=self._frame(),
                      status="roaming")
        self.due = time.perf_counter()

    def _step(self) -> None:
        obs = self.world.step(self.command, self.period_s)
        self.n += 1
        self.due += self.period_s
        if obs["fallen"]:
            self._publish(obs=obs)
            self._end_episode("fallen")
            self._publish(("notice", {"code": "fell", "message": "The dog fell over; a fresh episode starts shortly."}))
            self.due = time.perf_counter() + self.app["fall_restart_pause_s"]
            return
        self._publish(obs=obs if self.n % self.observation_every == 0 else None,
                      frame=self._frame() if self.n % self.frame_every == 0 else None)

    def _end_episode(self, status: str) -> None:
        outcome = self.world.finish_episode()
        del outcome["commands_received"]  # stays on the server: the page sees each command in the observations
        del outcome["trace_path"]  # a server-side file path, and roaming writes no trace anyway
        self.command = self.zero
        self._publish(("episode", {"phase": "end", "preset": self.world.preset, "outcome": outcome}), status=status)

    def _frame(self) -> bytes:
        return encode_png(self.world.render_frame(), self.app["png_compression_level"])

    def _publish(self, *events, obs=None, frame=None, status=None) -> None:
        """Publish events, an observation, a frame and a status together, so no reader sees one without the others."""
        observation = None if obs is None else describe(obs, self.command)
        with self.changed:
            self.event_log.extend(events)
            if observation is not None:
                self._add("observation", observation, latest_only=True)
            if frame is not None:
                self.frame, self.frame_n = frame, self.frame_n + 1
            if status is not None:
                self.status = status
            self.changed.notify_all()
