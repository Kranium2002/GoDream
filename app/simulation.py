"""The viewer's simulation thread: the only thread that touches the roaming Go2World, and the stage the page shows.

Other threads hand it requests through submit() and read what it publishes, all guarded by one condition: its
status, the latest PNG frame, the latest observation, pose and scene and the episode and notice events.

The stage is the roaming world unless the episode player (app/player.py) holds it: then roaming pauses where it is
and the player publishes its own world's observations, poses, scenes and frames through publish_stage and offer_frame
until it releases the stage. Stream frames are drawn by FrameRenderer, a thread of its own, from the newest state it
was offered, so rendering never holds up stepping or pose events.
"""
import copy
import logging
import math
import queue
import struct
import threading
import time
import zlib
from collections import deque
from concurrent.futures import Future

import mujoco
import numpy as np

from app import robot
from app.routes import to_json
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


def stream_settings(config: dict) -> dict:
    """config.yaml app.stream checked: {width, height, shadows}; ValueError naming the key otherwise. The size must fit
    the model's offscreen buffer, world.render's width x height."""
    stream, buffer = config["app"].get("stream") or {}, config["world"]["render"]
    for key, limit in (("width", buffer["width"]), ("height", buffer["height"])):
        value = stream.get(key)
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"config.yaml app.stream.{key} must be a whole number of pixels from 1 to {limit} (the "
                             f"offscreen buffer), got {value!r}")
    if type(stream.get("shadows")) is not bool:
        raise ValueError(f"config.yaml app.stream.shadows must be true or false, got {stream.get('shadows')!r}")
    return {"width": stream["width"], "height": stream["height"], "shadows": stream["shadows"]}


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


class FrameRenderer(threading.Thread):
    """Draws the stream's frames from the newest stage state offered (offer, from any thread): the state is copied
    (qpos and the mocap poses, all a frame needs) into a one-slot mailbox, so while a frame is being drawn a newer
    offer replaces an older one (counted in skipped) and no frame ever queues. The frame is MuJoCo's tracking camera
    on the trunk at world.render.tracking_distance_m, as Go2World.render_frame draws it, at app.stream's size, with
    app.stream.shadows; the pose comes from a private MjData per model (mj_kinematics, mj_comPos, mj_camlight).
    publish(png) gets each frame. Its OpenGL contexts live and die on this thread."""

    def __init__(self, stream: dict, distance_m: float, level: int, publish):
        super().__init__(name="stream-render", daemon=True)
        self.stream, self.distance_m, self.level, self.publish = stream, distance_m, level, publish
        self._lock, self._wake, self._stopping = threading.Lock(), threading.Event(), False
        self._slot = None
        self.offered = self.skipped = self.rendered = 0
        self.ms = deque(maxlen=200)  # wall ms per frame, render and PNG encoding, the newest 200

    def offer(self, model, data) -> None:
        snapshot = (model, data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy())
        with self._lock:
            self.offered += 1
            if self._slot is not None:
                self.skipped += 1
            self._slot = snapshot
        self._wake.set()

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self.is_alive():
            self.join(timeout=10)

    def stats(self) -> dict:
        with self._lock:
            ms = sorted(self.ms)
            return {"offered": self.offered, "skipped": self.skipped, "rendered": self.rendered,
                    "ms_median": ms[len(ms) // 2] if ms else None}

    def run(self) -> None:
        drawn = {}  # model -> (renderer, scratch MjData, camera), at most a few: the stage switches scenes
        try:
            while True:
                self._wake.wait()
                self._wake.clear()
                if self._stopping:
                    return
                with self._lock:
                    snapshot, self._slot = self._slot, None
                if snapshot is None:
                    continue
                started = time.perf_counter()
                model, qpos, mocap_pos, mocap_quat = snapshot
                if model not in drawn:
                    if len(drawn) >= 4:
                        oldest = next(iter(drawn))
                        drawn.pop(oldest)[0].close()
                    renderer = mujoco.Renderer(model, height=self.stream["height"], width=self.stream["width"])
                    camera = mujoco.MjvCamera()
                    mujoco.mjv_defaultFreeCamera(model, camera)
                    camera.type, camera.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, model.body("base_link").id
                    camera.distance = self.distance_m
                    drawn[model] = renderer, mujoco.MjData(model), camera
                renderer, scratch, camera = drawn[model]
                scratch.qpos[:], scratch.mocap_pos[:], scratch.mocap_quat[:] = qpos, mocap_pos, mocap_quat
                mujoco.mj_kinematics(model, scratch)
                mujoco.mj_comPos(model, scratch)
                mujoco.mj_camlight(model, scratch)
                renderer.update_scene(scratch, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.stream["shadows"]
                png = encode_png(renderer.render(), self.level)
                with self._lock:
                    self.rendered += 1
                    self.ms.append((time.perf_counter() - started) * 1000.0)
                self.publish(png)
                if self.rendered % 200 == 0:
                    stats = self.stats()
                    log.info("Stream frames: %d drawn, %d offers skipped while busy, median %.0f ms a frame (newest "
                             "200; %dx%d, shadows %s)", stats["rendered"], stats["skipped"], stats["ms_median"],
                             self.stream["width"], self.stream["height"], self.stream["shadows"])
        except Exception:
            log.exception("The stream renderer stopped on an error; the stream shows its last frame")
        finally:
            for renderer, _, _ in drawn.values():
                renderer.close()


class Simulation(threading.Thread):
    """Roams one Go2World at a time: one continuing episode, advanced one control period per step at the pace of the
    wall clock, with the latest drive command. A fall ends the episode and a fresh one starts after
    app.fall_restart_pause_s. Every episode starts on the zero command, so the dog first stands.

    Requests (submit): ("drive", command) from the next control period on; ("reset", preset or None) ends the episode
    and starts a fresh one, on a new world if the preset differs; ("hold", token) pauses roaming where it is, for the
    episode player, until ("release", token) with the same token. Status: resetting, roaming, fallen, or failed after
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
        self.pose_every = periods_per(self.app["pose_hz"], self.period_s, "pose_hz")
        self.stream = stream_settings(self.config)
        self.renderer = FrameRenderer(self.stream, self.config["world"]["render"]["tracking_distance_m"],
                                      self.app["png_compression_level"], self._new_frame)
        self.described = None  # the roaming model robot_text last described (this thread only)
        # GET /api/robot's JSON (robot_text) and its version, and the description without version (robot_body) to
        # tell whether another model draws the same; built once per model; guarded
        self.robot_text, self.robot_body, self.robot_version = None, None, 0
        self.zero = dict.fromkeys(self.schema["fields"], 0.0)
        self.command = self.zero
        self.requests = queue.Queue()
        self.ready = threading.Event()  # set once the first episode runs, or once the thread has stopped
        self.changed = threading.Condition()
        self.holder = None  # the episode player's token while it holds the stage (this thread only)
        # Guarded by self.changed and written by this thread only:
        self.status, self.preset, self.available = "resetting", None, []
        self.frame, self.frame_n = b"", 0  # written by the renderer thread
        self.stream_readers = 0  # open /api/stream responses: frames are drawn only while there is one
        # name -> (count, data) of the events only the newest of which matters: observation, pose, scene, infer
        self.latest = {}
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

    def robot_json(self) -> str | None:
        """GET /api/robot's JSON text for the stage's current model, or None before the first episode."""
        with self.changed:
            return self.robot_text

    def state(self) -> dict:
        with self.changed:
            return {"status": self.status, "preset": self.preset, "available_presets": list(self.available)}

    def frames(self):
        """Each PNG frame as it is published, starting with the latest; ends when the thread stops. Frames are drawn
        only while at least one of these is open."""
        seen = 0
        with self.changed:
            self.stream_readers += 1
        try:
            while True:
                with self.changed:
                    self.changed.wait_for(lambda: (self.frame_n != seen and self.frame) or self.stopped)
                    if self.stopped:
                        return
                    seen, frame = self.frame_n, self.frame
                yield frame
        finally:
            with self.changed:
                self.stream_readers -= 1

    def events(self):
        """(name, data) for every logged event (episode, notice, run) from now on and the latest observation, pose,
        scene and infer event after them; a reader too slow for every one of those gets the newest. Ends when the
        thread stops and all has been sent."""
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

    # ---- the stage, for the episode player (any thread) ----------------------------------------------------------

    def show_model(self, model) -> list:
        """Make model the one GET /api/robot describes. When its description differs from the current one, the version
        goes up and a ("robot", {reason, version}) event is returned for the caller to publish; else []."""
        started = time.perf_counter()
        described = robot.describe(model)
        body = to_json(described)
        with self.changed:
            if body == self.robot_body:
                return []
            version = self.robot_version + 1
            self.robot_body, self.robot_version = body, version
            self.robot_text = to_json(described | {"version": version})
        log.info("Robot description %d: %d bytes in %.0f ms", version, len(body), (time.perf_counter() - started) * 1000)
        return [("robot", {"reason": "world_rebuilt", "version": version})]

    def publish_stage(self, events=(), observation=None, pose=None, scene=None) -> None:
        """The episode player's events (logged), observation, pose ({t_s, pos, quat}; the version is added here) and
        scene, published together."""
        with self.changed:
            self.event_log.extend(events)
            if observation is not None:
                self._add("observation", observation, latest_only=True)
            if pose is not None:
                self._add("pose", pose | {"version": self.robot_version}, latest_only=True)
            if scene is not None:
                self._add("scene", scene, latest_only=True)
            self.changed.notify_all()

    def offer_frame(self, model, data) -> None:
        """A stage state for the stream; drawn when the renderer is free, else replaced by a newer one; nothing while
        no /api/stream response is open."""
        if self.stream_readers:
            self.renderer.offer(model, data)

    def _new_frame(self, png: bytes) -> None:  # the renderer thread
        with self.changed:
            self.frame, self.frame_n = png, self.frame_n + 1
            self.changed.notify_all()

    # ---- this thread ---------------------------------------------------------------------------------------------

    def run(self) -> None:
        self.renderer.start()
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
            self.renderer.stop()
            self.world = None
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
            timeout = None if self.holder is not None else max(0.0, self.due - time.perf_counter())
            try:
                kind, value, future = self.requests.get(timeout=timeout)
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
                self.world = None
                self.world = Go2World(self.config, self.seed, value)
                with self.changed:
                    self.preset = value
            self._start_episode()
        elif kind == "hold":
            self.holder = value
        elif kind == "release":
            if self.holder is not None and self.holder == value:
                self.holder = None
                self.described = None  # the player may have shown its own model: describe ours again if it differs
                events = self._describe_world()
                self._publish(*events, obs=self.last_obs, pose=self.last_obs["t_s"])  # the roaming dog, where it paused
                self.offer_frame(self.world.model, self.world.policy.data)
                self.due = time.perf_counter()
        else:
            raise ValueError(f"unknown request {kind!r}")

    def _describe_world(self) -> list:
        if self.world.model is self.described:
            return []
        self.described = self.world.model  # a new world, or a scene with an object: the page redraws if it differs
        return self.show_model(self.world.model)

    def _start_episode(self) -> None:
        with self.changed:
            self.status = "resetting"
        obs = self.last_obs = self.world.reset()
        self.n = 0  # control periods since the decision instant
        events = [*self._describe_world(), ("episode", {"phase": "start", "preset": self.world.preset})]
        self.offer_frame(self.world.model, self.world.policy.data)
        self._publish(*events, obs=obs, pose=obs["t_s"], status="roaming")
        self.due = time.perf_counter()

    def _step(self) -> None:
        obs = self.last_obs = self.world.step(self.command, self.period_s)
        self.n += 1
        self.due += self.period_s
        if obs["fallen"]:
            self._publish(obs=obs)
            self._end_episode("fallen")
            self._publish(("notice", {"code": "fell", "message": "The dog fell over; a fresh episode starts shortly."}))
            self.due = time.perf_counter() + self.app["fall_restart_pause_s"]
            return
        if self.n % self.frame_every == 0:
            self.offer_frame(self.world.model, self.world.policy.data)
        self._publish(obs=obs if self.n % self.observation_every == 0 else None,
                      pose=obs["t_s"] if self.n % self.pose_every == 0 else None)

    def _end_episode(self, status: str) -> None:
        outcome = self.world.finish_episode()
        del outcome["commands_received"]  # stays on the server: the page sees each command in the observations
        del outcome["trace_path"]  # a server-side file path, and roaming writes no trace anyway
        self.command = self.zero
        self._publish(("episode", {"phase": "end", "preset": self.world.preset, "outcome": outcome}), status=status)

    def _publish(self, *events, obs=None, pose=None, status=None) -> None:
        """Publish events, an observation, the bodies' poses and a status together, so no reader sees one without the
        others. pose is the episode time to stamp the bodies' poses with, or None for none."""
        observation = None if obs is None else describe(obs, self.command)
        poses = None if pose is None else robot.pose(self.world.policy.data, pose)
        with self.changed:
            self.event_log.extend(events)
            if observation is not None:
                self._add("observation", observation, latest_only=True)
            if poses is not None:
                self._add("pose", poses | {"version": self.robot_version}, latest_only=True)
            if status is not None:
                self.status = status
            self.changed.notify_all()
