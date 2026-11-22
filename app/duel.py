"""The real-run demos, paired live episodes and run-backed story data."""
from __future__ import annotations

import copy
import json
import logging
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mujoco
import numpy as np

from app import backend, robot, showcase
from app.routes import Refusal, route
from learned_judgment.env import Go2World

log = logging.getLogger(__name__)
_DEMO_SKILLS = {"hills": "reflex", "dodge": "dodge_vision"}
DEMOS = tuple(_DEMO_SKILLS)  # GET /api/demos order; each has config.yaml app.demos.<demo>


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _run_root(config):
    root, missing = backend.runs_dir(config)
    return root, missing


def _finished(result):
    if not isinstance(result, dict):
        return False
    return any(isinstance(value, dict) and value.get("state") == "done"
               for key, value in result.items() if key not in ("metrics", "paired"))


def _finished_time(folder, result):
    times = [result.get("updated_utc")]
    events = folder / "events.jsonl"
    if events.is_file():
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
                if event.get("state") == "done":
                    times.append(event.get("time_utc"))
            except ValueError:
                continue
    parsed = []
    for value in times:
        if isinstance(value, str):
            try: parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
            except ValueError: pass
    return max(parsed, default=(folder / "result.json").stat().st_mtime)


def _run_for(root, skill):
    if root is None or not root.is_dir():
        return None, None, None
    candidates = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        result = _read(folder / "result.json")
        if not _finished(result):
            continue
        declared = _read(folder / "declaration.json") or {}
        student = folder / "student" / "student.json"
        skill_name = result.get("skill") or ((declared.get("resolved") or {}).get("spec"))
        if skill_name == skill:
            candidates.append((folder, result, declared, student.is_file()))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda item: _finished_time(item[0], item[1]), reverse=True)
    folder, result, declared, has_student = candidates[0]
    return folder, result, declared


def _newest_student(root, skill):
    if root is None or not root.is_dir():
        return None
    found = []
    for path in root.glob("*/student/student.json"):
        if (_read(path) or {}).get("skill") == skill:
            found.append(path.parent)
    return max(found, key=lambda folder: max((p.stat().st_mtime for p in folder.rglob("*") if p.is_file()),
                                             default=folder.stat().st_mtime), default=None)


def _student_for(root, skill, config, finished_folder=None):
    configured = ((config.get("app") or {}).get("infer") or {}).get("student_dir") if isinstance(config, dict) else None
    if configured:
        path = backend.ROOT / configured
        data = _read(path / "student.json")
        if data and data.get("skill") == skill:
            return path
    if finished_folder is not None and (finished_folder / "student" / "student.json").is_file():
        data = _read(finished_folder / "student" / "student.json") or {}
        if data.get("skill") == skill:
            return finished_folder / "student"
    return None


def _reflex_handles(lesson):
    world = lesson.get("practice_world") or {}
    result = []
    for key, label, unit, kind in (
        ("radius_m", "Mound radius", "m", "amount"), ("height_m", "Mound height", "m", "amount"),
        ("centre_ahead_m", "Mound distance", "m", "amount")):
        span = world.get(key)
        if isinstance(span, dict) and all(isinstance(span.get(k), (int, float)) for k in ("min", "max")):
            result.append({"name": key, "kind": kind, "label": label, "unit": unit,
                           "min": span["min"], "max": span["max"], "step": 0.05,
                           "default": (span["min"] + span["max"]) / 2})
    return result


def _dodge_handles(config):
    declaration = ((config.get("skills") or {}).get("dodge_vision") or {})
    ranges = declaration.get("scenarios") or {}
    result = []
    for key, label, unit, source in (("distance_m", "Punch starts", "m", ranges.get("distance_m")),
                                     ("speed_m_s", "Punch speed", "m/s",
                                      (declaration.get("punch") or {}).get("speed_m_s"))):
        value = source
        if isinstance(value, (list, tuple)) and len(value) == 2:
            result.append({"name": key, "kind": "amount", "label": label, "unit": unit,
                           "min": value[0], "max": value[1], "step": 0.05,
                           "default": (value[0] + value[1]) / 2})
    return result


def _demo(config, demo):
    root, missing = _run_root(config)
    skill = _DEMO_SKILLS[demo]
    folder, result, declaration = _run_for(root, skill)
    skill_declaration = (declaration or {}).get("lesson") if skill == "reflex" else None
    if skill_declaration is None and folder is not None:
        skill_declaration = (_read(folder / "config.yaml") or {}).get("skills", {}).get(skill, {}).get("declaration")
    skill_declaration = skill_declaration or {}
    student_folder = _student_for(root, skill, config, folder)
    student_ok = student_folder is not None
    run_id = folder.name if folder else None
    metrics = result.get("metrics") if isinstance(result, dict) else None
    headline = None
    if isinstance(metrics, list) and metrics:
        metric = metrics[0]
        values = metric.get("values") or {}
        methods = ("no_op", "snn") if skill == "reflex" else ("no_op", "snn")
        before = values.get(methods[0])
        after = values.get(methods[1])
        if isinstance(before, dict) and isinstance(after, dict):
            headline = {"metric_label": metric.get("name"), "unit": metric.get("unit"),
                        "before": before.get("value"), "after": after.get("value"),
                        "n": min(before.get("n", 0), after.get("n", 0))}
    if skill == "reflex":  # the keys of reflex.run_episode's outcome, sent in duel events as sides.<lane>.outcome
        outcomes = [{"name": "crossed_goal_line", "kind": "flag", "label": "Reached the finish line"},
                    {"name": "fall", "kind": "flag", "label": "Fell"},
                    {"name": "end_s", "kind": "seconds", "label": "Time"}]
        handles = _reflex_handles(skill_declaration)
    else:
        outcomes = [{"name": "collision", "kind": "flag", "label": "Collision"},
                    {"name": "fall", "kind": "flag", "label": "Fell"},
                    {"name": "min_separation_m", "kind": "metres", "label": "Closest approach"},
                    {"name": "recovery_time_s", "kind": "seconds", "label": "Recovery time"}]
        run_config = _read_yaml(folder / "config.yaml") if folder else config
        handles = _dodge_handles(run_config or config)
        if student_folder is not None and student_folder.parent != folder:
            headline = None  # those test numbers belong to another run's student than the one in the lane
    words = showcase.demo_settings(config, demo)
    reasons = []
    if root is None:
        reasons.extend(missing)
    elif folder is None:
        reasons.append(f"No finished {skill} run with a declaration exists under observed.runs_dir.")
    elif not student_ok:
        reasons.append(f"No exported {skill} student/student.json exists under observed.runs_dir.")
    return {"id": demo, "name": words["name"], "title": words["title"], "tagline": words["tagline"],
            "lanes": words["lanes"], "skill": skill, "run_id": run_id,
            "headline": headline, "handles": handles, "outcome_fields": outcomes,
            "sense": "feel" if demo == "hills" else "eye", "live_camera": demo == "dodge",
            "available": bool(folder and student_ok), "missing": reasons,
            "showcase": showcase.build(skill, words, folder, student_folder, root)}


def _read_yaml(path):
    try:
        from learned_judgment.config import load_config
        return load_config(path)
    except Exception:
        return None


@route("GET", "/api/demos")
def demos(h):
    h.send_json(200, [_demo(h.server.config, name) for name in DEMOS])


class _DuelWorld:
    """Go2World facade emitting the actual observations and pacing every control period."""
    def __init__(self, world, manager, side, camera=None, visual=None, yaw_trim_rad_s=0.0):
        self.world, self.manager, self.side, self.camera, self.visual = world, manager, side, camera, visual
        self.yaw_trim_rad_s = float(yaw_trim_rad_s)
        self.config = world.config

    def __getattr__(self, key):
        return getattr(self.world, key)

    def reset(self, situation=None, attempt_seed=None):
        obs = self.world.reset(situation, attempt_seed=attempt_seed)
        self.manager.show(self.side, obs, self.world, True)
        return obs

    def step(self, command, duration_s, on_period=None):
        if self.yaw_trim_rad_s and "yaw_rate" in command:
            command = dict(command)
            command["yaw_rate"] = float(np.clip(float(command["yaw_rate"]) + self.yaw_trim_rad_s, -1.0, 1.0))
        def cb(obs):
            if on_period:
                on_period(obs)
            self.manager.show(self.side, obs, self.world, False, self.camera, self.visual)
            self.manager.pace(obs["t_s"])
        return self.world.step(command, duration_s, on_period=cb)

    def finish_episode(self):
        return self.world.finish_episode()


class Duel:
    def __init__(self, server):
        self.server = server
        self.lock = threading.RLock()
        self.active = False
        self.stop_event = threading.Event()
        self.seed = None
        self.started = 0.0
        self.outcomes = {"before": None, "after": None}
        self.status = {"before": "done", "after": "done"}
        self.clock_origin = 0.0
        self.scene_helpers = {}
        self.camera_lock = threading.RLock()
        self.side_ticks = {"before": 0, "after": 0}
        self.next_dodge_seed = None  # the seed of the next dodge round started without one

    def show(self, side, obs, world, reset=False, camera=None, visual=None):
        if reset:
            self.side_ticks[side] = 0
        else:
            self.side_ticks[side] += 1
            if self.side_ticks[side] % self.server.sim.pose_every:
                return
        d = world.policy.data
        if reset:  # reset placed the object by writing qpos only: derive its position, else it shows where it was parked
            mujoco.mj_kinematics(world.model, d)
        pose = robot.pose(d, obs["t_s"]) | {"side": side, "version": self.server.sim.robot_version}
        self.server.sim.announce("pose", pose, latest_only=True)
        from app.player import StageWorld
        helper = self.scene_helpers.get(world.model)
        if helper is None:
            helper = object.__new__(StageWorld)
            helper._robot_root = world.model.body_rootid[world.model.body("base_link").id]
            helper._objects = {}
            self.scene_helpers[world.model] = helper
        scene = {"side": side, "objects": StageWorld._scene_objects(helper, world.model, d), "markers": []}
        self.server.sim.announce("scene", scene, latest_only=True)
        q = obs["base_quat_wxyz"]
        w, x, y, z = map(float, q)
        nose = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        self.server.sim.announce("feel", {"side": side, "t_s": obs["t_s"], "tilt_deg": [nose, roll],
            "turn_rate": [float(v) for v in obs["base_ang_vel_rad_s"]],
            "feet": [int(v) for v in obs["foot_contact"]]}, latest_only=True)

    def pace(self, t_s):
        deadline = self.clock_origin + float(t_s)
        delay = deadline - time.perf_counter()
        if delay > 0 and self.stop_event.wait(delay):
            raise RuntimeError("duel stopped")
        if self.stop_event.is_set():
            raise RuntimeError("duel stopped")

    def brain(self, t_s, decision_ms, choice, **extra):
        """SSE brain (newest only): one decision of the learned lane's student, timed on this CPU around the policy
        call alone (time.perf_counter), with the name of the action it chose."""
        self.server.sim.announce("brain", {"side": "after", "t_s": float(t_s), "decision_ms": float(decision_ms),
                                           "choice": choice} | extra, latest_only=True)

    def state_event(self, state, demo, params, seed):
        self.server.sim.announce("duel", {"state": state, "demo": demo, "params": params, "seed": seed,
            "sides": {name: {"status": self.status[name], "outcome": self.outcomes[name]} for name in self.status}})

    def stop(self):
        with self.lock:
            if not self.active:
                return False
            self.stop_event.set()
            return True

    def run(self, demo, params, seed, plan):
        before = after = None
        token = getattr(self, "holder", None)
        try:
            before, after = self._worlds(plan)
            if demo == "hills":
                self._run_hills(plan, before, after, seed)
            else:
                self._run_dodge(plan, before, after, seed)
        except Exception:
            log.exception("Duel %s failed", demo)
        finally:
            for world in (before, after):
                if world is not None:
                    try: world.close()
                    except Exception: pass
            with self.lock:
                self.active = False
                self.status = {"before": "done", "after": "done"}
                self.state_event("end", demo, params, seed)
            if token is not None:
                try: self.server.sim.submit("release", token).result(timeout=5)
                except Exception: log.exception("Could not release the stage after a duel")

    def _worlds(self, plan):
        cfg = plan["config"]
        cfg["world"]["save_traces"] = False
        seed = int(cfg["observed"].get("world_seed", cfg["runtime"]["seed"]))
        preset = cfg["observed"].get("world_preset", "default")
        return (Go2World(cfg, seed, preset), Go2World(cfg, seed, preset))

    def _run_hills(self, plan, before, after, seed):
        from learned_judgment.skills import reflex
        cfg = plan["config"]
        scenario = reflex.sample_reflex_scenarios(
            cfg, np.random.default_rng(np.random.SeedSequence([cfg["runtime"]["seed"], 3, int(seed)])), 1)[0]
        scenario["scenario_id"] = seed
        self._apply_reflex_params(scenario, plan["params"])
        student_folder = Path(plan.get("student_folder", plan["folder"] / "student"))
        student = reflex._StudentPolicy(student_folder, cfg)  # checks student.json action_names against the lesson
        names = (_read(student_folder / "student.json") or {}).get("action_names") or []

        def learned(window, state):  # the student's own choice, unchanged; only timed and announced
            started = time.perf_counter()
            action = student(window, state)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            choice = names[action] if isinstance(action, int) and 0 <= action < len(names) else None
            self.brain(after._t(), elapsed_ms, choice)
            return action
        # The walking-alone lane's yaw trim (config.yaml app.duel.hills); the learned lane's commands are its own.
        baseline_trim = float(plan["duel"]["baseline_yaw_trim_rad_s"])
        self.clock_origin = time.perf_counter()
        def go(side, world, policy):
            try:
                trim = baseline_trim if side == "before" else 0.0
                wrapped = _DuelWorld(world, self, side, yaw_trim_rad_s=trim)
                answer = reflex.run_episode(wrapped, scenario, seed, policy)
                self.outcomes[side] = answer["outcome"]
            except Exception:
                log.exception("Hills duel %s side failed", side)
            finally:
                self.status[side] = "done"
        no_op = lambda _window, _state: 0
        a = threading.Thread(target=go, args=("before", before, no_op), daemon=True)
        b = threading.Thread(target=go, args=("after", after, learned), daemon=True)
        self.status = {"before": "running", "after": "running"}
        a.start(); b.start(); a.join(); b.join()

    @staticmethod
    def _apply_reflex_params(scenario, params):
        keys = {"radius_m": "mound_radius_m", "height_m": "mound_height_m",
                "centre_ahead_m": "mound_x_m", "centre_side_m": "mound_y_m"}
        for handle, field in keys.items():
            if handle in params and scenario.get("mound"):
                scenario["mound"][field] = float(params[handle])
                scenario["situation"][field] = float(params[handle])

    def _run_dodge(self, plan, before, after, seed):
        from learned_judgment.skills import dodge_vision
        from learned_judgment.vision import VisionSettings
        cfg = plan["config"]
        scenario = dodge_duel_scenario(cfg, plan["duel"], seed)
        scenario["scenario_id"] = seed
        self._apply_dodge_params(scenario, plan["params"], float(cfg["skills"]["dodge_vision"]["punch"]["aim_x_m"]))
        settings = VisionSettings.from_config(cfg, int(plan["student"]["input"]["frames"]))
        student = plan["runtime"]
        self.clock_origin = time.perf_counter()
        def go(side, world, learned):
            from learned_judgment.skills.dodge_vision import HeadView
            view = HeadView(cfg)
            relay = _EyeView(view, settings, self, side)
            try:
                wrapped = _DuelWorld(world, self, side, relay, scenario.get("visual"))
                if learned:
                    def policy(window, t_s):
                        started = time.perf_counter()
                        out = student.infer(window, t_s)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        name = out.get("action", "NO_OP")
                        self.brain(t_s, elapsed_ms, name, issued=bool(out.get("issued")))
                        return {"command": out["command"], "action": name, "issued": bool(out.get("issued"))}
                else:
                    def policy(_window, _t_s):
                        return {"command": dict(plan["actions"][plan["no_op"]]), "action": "NO_OP"}
                answer = dodge_vision.closed_loop(wrapped, relay, scenario, seed, policy, settings,
                    augment_rng=np.random.default_rng(np.random.SeedSequence([cfg["runtime"]["seed"], 3, int(seed), 1])))
                self.outcomes[side] = answer["measured"] | {"result": result_word(answer["measured"], answer["issued"])}
            except Exception:
                log.exception("Dodge duel %s side failed", side)
            finally:
                relay.close()
                self.status[side] = "done"
        self.status = {"before": "running", "after": "running"}
        a = threading.Thread(target=go, args=("before", before, False), daemon=True)
        b = threading.Thread(target=go, args=("after", after, True), daemon=True)
        a.start(); b.start(); a.join(); b.join()

    @staticmethod
    def _apply_dodge_params(scenario, params, aim_x):
        situation = scenario["situation"]
        if "distance_m" in params and "ball_rel_x_m" in situation:
            distance = math.hypot(situation["ball_rel_x_m"], situation["ball_rel_y_m"])
            scale = float(params["distance_m"]) / distance
            pos, vel = ("ball_rel_x_m", "ball_rel_y_m", "ball_rel_z_m"), ("ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s")
            start, v = np.array([situation[k] for k in pos]), np.array([situation[k] for k in vel])
            situation["ball_rel_x_m"] *= scale
            situation["ball_rel_y_m"] *= scale
            if v[0] < 0:  # re-aim at the point where the drawn path crossed the head plane x = aim_x, at the same speed
                aim = start + v * (aim_x - start[0]) / v[0]
                path = aim - np.array([situation[k] for k in pos])
                for key, value in zip(vel, np.linalg.norm(v) * path / np.linalg.norm(path)):
                    situation[key] = float(value)
        if "speed_m_s" in params and "ball_vel_x_m_s" in situation:
            speed = math.sqrt(sum(float(situation[k]) ** 2 for k in
                                  ("ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s")))
            scale = float(params["speed_m_s"]) / speed
            for key in ("ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s"):
                situation[key] *= scale


def dodge_duel_scenario(cfg, duel, seed):
    """The dodge duel's punch for a round seed: of the scenarios of split duel["split"] whose kind is in duel["kinds"]
    (config app.duel.dodge), in split order, the (seed mod their count)-th; a copy, never chosen by outcome."""
    from learned_judgment.skills import dodge_vision
    split = duel["split"]
    scenarios, _ = dodge_vision._split_plan(cfg, split, int(cfg["observed"]["dodge_vision"][f"{split}_scenarios"]))
    pool = [s for s in scenarios if s["kind"] in duel["kinds"]]
    return copy.deepcopy(pool[int(seed) % len(pool)])


def result_word(measured, issued):
    """A dodge lane's result in a plain word from its measured outcome: a fall first, then a hit; a lane that was not
    hit "Dodged" only if it issued a dodge, else "Not hit" (the punch missed a dog that stood still)."""
    if float(measured["fall"]) > 0.5:
        return "Fell"
    if float(measured["collision"]) > 0.5:
        return "Hit"
    return "Dodged" if issued else "Not hit"


class _EyeView:
    """Serialize Windows camera calls and emit the exact preprocessed frame used by the skill."""
    def __init__(self, view, settings, manager, side):
        self.view, self.settings, self.manager, self.side = view, settings, manager, side

    def frame(self, world, visual=None):
        from learned_judgment.skills.dodge_vision import preprocess_frame
        with self.manager.camera_lock:
            rgb = self.view.frame(world, visual)
        cells = preprocess_frame(rgb, self.settings).reshape(-1).astype(float).tolist()
        self.manager.server.sim.announce("eye", {"side": self.side, "t_s": world._t(),
                                                   "cells": cells}, latest_only=True)
        return rgb

    def close(self):
        self.view.close()


def _manager(server):
    if not hasattr(server, "duel"):
        server.duel = Duel(server)
    return server.duel


def stop_for_stage(server):
    manager = getattr(server, "duel", None)
    if manager is not None and manager.stop():
        thread = getattr(manager, "thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)


@route("POST", "/api/duel/start")
def start_duel(h):
    body = h.read_body()
    if set(body) - {"demo", "params", "seed"} or body.get("demo") not in _DEMO_SKILLS:
        raise Refusal(400, "invalid_body", "A duel needs demo, params, and an optional integer seed.")
    demo, params, seed = body["demo"], body.get("params", {}), body.get("seed")
    if not isinstance(params, dict) or (seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))):
        raise Refusal(400, "invalid_body", "Duel params must be an object and seed must be an integer or null.")
    info = _demo(h.server.config, demo)
    if not info["available"]:
        raise Refusal(409, "demo_unavailable", "; ".join(info["missing"]) or "The demo is unavailable.")
    handles = {item["name"]: item for item in info["handles"]}
    if set(params) - set(handles):
        raise Refusal(400, "invalid_params", "Duel params must use handles listed by this demo.")
    for name, value in params.items():
        handle = handles[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not handle["min"] <= value <= handle["max"]:
            raise Refusal(400, "invalid_params", f"{name} must be a finite number from {handle['min']} to {handle['max']}.")
    manager = _manager(h.server)
    h.server.player.stop()
    with manager.lock:
        if manager.active:
            raise Refusal(409, "duel_active", "A duel is already running.")
        folder = backend.runs_dir(h.server.config)[0] / info["run_id"]
        cfg = _read_yaml(folder / "config.yaml") or h.server.config
        student_folder = _student_for(backend.runs_dir(h.server.config)[0], _DEMO_SKILLS[demo],
                                      h.server.config, folder)
        if student_folder is not None:
            cfg = _read_yaml(student_folder.parent / "config.yaml") or cfg
        if demo == "hills":
            student_file = student_folder / "student.json" if student_folder else folder / "student" / "student.json"
            if not student_file.is_file():
                raise Refusal(409, "student_unavailable", f"Run {folder.name} has no exported student/student.json.")
            plan = {"folder": folder, "student_folder": student_file.parent, "config": cfg, "params": params,
                    "duel": h.server.config["app"]["duel"]["hills"]}
        else:
            student_folder = student_folder or folder / "student"
            student_data = _read(student_folder / "student.json") or {}
            try:
                from learned_judgment.runtime import DodgeRuntime
                runtime = DodgeRuntime.load(student_folder, cfg)
                from learned_judgment.skills.dodge_vision import _actions
                actions = _actions(cfg)
                plan = {"folder": folder, "student_folder": student_folder, "config": cfg, "params": params, "student": student_data,
                        "runtime": runtime, "actions": actions.commands, "no_op": actions.no_op,
                        "duel": h.server.config["app"]["duel"]["dodge"]}
            except Exception as exc:
                raise Refusal(409, "student_unavailable", f"The saved dodge student cannot run: {exc}") from None
            if seed is None:  # each new dodge round takes the next seed, so rounds walk through the punches in order
                seed = manager.next_dodge_seed if manager.next_dodge_seed is not None else h.server.config["runtime"]["seed"]
            manager.next_dodge_seed = int(seed) + 1
        actual_seed = int(seed if seed is not None else h.server.config["runtime"]["seed"])
        manager.active, manager.stop_event, manager.seed = True, threading.Event(), actual_seed
        manager.outcomes = {"before": None, "after": None}
        manager.status = {"before": "running", "after": "running"}
        manager.holder = f"duel-{time.monotonic_ns()}"
        h.server.sim.submit("hold", manager.holder).result()
        manager.state_event("start", demo, params, actual_seed)
        thread = threading.Thread(target=manager.run, args=(demo, params, actual_seed, plan), name="duel", daemon=True)
        manager.thread = thread
        thread.start()
    h.send_json(202, {"seed": actual_seed})


@route("POST", "/api/duel/stop")
def stop_duel(h):
    stop_for_stage(h.server)
    h.send_json(200, {"stopping": True})


@route("GET", "/api/story/*")
def story(h):
    run_id = unquote(urlsplit(h.path).path.rsplit("/", 1)[-1])
    from app.runs import RUN_ID
    root, missing = backend.runs_dir(h.server.config)
    if root is None:
        raise Refusal(503, "runs_unavailable", f"Run stories need {'; '.join(missing)}.")
    if not RUN_ID.fullmatch(run_id) or not (root / run_id).is_dir():
        raise Refusal(404, "unknown_run", "There is no run with that id.")
    h.send_json(200, _story(root / run_id))


def _story(folder):
    result = _read(folder / "result.json") or {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), list) else None
    paired = result.get("paired") if isinstance(result.get("paired"), list) else None
    story_dir = folder / "story"
    return {"attempts": _read(story_dir / "attempts.json"),
            "what_if": _read(story_dir / "what_if.json"),
            "imagination": _read(story_dir / "imagination.json"),
            "brain": _read(story_dir / "brain.json"),
            "test": {"metrics": metrics, "paired": paired} if metrics is not None or paired is not None else None}
