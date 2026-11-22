"""The real-run demos, paired live episodes and run-backed story data."""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from app import backend, robot
from app.routes import Refusal, route
from learned_judgment.env import Go2World

log = logging.getLogger(__name__)
_DEMO_SKILLS = {"hills": "reflex", "dodge": "dodge_vision"}


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
        ("centre_ahead_m", "Mound distance", "m", "amount"), ("centre_side_m", "Mound side offset", "m", "offset")):
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
    for key, label, unit, source in (("distance_m", "Threat distance", "m", ranges.get("distance_m")),
                                     ("speed_m_s", "Threat speed", "m/s",
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
    if skill == "reflex":
        outcomes = [{"name": "reached", "kind": "flag", "label": "Reached"},
                    {"name": "fall", "kind": "flag", "label": "Fell"},
                    {"name": "end_s", "kind": "seconds", "label": "Time"},
                    {"name": "final_sideways_m", "kind": "metres", "label": "Side offset"}]
        handles = _reflex_handles(skill_declaration)
        title = skill_declaration.get("title") or "Hill course"
        tagline = skill_declaration.get("summary") or skill_declaration.get("goal", {}).get("forward_m")
        tagline = tagline if isinstance(tagline, str) else "A blind walking lesson from the saved run declaration."
    else:
        outcomes = [{"name": "collision", "kind": "flag", "label": "Collision"},
                    {"name": "fall", "kind": "flag", "label": "Fell"},
                    {"name": "min_separation_m", "kind": "metres", "label": "Closest approach"},
                    {"name": "recovery_time_s", "kind": "seconds", "label": "Recovery time"}]
        run_config = _read_yaml(folder / "config.yaml") if folder else config
        handles = _dodge_handles(run_config or config)
        title, tagline = "Visual dodge", "A saved vision student reacts to a sampled threat scenario."
    reasons = []
    if root is None:
        reasons.extend(missing)
    elif folder is None:
        reasons.append(f"No finished {skill} run with a declaration exists under observed.runs_dir.")
    elif not student_ok:
        reasons.append(f"No exported {skill} student/student.json exists under observed.runs_dir.")
    return {"id": demo, "title": title, "tagline": tagline, "skill": skill, "run_id": run_id,
            "headline": headline, "handles": handles, "outcome_fields": outcomes,
            "sense": "feel" if demo == "hills" else "eye", "live_camera": demo == "dodge",
            "available": bool(folder and student_ok), "missing": reasons}


def _read_yaml(path):
    try:
        from learned_judgment.config import load_config
        return load_config(path)
    except Exception:
        return None


@route("GET", "/api/demos")
def demos(h):
    h.send_json(200, [_demo(h.server.config, name) for name in ("hills", "dodge")])


class _DuelWorld:
    """Go2World facade emitting the actual observations and pacing every control period."""
    def __init__(self, world, manager, side, camera=None, visual=None):
        self.world, self.manager, self.side, self.camera, self.visual = world, manager, side, camera, visual
        self.config = world.config

    def __getattr__(self, key):
        return getattr(self.world, key)

    def reset(self, situation=None, attempt_seed=None):
        obs = self.world.reset(situation, attempt_seed=attempt_seed)
        self.manager.show(self.side, obs, self.world, True)
        return obs

    def step(self, command, duration_s, on_period=None):
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

    def show(self, side, obs, world, reset=False, camera=None, visual=None):
        if reset:
            self.side_ticks[side] = 0
        else:
            self.side_ticks[side] += 1
            if self.side_ticks[side] % self.server.sim.pose_every:
                return
        d = world.policy.data
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
        student = reflex._StudentPolicy(plan.get("student_folder", plan["folder"] / "student"), cfg)
        self.clock_origin = time.perf_counter()
        def go(side, world, policy):
            try:
                wrapped = _DuelWorld(world, self, side)
                answer = reflex.run_episode(wrapped, scenario, seed, policy)
                self.outcomes[side] = answer["outcome"]
            except Exception:
                log.exception("Hills duel %s side failed", side)
            finally:
                self.status[side] = "done"
        no_op = lambda _window, _state: 0
        a = threading.Thread(target=go, args=("before", before, no_op), daemon=True)
        b = threading.Thread(target=go, args=("after", after, student), daemon=True)
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
        rng = np.random.default_rng(np.random.SeedSequence([cfg["runtime"]["seed"], 3, int(seed)]))
        scenario = dodge_vision.sample_threat_scenarios(cfg, rng, 1)[0]
        scenario["scenario_id"] = seed
        self._apply_dodge_params(scenario, plan["params"])
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
                        out = student.infer(window, t_s)
                        name = out.get("action", "NO_OP")
                        return {"command": out["command"], "action": name, "issued": bool(out.get("issued"))}
                else:
                    def policy(_window, _t_s):
                        return {"command": dict(plan["actions"][plan["no_op"]]), "action": "NO_OP"}
                answer = dodge_vision.closed_loop(wrapped, relay, scenario, seed, policy, settings,
                    augment_rng=np.random.default_rng(np.random.SeedSequence([cfg["runtime"]["seed"], 3, int(seed), 1])))
                self.outcomes[side] = answer["measured"]
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
    def _apply_dodge_params(scenario, params):
        situation = scenario["situation"]
        if "distance_m" in params and "ball_rel_x_m" in situation:
            distance = math.hypot(situation["ball_rel_x_m"], situation["ball_rel_y_m"])
            scale = float(params["distance_m"]) / distance
            situation["ball_rel_x_m"] *= scale
            situation["ball_rel_y_m"] *= scale
        if "speed_m_s" in params and "ball_vel_x_m_s" in situation:
            speed = math.sqrt(sum(float(situation[k]) ** 2 for k in
                                  ("ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s")))
            scale = float(params["speed_m_s"]) / speed
            for key in ("ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s"):
                situation[key] *= scale


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
            plan = {"folder": folder, "student_folder": student_file.parent, "config": cfg, "params": params}
        else:
            student_folder = student_folder or folder / "student"
            student_data = _read(student_folder / "student.json") or {}
            try:
                from learned_judgment.runtime import DodgeRuntime
                runtime = DodgeRuntime.load(student_folder, cfg)
                from learned_judgment.skills.dodge_vision import _actions
                actions = _actions(cfg)
                plan = {"folder": folder, "student_folder": student_folder, "config": cfg, "params": params, "student": student_data,
                        "runtime": runtime, "actions": actions.commands, "no_op": actions.no_op}
            except Exception as exc:
                raise Refusal(409, "student_unavailable", f"The saved dodge student cannot run: {exc}") from None
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
    rows_folder = folder / "context" if (folder / "context" / "rows.json").is_file() else folder / "dev"
    rows = _read(rows_folder / "rows.json")
    row_count = decision_count = None
    try:
        with np.load(rows_folder / "rows.npz", allow_pickle=False) as archive:
            indices = np.asarray(archive["scenario_index"])
            row_count, decision_count = int(np.unique(indices).size), int(indices.size)
    except (OSError, ValueError, KeyError):
        pass
    dream = _read(folder / "dream" / "dream.json")
    scores_path = folder / "scores.npz"
    scores_exists = scores_path.is_file()
    events = []
    event_path = folder / "events.jsonl"
    if event_path.is_file():
        for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try: events.append(json.loads(line))
            except ValueError: pass
    attempts = None
    episodes = ((rows or {}).get("episodes") or [])
    if rows is not None:
        scenarios = rows.get("scenarios") or []
        outcomes = ([{"kind": x.get("kind"), "ended": (x.get("outcome") or {}).get("end_reason")}
                     for x in episodes] if episodes else None)
        obstacles = [x.get("mound") for x in scenarios if isinstance(x, dict) and x.get("mound") is not None]
        attempts = {"n_tries": row_count if row_count is not None else (len(episodes) if episodes else None),
                    "n_decisions": decision_count if decision_count is not None else
                    (sum(int(x.get("decisions", 0)) for x in episodes) if episodes else None),
                    "paths": None, "outcomes": outcomes, "obstacles": obstacles or None}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), list) else None
    paired = result.get("paired") if isinstance(result.get("paired"), list) else None
    brain = None
    student = _read(folder / "student" / "student.json")
    if student:
        network = _read(folder / "student" / "network.json") or {}
        sizes = network.get("layer_sizes")
        brain = {"inputs": ((student.get("input") or {}).get("features") or
                            (sizes[0] if isinstance(sizes, list) and sizes else None)),
                 "neurons": ((student.get("training_settings") or {}).get("hidden_sizes") or
                             (sizes[1:-1] if isinstance(sizes, list) and len(sizes) > 2 else None)),
                 "choices": student.get("action_names"), "curve": None,
                 "spikes_per_decision": None, "decision_ms": None}
    return {"attempts": attempts, "what_if": None, "imagination": None, "brain": brain,
            "test": {"metrics": metrics, "paired": paired} if metrics is not None or paired is not None else None}
