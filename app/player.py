"""The episode player: a learning run's own episodes replayed on the stage, so the page shows what the run does.

    POST /api/episodes/start {"run_id"}  202 {run_id, skill, step, split, total, note}; 400 invalid_body; 404
                             unknown_run; 422 cannot_play naming why; 503 simulation_stopped. A player already playing
                             is stopped first: one at a time.
    POST /api/episodes/stop  200 {"playing": false}
    GET  /api/state          gains "episodes": {"playing", "run_id", "note"}
    SSE  episode             {state: start | end | idle, run_id, skill (DODGE | RAMP | MOVE), step (attempts | baseline |
                             evaluate), index, total, kind, action, caption, outcome, note}; outcome is null except at
                             an end, where it is what the skill's own episode function measured
    SSE  scene               {objects: [{id, shape, size, rgba, pos, quat}], markers: [{id, shape, pos, to, rgba}]} with
                             every pose event while an episode plays, one empty scene when the player stops. World frame
                             as the poses. objects are the geoms of the played MuJoCo model's bodies that are neither the
                             world's nor the robot's (the thrown object, the ramp and its plateau) at their pose in the
                             played MjData; shape sphere, box or cylinder and size in MuJoCo's geom_size convention, as
                             GET /api/robot: sphere [radius], box [half x, half y, half z], cylinder [radius, half
                             length]; rgba the geom's own. markers are computed from the scenario (MOVE: the target),
                             never simulated: disc at pos, line from pos to "to".

What is played. The run's skill, mode and split come from its files (declaration.json, result.json, rows.json,
student/student.json); its config is its own snapshot (config.yaml in the run folder) or, for a run recorded without
one, the repository's (the caption says which, and whether the run's recorded config digest matches). The scenarios
are the run's saved ones for that split (rows.json or the evaluation's run.json) or, before the run has saved them,
drawn with the same sampler and split seeds its commands use; attempt_seed is the scenario id, as in the run. Each
scenario is played twice in a row: the run's baseline, then
    a run with a trained student (<run>/student): the student deciding through the run's evaluation code path on the
        evaluation split (test): MOVE and RAMP the student's argmax (their run commands' _StudentPolicy); DODGE a
        DodgeRuntime behind its frozen gate in closed_loop (dodge_eval's evaluate), or, while the student has no frozen
        thresholds, the action-head argmax of observed.student_readout at the scenario's decision frame through
        observe_and_act;
    a run without a student yet: on the context split (the run's attempts) the candidates the run executes, cycled
        (full mode: every candidate is run per scenario; fast mode: the scenario's own attempt);
    a bare student folder (no run files, e.g. results/runs/v3-provisional): its student on the dev split.
Baselines: MOVE the physics rule, RAMP the nominal command (the detector observes, never acts), DODGE NO_OP. The
simulator is deterministic, so these are the run's episodes recomputed, not a feed from its background workers; where
the run saved the measured outcome of the same episode, the end event says whether it matches.

Pacing: the player's world is wrapped (StageWorld) so that every control period of the skill's own episode function
publishes the pose, scene, observation and stream frame at the app rates and waits for the wall clock; a reset's settle
runs unpaced and unshown. The drive pad, a reset, a stop and live inference take the stage back (stop the player).
"""
import copy
import json
import logging
import math
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from app import backend
from app.routes import Refusal, route
from app.simulation import describe as describe_observation

log = logging.getLogger(__name__)
SPECS = {"move": "MOVE", "ramp": "RAMP", "dodge_vision": "DODGE"}  # run spec -> the contract's skill name
NOTES = {
    "running": "showing this run's own episodes: same scenarios and seeds, replayed on the stage; the run computes them "
               "in parallel in the background",
    "ended": "showing this run's own episodes: same scenarios and seeds, replayed on the stage; the run computed them "
             "in parallel in the background",
    "student": "showing this student's own decisions on the dev split's scenarios and seeds, replayed on the stage; no "
               "run computes them",
}
SHAPES = {mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_BOX: "box",
          mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}
TARGET_RGBA, HEADING_RGBA = [0.15, 0.75, 0.35, 0.9], [0.6, 0.6, 0.6, 0.7]
MARKER_Z = 0.005  # m above the floor
HEADING_LINE_M = 0.6  # length of a turn's heading lines


class Stopped(Exception):
    """Raised inside an episode when the player is told to stop; the episode is abandoned without an outcome."""


# ---- the run: what to play ----------------------------------------------------------------------------------------

def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _run_spec(folder: Path) -> str | None:
    declaration = _json(folder / "declaration.json") or {}
    for value in ((declaration.get("resolved") or {}).get("spec"), (_json(folder / "result.json") or {}).get("skill"),
                  (_json(folder / "student" / "student.json") or {}).get("skill")):
        if isinstance(value, str):
            return value
    events = folder / "events.jsonl"
    if events.is_file():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                return json.loads(line)["skill"]
            except (ValueError, KeyError, TypeError):
                continue
    return None


def _recorded_digest(folder: Path) -> str | None:
    for path in (folder / "student" / "student.json",):
        digest = ((_json(path) or {}).get("provenance") or {}).get("config_digest")
        if digest:
            return digest
    for path in (folder / "test_run" / "run.json", folder / "context" / "rows.json", folder / "dev" / "rows.json"):
        digest = (_json(path) or {}).get("config_digest")
        if digest:
            return digest
    return None


def _run_config(folder: Path, server_config: dict):
    """(config, sentence saying which) for the run's episodes."""
    from learned_judgment.config import config_digest, load_config
    snapshot = folder / "config.yaml"
    if snapshot.is_file():
        return load_config(snapshot), "config: the run's own snapshot"
    recorded, current = _recorded_digest(folder), config_digest(server_config)
    if recorded is None:
        return server_config, "config: the repository's config.yaml (the run kept no snapshot or digest)"
    same = "matches" if recorded == current else "differs from"
    return server_config, (f"config: the repository's config.yaml, whose digest {same} the one the run recorded (the run "
                           f"kept no snapshot)")


def _keys(spec: str) -> tuple:
    return {"move": ("scenario_id", "kind", "target", "situation"), "ramp": ("scenario_id", "kind", "situation"),
            "dodge_vision": ("scenario_id", "kind", "object_kind", "situation", "visual")}[spec]


def _module(spec: str):
    if spec == "move":
        from learned_judgment.skills import move as module
    elif spec == "ramp":
        from learned_judgment.skills import ramp as module
    else:
        from learned_judgment.skills import dodge_vision as module
    return module


def _sampled(spec: str, config: dict, split: str, n: int):
    """(scenarios, per-scenario (frame seed, augmentation seed) or None) from the skill's own split plan."""
    module = _module(spec)
    if spec == "dodge_vision":
        scenarios, seeds = module._split_plan(config, split, n)
        return scenarios, [tuple(seed.spawn(2)) for seed in seeds]  # collect_rows' and dodge_eval's children
    return module._split_plan(config, split, n), None


def _saved_scenarios(folder: Path, spec: str, split: str):
    """The scenarios the run saved for a split, with any per-scenario extras (attempt), or None."""
    if split == "test":
        records = (_json(folder / "test_run" / "run.json") or {}).get("records")
        if records:
            return [{k: r[k] for k in _keys(spec) if k in r} for r in records]
        return None
    rows = _json(folder / split / "rows.json")
    if rows and rows.get("scenarios"):
        return [{k: s[k] for k in (*_keys(spec), "attempt") if k in s} for s in rows["scenarios"]]
    return None


def _count(config: dict, spec: str, split: str, fast: bool) -> int:
    observed = config["observed"][spec]
    return observed["fast_attempts"] if fast and split == "context" else observed[f"{split}_scenarios"]


class Plan:
    """What one start plays: the run, its skill, split and step, the config, and the episodes, each {scenario,
    method, candidate?, seeds?, recorded?}."""

    def __init__(self, run_id, folder, spec, config, config_note, student_dir, split, step, note):
        self.run_id, self.folder, self.spec, self.skill = run_id, folder, spec, SPECS[spec]
        self.config, self.config_note, self.student_dir = config, config_note, student_dir
        self.split, self.step, self.note = split, step, note
        self.episodes, self.source, self.student_mode, self.fast = [], "", None, False

    def summary(self) -> dict:
        return {"run_id": self.run_id, "skill": self.skill, "step": self.step, "split": self.split,
                "total": len(self.episodes), "note": self.note, "scenarios": self.source, "config": self.config_note}


def build_plan(run_id: str, server_config: dict) -> Plan:
    """The episodes of a run, or Refusal 404 unknown_run / 422 cannot_play."""
    from app.runs import RUN_ID
    root, missing = backend.runs_dir(server_config)
    if root is None:
        raise Refusal(503, "runs_unavailable", f"Run results need {'; and '.join(missing)}.")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or not (root / run_id).is_dir():
        raise Refusal(404, "unknown_run", "There is no run with that id.")
    folder = root / run_id
    spec = _run_spec(folder)
    if spec not in SPECS:
        raise Refusal(422, "cannot_play", "The player shows DODGE, RAMP and MOVE runs only, and this run is of none of "
                                          "them.")
    student_dir = folder / "student" if (folder / "student" / "student.json").is_file() else None
    bare = student_dir is not None and not any((folder / name).is_file() for name in (
        "events.jsonl", "declaration.json", "result.json"))
    if bare:
        config, config_note = server_config, "config: the repository's config.yaml"
        split, step, note = "dev", "evaluate", NOTES["student"]
    else:
        config, config_note = _run_config(folder, server_config)
        status = _json(folder / "run_status.json") or {}
        note = NOTES["running" if status.get("state") == "running" else "ended"]
        split, step = ("test", "evaluate") if student_dir is not None else ("context", "attempts")
    config = copy.deepcopy(config)
    config["world"]["save_traces"] = False
    plan = Plan(run_id, folder, spec, config, config_note, student_dir, split, step, note)
    declaration = _json(folder / "declaration.json") or {}
    context_rows = _json(folder / "context" / "rows.json") or {}
    plan.fast = declaration.get("mode") == "fast" or context_rows.get("collection") == "attempts"
    saved = _saved_scenarios(folder, spec, split)
    n = len(saved) if saved else _count(config, spec, split, plan.fast)
    try:
        sampled, seeds = _sampled(spec, config, split, n)
    except Exception as e:
        log.exception("The scenarios of run %s cannot be drawn", run_id)
        raise Refusal(422, "cannot_play", "The run's scenarios cannot be drawn with its config; the server log has "
                                          "why.") from e
    if saved:
        same = all({k: s.get(k) for k in _keys(spec)} == {k: t.get(k) for k in _keys(spec)}
                   for s, t in zip(saved, sampled))
        scenarios = saved
        plan.source = (f"the run's saved {split} scenarios, " + ("equal to" if same else "NOT equal to") +
                       " the sampler's draw with the split seeds")
    else:
        scenarios = sampled
        plan.source = (f"drawn by the skill's sampler with the {split} split seeds" if bare else
                       f"drawn with the run's {split} split seeds (the run has not saved them)")
    _add_episodes(plan, scenarios, seeds)
    if not plan.episodes:
        raise Refusal(422, "cannot_play", "The run has no scenarios to play.")
    try:
        plan.loaded = _load_student(plan)
    except Exception as e:
        log.exception("The student of run %s does not load", run_id)
        raise Refusal(422, "cannot_play", "The run's student does not load with the run's config; the server log has "
                                          "why.") from e
    return plan


def _load_student(plan: Plan):
    """The student as the run's evaluation uses it: MOVE and RAMP their _StudentPolicy, DODGE a DodgeRuntime (frozen
    thresholds) or (network, heads, input) for the action-head argmax; None without a student."""
    if plan.student_dir is None:
        return None
    if plan.spec in ("move", "ramp"):
        return _module(plan.spec)._StudentPolicy(plan.student_dir, plan.config)
    if plan.student_mode == "closed_loop":
        from learned_judgment.runtime import DodgeRuntime
        return DodgeRuntime.load(plan.student_dir, plan.config)
    from learned_judgment.students.snn_core import load_network
    student = _json(plan.student_dir / "student.json")
    return load_network(plan.student_dir / "network")[0], student["heads"], student.get("input")


def _add_episodes(plan: Plan, scenarios: list, seeds) -> None:
    spec, config = plan.spec, plan.config
    module = _module(spec)
    if plan.student_dir is not None:
        if spec == "dodge_vision":
            frozen = (_json(plan.student_dir / "student.json") or {}).get("thresholds") is not None
            plan.student_mode = "closed_loop" if frozen else "argmax"
        else:
            plan.student_mode = "argmax"
    no_op = None
    if spec == "dodge_vision":
        actions = config["skills"]["dodge_vision"]["actions"]
        names = list(actions)
        no_op = names.index(next(n for n, c in actions.items() if not any(float(v) for v in c.values())))
        others = [i for i in range(len(names)) if i != no_op]
    else:
        names = list(module.build_observed_spec(config).action_names)
        others = list(range(len(names)))
    plan.no_op = no_op
    for i, scenario in enumerate(scenarios):
        extra = {} if seeds is None else {"seeds": seeds[i]}
        baseline = {"move": "physics", "ramp": "nominal", "dodge_vision": "no_op"}[spec]
        plan.episodes.append({"scenario": scenario, "method": baseline, **extra})
        if plan.student_dir is not None:
            plan.episodes.append({"scenario": scenario, "method": "student", **extra})
        elif plan.fast:
            plan.episodes.append({"scenario": scenario, "method": "attempt", **extra})
        else:
            plan.episodes.append({"scenario": scenario, "method": "candidate", "candidate": others[i % len(others)],
                                  **extra})
    plan.names = names


# ---- the stage world: the skill's own world, shown and paced -------------------------------------------------------

class StageWorld:
    """A Go2World as the skills' episode functions take it, whose every control period is shown on the stage and
    paced to the wall clock. Everything but reset, step and finish_episode is the world's own (attribute access is
    passed through), so the episode is the world's episode, unchanged."""

    def __init__(self, world, player, plain_model, stopping: threading.Event):
        self._world, self._player, self._stopping = world, player, stopping
        self._nbody = plain_model.nbody
        self._robot_root = plain_model.body_rootid[plain_model.body("base_link").id]
        self._names = [mujoco.mj_id2name(plain_model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self._nbody)]
        self._checked = set()  # ids of models whose first bodies are the plain model's
        self._objects = {}  # model -> [(geom id, object id, shape, size, rgba)]
        self._scratch = None  # (model, MjData) for the poses shown
        self.markers_for = None  # fn(reset observation) -> markers, set per episode
        self.markers, self.outcome, self.wall_s, self.sim_periods, self.resyncs = [], None, 0.0, 0, 0

    def __getattr__(self, name):
        return getattr(self._world, name)

    def reset(self, situation=None, attempt_seed=None):
        obs = self._world.reset(situation, attempt_seed=attempt_seed)
        self.markers = self.markers_for(obs) if self.markers_for else []
        self.outcome, self.sim_periods, self.n = None, 0, 0
        self.started = self.due = time.perf_counter()
        self._show(obs, observation=True, frame=True)
        return obs

    def step(self, command, duration_s, on_period=None):
        def period(obs):
            if on_period is not None:
                on_period(obs)
            self._tick(obs)
        return self._world.step(command, duration_s, period)

    def finish_episode(self):
        self.outcome = self._world.finish_episode()
        self.wall_s = time.perf_counter() - self.started
        return self.outcome

    def _tick(self, obs) -> None:
        p = self._player
        self.n += 1
        self.sim_periods += 1
        pose = self.n % p.pose_every == 0
        if pose or self.n % p.observation_every == 0 or self.n % p.frame_every == 0:
            self._show(obs if pose else None, observation=self.n % p.observation_every == 0,
                       frame=self.n % p.frame_every == 0, obs=obs)
        self.due += p.period_s
        wait = self.due - time.perf_counter()
        if wait > 0:
            if self._stopping.wait(wait):
                raise Stopped
        elif -wait > p.resync_after_s:
            self.due, self.resyncs = time.perf_counter(), self.resyncs + 1
        if self._stopping.is_set():
            raise Stopped

    def _show(self, pose_obs, observation=False, frame=False, obs=None) -> None:
        obs = pose_obs if obs is None else obs
        model, data = self._world.model, self._world.policy.data
        pose = scene = None
        if pose_obs is not None:
            if id(model) not in self._checked:
                if model.nbody < self._nbody or any(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) != name
                                                    for b, name in enumerate(self._names)):
                    raise RuntimeError("the played scene's first bodies are not the robot's: its poses cannot be shown")
                self._checked.add(id(model))
            # Poses from a private MjData: the episode's own data is only read (qpos and the mocap poses), and its
            # derived positions can lag qpos (right after reset the thrown object was placed by writing qpos only).
            if self._scratch is None or self._scratch[0] is not model:
                self._scratch = model, mujoco.MjData(model)
            scratch = self._scratch[1]
            scratch.qpos[:], scratch.mocap_pos[:], scratch.mocap_quat[:] = data.qpos, data.mocap_pos, data.mocap_quat
            mujoco.mj_kinematics(model, scratch)
            pose = {"t_s": obs["t_s"], "pos": scratch.xpos[:self._nbody].tolist(),
                    "quat": scratch.xquat[:self._nbody].tolist()}
            scene = {"objects": self._scene_objects(model, scratch), "markers": self.markers}
        self._player.sim.publish_stage(
            observation=describe_observation(obs, obs["command_active"]) if observation else None, pose=pose,
            scene=scene)
        if frame:
            self._player.sim.offer_frame(model, data)

    def _scene_objects(self, model, data) -> list:
        if model not in self._objects:
            found = []
            for g in range(model.ngeom):
                body = int(model.geom_bodyid[g])
                if body == 0 or model.body_rootid[body] == self._robot_root:
                    continue  # the floor (in GET /api/robot) and the robot
                shape = SHAPES.get(mujoco.mjtGeom(model.geom_type[g]))
                if shape is None:
                    log.warning("Scene geom %d has a shape the scene event cannot carry; it is not shown", g)
                    continue
                matid = model.geom_matid[g]
                found.append((g, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", shape,
                              model.geom_size[g][:{"sphere": 1, "box": 3, "cylinder": 2}[shape]].tolist(),
                              (model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[g]).tolist()))
            self._objects = {model: found}  # one scene at a time
        objects, quat = [], np.empty(4)
        for g, name, shape, size, rgba in self._objects[model]:
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
            objects.append({"id": name, "shape": shape, "size": size, "rgba": rgba,
                            "pos": data.geom_xpos[g].tolist(), "quat": quat.tolist()})
        return objects


def _yaw(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def move_markers(scenario: dict):
    """The MOVE target as markers, from the scenario and the dog's pose at the decision instant: forward and sideways
    a line from the start to the target point and a disc there; a turn a line along the start heading and one along
    the target heading."""
    def markers(obs):
        x, y = float(obs["base_pos_m"][0]), float(obs["base_pos_m"][1])
        yaw, kind, target = _yaw(obs["base_quat_wxyz"]), scenario["kind"], float(scenario["target"])
        start = [x, y, MARKER_Z]
        if kind == "turn":
            def ray(angle):
                return [x + HEADING_LINE_M * math.cos(angle), y + HEADING_LINE_M * math.sin(angle), MARKER_Z]
            return [{"id": "start_heading", "shape": "line", "pos": start, "to": ray(yaw), "rgba": HEADING_RGBA},
                    {"id": "target_heading", "shape": "line", "pos": start, "to": ray(yaw + target), "rgba": TARGET_RGBA}]
        dx, dy = (math.cos(yaw), math.sin(yaw)) if kind == "forward" else (-math.sin(yaw), math.cos(yaw))
        end = [x + target * dx, y + target * dy, MARKER_Z]
        return [{"id": "target_path", "shape": "line", "pos": start, "to": end, "rgba": TARGET_RGBA},
                {"id": "target", "shape": "disc", "pos": end, "to": None, "rgba": TARGET_RGBA}]
    return markers


# ---- running one episode per skill ---------------------------------------------------------------------------------

def _recorded(plan: Plan):
    """{(scenario_id, method or candidate index): measured dict} the run saved for the played split, for the check."""
    out, folder, spec = {}, plan.folder, plan.spec
    if plan.split == "test" and spec in ("move", "ramp"):
        methods = {"move": {"physics_only": "physics", "snn": "student"},
                   "ramp": {"nominal": "nominal", "student": "student"}}[spec]
        for r in (_json(folder / "test_run" / "run.json") or {}).get("records", []):
            for saved, method in methods.items():
                if isinstance(r.get(saved), dict) and "measured" in r[saved]:
                    out[(r["scenario_id"], method)] = r[saved]["measured"]
    if spec == "dodge_vision" and plan.student_mode == "closed_loop":
        for r in (_json(folder / f"closed_loop_{plan.split}.json") or {}).get("records", []):
            for saved, method in (("no_op", "no_op"), ("snn", "student")):
                o = (r.get("outcomes") or {}).get(saved)
                if isinstance(o, dict):
                    out[(r["scenario_id"], method)] = {k: v for k, v in o.items() if k != "first_collision_time_s"}
    rows = folder / plan.split / "rows.npz"
    if plan.student_dir is None and rows.is_file():
        try:
            with np.load(rows) as arrays:
                names = _module(spec).OUTCOME_NAMES
                for sid, action, measured in zip(arrays["scenario_index"], arrays["action"], arrays["measured"]):
                    out[(int(sid), int(action))] = dict(zip(names, measured.tolist()))
        except (OSError, ValueError, KeyError):
            log.warning("Run %s: %s cannot be read for the outcome check", plan.run_id, rows)
    return out


def _check(recorded: dict | None, measured: dict):
    """True / False whether the replayed outcome equals the run's saved one, None when the run saved none."""
    if recorded is None:
        return None
    return all(k in measured and math.isclose(float(recorded[k]), float(measured[k]), rel_tol=1e-9, abs_tol=1e-12)
               for k in recorded)


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Episodes:
    """The per-skill side of playing: who decides, through the skill's own functions, and how to say it."""

    def __init__(self, plan: Plan, world, stage: StageWorld):
        self.plan, self.world, self.stage, self.config = plan, world, stage, plan.config
        self.module = _module(plan.spec)
        self.recorded = _recorded(plan)
        self.student = self.view = self.runtime = None
        if plan.spec in ("move", "ramp"):
            self.student = plan.loaded
        else:
            from learned_judgment.vision import VisionSettings
            self.view = self.module.HeadView(self.config)  # its OpenGL context belongs to this (the player's) thread
            if plan.student_mode == "closed_loop":
                self.runtime = plan.loaded
                self.settings = VisionSettings.from_config(self.config)  # dodge_eval: the vision.frames window
            elif plan.student_mode == "argmax":
                self.network, self.heads, self.input = plan.loaded
                self.settings = VisionSettings.from_config(self.config, int(self.input["frames"]))
            else:
                self.settings = VisionSettings.from_config(self.config)  # collect_rows' default window
            self.eligible = self.module._eligible_frames(self.config, self.settings.frames)

    def close(self) -> None:
        if self.view is not None:
            self.view.close()

    # Each returns (action name, outcome dict, caption sentence after the outcome).

    def who(self, episode: dict) -> tuple[str, str]:
        """(the start event's action, who decides, as a phrase)."""
        method, names = episode["method"], self.plan.names
        if method == "student":
            how = {"closed_loop": "the student behind its frozen threat gate (DodgeRuntime in closed loop, as the "
                                  "run's evaluation)",
                   "argmax": "the student's action-head argmax" + (
                       " at the scenario's decision frame (no frozen thresholds yet; observe_and_act)"
                       if self.plan.spec == "dodge_vision" else ", as the run's evaluation")}[self.plan.student_mode]
            return "student", how
        if method == "candidate":
            name = names[episode["candidate"]]
            return name, f"candidate {name}, as the run's attempts try every candidate"
        if method == "attempt":
            return "attempt", "the run's own attempt for this scenario (a random candidate or the physics rule)"
        return {"physics": ("physics rule", "the physics rule (the run's baseline)"),
                "nominal": ("nominal", "the nominal command throughout (the run's baseline: the detector observes, "
                                       "never acts)"),
                "no_op": ("NO_OP", "NO_OP throughout (the run's baseline)")}[method]

    def request(self, scenario: dict) -> str:
        spec, kind = self.plan.spec, scenario["kind"]
        if spec == "move":
            unit, scale = self.module.DISPLAY[kind]
            target = float(scenario["target"]) * scale
            what = {"forward": "move forward" if target > 0 else "move backward",
                    "sideways": "move left" if target > 0 else "move right",
                    "turn": "turn anticlockwise" if target > 0 else "turn clockwise"}[kind]
            return f"{what} {abs(target):.0f} {unit}"
        if spec == "ramp":
            speed = scenario["situation"]["dog_forward_speed_m_s"]
            if kind == "slope":
                return f"walk at {speed:.2f} m/s onto a {scenario['situation']['incline_deg']:.1f} degree ramp"
            return f"walk at {speed:.2f} m/s on flat ground"
        return f"{kind.replace('_', ' ')} ({scenario.get('object_kind', 'ball')})"

    def run(self, episode: dict):
        return {"move": self._move, "ramp": self._ramp, "dodge_vision": self._dodge}[self.plan.spec](episode)

    def _move(self, episode):
        m, config, scenario = self.module, self.config, episode["scenario"]
        method = episode["method"]
        if method == "student":
            policy = self.student
        elif method == "physics":
            policy = lambda features: int(m.physics_choice(config, np.asarray(features)[None, :])[0])  # noqa: E731
        elif method == "attempt":
            policy = self._move_attempt(scenario)
        else:
            policy = lambda features, a=episode["candidate"]: a  # noqa: E731
        self.stage.markers_for = move_markers(scenario)
        result = m.run_episode(self.stage, scenario, scenario["scenario_id"], policy)
        unit, scale = m.DISPLAY[scenario["kind"]]
        name = self.plan.names[result["action"]]
        outcome = {**result["measured"], "error_units": result["error_units"], "unit": m.UNITS[scenario["kind"]],
                   "target": result["target"], "final_forward_m": result["final"]["forward_m"],
                   "final_sideways_m": result["final"]["sideways_m"],
                   "final_yaw_change_rad": result["final"]["yaw_change_rad"], "elapsed_s": result["elapsed_s"],
                   "fall_time_s": result["fall_time_s"]}
        keys = [(scenario["scenario_id"], method), (scenario["scenario_id"], result["action"])]
        said = (f"chose {name}, missed the target by {result['error_units'] * scale:.0f} {unit}, drifted "
                f"{result['measured']['drift_m'] * 100:.0f} cm" + (", and fell" if result["measured"]["fall"] else ""))
        return name, outcome, said, keys, result["measured"]

    def _move_attempt(self, scenario):
        attempt = scenario.get("attempt")
        if isinstance(attempt, dict) and isinstance(attempt.get("action"), int):
            return lambda features, a=attempt["action"]: a
        m, config = self.module, self.config  # collect_attempts' own draw, before the run saved it
        u = np.random.default_rng(np.random.SeedSequence(
            [int(config["runtime"]["seed"]), m.SPLIT_CODES["context"], scenario["scenario_id"]])).random(2)
        c = len(self.plan.names)
        if bool(u[0] < m._exploration_fraction(config)):
            return lambda features, a=min(int(math.floor(u[1] * c)), c - 1): a
        return lambda features: int(m.physics_choice(config, np.asarray(features)[None, :])[0])

    def _ramp(self, episode):
        m, config, scenario = self.module, self.config, episode["scenario"]
        method = episode["method"]
        if method == "student":
            policy = self.student
        elif method == "nominal":
            policy = None
        elif method == "attempt":
            u = np.random.default_rng(np.random.SeedSequence(
                [int(config["runtime"]["seed"]), m.SPLIT_CODES["context"], scenario["scenario_id"]])).random(2)
            c = len(self.plan.names)
            explore = bool(u[0] < float(config["engine"]["attempt_exploration_fraction"]))
            policy = m._AttemptPolicy(config, min(int(math.floor(u[1] * c)), c - 1) if explore else None)
        else:
            policy = lambda features, a=episode["candidate"]: a  # noqa: E731
        self.stage.markers_for = None
        result = m.run_episode(self.stage, scenario, scenario["scenario_id"], policy)
        name = None if result["action"] is None else self.plan.names[result["action"]]
        measured = result["measured"]
        outcome = {**measured, "entered": result["entered"], "decision_time_s": result["decision_time_s"],
                   "action": name, "ramp_start_m": result["ramp_start_m"], "ramp_end_m": result["ramp_end_m"],
                   "reached_ramp_end_s": result["reached_ramp_end_s"], "elapsed_s": result["elapsed_s"]}
        decided = (f"switched to {name} at {result['decision_time_s']:.2f} s" if name is not None else
                   "kept the nominal command" + ("" if result["entered"] or scenario["kind"] == "flat" else
                                                 " (the detector saw no slope)"))
        said = (f"{decided}, progress {measured['progress_m']:.2f} m, " +
                ("reached the top" if measured["reached_top"] else "did not reach the top" if scenario["kind"] == "slope"
                 else "flat ground") + (", fell" if measured["fall"] else "") +
                f", {measured['sideways_m']:.2f} m sideways")
        keys = [(scenario["scenario_id"], method)] + (
            [] if result["action"] is None or method == "nominal" else [(scenario["scenario_id"], result["action"])])
        return name or "nominal", outcome, said, keys, measured

    def _dodge(self, episode):
        m, config, scenario = self.module, self.config, episode["scenario"]
        method, sid = episode["method"], scenario["scenario_id"]
        frame_seed, augment_seed = episode["seeds"]
        self.stage.markers_for = None
        if self.plan.student_mode == "closed_loop" and method in ("no_op", "student"):
            from learned_judgment.vision import take_frames
            runtime = self.runtime
            zero = {k: float(v) for k, v in config["skills"]["dodge_vision"]["actions"][runtime.settings.no_op].items()}
            if method == "student":
                def policy(window, t_s):
                    return runtime.infer(take_frames(window, runtime.frames), t_s)
            else:
                def policy(window, t_s):
                    return {"command": zero, "action": runtime.settings.no_op}
            runtime.reset()
            out = m.closed_loop(self.stage, self.view, scenario, sid, policy, self.settings,
                                np.random.default_rng(augment_seed))
            measured, issued = out["measured"], [[float(t), a] for t, a in out["issued"]]
            name = ", ".join(f"{a} at {t:.2f} s" for t, a in issued) if issued else "NO_OP"
            outcome = {**measured, "issued": issued,
                       "first_collision_time_s": self.stage.outcome["first_collision_time_s"]}
            said = (f"issued {name}" if issued else "issued nothing") + self._dodge_said(measured)
            return name, outcome, said, [(sid, method)], measured
        own = m._frames_before_arrival(config, scenario, self.eligible)
        frame = int(own[np.random.default_rng(frame_seed).integers(own.size)])  # collect_rows' decision frame
        t_decide = float(m.frame_times(config)[frame])
        check = ""
        if method == "student":  # argmax: read the window the episode will show, then act on it
            from learned_judgment.observed import student_readout
            window = m.render_windows(self.world, self.view, scenario, sid, [frame], self.settings,
                                      [np.random.default_rng(augment_seed)])
            action = int(np.argmax(student_readout(self.network, window, self.heads, self.input)[1][0]))
        else:
            window = None
            action = self.plan.no_op if method == "no_op" else episode["candidate"]
        features, measured = m.observe_and_act(self.stage, self.view, scenario, sid, frame, action, self.settings,
                                               np.random.default_rng(augment_seed))
        if window is not None and not np.array_equal(features, window[0]):
            check = " (the episode's window differed from the one the student read)"
        name = self.plan.names[action]
        outcome = {**measured, "action": name, "decision_time_s": t_decide}
        said = f"{name} at {t_decide:.2f} s{check}" + self._dodge_said(measured)
        return name, outcome, said, [(sid, method), (sid, action)], measured

    @staticmethod
    def _dodge_said(measured) -> str:
        return (", " + ("hit" if measured["collision"] else "no contact") +
                f", closest {measured['min_separation_m'] * 100:.0f} cm" + (", fell" if measured["fall"] else ""))


# ---- the player --------------------------------------------------------------------------------------------------

class Player:
    """One episode player per server; one plan plays at a time, on a thread of its own."""

    def __init__(self, server):
        self.server, self.sim = server, server.sim
        app = server.config["app"]
        self.period_s = self.sim.period_s
        self.pose_every, self.observation_every, self.frame_every = (
            self.sim.pose_every, self.sim.observation_every, self.sim.frame_every)
        episodes = app["episodes"]
        self.pause_s, self.resync_after_s, self.stop_timeout_s = (
            float(episodes["pause_s"]), float(episodes["resync_after_s"]), float(episodes["stop_timeout_s"]))
        self.lock = threading.Lock()
        self.thread, self.plan, self.stopping = None, None, threading.Event()

    def state(self) -> dict:
        with self.lock:
            playing = self.thread is not None and self.thread.is_alive()
            return {"playing": playing, "run_id": self.plan.run_id if playing else None,
                    "note": self.plan.note if playing else ""}

    def start(self, run_id: str) -> dict:
        plan = build_plan(run_id, self.server.config)
        self.stop()
        token = object()
        try:
            self.sim.submit("hold", token).result()
        except Exception:
            raise Refusal(503, "simulation_stopped", "The simulation is not running; restart the server.") from None
        stopping = threading.Event()
        thread = threading.Thread(target=self._play, args=(plan, token, stopping), name="episode-player", daemon=True)
        with self.lock:
            self.thread, self.plan, self.stopping = thread, plan, stopping
        thread.start()
        log.info("Playing run %s: %s, %d episodes of the %s split (%s; %s)", plan.run_id, plan.skill,
                 len(plan.episodes), plan.split, plan.source, plan.config_note)
        return plan.summary()

    def stop(self) -> bool:
        """Stop the playing plan, if any, and wait for its thread (app.episodes.stop_timeout_s at most). True if one
        was playing."""
        with self.lock:
            thread, stopping = self.thread, self.stopping
        if thread is None or not thread.is_alive():
            return False
        stopping.set()
        thread.join(timeout=self.stop_timeout_s)
        if thread.is_alive():
            log.warning("The episode player did not stop within %s s; it stops at its next control period",
                        self.stop_timeout_s)
        return True

    def _event(self, plan, state, index=None, episode=None, action="", caption="", outcome=None) -> tuple:
        scenario = (episode or {}).get("scenario") or {}
        return ("episode", {"state": state, "run_id": plan.run_id, "skill": plan.skill, "step": plan.step,
                            "index": index, "total": len(plan.episodes), "kind": scenario.get("kind", ""),
                            "action": action, "caption": caption, "outcome": outcome, "note": plan.note})

    def _play(self, plan: Plan, token, stopping: threading.Event) -> None:
        from learned_judgment.env import Go2World
        sim, episodes, played = self.sim, None, 0
        final = "Finished: every episode of the plan was played."
        try:
            world = Go2World(plan.config, plan.config["observed"]["world_seed"], plan.config["observed"]["world_preset"])
            sim.publish_stage(events=sim.show_model(world.model))  # the plain scene: the robot the poses move
            stage = StageWorld(world, self, world.model, stopping)
            episodes = Episodes(plan, world, stage)
            for index, episode in enumerate(plan.episodes):
                if stopping.is_set():
                    final = "Stopped."
                    break
                scenario = episode["scenario"]
                action, who = episodes.who(episode)
                head = f"{plan.skill}, {plan.split} scenario {scenario['scenario_id']}: {episodes.request(scenario)}"
                sim.publish_stage(events=[self._event(
                    plan, "start", index, episode, action,
                    f"{head}. Deciding: {who}. Scenarios: {plan.source}; {plan.config_note}.")])
                try:
                    name, outcome, said, keys, measured = episodes.run(episode)
                except Stopped:
                    sim.publish_stage(events=[self._event(plan, "end", index, episode, action,
                                                          f"{head}. Stopped before the episode ended: no outcome.")])
                    final = "Stopped."
                    break
                except Exception as e:
                    log.exception("Run %s: episode %d failed", plan.run_id, index)
                    sim.publish_stage(events=[self._event(
                        plan, "end", index, episode, action,
                        f"{head}. The episode stopped with an error ({type(e).__name__}); the server log has the "
                        f"details. No outcome.")])
                    final = "Stopped after an episode failed."
                    break
                played = index + 1
                match = _check(next((episodes.recorded[k] for k in keys if k in episodes.recorded), None), measured)
                checked = {None: "", True: " Same outcome as the run saved for this episode.",
                           False: " NOT the outcome the run saved for this episode."}[match]
                simulated = stage.sim_periods * self.period_s
                log.info("Run %s episode %d/%d: %.2f s simulated in %.2f s wall (%.2f wall s per simulated s, %d "
                         "resyncs); matches the run's record: %s", plan.run_id, played, len(plan.episodes),
                         simulated, stage.wall_s, stage.wall_s / max(simulated, 1e-9), stage.resyncs, match)
                sim.publish_stage(events=[self._event(
                    plan, "end", index, episode, name, f"{head}. Deciding: {who}. Result: {said}.{checked}",
                    _plain({**outcome, "matches_run_record": match}))])
                if stopping.wait(self.pause_s):
                    final = "Stopped."
                    break
        except Exception as e:
            log.exception("The episode player stopped on an error")
            final = f"The player stopped with an error ({type(e).__name__}); the server log has the details."
        finally:
            if episodes is not None:
                episodes.close()
            sim.publish_stage(events=[self._event(plan, "idle", played, caption=f"{final} {played} of "
                                                                                f"{len(plan.episodes)} played.")],
                              scene={"objects": [], "markers": []})
            sim.submit("release", token)
            with self.lock:
                if self.thread is threading.current_thread():
                    self.thread = None


@route("POST", "/api/episodes/start")
def start_episodes(h) -> None:
    body = h.read_body()
    if set(body) != {"run_id"} or not isinstance(body["run_id"], str):
        raise Refusal(400, "invalid_body", "An episode request is a run_id and nothing else.")
    h.send_json(202, h.server.player.start(body["run_id"]))


@route("POST", "/api/episodes/stop")
def stop_episodes(h) -> None:
    h.read_body()
    h.server.player.stop()
    h.send_json(200, {"playing": False})
