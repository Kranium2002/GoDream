"""REFLEX (Person B): a blind closed-loop steering reflex from RAW body sensors, built entirely from a lesson declaration.

The simulated Go2 walks toward a spot straight ahead with its frozen, borrowed walking policy; a smooth mound may lie
in the way (Go2World's mound keys). Every few tenths of a second a small spiking network reads only a short raw history
of body sensors and picks one steering command. Everything about the lesson (senses, timing, steering options, goal,
practice world) comes from config.yaml skills.reflex.declaration, a lesson a language model designed as data from
skills.reflex.menu (learned_judgment/reflex_lesson.py), checked against the menu here before anything runs; this module
hard-codes none of it. Hidden simulator truth (position, the mound) shapes the scenarios and is read by the scorer and
the physics guess only, never by a policy that the student imitates.

Lane contract (Person B, reflex)
--------------------------------
    TIME        whole control periods (robot.walking_policy.command_schema.update_period_s, 0.02 s) from t_s 0 of
                Go2World.reset (start "trotting": after world.settle_s at the nominal speed; "standing": after
                skills.reflex.standing_settle_s on the zero command). The nominal command (vel_x = nominal_speed_m_s,
                vel_y 0, yaw_rate 0) is sent from t_s 0.
    WINDOW      the policy's ONLY input: K = history_s / sample_every_s + 1 samples at t, t - sample_every_s, ...,
                t - history_s, flattened sample-major OLDEST FIRST; each sample holds the channels of the lesson's sense
                groups in menu order (skills.reflex.menu.sense_groups), each scaled to [0, 1] with its fixed [low,
                high] of the menu and clipped. D = K x channels. Channels, from the observation of that control period:
                  tilt          pitch_deg = degrees(asin(2 (x z - w y))) nose up +, roll_deg = degrees(asin(2 (y z +
                                w x))) left side up +, of base_quat_wxyz (w, x, y, z)
                  turn_rates    base_ang_vel_rad_s turned into the trunk frame (R^T w): roll, pitch, yaw rate
                  heading       yaw(base_quat_wxyz) - yaw at t_s 0, wrapped to (-180, 180], degrees, + left
                  joint_angles  joint_pos_rad (FL, FR, RL, RR x hip, thigh, calf) - the policy's default pose
                  joint_speeds  joint_vel_rad_s
                  feet          foot_contact as 0 / 1
                  last_command  command_active vel_x, vel_y, yaw_rate (what the dog itself sent last)
                  path_memory   DEAD RECKONING from the dog's own signals only: 0, 0 at t_s 0 (the decision instant
                                of Go2World.reset); every later control period adds command_active vel_x x period
                                (the forward speed the dog itself commanded) along the IMU heading of that period
                                (the "heading" yaw relative to t_s 0, midway between the period's first and last
                                observation): path_forward_m along the heading at t_s 0, path_sideways_m to its left.
                                An estimate that drifts (the walking policy does not follow commands exactly and
                                slopes push the trunk sideways); each window sample carries the estimate at its time.
                Never an input: base_pos_m, base_lin_vel_m_s, anything about the mound.
    DECISIONS   first at t = history_s (the window is full), then every decide_every_s; the picked candidate is held
                until the next decision. CANDIDATES = steer.yaw_rate x steer.vel_x of the lesson, vel_y 0, each
                checked against the verified command schema; ORDER: straight (yaw 0, vel_x = nominal) first, then by
                |yaw_rate|, left (+) before right (-), then by |vel_x - nominal|, faster first. action_features (C, 2)
                = (vel_x, yaw_rate). no_op_index None.
    EPISODE     ends when the trunk's progress along the heading at t_s 0 reaches goal.forward_m ("goal"), at
                time_limit_s ("time") or at a fall ("fall"). reached = "goal" and |sideways offset| <= goal.tolerance_m
                at that moment. For the teacher's rows (attempts) the episode then goes on for horizon_s more under the
                same closed loop (scoring tail: the last decisions' horizons are measured, not cut; tail decisions are
                no rows, the outcome stays the end moment's); the evaluation (run) has no tail. The outcome also keeps
                the path_memory estimate at the end moment (path_memory_forward_m, path_memory_sideways_m; computed
                in every episode, whatever the lesson's senses) beside the truth (final_forward_m, final_sideways_m):
                scorer-side, to measure the estimate's drift.
    OUTCOMES    K = 4 short-horizon outcomes per decision at t, measured from the scorer's per-period track over
                [t, t + horizon_s] (truncated at a fall), OUTCOME_NAMES order:
                  progress_m  regression  gain along the heading at t_s 0
                  offset_m    regression  |sideways offset from the line to the spot| at the end of the horizon
                  tilt_deg    regression  mean trunk tilt (from upright) over the horizon's control periods
                  fall        binary      a fall within the horizon
                bounds skills.reflex.outcome_bounds (fall [0, 1]). Later decisions inside the horizon act too: it is
                the closed loop's result after this pick.
    PHYSICS     (scorer side: uses the true state at the decision; a teacher column, never a policy input) a unicycle
                over horizon_s H from the true heading error e (rad, heading - heading at t_s 0) and offset y0 (m, +
                left): v = track.forward x vel_x, w = track.turn x yaw_rate (skills.move.physics.track);
                  w != 0: progress = v / w (sin(e + w H) - sin e), offset = |y0 + v / w (cos e - cos(e + w H))|
                  w == 0: progress = v H cos e,                     offset = |y0 + v H sin e|
                tilt_deg = the tilt at the decision (persistence); fall = 0. Clipped to the outcome bounds.
    OBJECTIVE   goal.weights w: + w.progress x mean progress_m - w.offset x mean offset_m - w.tilt x mean tilt_deg -
                w.fall x mean fall - w.spread x spread of progress_m.
    RULE        "physics_only" / the rule of thumb: argmax over the candidates of the objective of the physics guess
                (point mass), ties to the first in candidate order. It reads the true state: a reference, not a blind
                policy.
    SPLITS      context 1 (attempts), dev 2 (round choice), test 3 (held out until run reflex2, which looked at it after
                every round), dream 4, test2 5 (the held-out split from run reflex3 on: looked at once per run).
    SCENARIOS   per split np.random.default_rng(np.random.SeedSequence([runtime.seed, SPLIT_CODES[split]])), per
                scenario u = rng.random(6): u0 < practice_world.flat_share -> flat (no mound keys); else radius, height,
                centre ahead, centre side uniform in the lesson's ranges (u1..u4); the height lowered to the
                world.mound.max_rim_slope_deg height of that radius when above it; the centre moved ahead to radius +
                world.mound.min_gap_m when closer. attempt_seed = scenario_id.

Files of the command line (the observed.py array contract; "frames": null, D window features in [0, 1]):

    rows.npz    scenario_index, decision_time_s, action, features (n, D) float32, physics (n, 4), measured (n, 4); and,
                outside the contract, state (n, 4) [forward_m, sideways_m, heading_error_rad, tilt_deg] at the decision
                (scorer truth), horizon_s (n,) measured horizon, source (n,) 0 rule of thumb / 1 exploration / 2 student
    rows.json   skill, split, action_names, outcome_names/kinds/bounds, feature_names, frames null, window, lesson,
                seeds, policy, scenarios (index = scenario_index), episodes (per scenario: kind, mound, outcome,
                decisions), wall times; a merged file also "parts"
    dream.npz   scenario_index (m,) = dream episode index, decision_time_s, features (m, D), physics (m, C, 4); state
                (m, 4), source (m,) 0 attempts / 1 extra rule-of-thumb episode
    dream.json  the meta plus "sampling": the dream situations are REAL sensor windows (see dream)
    run.json    records per scenario and method, summary per method (all, flat, mound), by_difficulty per method (flat,
                gentle, steep: see DIFFICULTY), paired differences
    OUT/tracks/<method>_s<scenario>.npz  t_s (N,), qpos (N, 19) per control period, for the first
                observed.reflex.tracks_per_method scenarios of each method; OUT/tracks/index.json describes them (mound
                in the world frame, outcome)

    python -m learned_judgment.skills.reflex attempts --out DIR [--n N] [--first S] [--policy {rule|student}
        [--student DIR]] [--explore FRACTION]
    python -m learned_judgment.skills.reflex merge --inputs DIR DIR ... --out DIR
    python -m learned_judgment.skills.reflex dream --out DIR [--n N] [--context DIR [DIR ...]]
    python -m learned_judgment.skills.reflex run (--student DIR | --no-student) --split {dev|test|test2} --out DIR
        [--n N] [--run RUN] [--methods M ...]
    python -m learned_judgment.run_skill _exec --config RUN_DIR/config.yaml learned_judgment.skills.reflex loop
        --run RUN --rounds N [--test-split {test2|test}] [--confirm-billing]

attempts: N (default observed.reflex.attempt_episodes) episodes on the context scenarios S .. S + N - 1 (default S 0;
scenario_id = the index in the context split, rows' scenario_index = scenario_id - S); at decision j of scenario i, u =
np.random.default_rng(np.random.SeedSequence([runtime.seed, SPLIT_CODES["context"], i, j])).random(2): u[0] < the
exploration fraction -> candidate min(floor(u[1] C), C - 1), else the policy's pick: --policy rule (default) the rule
of thumb, fraction engine.attempt_exploration_fraction; --policy student the trained student from the window only
(on-policy: the states its own choices lead to), fraction observed.reflex.student_explore_fraction; --explore overrides
the fraction. EVERY decision is a row with its measured short-horizon outcome. merge: rows files of this lesson
concatenated in the given order; part p's scenario_index is offset by the scenarios of the parts before it, so indices
stay distinct and index the merged "scenarios"; rows.json "parts" records each part (path, rows, scenarios, offset,
policy, student, exploration). dream: raw sensor histories cannot be invented, so the dream situations are all windows of
the attempts' episodes (read from --context, one or several attempts directories merged as merge does, default the
"context" directory beside --out) plus the windows of N (default observed.reflex.dream_episodes) further unbilled
rule-of-thumb episodes on dream scenarios; each with the physics guess of EVERY candidate from the true state its
episode recorded. run: full closed-loop episodes on the same scenarios and attempt seeds for "no_op" (always the
straight nominal command: the walking policy by itself), "physics_only" (the rule of thumb; it reads the truth, a
reference) and "snn" (the trained student, observed.student_readout argmax of the action head, from the window only);
--no-student skips "snn", --methods keeps only the named ones. With --run RUN it merges the UI keys into
<observed.runs_dir>/RUN/result.json and appends "evaluate" events to its events.jsonl.

DIFFICULTY (run summary, scorer truth): "flat"; a mound is "gentle" below skills.reflex.report.steep_from_deg rim slope
(env.mound.rim_slope_deg), "steep" at or above it. How the dog passed a mound, from its per-period path in the start
frame (the mound's centre is (mound_x_m, mound_y_m) there): "over" when the path came within half the mound radius of
the centre; otherwise "around" when it came level with the centre (forward >= mound_x_m), else "not_past".

loop (the engine adapts; run it under the run's config through run_skill _exec, as the line above; it checks that the
loaded config is RUN_DIR/config.yaml's): round 0 = attempts with the rule of thumb on context scenarios 0 .. N - 1 (N =
observed.reflex.attempt_episodes) -> RUN/round_0/context, dream over its windows -> round_0/dream, observed teach
(--confirm-billing) and observed train as subprocesses with the run's config (they write RUN/scores.npz and
RUN/student; copied to round_0/scores.npz and round_0/student). Round k >= 1: attempts --policy student --student
round_<k-1>/student on scenarios k N .. (k + 1) N - 1 -> round_k/context, merge of round_0 .. round_k contexts ->
round_k/rows (the teacher's context: every real decision so far), dream over all those windows, teach, train (the
train step's result.json "train" record is kept as round_k/train.json). Every round's student is then run on the DEV
split with "snn" only -> round_k/dev_run; the held-out split is not looked at during the rounds.
ROUND CHOICE (fixed before any run of it): the run's student is the round whose student reached the most dev courses;
ties -> the smaller mean |sideways offset| at the goal line on dev (no crossing at all counts as the worst); ties ->
the later round. That round's student is copied to RUN/student (result.json "train" = its train.json, with
"chosen_round") and run ONCE on the held-out split (--test-split, default test2) with no_op, physics_only and snn,
tracks and the UI keys (--run RUN) -> RUN/test_run. result.json "rounds": per round the real decisions used, the dev
result (reached, n, mean miss, by difficulty too) and "chosen"; "round_choice": the rule, the chosen round, its dev
result and the held-out snn result; "loop": the steps with exit codes and wall times; every step's output also goes to
RUN/loop.log. --rounds N is the number of improvement rounds after round 0. Without --confirm-billing it stops after
round 0's attempts and dream with the free quote (observed quote; nothing billed) and exit code 3. Teacher requests are
cached by content, and every collection is deterministic, so a re-run sends nothing already sent.

This module defines no SPEC_CLASS, so the SkillSpec registry (learned_judgment.skills) does not load it.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import multiprocessing
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from learned_judgment.config import ConfigError, config_digest, load_config, optional, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.fileio import atomic_write_bytes, atomic_write_text
from learned_judgment.skills.ramp import _count, _jsonable, _write

__all__ = [
    "ACTION_FEATURE_NAMES",
    "OUTCOME_KINDS",
    "OUTCOME_NAMES",
    "SKILL",
    "SPLIT_CODES",
    "ReflexSpec",
    "build_observed_spec",
    "collect_attempts",
    "collect_dream",
    "lesson_from_config",
    "main",
    "merge_rows",
    "objective",
    "physics_choice",
    "physics_guess",
    "run_episode",
    "run_methods",
    "sample_reflex_scenarios",
    "sense_channels",
]

SKILL = "reflex"
OUTCOME_NAMES = ("progress_m", "offset_m", "tilt_deg", "fall")
OUTCOME_KINDS = ("regression", "regression", "regression", "binary")
ACTION_FEATURE_NAMES = ("vel_x", "yaw_rate")
STATE_NAMES = ("forward_m", "sideways_m", "heading_error_rad", "tilt_deg")
SPLIT_CODES = {"context": 1, "dev": 2, "test": 3, "dream": 4, "test2": 5}
RUN_SPLITS = ("context", "dev", "test", "test2")  # splits `run` evaluates (observed.reflex.<split>_scenarios)
HELD_OUT_SPLITS = ("test2", "test")  # the loop's final run; test2 the default (test was looked at in every reflex2 round)
ROUND_CHOICE_RULE = ("most dev courses reached; ties -> smaller mean |sideways offset| at the goal line on dev (no "
                     "crossing counts as the worst); ties -> the later round")
KINDS = ("flat", "mound")
DIFFICULTIES = ("flat", "gentle", "steep")
PASSAGES = ("over", "around", "not_past")
METHODS = ("no_op", "physics_only", "snn")
SOURCES = ("rule", "explore", "student")  # rows.npz source codes 0, 1, 2
_FIELDS = ("vel_x", "vel_y", "yaw_rate")
_ROOT = "skills.reflex"
_OBSERVED = "observed.reflex"
_LEGS = ("FL", "FR", "RL", "RR")
_PARTS = ("hip", "thigh", "calf")
_WHERE = "simulation (MuJoCo Go2World, frozen walking policy) on this computer's CPU"


# ----- the lesson ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lesson:
    """The checked declaration, in control periods."""

    declaration: Mapping[str, Any]
    period_s: float
    groups: tuple[str, ...]
    channels: tuple[tuple[str, float, float], ...]  # (name, low, high) per channel, window order
    samples: int  # K
    sample_n: int  # control periods between samples
    history_n: int  # (K - 1) * sample_n
    decide_n: int
    horizon_n: int
    limit_n: int
    start: str
    nominal: float
    names: tuple[str, ...]
    commands: tuple[dict[str, float], ...]
    features: np.ndarray  # (C, 2) vel_x, yaw_rate
    forward_m: float
    tolerance_m: float
    weights: Mapping[str, float]
    world: Mapping[str, Any]

    @property
    def width(self) -> int:
        return self.samples * len(self.channels)

    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"s{k}_{name}" for k in range(self.samples) for name, _, _ in self.channels)

    def sample_times_s(self) -> list[float]:
        """Offsets of the window's samples from the decision, oldest first (s, <= 0)."""
        return [-(self.samples - 1 - k) * self.sample_n * self.period_s for k in range(self.samples)]


def _order(yaw: float, vx: float, nominal: float) -> tuple:
    straight = yaw == 0.0 and vx == nominal
    return (not straight, abs(yaw), 0 if yaw >= 0 else 1, abs(vx - nominal), -vx)


def lesson_from_config(config: Mapping[str, Any]) -> Lesson:
    """The lesson of skills.reflex.declaration, checked against skills.reflex.menu. ConfigError without one or with a
    problem (the problems listed)."""
    from learned_judgment.reflex_lesson import check_lesson

    declaration = optional(config, f"{_ROOT}.declaration")
    if declaration is None:
        raise ConfigError(f"config.yaml has no {_ROOT}.declaration: a reflex run needs a lesson designed with "
                          f"learned_judgment.reflex_lesson (design, then config) in its run config")
    problems = check_lesson(declaration, config)
    if problems:
        raise ConfigError(f"{_ROOT}.declaration breaks the menu: " + " ".join(p["message"] for p in problems))
    menu = require(config, f"{_ROOT}.menu")
    period = float(require(config, "robot.walking_policy.command_schema.update_period_s"))
    d = declaration
    groups = tuple(g for g in menu["sense_groups"] if g in d["senses"])
    channels = tuple((str(c[0]), float(c[1]), float(c[2])) for g in groups for c in menu["sense_groups"][g]["channels"])
    every, history = float(d["sense"]["sample_every_s"]), float(d["sense"]["history_s"])
    sample_n = round(every / period)
    samples = round(history / every) + 1
    nominal = float(d["nominal_speed_m_s"])
    pairs = sorted(((float(w), float(v)) for w in d["steer"]["yaw_rate"] for v in d["steer"]["vel_x"]),
                   key=lambda p: _order(p[0], p[1], nominal))
    commands = tuple({"vel_x": v, "vel_y": 0.0, "yaw_rate": w} for w, v in pairs)
    names = tuple(f"yaw{w:+.2f}_vx{v:.2f}" for w, v in pairs)
    if len(set(names)) != len(names):
        raise ConfigError(f"the lesson's candidates collide at 2 decimals: {names}")
    features = np.array([[v, w] for w, v in pairs], dtype=np.float64)
    features.setflags(write=False)
    w = d["practice_world"]
    return Lesson(
        declaration=d, period_s=period, groups=groups, channels=channels, samples=samples, sample_n=sample_n,
        history_n=(samples - 1) * sample_n, decide_n=round(float(d["decide_every_s"]) / period),
        horizon_n=round(float(d["horizon_s"]) / period), limit_n=round(float(d["time_limit_s"]) / period),
        start=str(d["start"]), nominal=nominal, names=names, commands=commands, features=features,
        forward_m=float(d["goal"]["forward_m"]), tolerance_m=float(d["goal"]["tolerance_m"]),
        weights={k: float(v) for k, v in d["goal"]["weights"].items()},
        world={"flat_share": float(w["flat_share"]),
               **{k: (float(w[k]["min"]), float(w[k]["max"])) if isinstance(w[k], Mapping) else
                  (float(w[k][0]), float(w[k][1])) for k in ("radius_m", "height_m", "centre_ahead_m", "centre_side_m")}},
    )


def _outcome_bounds(config: Mapping[str, Any]) -> np.ndarray:
    from learned_judgment.skills.ramp import _pair

    key = f"{_ROOT}.outcome_bounds"
    bounds = np.array([_pair(config, f"{key}.progress_m", strict=True),
                       _pair(config, f"{key}.offset_m", minimum=0.0, strict=True),
                       _pair(config, f"{key}.tilt_deg", minimum=0.0, strict=True), (0.0, 1.0)], dtype=np.float64)
    bounds.setflags(write=False)
    return bounds


def _track_factors(config: Mapping[str, Any]) -> tuple[float, float]:
    """(forward, turn) of skills.move.physics.track: response per unit command."""
    from learned_judgment.skills.ramp import _positive

    return _positive(config, "skills.move.physics.track.forward"), _positive(config, "skills.move.physics.track.turn")


# ----- senses ----------------------------------------------------------------------------------------------------


def _yaw(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tilt_deg(q: Sequence[float]) -> float:
    _, x, y, _ = (float(v) for v in q)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _default_pose(config: Mapping[str, Any]) -> np.ndarray:
    """The policy's default joint angles in the model's joint order (FL, FR, RL, RR x hip, thigh, calf)."""
    action = require(config, "robot.walking_policy.action_schema")
    by_name = dict(zip(action["joint_order"], action["default_pose"]))
    return np.array([float(by_name[f"{leg}_{part}_joint"]) for leg in _LEGS for part in _PARTS])


def sense_channels(obs: Mapping[str, Any], groups: Sequence[str], yaw0: float, default_pose: np.ndarray,
                   path: Sequence[float] | None = None) -> np.ndarray:
    """The raw channel values (unscaled) of one observation, groups in the given order (see the module docstring).
    ``path`` is the dead-reckoning estimate (forward, sideways) at this observation, needed by "path_memory"."""
    q = np.asarray(obs["base_quat_wxyz"], dtype=np.float64)
    w, x, y, z = q
    out: list[float] = []
    for group in groups:
        if group == "tilt":
            out += [math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y))))),
                    math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (y * z + w * x)))))]
        elif group == "turn_rates":
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            out += (r.T @ np.asarray(obs["base_ang_vel_rad_s"], dtype=np.float64)).tolist()
        elif group == "heading":
            out.append(math.degrees(_wrap(_yaw(q) - yaw0)))
        elif group == "joint_angles":
            out += (np.asarray(obs["joint_pos_rad"], dtype=np.float64) - default_pose).tolist()
        elif group == "joint_speeds":
            out += np.asarray(obs["joint_vel_rad_s"], dtype=np.float64).tolist()
        elif group == "feet":
            out += [float(bool(c)) for c in obs["foot_contact"]]
        elif group == "last_command":
            out += [float(obs["command_active"][f]) for f in _FIELDS]
        elif group == "path_memory":
            if path is None:
                raise ValueError("the path_memory sense needs the dead-reckoning estimate (run_episode keeps it)")
            out += [float(path[0]), float(path[1])]
        else:
            raise ValueError(f"unknown sense group {group!r}")
    return np.array(out, dtype=np.float64)


def _scaler(lesson: Lesson) -> tuple[np.ndarray, np.ndarray]:
    low = np.array([c[1] for c in lesson.channels])
    return low, np.array([c[2] for c in lesson.channels]) - low


# ----- physics guess, objective, rule of thumb -------------------------------------------------------------------


def _guess(lesson: Lesson, bounds: np.ndarray, factors: tuple[float, float], states: np.ndarray,
           actions: np.ndarray) -> np.ndarray:
    states, actions = np.asarray(states, dtype=np.float64).reshape(-1, 4), np.asarray(actions)
    n = states.shape[0]
    if actions.shape != (n,):
        raise ValueError(f"states (n, 4) and actions (n,) must agree; got {states.shape}, {actions.shape}")
    if n and (actions.dtype.kind not in "iu" or actions.min() < 0 or actions.max() >= len(lesson.names)):
        raise ValueError(f"actions must be ints in [0, {len(lesson.names)})")
    h = lesson.horizon_n * lesson.period_s
    command = lesson.features[actions.astype(np.int64)] if n else np.zeros((0, 2))
    v, w = factors[0] * command[:, 0], factors[1] * command[:, 1]
    e, y0 = states[:, 2], states[:, 1]
    turning = np.abs(w) > 1e-12
    safe = np.where(turning, w, 1.0)
    progress = np.where(turning, v / safe * (np.sin(e + w * h) - np.sin(e)), v * h * np.cos(e))
    dy = np.where(turning, v / safe * (np.cos(e) - np.cos(e + w * h)), v * h * np.sin(e))
    guess = np.column_stack([progress, np.abs(y0 + dy), states[:, 3], np.zeros(n)])
    return np.clip(guess, bounds[:, 0], bounds[:, 1])


def physics_guess(config: Mapping[str, Any], states: np.ndarray, actions: Sequence[int] | np.ndarray) -> np.ndarray:
    """(n, 4) outcome guess in OUTCOME_NAMES order from the true states (n, 4) [forward_m, sideways_m,
    heading_error_rad, tilt_deg] at the decisions and candidate indices (n,) (formulas: the module docstring)."""
    return _guess(lesson_from_config(config), _outcome_bounds(config), _track_factors(config), states, np.asarray(actions))


def _objective(weights: Mapping[str, float], predicted: OutcomeDistribution) -> np.ndarray:
    if not isinstance(predicted, OutcomeDistribution):
        raise TypeError(f"predicted must be an OutcomeDistribution; got {type(predicted).__name__}")
    if tuple(predicted.names) != OUTCOME_NAMES:
        raise ValueError(f"predicted outcomes {predicted.names} are not {OUTCOME_NAMES}")
    return (weights["progress"] * predicted.mean_of("progress_m") - weights["offset"] * predicted.mean_of("offset_m")
            - weights["tilt"] * predicted.mean_of("tilt_deg") - weights["fall"] * predicted.mean_of("fall")
            - weights["spread"] * predicted.spread_of("progress_m"))


def objective(config: Mapping[str, Any], predicted: OutcomeDistribution) -> np.ndarray:
    """(n,) utility, higher is better, with the lesson's goal.weights (the module docstring)."""
    return _objective(lesson_from_config(config).weights, predicted)


class _Rule:
    """The rule of thumb on one lesson: argmax of the objective of the physics guess, first in candidate order."""

    def __init__(self, config: Mapping[str, Any], lesson: Lesson) -> None:
        self.lesson, self.bounds, self.factors = lesson, _outcome_bounds(config), _track_factors(config)
        self.levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)

    def choose(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64).reshape(-1, 4)
        n, c = states.shape[0], len(self.lesson.names)
        physics = _guess(self.lesson, self.bounds, self.factors, np.repeat(states, c, axis=0), np.tile(np.arange(c), n))
        scores = _objective(self.lesson.weights, OutcomeDistribution.point(OUTCOME_NAMES, physics, self.levels))
        return np.argmax(scores.reshape(n, c), axis=1)


def physics_choice(config: Mapping[str, Any], states: np.ndarray) -> np.ndarray:
    """(n,) the rule of thumb's candidate for true states (n, 4)."""
    return _Rule(config, lesson_from_config(config)).choose(states)


@dataclass(frozen=True)
class ReflexSpec:
    """What learners and scorers need of reflex (not a SkillSpec; the registry does not load it)."""

    name: str
    action_names: tuple[str, ...]
    action_features: np.ndarray
    action_feature_names: tuple[str, ...]
    no_op_index: None
    outcome_names: tuple[str, ...]
    outcome_kinds: tuple[str, ...]
    outcome_bounds: np.ndarray
    feature_names: tuple[str, ...]
    weights: Mapping[str, float] = field(repr=False, compare=False)

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        return _objective(self.weights, predicted)


def build_observed_spec(config: Mapping[str, Any]) -> ReflexSpec:
    """The ReflexSpec of the config's lesson. ConfigError without a declaration or with a problem."""
    lesson = lesson_from_config(config)
    _track_factors(config)
    return ReflexSpec(name=SKILL, action_names=lesson.names, action_features=lesson.features,
                      action_feature_names=ACTION_FEATURE_NAMES, no_op_index=None, outcome_names=OUTCOME_NAMES,
                      outcome_kinds=OUTCOME_KINDS, outcome_bounds=_outcome_bounds(config),
                      feature_names=lesson.feature_names(), weights=lesson.weights)


# ----- scenarios -------------------------------------------------------------------------------------------------


def sample_reflex_scenarios(config: Mapping[str, Any], rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    """n scenarios from the lesson's practice world (module docstring); no world needed."""
    from learned_judgment.env.mound import height_for_slope

    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator; got {type(rng).__name__}")
    lesson = lesson_from_config(config)
    mound = require(config, "world.mound")
    w = lesson.world
    if lesson.start == "trotting":
        situation = {"dog_forward_speed_m_s": lesson.nominal}
    else:
        situation = {"dog_forward_speed_m_s": 0.0, "settle_s": float(require(config, f"{_ROOT}.standing_settle_s"))}
    out = []
    for index in range(int(n)):
        u = rng.random(6)
        scenario: dict[str, Any] = {"scenario_id": index, "kind": "flat", "mound": None, "situation": dict(situation)}
        if u[0] >= w["flat_share"]:
            radius = w["radius_m"][0] + u[1] * (w["radius_m"][1] - w["radius_m"][0])
            height = w["height_m"][0] + u[2] * (w["height_m"][1] - w["height_m"][0])
            height = min(height, height_for_slope(radius, float(mound["max_rim_slope_deg"])) - 1e-9)
            ahead = w["centre_ahead_m"][0] + u[3] * (w["centre_ahead_m"][1] - w["centre_ahead_m"][0])
            ahead = max(ahead, radius + float(mound["min_gap_m"]) + 1e-6)
            side = w["centre_side_m"][0] + u[4] * (w["centre_side_m"][1] - w["centre_side_m"][0])
            keys = {"mound_x_m": float(ahead), "mound_y_m": float(side), "mound_radius_m": float(radius),
                    "mound_height_m": float(height)}
            scenario.update(kind="mound", mound=keys)
            scenario["situation"].update(keys)
        out.append(scenario)
    return out


def _split_plan(config: Mapping[str, Any], split: str, n: int) -> list[dict[str, Any]]:
    seed = require(config, "runtime.seed")
    return sample_reflex_scenarios(config, np.random.default_rng(np.random.SeedSequence([seed, SPLIT_CODES[split]])), n)


# ----- episodes --------------------------------------------------------------------------------------------------


def run_episode(world: Any, scenario: Mapping[str, Any], attempt_seed: int,
                policy: Callable[[np.ndarray, Mapping[str, float]], int], *, score_tail: bool = True,
                keep_qpos: bool = False) -> dict[str, Any]:
    """One closed-loop episode on the real world (module docstring). ``policy(window, state)`` returns a candidate
    index; ``window`` is the (D,) float32 window in [0, 1] (the ONLY thing a blind policy may read) and ``state`` the
    scorer's truth at the decision (used by the rule of thumb only). Returns {"scenario_id", "kind", "mound",
    "decisions": [{"t_s", "action", "window", "state", "measured", "horizon_s"}], "track": {"t_s", "forward_m",
    "sideways_m", "heading_rad", "tilt_deg", "fallen", "path_forward_m", "path_sideways_m"} (+ "qpos" (N, 19) with
    keep_qpos), "outcome": {...}, "start", "wall_s"}."""
    config = world.config
    lesson = lesson_from_config(config)
    dt = lesson.period_s
    if abs(world.policy.control_dt - dt) > 1e-12:
        raise ValueError(f"the world's control period {world.policy.control_dt} s is not update_period_s {dt} s")
    low, span = _scaler(lesson)
    default = _default_pose(config)
    c = len(lesson.names)
    clock = time.perf_counter()
    obs = world.reset(dict(scenario["situation"]), attempt_seed=int(attempt_seed))
    p0, yaw0 = np.asarray(obs["base_pos_m"][:2], dtype=np.float64), _yaw(obs["base_quat_wxyz"])
    ahead, left = np.array([math.cos(yaw0), math.sin(yaw0)]), np.array([-math.sin(yaw0), math.cos(yaw0)])
    keep = lesson.history_n + 1
    ring: list[np.ndarray] = []
    track = {"t_s": [], "forward_m": [], "sideways_m": [], "heading_rad": [], "tilt_deg": [], "fallen": [],
             "path_forward_m": [], "path_sideways_m": []}
    qpos: list[np.ndarray] = []
    path = [0.0, 0.0]  # dead reckoning (path_memory): the dog's own commands along its own IMU heading, from t_s 0

    def observe(o: Mapping[str, Any]) -> None:
        heading = _wrap(_yaw(o["base_quat_wxyz"]) - yaw0)
        if track["heading_rad"]:  # one control period since the last observation, under the command it issued
            mid = track["heading_rad"][-1] + 0.5 * _wrap(heading - track["heading_rad"][-1])
            step_m = float(o["command_active"]["vel_x"]) * dt
            path[0] += step_m * math.cos(mid)
            path[1] += step_m * math.sin(mid)
        scaled = np.clip((sense_channels(o, lesson.groups, yaw0, default, path) - low) / span, 0.0, 1.0)
        ring.append(scaled)
        if len(ring) > keep:
            del ring[0]
        d = np.asarray(o["base_pos_m"][:2], dtype=np.float64) - p0
        track["t_s"].append(float(o["t_s"]))
        track["forward_m"].append(float(d @ ahead))
        track["sideways_m"].append(float(d @ left))
        track["heading_rad"].append(heading)
        track["path_forward_m"].append(path[0])
        track["path_sideways_m"].append(path[1])
        track["tilt_deg"].append(_tilt_deg(o["base_quat_wxyz"]))
        track["fallen"].append(bool(o["fallen"]))
        if keep_qpos:
            qpos.append(np.concatenate([o["base_pos_m"], o["base_quat_wxyz"], o["joint_pos_rad"]]).astype(np.float64))

    observe(obs)
    command = dict(lesson.commands[0])  # the nominal command (candidate 0, straight)
    decisions: list[dict[str, Any]] = []
    p, next_decision, end, reason = 0, lesson.history_n, None, None
    while True:
        if end is None:
            if track["fallen"][p]:
                end, reason = p, "fall"
            elif track["forward_m"][p] >= lesson.forward_m:
                end, reason = p, "goal"
            elif p >= lesson.limit_n:
                end, reason = p, "time"
        if end is not None and (reason == "fall" or not score_tail or p >= end + lesson.horizon_n):
            break
        if p == next_decision:
            window = np.concatenate([ring[len(ring) - 1 - (lesson.samples - 1 - k) * lesson.sample_n]
                                     for k in range(lesson.samples)]).astype(np.float32)
            state = {"forward_m": track["forward_m"][p], "sideways_m": track["sideways_m"][p],
                     "heading_error_rad": track["heading_rad"][p], "tilt_deg": track["tilt_deg"][p]}
            action = policy(window.copy(), dict(state))
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < c:
                raise ValueError(f"policy must return a candidate index in [0, {c}); got {action!r}")
            command = dict(lesson.commands[int(action)])
            if end is None:
                decisions.append({"t_s": p * dt, "period": p, "action": int(action), "window": window,
                                  "state": [state[k] for k in STATE_NAMES]})
            next_decision += lesson.decide_n
        world.step(command, dt, on_period=observe)
        p += 1
    outcome_world = world.finish_episode()
    if outcome_world["sim_unstable"]:
        raise RuntimeError("MuJoCo flagged a bad state in this episode (sim_unstable 1); its outcome is no measurement")
    last = len(track["t_s"]) - 1
    f, s, tilt, fallen = (np.array(track[k]) for k in ("forward_m", "sideways_m", "tilt_deg", "fallen"))
    for d in decisions:
        a, b = d["period"], min(d["period"] + lesson.horizon_n, last)
        d["measured"] = [float(f[b] - f[a]), float(abs(s[b])), float(tilt[a + 1:b + 1].mean() if b > a else tilt[a]),
                         float(fallen[b])]
        d["horizon_s"] = (b - a) * dt
    outcome = {
        "end_reason": reason, "reached": int(reason == "goal" and abs(s[end]) <= lesson.tolerance_m),
        "crossed_goal_line": int(reason == "goal"), "time_to_goal_s": end * dt if reason == "goal" else None,
        "end_s": end * dt, "final_forward_m": float(f[end]), "final_sideways_m": float(s[end]),
        "offset_at_goal_line_m": float(abs(s[end])) if reason == "goal" else None,
        "path_memory_forward_m": float(track["path_forward_m"][end]),
        "path_memory_sideways_m": float(track["path_sideways_m"][end]),
        "max_tilt_deg": float(tilt[:end + 1].max()), "fall": int(bool(fallen[:end + 1].any())),
        "foot_slip_m": float(outcome_world["foot_slip_m"]),
        "foot_slip_note": "over the whole stepped episode" + (", scoring tail included" if score_tail else ""),
        "n_decisions": len(decisions),
    }
    result = {
        "scenario_id": scenario.get("scenario_id"), "kind": scenario.get("kind"), "mound": scenario.get("mound"),
        "decisions": decisions, "outcome": outcome,
        "track": {k: np.array(v) for k, v in track.items()},
        "start": {"base_pos_m": np.asarray(obs["base_pos_m"], dtype=np.float64).tolist(), "yaw_rad": yaw0},
        "wall_s": time.perf_counter() - clock,
    }
    if keep_qpos:
        result["track"]["qpos"] = np.stack(qpos)
    return result


# ----- policies and workers --------------------------------------------------------------------------------------


class _StudentPolicy:
    """The exported reflex student (observed.save_student, frames None): argmax of the action head, window only."""

    def __init__(self, directory: str | Path, config: Mapping[str, Any]) -> None:
        from learned_judgment.observed import INPUT_ENCODING, STUDENT_FORMAT, student_readout
        from learned_judgment.students.snn_core import load_network

        folder = Path(directory)
        student = json.loads((folder / "student.json").read_text(encoding="utf-8"))
        spec = build_observed_spec(config)
        c, d = len(spec.action_names), len(spec.feature_names)
        expected = {"format": STUDENT_FORMAT, "skill": SKILL, "action_names": list(spec.action_names),
                    "heads": {"action": [0, c]}, "input": {"features": d, "encoding": INPUT_ENCODING}}
        for key, value in expected.items():
            if student.get(key) != value:
                raise ValueError(f"{folder / 'student.json'} {key} is {student.get(key)!r}; the lesson needs {value!r}")
        self.network, _ = load_network(folder / "network")
        if self.network.n_inputs != d or self.network.n_outputs != c:
            raise ValueError(f"the network is {self.network.layer_sizes}; the reflex student needs {d} inputs, {c} outputs")
        self.heads, self.digest = expected["heads"], student.get("network", {}).get("sha256")
        self._readout = student_readout

    def __call__(self, window: np.ndarray, _state: Mapping[str, float] | None = None) -> int:
        _, scores = self._readout(self.network, np.asarray(window)[None, :], self.heads)
        return int(np.argmax(scores[0]))


_WORKER: dict[str, Any] = {}


def _worker_init(config: Mapping[str, Any], student_dir: str | None) -> None:
    """Per process: its own Go2World (observed.world_seed, observed.world_preset; no trace files) and the student."""
    from learned_judgment.env import Go2World

    config = copy.deepcopy(dict(config))
    config["world"]["save_traces"] = False
    _WORKER["config"] = config
    _WORKER["world"] = Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    _WORKER["rule"] = _Rule(config, lesson_from_config(config))
    _WORKER["student"] = None if student_dir is None else _StudentPolicy(student_dir, config)


def _episode_task(task: tuple) -> dict[str, Any]:
    """(scenario, mode, score_tail, keep_qpos[, explore]): mode attempt | attempt_student | rule | no_op |
    physics_only | snn; explore = the exploration fraction of the attempt modes."""
    scenario, mode, score_tail, keep_qpos = task[:4]
    config, rule = _WORKER["config"], _WORKER["rule"]
    sources: list[int] = []

    def rule_policy(_window: np.ndarray, state: Mapping[str, float]) -> int:
        return int(rule.choose(np.array([[state[k] for k in STATE_NAMES]]))[0])

    if mode in ("rule", "physics_only"):
        policy = rule_policy
    elif mode == "no_op":
        def policy(_window, _state):
            return 0
    elif mode == "snn":
        student = _WORKER["student"]

        def policy(window, _state):
            return student(window)  # the window only: the student never sees the scorer's state
    elif mode in ("attempt", "attempt_student"):
        seed, fraction = int(require(config, "runtime.seed")), float(task[4])
        c = len(rule.lesson.names)
        if mode == "attempt_student":
            student = _WORKER["student"]
            own, own_source = (lambda window, _state: student(window)), SOURCES.index("student")
        else:
            own, own_source = rule_policy, SOURCES.index("rule")

        def policy(window, state):
            u = np.random.default_rng(np.random.SeedSequence(
                [seed, SPLIT_CODES["context"], int(scenario["scenario_id"]), len(sources)])).random(2)
            explore = bool(u[0] < fraction)
            sources.append(SOURCES.index("explore") if explore else own_source)
            return min(int(math.floor(u[1] * c)), c - 1) if explore else own(window, state)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    result = run_episode(_WORKER["world"], scenario, int(scenario["scenario_id"]), policy, score_tail=score_tail,
                         keep_qpos=keep_qpos)
    result["policy"] = mode
    if mode in ("attempt", "attempt_student"):
        for decision, source in zip(result["decisions"], sources):
            decision["source"] = source
    else:
        for decision in result["decisions"]:
            decision["source"] = 0
    return result


@contextmanager
def _pool(config: Mapping[str, Any], student_dir: str | None = None) -> Iterator[Any]:
    workers = require(config, "observed.workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"config.yaml key 'observed.workers' must be an int >= 1; got {workers!r}")
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_worker_init,
                                                   initargs=(config, student_dir)) as pool:
        yield pool


def _map(pool: Any, tasks: list[tuple], label: str, on_result: Callable[[int], None] | None = None) -> list[dict]:
    every = max(1, len(tasks) // 10)
    results = []
    for result in pool.imap(_episode_task, tasks):
        results.append(result)
        if len(results) % every == 0 or len(results) == len(tasks):
            print(f"{label} {len(results)}/{len(tasks)}", flush=True)
            if on_result is not None:
                on_result(len(results))
    return results


def _exploration_fraction(config: Mapping[str, Any], key: str = "engine.attempt_exploration_fraction") -> float:
    from learned_judgment.skills.ramp import _value

    fraction = _value(config, key)
    if not 0.0 <= fraction <= 1.0:
        raise ConfigError(f"{key} must lie in [0, 1]; got {fraction}")
    return fraction


# ----- files -----------------------------------------------------------------------------------------------------


def _seeds(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {"runtime_seed": require(config, "runtime.seed"), "split": split, "split_code": SPLIT_CODES[split],
            "scenarios": "np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code])), rng.random(6) "
                         "per scenario", "attempt_seed": "scenario_id",
            "world_seed": require(config, "observed.world_seed"), "world_preset": require(config, "observed.world_preset")}


def _meta(config: Mapping[str, Any], split: str, n_situations: int) -> dict[str, Any]:
    lesson = lesson_from_config(config)
    spec = build_observed_spec(config)
    return {
        "skill": SKILL, "split": split, "action_names": list(spec.action_names),
        "action_feature_names": list(ACTION_FEATURE_NAMES), "action_features": spec.action_features.tolist(),
        "outcome_names": list(OUTCOME_NAMES), "outcome_kinds": list(OUTCOME_KINDS),
        "outcome_bounds": spec.outcome_bounds.tolist(), "feature_names": list(spec.feature_names), "frames": None,
        "window": {"groups": list(lesson.groups), "channels": [list(c) for c in lesson.channels],
                   "samples": lesson.samples, "sample_every_s": lesson.sample_n * lesson.period_s,
                   "sample_times_s": lesson.sample_times_s(), "width": lesson.width,
                   "layout": "sample-major, oldest sample first; each channel (value - low) / (high - low) clipped to [0, 1]"},
        "state_names": list(STATE_NAMES),
        "state_note": "scorer truth at the decision (physics guess and evaluation only), never a policy input",
        "lesson": dict(lesson.declaration), "n_situations": n_situations, "seeds": _seeds(config, split),
        "world_seed": require(config, "observed.world_seed"), "world_preset": require(config, "observed.world_preset"),
        "config_digest": config_digest(config), "where": _WHERE,
    }


def _episode_record(result: Mapping[str, Any]) -> dict[str, Any]:
    actions = Counter(d["action"] for d in result["decisions"])
    return {"scenario_id": result["scenario_id"], "kind": result["kind"], "mound": result["mound"],
            "policy": result["policy"], "outcome": result["outcome"], "wall_s": result["wall_s"],
            "decisions": len(result["decisions"]),
            "explored": sum(d.get("source", 0) == SOURCES.index("explore") for d in result["decisions"]),
            "action_counts": dict(sorted(actions.items()))}


def _row_arrays(results: Sequence[Mapping[str, Any]], width: int, k: int, index_of: Callable[[Mapping], int]) -> dict:
    decisions = [(index_of(r), d) for r in results for d in r["decisions"]]
    return {
        "scenario_index": np.array([i for i, _ in decisions], dtype=np.int64),
        "decision_time_s": np.array([d["t_s"] for _, d in decisions], dtype=np.float64),
        "action": np.array([d["action"] for _, d in decisions], dtype=np.int64),
        "features": np.array([d["window"] for _, d in decisions], dtype=np.float32).reshape(-1, width),
        "measured": np.array([d.get("measured", [0.0] * k) for _, d in decisions], dtype=np.float64).reshape(-1, k),
        "state": np.array([d["state"] for _, d in decisions], dtype=np.float64).reshape(-1, len(STATE_NAMES)),
        "horizon_s": np.array([d.get("horizon_s", 0.0) for _, d in decisions], dtype=np.float64),
        "source": np.array([d.get("source", 0) for _, d in decisions], dtype=np.int64),
    }


def _print_episodes(label: str, results: Sequence[Mapping[str, Any]], wall_s: float) -> None:
    for kind in KINDS:
        own = [r for r in results if r["kind"] == kind]
        if own:
            print(f"{label} {kind}: {len(own)} episodes, reached {sum(r['outcome']['reached'] for r in own)}, crossed "
                  f"the goal line {sum(r['outcome']['crossed_goal_line'] for r in own)}, falls "
                  f"{sum(r['outcome']['fall'] for r in own)}, decisions per episode "
                  f"{np.mean([r['outcome']['n_decisions'] for r in own]):.1f}", flush=True)
    episode_walls = [r["wall_s"] for r in results]
    print(f"wall {wall_s:.1f} s for {len(results)} episodes (pool start-up included); per episode in its worker "
          f"mean {np.mean(episode_walls):.1f} s, max {np.max(episode_walls):.1f} s", flush=True)


def _source_counts(source: np.ndarray) -> dict[str, int]:
    return {name: int((np.asarray(source) == code).sum()) for code, name in enumerate(SOURCES)}


def collect_attempts(config: Mapping[str, Any], out_dir: str | Path, n: int | None = None, *, policy: str = "rule",
                     student_dir: str | Path | None = None, explore: float | None = None,
                     first: int = 0) -> dict[str, Any]:
    """N exploration episodes on context scenarios first .. first + N - 1, picked by the rule of thumb (policy "rule")
    or the trained student (policy "student", on-policy) apart from the exploration share; every decision a row (module
    docstring). Writes rows.npz/json."""
    n = _count("n", require(config, f"{_OBSERVED}.attempt_episodes") if n is None else n)
    if isinstance(first, bool) or not isinstance(first, int) or first < 0:
        raise ValueError(f"first must be an int >= 0; got {first!r}")
    if policy not in ("rule", "student"):
        raise ValueError(f"policy must be rule or student; got {policy!r}")
    if (policy == "student") != (student_dir is not None):
        raise ValueError("policy student needs a student directory, and only it takes one")
    spec, lesson = build_observed_spec(config), lesson_from_config(config)
    key = "engine.attempt_exploration_fraction" if policy == "rule" else f"{_OBSERVED}.student_explore_fraction"
    fraction = _exploration_fraction(config, key) if explore is None else float(explore)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"the exploration fraction must lie in [0, 1]; got {fraction}")
    student = None if student_dir is None else _StudentPolicy(student_dir, config)  # checked before any worker
    scenarios = _split_plan(config, "context", first + n)[first:]
    print(f"attempts: {n} context episodes, scenarios {first}..{first + n - 1} ({Counter(s['kind'] for s in scenarios)}), "
          f"policy {policy}{'' if student is None else ' ' + str(student_dir)}, window D {lesson.width}, "
          f"{len(spec.action_names)} candidates {list(spec.action_names)}, exploration fraction {fraction}", flush=True)
    mode = "attempt" if policy == "rule" else "attempt_student"
    start = time.perf_counter()
    with _pool(config, None if student_dir is None else str(student_dir)) as pool:
        results = _map(pool, [(s, mode, True, False, fraction) for s in scenarios], "attempts:")
    wall_s = time.perf_counter() - start
    results = sorted(results, key=lambda r: r["scenario_id"])
    arrays = _row_arrays(results, lesson.width, len(OUTCOME_NAMES), lambda r: r["scenario_id"] - first)
    arrays["physics"] = _guess(lesson, spec.outcome_bounds, _track_factors(config), arrays["state"], arrays["action"])
    arrays["measured"] = np.clip(arrays["measured"], spec.outcome_bounds[:, 0], spec.outcome_bounds[:, 1])
    meta = _meta(config, "context", len(results))
    meta.update(collection="attempts", policy=policy,
                student=None if student is None else {"path": Path(student_dir).resolve().as_posix(),
                                                      "network_sha256": student.digest},
                first_scenario=first, n_attempts=n, n_scenarios=n, exploration_fraction=fraction,
                exploration_key=key if explore is None else "--explore", wall_s=wall_s,
                kind_counts=dict(Counter(s["kind"] for s in scenarios)), scenarios=scenarios,
                episodes=[_episode_record(r) for r in results], attempt_sources=_source_counts(arrays["source"]))
    meta["seeds"]["attempt_choice"] = ("np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code, "
                                       "scenario_id, decision_index])).random(2): u[0] < exploration_fraction -> "
                                       f"candidate min(floor(u[1] * C), C - 1), else the {policy} policy's pick")
    meta["seeds"]["scenario_ids"] = f"context split scenarios {first} .. {first + n - 1}; scenario_index = scenario_id - {first}"
    meta["measured_note"] = ("each row's outcomes over the next horizon_s of the closed loop (later decisions inside the "
                             "horizon act too); clipped to outcome_bounds; horizon_s per row in rows.npz")
    _write(Path(out_dir), "rows", meta, **arrays)
    _print_episodes("attempts", results, wall_s)
    print(f"rows {arrays['action'].shape[0]} (every decision); by source {meta['attempt_sources']}; D "
          f"{lesson.width}; wrote {Path(out_dir) / 'rows.npz'} and rows.json", flush=True)
    return meta


_ROW_EXTRAS = ("state", "horizon_s", "source")


def _merged_rows(config: Mapping[str, Any], inputs: Sequence[str | Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """(arrays, meta) of rows files of this lesson concatenated in order; part p's scenario_index offset by the
    scenarios of the parts before it (module docstring: merge)."""
    from learned_judgment.observed import load_rows

    if not inputs:
        raise ValueError("merge needs at least one rows directory")
    spec, lesson = build_observed_spec(config), lesson_from_config(config)
    arrays: dict[str, list[np.ndarray]] = {}
    scenarios, episodes, parts, kinds = [], [], [], Counter()
    offset = 0
    for p, directory in enumerate(Path(d) for d in inputs):
        rows = load_rows(directory)
        meta = rows["meta"]
        for key, expected in (("skill", SKILL), ("action_names", list(spec.action_names)),
                              ("feature_names", list(spec.feature_names)), ("outcome_names", list(OUTCOME_NAMES)),
                              ("lesson", _jsonable(dict(lesson.declaration)))):
            if _jsonable(meta.get(key)) != expected:
                raise ValueError(f"{directory} is not a rows file of this lesson: its {key} differs")
        with np.load(directory / "rows.npz") as archive:
            extras = {k: np.array(archive[k]) for k in _ROW_EXTRAS}
        n_scenarios = len(meta["scenarios"])
        if rows["action"].shape[0] and int(rows["scenario_index"].max()) >= n_scenarios:
            raise ValueError(f"{directory}: a scenario_index is beyond its {n_scenarios} scenarios")
        part = {"scenario_index": np.asarray(rows["scenario_index"], dtype=np.int64) + offset,
                **{k: np.asarray(rows[k]) for k in ("decision_time_s", "action", "features", "physics", "measured")},
                **extras}
        for k, v in part.items():
            arrays.setdefault(k, []).append(v)
        scenarios += [{**s, "part": p} for s in meta["scenarios"]]
        episodes += [{**e, "part": p} for e in meta.get("episodes", [])]
        kinds.update(meta.get("kind_counts", {}))
        parts.append({"path": directory.as_posix(), "n_rows": int(rows["action"].shape[0]), "n_scenarios": n_scenarios,
                      "index_offset": offset, "collection": meta.get("collection"), "policy": meta.get("policy", "rule"),
                      "student": meta.get("student"), "first_scenario": meta.get("first_scenario", 0),
                      "exploration_fraction": meta.get("exploration_fraction"), "config_digest": meta.get("config_digest"),
                      "attempt_sources": _source_counts(extras["source"])})
        offset += n_scenarios
    out = {k: np.concatenate(v) for k, v in arrays.items()}
    out["features"] = out["features"].astype(np.float32)
    meta = _meta(config, "context", offset)
    meta.update(collection="merged", parts=parts, n_attempts=offset, n_scenarios=offset, kind_counts=dict(kinds), scenarios=scenarios, episodes=episodes,
                attempt_sources=_source_counts(out["source"]),
                measured_note="the parts' measured outcomes as stored (each part's own measured_note)")
    meta["seeds"]["merge"] = "parts concatenated in the given order; part p's scenario_index + its index_offset"
    return out, meta


def merge_rows(config: Mapping[str, Any], inputs: Sequence[str | Path], out_dir: str | Path) -> dict[str, Any]:
    """Rows files of this lesson concatenated (module docstring: merge). Writes rows.npz/json."""
    arrays, meta = _merged_rows(config, inputs)
    _write(Path(out_dir), "rows", meta, **arrays)
    print(f"merge: {len(meta['parts'])} parts, {arrays['action'].shape[0]} rows ("
          + ", ".join(f"{q['path']} {q['n_rows']}" for q in meta["parts"])
          + f"), {meta['n_scenarios']} scenarios, by source {meta['attempt_sources']}; wrote {Path(out_dir) / 'rows.npz'}",
          flush=True)
    return meta


def collect_dream(config: Mapping[str, Any], out_dir: str | Path, n: int | None = None,
                  context_dir: str | Path | Sequence[str | Path] | None = None) -> dict[str, Any]:
    """The dream: every window of the attempts' episodes (one attempts directory, or several merged as merge does)
    plus the windows of N further rule-of-thumb episodes on dream scenarios, each with the physics guess of EVERY
    candidate from its recorded true state. Writes dream.npz/json."""
    n = _count("n", require(config, f"{_OBSERVED}.dream_episodes") if n is None else n)
    spec, lesson = build_observed_spec(config), lesson_from_config(config)
    if context_dir is None:
        context_dirs = [Path(out_dir).parent / "context"]
    elif isinstance(context_dir, (str, Path)):
        context_dirs = [Path(context_dir)]
    else:
        context_dirs = [Path(d) for d in context_dir]
    rows, rows_meta = _merged_rows(config, context_dirs)  # checks every part against this lesson
    rows["meta"] = rows_meta
    if rows["features"].shape[1] != lesson.width:
        raise ValueError(f"{context_dirs} hold windows of width {rows['features'].shape[1]}; the lesson's is {lesson.width}")
    attempt_state = rows["state"]
    context_text = " + ".join(d.as_posix() for d in context_dirs)
    scenarios = _split_plan(config, "dream", n)
    print(f"dream: {rows['action'].shape[0]} windows of {len(rows['meta']['episodes'])} attempt episodes ({context_text}; "
          f"by source {rows_meta['attempt_sources']}) + {n} rule-of-thumb episodes on dream scenarios "
          f"({Counter(s['kind'] for s in scenarios)})", flush=True)
    start = time.perf_counter()
    with _pool(config) as pool:
        results = _map(pool, [(s, "rule", False, False) for s in scenarios], "dream episodes:")
    wall_s = time.perf_counter() - start
    results = sorted(results, key=lambda r: r["scenario_id"])
    attempt_ids = sorted({int(i) for i in rows["scenario_index"]})
    n_attempts = len(attempt_ids)
    extra = _row_arrays(results, lesson.width, len(OUTCOME_NAMES), lambda r: n_attempts + r["scenario_id"])
    features = np.concatenate([np.asarray(rows["features"], dtype=np.float32), extra["features"]])
    state = np.concatenate([attempt_state, extra["state"]])
    episode = np.concatenate([np.searchsorted(attempt_ids, rows["scenario_index"]).astype(np.int64),
                              extra["scenario_index"]])
    times = np.concatenate([rows["decision_time_s"], extra["decision_time_s"]])
    source = np.concatenate([np.zeros(rows["action"].shape[0], dtype=np.int64), np.ones(extra["action"].shape[0], dtype=np.int64)])
    m, c = features.shape[0], len(spec.action_names)
    physics = _guess(lesson, spec.outcome_bounds, _track_factors(config), np.repeat(state, c, axis=0),
                     np.tile(np.arange(c), m)).reshape(m, c, len(OUTCOME_NAMES))
    meta = _meta(config, "dream", m)
    meta.update(
        wall_s=wall_s, context=context_dirs[0].as_posix() if len(context_dirs) == 1 else [d.as_posix() for d in context_dirs],
        context_parts=rows_meta["parts"], attempt_window_sources=rows_meta["attempt_sources"], n_scenarios=n,
        scenarios=scenarios,
        episodes=[{"index": i, "source": "attempts", "split": "context", "scenario_index": sid,
                   "scenario_id": rows_meta["scenarios"][sid]["scenario_id"], "part": rows_meta["scenarios"][sid]["part"]}
                  for i, sid in enumerate(attempt_ids)]
        + [{"index": n_attempts + r["scenario_id"], "source": "rule_of_thumb", "split": "dream", **_episode_record(r)}
           for r in results],
        sampling={
            "simulator": "real MuJoCo Go2World episodes: raw sensor histories cannot be invented, so every dream situation "
                         "is a sensor window the real simulator produced",
            "situations": f"all {rows['action'].shape[0]} windows of the attempts' episodes ({context_text}) and "
                          f"the {extra['action'].shape[0]} windows of {n} further unbilled episodes walked with the rule of "
                          "thumb on dream-split scenarios",
            "physics": "the physics guess of EVERY candidate from the true state the episode recorded at that window "
                       "(scorer side; never a student input)",
            "scenario_index": "the dream episode index (attempt episodes first, in scenario order, then the extra ones)",
        })
    _write(Path(out_dir), "dream", meta, scenario_index=episode, decision_time_s=times, features=features,
           physics=physics, state=state, source=source)
    rule = _Rule(config, lesson).choose(state)
    counts = Counter(spec.action_names[i] for i in rule)
    print("rule-of-thumb candidate counts over the dream: " + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    _print_episodes("dream", results, wall_s)
    print(f"dream situations {m} ({rows['action'].shape[0]} from attempts, {extra['action'].shape[0]} extra) x {c} "
          f"candidates; wrote {Path(out_dir) / 'dream.npz'} and dream.json", flush=True)
    return meta


# ----- evaluation ------------------------------------------------------------------------------------------------


def _wilson(k: int, n: int, level: float) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    z = statistics.NormalDist().inv_cdf(0.5 + level / 2.0)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _paired_bootstrap(differences: Sequence[float], resamples: int, level: float, rng: np.random.Generator) -> dict:
    d = np.asarray(differences, dtype=np.float64)
    if d.size == 0:
        return {"diff": None, "ci_low": None, "ci_high": None, "n": 0}
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        means[start:stop] = d[rng.integers(0, d.size, size=(stop - start, d.size))].mean(axis=1)
    low, high = np.quantile(means, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return {"diff": float(d.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(d.size)}


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def _summary(outcomes: Sequence[Mapping[str, Any]], level: float) -> dict[str, Any]:
    n = len(outcomes)
    reached = sum(o["reached"] for o in outcomes)
    low, high = _wilson(reached, n, level)
    return {
        "n": n, "reached": reached, "reached_rate": reached / n if n else None, "reached_ci": [low, high],
        "falls": sum(o["fall"] for o in outcomes),
        "crossed_goal_line": sum(o["crossed_goal_line"] for o in outcomes),
        "mean_offset_at_goal_line_m": _mean([o["offset_at_goal_line_m"] for o in outcomes]),
        "mean_time_to_goal_s": _mean([o["time_to_goal_s"] for o in outcomes if o["reached"]]),
        "mean_max_tilt_deg": _mean([o["max_tilt_deg"] for o in outcomes]),
        "mean_foot_slip_m": _mean([o["foot_slip_m"] for o in outcomes]),
        "mean_final_abs_sideways_m": _mean([abs(o["final_sideways_m"]) for o in outcomes]),
    }


def _difficulty(scenario: Mapping[str, Any], steep_from_deg: float) -> tuple[str, float | None]:
    """("flat" | "gentle" | "steep", rim slope in degrees or None) of a scenario (scorer truth)."""
    from learned_judgment.env.mound import rim_slope_deg

    mound = scenario.get("mound")
    if mound is None:
        return "flat", None
    slope = float(rim_slope_deg(mound["mound_radius_m"], mound["mound_height_m"]))
    return ("steep" if slope >= steep_from_deg else "gentle"), slope


def _passage(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """How the episode's path passed its mound (module docstring: DIFFICULTY); None on a flat course."""
    mound = result.get("mound")
    if mound is None:
        return None
    from learned_judgment.env.mound import sphere_radius

    f, s = np.asarray(result["track"]["forward_m"]), np.asarray(result["track"]["sideways_m"])
    cx, cy, radius = float(mound["mound_x_m"]), float(mound["mound_y_m"]), float(mound["mound_radius_m"])
    height = float(mound["mound_height_m"])
    closest = float(np.min(np.hypot(f - cx, s - cy)))
    how = "over" if closest < 0.5 * radius else ("around" if float(f.max()) >= cx else "not_past")
    big_r = sphere_radius(radius, height)
    ground = math.sqrt(big_r * big_r - closest * closest) - (big_r - height) if closest < radius else 0.0
    return {"how": how, "closest_to_centre_m": closest, "closest_over_radius": closest / radius,
            "ground_at_closest_m": ground, "ground_at_closest_over_height": ground / height}


def _by_difficulty(records: Sequence[Mapping[str, Any]], methods: Sequence[str], level: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in methods:
        out[m] = {}
        for difficulty in DIFFICULTIES:
            own = [r for r in records if r["difficulty"] == difficulty]
            entry = _summary([r[m]["outcome"] for r in own], level)
            if difficulty != "flat":
                hows = Counter(r[m]["passage"]["how"] for r in own)
                entry["passage"] = {how: hows.get(how, 0) for how in PASSAGES}
                entry["reached_by_passage"] = {how: sum(r[m]["outcome"]["reached"] for r in own
                                                        if r[m]["passage"]["how"] == how) for how in PASSAGES}
            out[m][difficulty] = entry
    return out


def _print_difficulty(by: Mapping[str, Any], methods: Sequence[str], steep_from_deg: float) -> None:
    print(f"by difficulty (steep = rim slope >= {steep_from_deg:g} deg; passage over = path within half the radius of "
          f"the centre, around = farther out and came level with the centre, not_past = never came level):", flush=True)
    for m in methods:
        for difficulty in DIFFICULTIES:
            s = by[m][difficulty]
            if not s["n"]:
                continue
            miss = s["mean_offset_at_goal_line_m"]
            passage = "" if difficulty == "flat" else (
                " | " + ", ".join(f"{how} {s['passage'][how]} (reached {s['reached_by_passage'][how]})" for how in PASSAGES))
            print(f"  {m:12s} {difficulty:6s} n {s['n']:2d}: reached {s['reached']:2d}, crossed {s['crossed_goal_line']:2d}, "
                  f"falls {s['falls']}, mean miss at the line {'-' if miss is None else f'{miss:.2f} m'}, final |sideways| "
                  f"{s['mean_final_abs_sideways_m']:.2f} m{passage}", flush=True)


def _merge_result(run_dir: Path, run_id: str, updates: Mapping[str, Any]) -> Path:
    from learned_judgment.observed import _dumps, _utc_now

    path = run_dir / "result.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": run_id, "skill": SKILL}
    if result.get("run_id") != run_id or result.get("skill") != SKILL:
        raise ValueError(f"{path} belongs to run {result.get('run_id')!r} of skill {result.get('skill')!r}")
    result.update(_jsonable(dict(updates)))
    result["updated_utc"] = _utc_now().isoformat()
    atomic_write_text(path, _dumps(result, indent=1) + "\n")
    return path


def _save_tracks(folder: Path, results: Sequence[Mapping[str, Any]], scenarios: Mapping[int, Mapping[str, Any]],
                 config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from learned_judgment.env.mound import rim_slope_deg, sphere_radius

    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for r in results:
        if "qpos" not in r["track"]:
            continue
        name = f"{r['policy']}_s{r['scenario_id']}.npz"
        buffer = io.BytesIO()
        np.savez_compressed(buffer, t_s=r["track"]["t_s"], qpos=r["track"]["qpos"])
        atomic_write_bytes(folder / name, buffer.getvalue())
        mound = None
        if r["mound"] is not None:
            m, yaw, xy = r["mound"], r["start"]["yaw_rad"], np.array(r["start"]["base_pos_m"][:2])
            centre = xy + m["mound_x_m"] * np.array([math.cos(yaw), math.sin(yaw)]) + m["mound_y_m"] * np.array(
                [-math.sin(yaw), math.cos(yaw)])
            big_r = sphere_radius(m["mound_radius_m"], m["mound_height_m"])
            mound = {**m, "centre_xy_m": centre.tolist(), "sphere_radius_m": big_r,
                     "sphere_centre_z_m": m["mound_height_m"] - big_r,
                     "rim_slope_deg": rim_slope_deg(m["mound_radius_m"], m["mound_height_m"]),
                     "rgba": require(config, "world.mound.rgba")}
        index.append({"file": name, "method": r["policy"], "scenario_id": r["scenario_id"], "kind": r["kind"],
                      "mound_world": mound, "start": r["start"], "outcome": r["outcome"],
                      "qpos_layout": "base_pos_m (3), base_quat_wxyz (4), joint_pos_rad (12, FL FR RL RR x hip thigh "
                                     "calf): the world's qpos, one row per control period from t_s 0"})
    atomic_write_text(folder / "index.json", json.dumps(_jsonable(index), indent=1) + "\n")
    return index


def run_methods(config: Mapping[str, Any], student_dir: str | Path | None, split: str, out_dir: str | Path,
                n: int | None = None, run_id: str | None = None, methods: Sequence[str] | None = None) -> dict[str, Any]:
    """Closed-loop episodes of a split for no_op, physics_only and (with a student) snn (or only the given methods),
    same scenarios and attempt seeds; writes <out_dir>/run.json and tracks; with run_id also the result.json UI keys
    and "evaluate" events."""
    from learned_judgment.engine.streams import stream_rng
    from learned_judgment.observed import EventLog, _run_dir

    if split not in RUN_SPLITS:
        raise ValueError(f"split must be one of {RUN_SPLITS}; got {split!r}")
    n = _count("n", require(config, f"{_OBSERVED}.{split}_scenarios") if n is None else n)
    spec, lesson = build_observed_spec(config), lesson_from_config(config)
    student = None if student_dir is None else _StudentPolicy(student_dir, config)  # checked before any worker
    if methods is not None and (not methods or any(m not in METHODS for m in methods)
                                or ("snn" in methods and student is None)):
        raise ValueError(f"methods must be a non-empty subset of {METHODS} (snn needs a student); got {methods!r}")
    methods = [m for m in METHODS if (m != "snn" or student is not None) and (methods is None or m in methods)]
    steep_from = float(require(config, f"{_ROOT}.report.steep_from_deg"))
    level = float(require(config, "evaluation.bootstrap_confidence_level"))
    resamples = _count("evaluation.bootstrap_resamples", require(config, "evaluation.bootstrap_resamples"))
    seed = int(require(config, "runtime.seed"))
    tracks = int(require(config, f"{_OBSERVED}.tracks_per_method"))
    scenarios = _split_plan(config, split, n)
    run_dir = None if run_id is None else _run_dir(config, run_id)
    events = None if run_dir is None else EventLog(run_dir / "events.jsonl", run_id, SKILL)
    section = f"closed_loop_{split}"
    common = {"student": None if student_dir is None else Path(student_dir).resolve().as_posix(),
              "network_sha256": None if student is None else student.digest, "split": split, "n": n, "methods": methods}
    total = len(methods) * n
    if events is not None:
        _merge_result(run_dir, run_id, {section: {"state": "running", **common}})
        events.emit("evaluate", "running", done=0, total=total, data={"split": split, "methods": methods})
    print(f"run: split {split}, {n} scenarios ({Counter(s['kind'] for s in scenarios)}) x {methods}", flush=True)
    folder = Path(out_dir)
    try:
        start = time.perf_counter()
        tasks = [(s, m, False, s["scenario_id"] < tracks) for s in scenarios for m in methods]
        with _pool(config, None if student_dir is None else str(student_dir)) as pool:
            progress = None if events is None else (lambda done: events.emit("evaluate", "running", done=done, total=total))
            results = _map(pool, tasks, "episodes:", progress)
        wall_s = time.perf_counter() - start
        by = {(r["scenario_id"], r["policy"]): r for r in results}
        records = []
        for s in scenarios:
            difficulty, slope = _difficulty(s, steep_from)
            record = {"scenario_id": s["scenario_id"], "kind": s["kind"], "mound": s["mound"], "difficulty": difficulty,
                      "rim_slope_deg": slope}
            for m in methods:
                r = by[(s["scenario_id"], m)]
                record[m] = {"outcome": r["outcome"], "wall_s": r["wall_s"], "passage": _passage(r),
                             "action_counts": {spec.action_names[a]: c for a, c in
                                               sorted(Counter(d["action"] for d in r["decisions"]).items())}}
            records.append(record)
        summary = {m: {"all": _summary([r[m]["outcome"] for r in records], level),
                       **{kind: _summary([r[m]["outcome"] for r in records if r["kind"] == kind], level) for kind in KINDS}}
                   for m in methods}
        by_difficulty = _by_difficulty(records, methods, level)
        difficulty_rule = {"steep_from_deg": steep_from, "key": f"{_ROOT}.report.steep_from_deg",
                           "counts": dict(Counter(r["difficulty"] for r in records)),
                           "passage": "over: the path came within half the mound radius of its centre; around: it stayed "
                                      "farther out and came level with the centre; not_past: farther out and never level "
                                      "with the centre (scorer truth, the start frame)"}
        paired = []
        for a in [m for m in ("snn", "physics_only") if m in methods and "no_op" in methods]:
            stream = f"{SKILL}/run/{split}/paired_bootstrap/{a}"
            d = [r[a]["outcome"]["reached"] - r["no_op"]["outcome"]["reached"] for r in records]
            paired.append({"name": "reached", "a": a, "b": "no_op",
                           **_paired_bootstrap(d, resamples, level, stream_rng(seed, stream, 0)),
                           "method": "percentile bootstrap over scenarios", "resamples": resamples, "level": level,
                           "stream": f"{stream}[0]"})
        index = _save_tracks(folder / "tracks", results, {s["scenario_id"]: s for s in scenarios}, config)
        seeds = _seeds(config, split)
        out = {
            "skill": SKILL, **common, "seeds": seeds, "config_digest": config_digest(config),
            "lesson": dict(lesson.declaration), "action_names": list(spec.action_names),
            "methods_described": {
                "no_op": "the nominal command, never steering: what the walking policy does by itself",
                "physics_only": "the rule of thumb (argmax of the objective of the physics guess); it reads the true "
                                "position and heading, so it is a reference, not a blind policy",
                "snn": "the trained spiking student's argmax candidate from the raw sensor window only"},
            "where": "simulation (MuJoCo Go2World); student inference on this computer's CPU (numpy)",
            "records": records, "summary": summary, "by_difficulty": by_difficulty, "difficulty_rule": difficulty_rule,
            "paired_reached": paired, "tracks": index, "wall_s": wall_s,
            "episode_wall_s": {m: _mean([by[(s["scenario_id"], m)]["wall_s"] for s in scenarios]) for m in methods},
        }
        atomic_write_text(folder / "run.json", json.dumps(_jsonable(out), indent=1, allow_nan=False) + "\n")
        if events is not None:
            metrics = []
            for name, unit, better, getter in (
                    ("reached_rate", "fraction", True, lambda s: (s["reached_rate"], *s["reached_ci"])),
                    ("fall_rate", "fraction", False, lambda s: (s["falls"] / s["n"] if s["n"] else None,
                                                                *_wilson(s["falls"], s["n"], level))),
                    ("mean_offset_m", "m", False, lambda s: (s["mean_offset_at_goal_line_m"], None, None)),
                    ("mean_time_to_goal_s", "s", False, lambda s: (s["mean_time_to_goal_s"], None, None))):
                values = {}
                for m in methods:
                    value, low, high = getter(summary[m]["all"])
                    values[m] = {"value": value, "ci_low": low, "ci_high": high, "n": summary[m]["all"]["n"]}
                metrics.append({"name": name, "unit": unit, "higher_is_better": better, "values": values})
            _merge_result(run_dir, run_id, {
                section: {"state": "done", **common, "records": (folder / "run.json").as_posix(), "summary": summary,
                          "by_difficulty": by_difficulty, "difficulty_rule": difficulty_rule,
                          "paired_reached": paired, "wall_s": wall_s, "seeds": seeds, "where": out["where"],
                          "tracks": (folder / "tracks" / "index.json").as_posix()},
                "seeds": seeds, "scenario_ids": [s["scenario_id"] for s in scenarios], "metrics": metrics,
                "paired": [{k: p[k] for k in ("name", "a", "b", "diff", "ci_low", "ci_high", "n")} for p in paired]})
            events.emit("evaluate", "done", done=total, total=total, data={
                "split": split, "summary": {m: summary[m]["all"] for m in methods}, "paired_reached": paired,
                "records": (folder / "run.json").as_posix()})
    except Exception as error:
        if events is not None:
            events.emit("evaluate", "failed", data={"type": type(error).__name__, "message": str(error)})
            _merge_result(run_dir, run_id, {section: {"state": "failed", **common,
                                                      "error": {"type": type(error).__name__, "message": str(error)}}})
        raise
    for m in methods:
        for kind in ("all", *KINDS):
            s = summary[m][kind]
            if s["n"]:
                print(f"{m} {kind}: n {s['n']}, reached {s['reached']} ({s['reached_rate']:.2f}, Wilson "
                      f"[{s['reached_ci'][0]:.2f}, {s['reached_ci'][1]:.2f}]), crossed {s['crossed_goal_line']}, falls "
                      f"{s['falls']}, |offset| at the line {s['mean_offset_at_goal_line_m']}, time to goal "
                      f"{s['mean_time_to_goal_s']}, max tilt {s['mean_max_tilt_deg']:.1f} deg, slip "
                      f"{s['mean_foot_slip_m']:.2f} m, final |sideways| {s['mean_final_abs_sideways_m']:.2f} m")
    _print_difficulty(by_difficulty, methods, steep_from)
    for p in paired:
        print(f"paired reached ({p['a']} - no_op): {p['diff']} [{p['ci_low']}, {p['ci_high']}]")
    print(f"wall {wall_s:.1f} s; wrote {folder / 'run.json'} and {len(index)} tracks under {folder / 'tracks'}", flush=True)
    return out


# ----- the improvement loop --------------------------------------------------------------------------------------


def _test_digest(run_json: Path, method: str) -> dict[str, Any]:
    """One method's numbers from a run.json (any split): all courses and by difficulty."""
    run = json.loads(run_json.read_text(encoding="utf-8"))
    keep = ("n", "reached", "crossed_goal_line", "falls", "mean_offset_at_goal_line_m", "mean_final_abs_sideways_m",
            "mean_time_to_goal_s")
    return {"split": run["split"], "method": method, "run_json": run_json.as_posix(),
            **{k: run["summary"][method]["all"][k] for k in keep},
            "by_difficulty": {d: {**{k: s[k] for k in keep}, **({"passage": s["passage"]} if "passage" in s else {})}
                              for d, s in run["by_difficulty"][method].items()}}


def _round_rank(record: Mapping[str, Any]) -> tuple[int, float, int]:
    """ROUND_CHOICE_RULE as a sort key, larger is better: dev reached, then - mean miss (None worst), then the round."""
    miss = record["dev"]["mean_offset_at_goal_line_m"]
    return int(record["dev"]["reached"]), -(math.inf if miss is None else float(miss)), int(record["round"])


def run_loop(config: Mapping[str, Any], run_id: str, rounds: int, confirm_billing: bool,
             test_split: str = "test2") -> int:
    """The improvement loop (module docstring: loop). Each round's student is judged on the dev split; the run keeps
    the round chosen by ROUND_CHOICE_RULE (most dev courses reached; ties -> smaller mean |sideways offset| at the goal
    line on dev, no crossing the worst; ties -> the later round) and runs only that student, once, on test_split.
    Returns 0 done, 3 stopped for billing (or the teacher refused), 1 a step failed."""
    from learned_judgment.config import CONFIG_PATH
    from learned_judgment.observed import EXIT_REFUSED, _run_dir, _utc_now

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise ValueError(f"--rounds must be an int >= 0; got {rounds!r}")
    if test_split not in HELD_OUT_SPLITS:
        raise ValueError(f"--test-split must be one of {HELD_OUT_SPLITS}; got {test_split!r}")
    for split in ("dev", test_split):
        _count(f"{_OBSERVED}.{split}_scenarios", require(config, f"{_OBSERVED}.{split}_scenarios"))
    root = CONFIG_PATH.parent
    run_dir = _run_dir(config, run_id)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"{config_path} does not exist: write the run's config first (reflex_lesson config)")
    if config_digest(load_config(config_path)) != config_digest(config):
        raise ValueError(f"this process's config is not {config_path}: run the loop through python -m "
                         f"learned_judgment.run_skill _exec --config {config_path} learned_judgment.skills.reflex loop ...")
    n = _count(f"{_OBSERVED}.attempt_episodes", require(config, f"{_OBSERVED}.attempt_episodes"))
    build_observed_spec(config)  # the lesson is usable before anything runs

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    log = open(run_dir / "loop.log", "a", encoding="utf-8")
    steps: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    clock = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

    def say(text: str) -> None:
        line = f"[loop {_utc_now().isoformat(timespec='seconds')}] {text}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def state(value: str, **extra: Any) -> None:
        _merge_result(run_dir, run_id, {"loop": {"state": value, "rounds": rounds, "confirm_billing": confirm_billing,
                                                 "attempt_episodes_per_round": n, "test_split": test_split,
                                                 "round_choice_rule": ROUND_CHOICE_RULE, "steps": steps,
                                                 "wall_s": time.perf_counter() - clock, "log": rel(run_dir / "loop.log"),
                                                 **extra}})

    def step(k: int, name: str, module: str, args: Sequence[str]) -> int:
        argv = [sys.executable, "-m", "learned_judgment.run_skill", "_exec", "--config", rel(config_path), module, *args]
        say(f"round {k} step {name}: {subprocess.list2cmdline(argv[1:])}")
        start, started = time.perf_counter(), _utc_now().isoformat()
        process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        log.flush()
        code = process.wait()
        wall = round(time.perf_counter() - start, 3)
        steps.append({"round": k, "step": name, "argv": argv[1:], "returncode": code, "started_utc": started,
                      "wall_s": wall})
        say(f"round {k} step {name}: exit {code} after {wall} s")
        state("running")
        return code

    try:
        say(f"loop {run_id}: round 0 + {rounds} improvement round(s), {n} attempt episodes per round, config "
            f"{rel(config_path)} (digest {config_digest(config)[:12]}), confirm_billing {confirm_billing}; each round's "
            f"student judged on dev, the run keeps the round with {ROUND_CHOICE_RULE}; that student alone then runs "
            f"once on the held-out split {test_split}")
        state("running")
        for k in range(rounds + 1):
            round_clock = time.perf_counter()
            folder = run_dir / f"round_{k}"
            context = folder / "context"
            previous = None if k == 0 else run_dir / f"round_{k - 1}" / "student"
            args = ["attempts", "--out", rel(context), "--n", str(n), "--first", str(k * n)]
            if previous is not None:
                args += ["--policy", "student", "--student", rel(previous)]
            if step(k, "attempts", "learned_judgment.skills.reflex", args):
                state("failed")
                return 1
            contexts = [run_dir / f"round_{j}" / "context" for j in range(k + 1)]
            rows_dir = context
            if k > 0:
                rows_dir = folder / "rows"
                if step(k, "merge", "learned_judgment.skills.reflex",
                        ["merge", "--inputs", *map(rel, contexts), "--out", rel(rows_dir)]):
                    state("failed")
                    return 1
            dream_dir = folder / "dream"
            if step(k, "dream", "learned_judgment.skills.reflex",
                    ["dream", "--out", rel(dream_dir), "--context", *map(rel, contexts)]):
                state("failed")
                return 1
            teacher = ["--run", run_id, "--skill", SKILL, "--context", rel(rows_dir), "--dream", rel(dream_dir)]
            if not confirm_billing:
                code = step(k, "quote", "learned_judgment.observed", ["quote", *teacher])
                quote = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("quote", {}).get("quote")
                total = None if not quote else quote["dream"]["total_quoted_tokens"]
                say(f"stopped before the first teach (no --confirm-billing): free quote of round {k}'s dream "
                    f"{total} tokens; nothing billed. Rerun with --confirm-billing to go on.")
                state("stopped_for_billing" if code == 0 else "failed", quoted_tokens_round_0=total)
                return EXIT_REFUSED if code == 0 else 1
            code = step(k, "teach", "learned_judgment.observed", ["teach", *teacher, "--confirm-billing"])
            if code:
                state("refused" if code == EXIT_REFUSED else "failed")
                return EXIT_REFUSED if code == EXIT_REFUSED else 1
            shutil.copy2(run_dir / "scores.npz", folder / "scores.npz")
            if step(k, "train", "learned_judgment.observed",
                    ["train", "--run", run_id, "--skill", SKILL, "--dream", rel(dream_dir),
                     "--scores", rel(folder / "scores.npz")]):
                state("failed")
                return 1
            if (folder / "student").exists():
                shutil.rmtree(folder / "student")
            shutil.copytree(run_dir / "student", folder / "student")
            trained = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("train")
            atomic_write_text(folder / "train.json", json.dumps(trained, indent=1) + "\n")
            if step(k, "dev", "learned_judgment.skills.reflex",
                    ["run", "--student", rel(folder / "student"), "--split", "dev", "--out", rel(folder / "dev_run"),
                     "--methods", "snn"]):
                state("failed")
                return 1
            with np.load(rows_dir / "rows.npz") as archive:
                used, used_sources = int(archive["action"].shape[0]), _source_counts(archive["source"])
            with np.load(context / "rows.npz") as archive:
                new, new_sources = int(archive["action"].shape[0]), _source_counts(archive["source"])
            with np.load(dream_dir / "dream.npz") as archive:
                windows = int(archive["scenario_index"].shape[0])
            student_json = json.loads((folder / "student" / "student.json").read_text(encoding="utf-8"))
            records.append({
                "round": k, "policy": "rule" if k == 0 else "student",
                "acting_student": None if previous is None else rel(previous), "attempt_episodes": n,
                "context_scenarios": [k * n, (k + 1) * n - 1], "new_rows": new, "new_rows_by_source": new_sources,
                "n_real_decisions": used, "real_decisions_by_source": used_sources, "teacher_context": rel(rows_dir),
                "dream_windows": windows, "dream": rel(dream_dir), "student": rel(folder / "student"),
                "network_sha256": student_json.get("network", {}).get("sha256"), "train": rel(folder / "train.json"),
                "dev": _test_digest(folder / "dev_run" / "run.json", "snn"), "chosen": False,
                "wall_s": round(time.perf_counter() - round_clock, 3)})
            _merge_result(run_dir, run_id, {"rounds": records})
            t = records[-1]["dev"]
            say(f"round {k} done: {used} real decisions, {windows} dream windows; student on the DEV split reached "
                f"{t['reached']} of {t['n']}, mean miss at the line {t['mean_offset_at_goal_line_m']}; "
                f"{records[-1]['wall_s']} s")
        best = max(records, key=_round_rank)
        k_best = int(best["round"])
        for r in records:
            r["chosen"] = r["round"] == k_best
        chosen_folder = run_dir / f"round_{k_best}"
        if (run_dir / "student").exists():
            shutil.rmtree(run_dir / "student")
        shutil.copytree(chosen_folder / "student", run_dir / "student")
        trained = json.loads((chosen_folder / "train.json").read_text(encoding="utf-8"))
        choice = {"rule": ROUND_CHOICE_RULE, "round": k_best, "student": rel(chosen_folder / "student"),
                  "network_sha256": best["network_sha256"], "copied_to": rel(run_dir / "student"),
                  "dev": {k: best["dev"][k] for k in ("split", "n", "reached", "mean_offset_at_goal_line_m")},
                  "dev_per_round": [{"round": r["round"], "reached": r["dev"]["reached"], "n": r["dev"]["n"],
                                     "mean_offset_at_goal_line_m": r["dev"]["mean_offset_at_goal_line_m"]}
                                    for r in records],
                  "test_split": test_split, "test": None}
        _merge_result(run_dir, run_id, {"rounds": records, "round_choice": choice,
                                        "train": {**(trained or {}), "chosen_round": k_best}})
        say("dev per round " + ", ".join(f"{r['round']}: {r['dev']['reached']}/{r['dev']['n']} (miss "
                                         f"{r['dev']['mean_offset_at_goal_line_m']})" for r in records)
            + f"; chosen round {k_best} ({ROUND_CHOICE_RULE}); its student is now {rel(run_dir / 'student')}")
        if step(k_best, "test", "learned_judgment.skills.reflex",
                ["run", "--student", rel(run_dir / "student"), "--split", test_split,
                 "--out", rel(run_dir / "test_run"), "--run", run_id]):
            state("failed")
            return 1
        choice["test"] = {m: _test_digest(run_dir / "test_run" / "run.json", m) for m in METHODS}
        _merge_result(run_dir, run_id, {"round_choice": choice})
        state("done", chosen_round=k_best)
        say(f"loop done in {time.perf_counter() - clock:.1f} s: chosen round {k_best}; held-out {test_split} reached "
            + ", ".join(f"{m} {choice['test'][m]['reached']}/{choice['test'][m]['n']}" for m in METHODS))
        return 0
    except BaseException as error:
        state("failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        log.close()


# ----- command line ----------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.skills.reflex", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    attempts = commands.add_parser("attempts", help="exploration episodes on context scenarios; every decision a row")
    attempts.add_argument("--out", required=True)
    attempts.add_argument("--n", type=int, help="episodes (default observed.reflex.attempt_episodes)")
    attempts.add_argument("--first", type=int, default=0, help="first context scenario index (default 0)")
    attempts.add_argument("--policy", choices=("rule", "student"), default="rule",
                          help="who picks apart from exploration: the rule of thumb (default) or the student (on-policy)")
    attempts.add_argument("--student", help="exported student directory (--policy student)")
    attempts.add_argument("--explore", type=float, help="exploration fraction (default engine.attempt_exploration_fraction "
                                                        "for rule, observed.reflex.student_explore_fraction for student)")
    merge = commands.add_parser("merge", help="concatenate rows files of this lesson (scenario indices kept distinct)")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    dream = commands.add_parser("dream", help="real sensor windows of the attempts plus extra rule-of-thumb episodes")
    dream.add_argument("--out", required=True)
    dream.add_argument("--n", type=int, help="extra episodes (default observed.reflex.dream_episodes)")
    dream.add_argument("--context", nargs="+", help="attempts rows directories (default: 'context' beside --out)")
    run = commands.add_parser("run", help="closed-loop episodes: no_op, physics_only and the student")
    who = run.add_mutually_exclusive_group(required=True)
    who.add_argument("--student", help="exported student directory (observed train)")
    who.add_argument("--no-student", action="store_true", help="run only no_op and physics_only (no student yet)")
    run.add_argument("--split", required=True, choices=RUN_SPLITS)
    run.add_argument("--out", required=True)
    run.add_argument("--n", type=int, help="scenarios (default observed.reflex.<split>_scenarios)")
    run.add_argument("--run", help="run id: also merge the UI keys into <observed.runs_dir>/RUN/result.json")
    run.add_argument("--methods", nargs="+", choices=METHODS, help="only these methods (default all available)")
    loop = commands.add_parser("loop", help="round 0 + N on-policy improvement rounds, each judged on dev; the best "
                                            "round's student then runs once on the held-out split (billed teach)")
    loop.add_argument("--run", required=True, help="run id; its directory holds the run's config.yaml")
    loop.add_argument("--rounds", type=int, required=True, help="improvement rounds after round 0")
    loop.add_argument("--test-split", choices=HELD_OUT_SPLITS, default="test2",
                      help="held-out split of the final run (default test2)")
    loop.add_argument("--confirm-billing", action="store_true", help="allow the billed teach steps")
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "attempts":
        if args.student is not None and args.policy != "student":
            parser.error("--student goes with --policy student")
        collect_attempts(config, args.out, args.n, policy=args.policy, student_dir=args.student, explore=args.explore,
                         first=args.first)
    elif args.command == "merge":
        merge_rows(config, args.inputs, args.out)
    elif args.command == "dream":
        collect_dream(config, args.out, args.n, args.context)
    elif args.command == "loop":
        return run_loop(config, args.run, args.rounds, args.confirm_billing, args.test_split)
    else:
        run_methods(config, None if args.no_student else args.student, args.split, args.out, args.n, args.run,
                    args.methods)
    return 0


if __name__ == "__main__":
    # Through the importable module, so spawned workers find _worker_init and the tasks by their module name.
    from learned_judgment.skills import reflex as _module

    raise SystemExit(_module.main())
