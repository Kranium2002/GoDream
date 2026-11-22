"""DODGE from the head camera (Person B): threat scenarios, the head view, episodes on the real Go2 world, the
physics guess, the objective and the collection command line.

Lane contract (Person B, dodge_vision)
--------------------------------------
The dog stands on the zero command; at t_s 0 (the launch) an object may move in the scene: a ball (under gravity) or
a punch (a slow straight flight at head height), by skills.dodge_vision.object_kinds. A student sees only
image windows from the head camera and picks one of C commands; the scorer side (physics guess, objective, and the
teacher's "truth" block: truth_features) may use the scenario's truth, which never becomes a student input. Shared arrays:

    ACTIONS   C = len(skills.dodge_vision.actions) in config order (NO_OP, DODGE_LEFT, DODGE_RIGHT, RETREAT); the
              candidate index is the position. An active command is held for hold_s, then the zero command is sent.
    OUTCOMES  K = 5 in OUTCOME_NAMES order:
                  min_separation_m  regression, [0, separation_clip_m]; the clip when the scene has no object
                  collision         binary
                  fall              binary
                  recovery_time_s   regression, [0, episode_duration_s]
                  command_travel_m  regression, [0, max |command field| x hold_s]: |commanded speed| x hold_s
    FEATURES  D = frames x N x N float32 in [0, 1] (learned_judgment.vision.flatten_window, oldest frame first).
    TIME      whole control periods of robot.walking_policy.command_schema.update_period_s from the launch. Frames
              are taken at learned_judgment.vision.frame_control_periods for vision.frame_rate_hz, at periods
              before the end of the episode; a decision frame is an index into frame_times(config).

Files of the command line, one directory per call:

    rows.npz    scenario_index (n,) int64, decision_time_s (n,) float64, action (n,) int64,
                features (n, D) float32, physics (n, K) float64, measured (n, K) float64
    rows.json   skill, split, action_names, outcome_names, outcome_kinds, outcome_bounds, frames, n_situations,
                seeds, world_seed, world_preset, config_digest, kind_counts, punch_stand_in (a punch ran as the
                ball with its weight cancelled), scenarios (index = scenario_index)
    dream.npz   scenario_index (m,) int64, decision_time_s (m,) float64, features (m, D) float32,
                physics (m, C, K) float64, kinematic (m,) bool (the window was rendered by render_windows_kinematic)
    dream.json  the keys of rows.json plus n_episodes, windows_per_dream_episode and render {mode,
                kinematic_windows, physics_windows}
    stand.npz   (dream --render kinematic) the recorded zero-command stand of record_stand

    python -m learned_judgment.skills.dodge_vision rows --split {context|dev|test} --out DIR [--n N] [--frames F]
    python -m learned_judgment.skills.dodge_vision dream --out DIR [--n N] [--render {kinematic|physics}]

Seeds. The scenarios of a split come from np.random.default_rng(np.random.SeedSequence([runtime.seed, code]))
(code: SPLIT_CODES). Scenario i's decision frame(s) and image augmentation come from the two children of
np.random.SeedSequence([runtime.seed, code], spawn_key=(i,)).spawn(2), and its episodes use attempt_seed i.

This module defines no SPEC_CLASS, so the SkillSpec registry (learned_judgment.skills) does not load it.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import io
import json
import math
import multiprocessing
import numbers
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import mujoco
import numpy as np

from learned_judgment.config import ConfigError, config_digest, load_config, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.fileio import atomic_write_bytes, atomic_write_text
from learned_judgment.vision import (
    FrameWindow,
    VisionSettings,
    _box_mean,
    _range,
    augment_window,
    flatten_window,
    frame_control_periods,
    preprocess_frame,
)

__all__ = [
    "KINDS",
    "OUTCOME_KINDS",
    "OUTCOME_NAMES",
    "SKILL",
    "SPLIT_CODES",
    "TRUTH_NAMES",
    "DodgeVisionSpec",
    "HeadView",
    "arrival_time_s",
    "build_observed_spec",
    "closed_loop",
    "collect_dream",
    "collect_rows",
    "frame_times",
    "main",
    "objective",
    "observe_and_act",
    "physics_guess",
    "record_stand",
    "render_windows",
    "render_windows_kinematic",
    "sample_threat_scenarios",
    "truth_features",
]

SKILL = "dodge_vision"
OUTCOME_NAMES = ("min_separation_m", "collision", "fall", "recovery_time_s", "command_travel_m")
OUTCOME_KINDS = ("regression", "binary", "binary", "regression", "regression")
SPLIT_CODES = {"context": 1, "dev": 2, "test": 3, "dream": 4}

# Kinds whose ball moves toward the dog: the config range of |aim_y| (the sideways offset where the path crosses the
# plane x = 0) and its sign, 0.0 meaning a random sign.
_AIMS = {
    "incoming_center": ("aim_center_m", 0.0),
    "incoming_left": ("aim_side_m", 1.0),
    "incoming_right": ("aim_side_m", -1.0),
    "ambiguous": ("aim_ambiguous_m", 0.0),
    "off_path": ("aim_off_path_m", 0.0),
}
KINDS = ("empty", "static", "receding", "background_motion", *_AIMS)
_LAUNCH_MODES = ("roll", "throw")
_OBJECT_KINDS = ("ball", "punch")  # values of skills.dodge_vision.object_kinds and of a scenario's "object_kind"
_OBJECT_GEOMS = ("ball", "punch")  # geoms HeadView draws in the scenario's object grey, when the model has them
# Emission of the object's render-only geom when HeadView paints a scenario's visual: with the model's lights the object's
# mean rendered grey is then about the drawn grey (drawn 0.1 / 0.5 / 1.0 -> mean 0.106 / 0.492 / 0.888 over the object's
# pixels, a punch mid-flight; emission 0 gives 0.059 / 0.253 / 0.496 and a 0.07-0.96 spread at grey 1.0). Probe of
# 2026-11-21, Person B.
_OBJECT_EMISSION = 0.5
_MAX_GREY_DRAWS = 10_000  # object-grey redraws before a scenario's visual draw gives up (the config check makes it rare)
_VISUAL_UNMAPPED = (
    "Go2World.reset accepts the situation key 'visual' (Person A's world draws the scene's look): the mapping of a "
    "scenario's visual {object_grey, backdrop_grey} to Person A's keys is decided when their format is known, and the "
    "stopgap paint of HeadView stays off for such a world"
)
_PLANAR = ("vel_x", "vel_y")  # verified command fields of the dog's planar velocity (x ahead, y left)
_ROOT = "skills.dodge_vision"
_TOLERANCE = 1e-9  # s or m: times are sums of 0.02 s periods
_DREAM_RENDERS = ("kinematic", "physics")  # values of skills.dodge_vision.dream_render and of dream --render
# The stand's punch (record_stand), launch state in the situation frame: 50 m behind the dog, flying away at 1 m/s, so
# the stand runs the punch scene (the robot's physics of a punch episode) with the object never near the dog or in view.
_STAND_OBJECT = (-50.0, 0.0, 0.3, -1.0, 0.0, 0.0)
# The standing dog steps in place (its feet move up to 0.05 m between two camera frames, world seed 0): a geom's surface
# may come this many times its move between the poses of two frames (centre move + rbound x rotation angle) closer to
# the punch between those frames (_kinematic_limit_s).
_SWAY_FACTOR = 2.0


# ----- config ----------------------------------------------------------------------------------------------------


def _number(key: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ConfigError(f"config.yaml key {key!r} must be a finite number; got {value!r}")
    return float(value)


def _grey(visual: Any, key: str) -> float:
    """visual[key] as a float in [0, 1] (a scenario's object_grey or backdrop_grey)."""
    value = visual.get(key) if isinstance(visual, Mapping) else None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not 0.0 <= value <= 1.0:
        raise ValueError(f"visual {key!r} must be a number in [0, 1]; got visual {visual!r}")
    return float(value)


def _value(config: Mapping[str, Any], key: str) -> float:
    return _number(key, require(config, key))


def _positive(config: Mapping[str, Any], key: str) -> float:
    number = _value(config, key)
    if number <= 0.0:
        raise ConfigError(f"config.yaml key {key!r} must be > 0; got {number}")
    return number


def _pair(config: Mapping[str, Any], key: str, minimum: float | None = None) -> tuple[float, float]:
    """[low, high] of finite numbers with low <= high (and low >= minimum when given)."""
    pair = require(config, key)
    if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
        raise ConfigError(f"config.yaml key {key!r} must be [low, high]; got {pair!r}")
    low, high = _number(key, pair[0]), _number(key, pair[1])
    if low > high or (minimum is not None and low < minimum):
        raise ConfigError(f"config.yaml key {key!r} must have {minimum} <= low <= high; got {list(pair)}")
    return low, high


def _weights(config: Mapping[str, Any], key: str, names: Sequence[str]) -> dict[str, float]:
    """name -> weight in config order; the names must be exactly ``names``, weights >= 0 with a positive sum."""
    weights = require(config, key)
    if not isinstance(weights, Mapping) or set(weights) != set(names):
        raise ConfigError(f"config.yaml key {key!r} must map exactly {sorted(names)} to weights; got {weights!r}")
    out = {str(name): _number(f"{key}.{name}", weight) for name, weight in weights.items()}
    if any(weight < 0.0 for weight in out.values()) or sum(out.values()) <= 0.0:
        raise ConfigError(f"config.yaml key {key!r} needs weights >= 0 with a positive sum; got {weights!r}")
    return out


def _ball_keys() -> tuple[str, ...]:
    # Imported on use: learned_judgment.env loads the walking policy's runtime (about 0.7 s), and the SkillSpec
    # registry imports every module of this package.
    from learned_judgment.env.throws import BALL_KEYS

    return BALL_KEYS


def _object_kinds(config: Mapping[str, Any]) -> tuple[str, ...]:
    """skills.dodge_vision.object_kinds: a non-empty list of distinct members of _OBJECT_KINDS."""
    key = f"{_ROOT}.object_kinds"
    kinds = require(config, key)
    if (
        isinstance(kinds, (str, bytes))
        or not isinstance(kinds, Sequence)
        or not kinds
        or not all(isinstance(kind, str) and kind in _OBJECT_KINDS for kind in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ConfigError(f"config.yaml key {key!r} must be a non-empty list of distinct {list(_OBJECT_KINDS)}; got {kinds!r}")
    return tuple(kinds)


def _punch_runs_as_stand_in() -> bool:
    """True while Go2World.reset has no "object_kind" key: a punch then runs as the ball with its weight cancelled."""
    from learned_judgment.env.world import SITUATION_KEYS  # on use: see _ball_keys

    return "object_kind" not in SITUATION_KEYS


def _world_takes_visual() -> bool:
    """True when Go2World.reset accepts a "visual" situation key (Person A's world; not on 2026-11-21)."""
    from learned_judgment.env.world import SITUATION_KEYS  # on use: see _ball_keys

    return "visual" in SITUATION_KEYS


def _reset(world: Any, scenario: Mapping[str, Any], attempt_seed: int) -> None:
    """world.reset(scenario's situation, attempt_seed) for every object kind.

    A punch scenario with an object: when Go2World.reset accepts "object_kind" (Person A's punch body) the situation
    gets "object_kind": "punch" and nothing else is done. Otherwise the stand-in: right after the reset, a constant
    force -mass x gravity on the ball body (world.policy.data.xfrc_applied; world.model is the ball scene then)
    cancels its weight, so it flies straight at its launch velocity until a contact. Go2World.reset zeroes
    xfrc_applied of the scene it resets (its walking-policy backend resets the MjData), measured 2026-11-21, so the
    force never reaches the next episode of either scene.

    The scenario's "visual" is drawn by HeadView's stopgap paint, never sent to the world. A Go2World whose
    SITUATION_KEYS holds "visual" (Person A's world draws the scene's look itself) would get the scenario's visual
    passed through in the situation, but the mapping of {object_grey, backdrop_grey} to Person A's keys is decided
    when their format is known: until then such a world raises NotImplementedError naming the key, so the look can
    never be silently wrong.
    """
    if _world_takes_visual():
        raise NotImplementedError(_VISUAL_UNMAPPED)
    situation = dict(scenario["situation"])
    punch = scenario.get("object_kind", "ball") == "punch" and all(key in situation for key in _ball_keys())
    if punch and not _punch_runs_as_stand_in():
        situation["object_kind"] = "punch"
        punch = False
    world.reset(situation, attempt_seed=int(attempt_seed))
    if punch:
        model, data = world.model, world.policy.data
        body = model.body("ball").id
        data.xfrc_applied[body, :3] = -model.body_mass[body] * model.opt.gravity  # z: mass x |gravity z|


def _camera(config: Mapping[str, Any]) -> tuple[np.ndarray, float, float]:
    """(pos_m (3,), pitch_up_deg, fovy_deg) of vision.head_camera."""
    pos = require(config, "vision.head_camera.pos_m")
    if isinstance(pos, (str, bytes)) or not isinstance(pos, Sequence) or len(pos) != 3:
        raise ConfigError(f"config.yaml key 'vision.head_camera.pos_m' must be [x, y, z]; got {pos!r}")
    eye = np.array([_number("vision.head_camera.pos_m", v) for v in pos], dtype=np.float64)
    fovy = _positive(config, "vision.head_camera.fovy_deg")
    if fovy >= 180.0:
        raise ConfigError(f"config.yaml key 'vision.head_camera.fovy_deg' must be below 180; got {fovy}")
    return eye, _value(config, "vision.head_camera.pitch_up_deg"), fovy


@dataclass(frozen=True)
class _Actions:
    names: tuple[str, ...]
    commands: tuple[dict[str, float], ...]  # each in command-schema field order
    fields: tuple[str, ...]  # command-schema fields by native index
    features: np.ndarray  # (C, len(fields)) float64
    no_op: int


def _actions(config: Mapping[str, Any]) -> _Actions:
    """skills.dodge_vision.actions, each checked against the verified command schema; exactly one is all zero."""
    from learned_judgment.env.commands import validate_command  # on use: see _ball_keys

    schema = require(config, "robot.walking_policy.command_schema")
    fields = tuple(sorted(schema["fields"], key=lambda name: schema["fields"][name]["native_index"]))
    missing = [name for name in _PLANAR if name not in fields]
    if missing:
        raise ConfigError(f"the verified command schema has no field(s) {missing}; it has {list(fields)}")
    actions = require(config, f"{_ROOT}.actions")
    if not isinstance(actions, Mapping) or not actions:
        raise ConfigError(f"config.yaml key '{_ROOT}.actions' must be a non-empty mapping; got {actions!r}")
    names, commands = [], []
    for name, command in actions.items():
        if not isinstance(command, Mapping):
            raise ConfigError(f"config.yaml key '{_ROOT}.actions.{name}' must be a command mapping; got {command!r}")
        try:
            validate_command(dict(command), schema)
        except ValueError as error:
            raise ConfigError(f"config.yaml key '{_ROOT}.actions.{name}' is not a verified command: {error}") from error
        names.append(str(name))
        commands.append({f: float(command[f]) for f in fields})
    features = np.array([[command[f] for f in fields] for command in commands], dtype=np.float64)
    zero = [index for index, row in enumerate(features) if not np.any(row)]
    if len(zero) != 1:
        raise ConfigError(f"exactly one action of '{_ROOT}.actions' must be the zero command; got {[names[i] for i in zero]}")
    features.setflags(write=False)
    return _Actions(tuple(names), tuple(commands), fields, features, zero[0])


def _whole_periods(key: str, seconds: float, period_s: float) -> int:
    count = round(seconds / period_s)
    if count < 1 or abs(count * period_s - seconds) > _TOLERANCE:
        raise ConfigError(f"config.yaml key {key!r} ({seconds} s) must be a whole number >= 1 of {period_s} s periods")
    return count


@dataclass(frozen=True)
class _Timing:
    period_s: float  # control period
    n_periods: int  # control periods per episode after the launch
    frame_periods: np.ndarray  # (n_frames,) int64, the control period of each frame, all < n_periods


def _timing(config: Mapping[str, Any]) -> _Timing:
    period_s = _positive(config, "robot.walking_policy.command_schema.update_period_s")
    duration = _positive(config, f"{_ROOT}.episode_duration_s")
    n_periods = _whole_periods(f"{_ROOT}.episode_duration_s", duration, period_s)
    rate = _positive(config, "vision.frame_rate_hz")
    periods = frame_control_periods(int(math.floor(duration * rate)) + 2, rate, period_s)
    return _Timing(period_s, n_periods, periods[periods < n_periods])


@dataclass(frozen=True)
class _Episode:
    timing: _Timing
    actions: _Actions
    hold_s: float
    hold_periods: int
    travel_m: tuple[float, ...]  # (C,) max |command field| x hold_s
    separation_clip_m: float
    augmentation: Mapping[str, Any]


def _episode(config: Mapping[str, Any]) -> _Episode:
    timing = _timing(config)
    actions = _actions(config)
    hold_s = _positive(config, f"{_ROOT}.hold_s")
    hold_periods = _whole_periods(f"{_ROOT}.hold_s", hold_s, timing.period_s)
    travel = tuple(float(np.abs(row).max()) * hold_s for row in actions.features)
    augmentation = require(config, "vision.augmentation")
    if not isinstance(augmentation, Mapping):
        raise ConfigError(f"config.yaml key 'vision.augmentation' must be a mapping; got {augmentation!r}")
    clip = _positive(config, f"{_ROOT}.separation_clip_m")
    return _Episode(timing, actions, hold_s, hold_periods, travel, clip, augmentation)


# ----- scenarios -------------------------------------------------------------------------------------------------


def _pick(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    """One name, drawn with probability proportional to its weight (one rng.random() draw)."""
    names = list(weights)
    cumulative = np.cumsum([weights[name] for name in names])
    return names[int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))]


def _uniform(rng: np.random.Generator, pair: tuple[float, float]) -> float:
    return float(rng.uniform(pair[0], pair[1]))


def _sign(rng: np.random.Generator) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def _visual_ranges(config: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float], float]:
    """(object_grey, backdrop_grey, min_contrast) of skills.dodge_vision.visual: greys inside [0, 1], min_contrast
    >= 0, and every backdrop grey of the range leaves an object grey of the range at least min_contrast away."""
    v = f"{_ROOT}.visual"
    obj, back = _pair(config, f"{v}.object_grey", minimum=0.0), _pair(config, f"{v}.backdrop_grey", minimum=0.0)
    if obj[1] > 1.0 or back[1] > 1.0:
        raise ConfigError(f"config.yaml keys '{v}.object_grey' and '{v}.backdrop_grey' must lie in [0, 1]; got {list(obj)}, {list(back)}")
    contrast = _value(config, f"{v}.min_contrast")
    if contrast < 0.0:
        raise ConfigError(f"config.yaml key '{v}.min_contrast' must be >= 0; got {contrast}")
    low, high = obj[1] - contrast, obj[0] + contrast  # a backdrop grey strictly between these has no object grey
    if low < high and low < back[1] and back[0] < high:
        raise ConfigError(
            f"config.yaml key '{v}.backdrop_grey' {list(back)} reaches ({low:g}, {high:g}), where no '{v}.object_grey' "
            f"{list(obj)} differs by '{v}.min_contrast' {contrast}"
        )
    return obj, back, contrast


def _draw_visual(
    rng: np.random.Generator, obj: tuple[float, float], back: tuple[float, float], contrast: float
) -> dict[str, float]:
    """{"object_grey", "backdrop_grey"}: the backdrop grey uniform, then the object grey uniform, redrawn until it
    differs from the backdrop grey by at least ``contrast``."""
    backdrop = _uniform(rng, back)
    for _ in range(_MAX_GREY_DRAWS):
        grey = _uniform(rng, obj)
        if abs(grey - backdrop) >= contrast:
            return {"object_grey": grey, "backdrop_grey": backdrop}
    raise RuntimeError(f"no object grey in {list(obj)} differed from backdrop grey {backdrop} by {contrast} in {_MAX_GREY_DRAWS} draws")


def sample_threat_scenarios(config: Mapping[str, Any], rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    """``n`` scenarios drawn from config.yaml skills.dodge_vision; no world needed.

    Each is {"scenario_id": i (0 .. n - 1), "kind": a key of kind_weights, "object_kind": a member of object_kinds,
    "situation": a dict Go2World.reset accepts, "visual": {"object_grey", "backdrop_grey"}} (the look HeadView paints;
    skills.dodge_vision.visual). The situation always has dog_forward_speed_m_s 0.0 and
    settle_s = skills.dodge_vision.settle_s; kind "empty" has no ball key, every other kind all six BALL_KEYS (the
    object's launch state in Go2World's frame at the launch: origin at the trunk origin, x ahead, y left, z up;
    velocity over the ground), whatever the object kind (run it with _reset). Values are floats.

    Draws, row by row in this order (so row i does not depend on n), each uniform over its scenarios range:
      kind                 by kind_weights
      object kind          uniform over object_kinds (one rng.integers draw; none when the list has one member)
    then, for object kind "punch" (skills.dodge_vision.punch; a straight flight at constant velocity):
      incoming_*, ambiguous, off_path
                           distance, bearing (launch point (x0, y0)); |aim_y| and sign as for the ball; speed;
                           z0 from launch_height_m; aim_z from aim_height_m. The velocity, of that speed, points
                           from (x0, y0, z0) to the aim point (aim_x_m, aim_y, aim_z).
      static               static_distance, bearing, z from launch_height_m; at rest (hovering)
      receding             receding_distance, bearing, z from aim_height_m, speed: straight away (+x)
      background_motion    background_distance (x0), a random side, speed, z from launch_height_m; at the edge of the
                           view as for the ball, moving sideways (pure y velocity) toward the other side
      empty                nothing more
    or, for object kind "ball":
      incoming_*, ambiguous, off_path
                           distance, bearing (launch point (x0, y0)); launch mode by launch_modes; |aim_y|
                           (aim_center_m, aim_side_m, aim_ambiguous_m or aim_off_path_m) and, except for
                           incoming_left (aim_y > 0) and incoming_right (aim_y < 0), a random sign; then
                           roll:  roll_speed; z0 = ball.radius_m - physics.base_height_m (resting on the floor), vz 0
                           throw: throw_speed, release height z0; vz such that the ball's centre is at the head
                                  camera's height (vision.head_camera.pos_m z) when it reaches the plane x = 0,
                                  under ball.gravity_m_s2
                           The horizontal velocity, of that speed, points from (x0, y0) to (0, aim_y).
      static               static_distance, bearing; at rest on the floor
      receding             receding_distance, bearing; on the floor, rolling straight away (+x) at roll_speed
      background_motion    background_distance (x0), a random side, roll_speed: on the floor at the edge of the
                           square view on that side, y0 = side x (x0 - camera x) x tan(fovy / 2), rolling sideways
                           across the view (pure y velocity, toward the other side)
      empty                nothing more
    and last, for every scenario (empty ones too), the visual from skills.dodge_vision.visual: backdrop_grey, then
    object_grey redrawn until |object_grey - backdrop_grey| >= min_contrast.

    Raises TypeError for a non-Generator rng, ValueError for a bad n, ConfigError for a missing or invalid key.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator; got {type(rng).__name__}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be an int >= 0; got {n!r}")
    ball_keys = _ball_keys()
    s = f"{_ROOT}.scenarios"
    kinds = _weights(config, f"{_ROOT}.kind_weights", KINDS)
    modes = _weights(config, f"{s}.launch_modes", _LAUNCH_MODES)
    eye, _, fovy = _camera(config)
    distance = _pair(config, f"{s}.distance_m", minimum=_TOLERANCE)
    bearing = _pair(config, f"{s}.bearing_deg", minimum=-89.0)
    if bearing[1] > 89.0:
        raise ConfigError(f"config.yaml key '{s}.bearing_deg' must lie within (-90, 90) degrees; got {list(bearing)}")
    roll_speed = _pair(config, f"{s}.roll_speed_m_s", minimum=_TOLERANCE)
    throw_speed = _pair(config, f"{s}.throw_speed_m_s", minimum=_TOLERANCE)
    release = _pair(config, f"{s}.throw_release_height_m")
    aims = {key: _pair(config, f"{s}.{key}", minimum=0.0) for key, _ in _AIMS.values()}
    static = _pair(config, f"{s}.static_distance_m", minimum=0.0)
    receding = _pair(config, f"{s}.receding_distance_m", minimum=0.0)
    background = _pair(config, f"{s}.background_distance_m", minimum=float(eye[0]) + _TOLERANCE)
    settle_s = _value(config, f"{_ROOT}.settle_s")
    if settle_s < 0.0:
        raise ConfigError(f"config.yaml key '{_ROOT}.settle_s' must be >= 0; got {settle_s}")
    gravity = _positive(config, "ball.gravity_m_s2")
    floor_z = _positive(config, "ball.radius_m") - _positive(config, f"{_ROOT}.physics.base_height_m")
    half_view = math.tan(math.radians(fovy) / 2.0)
    visual_ranges = _visual_ranges(config)
    objects = _object_kinds(config)
    if "punch" in objects:
        p = f"{_ROOT}.punch"
        punch_speed = _pair(config, f"{p}.speed_m_s", minimum=_TOLERANCE)
        punch_launch, punch_aim = _pair(config, f"{p}.launch_height_m"), _pair(config, f"{p}.aim_height_m")
        aim_x = _value(config, f"{p}.aim_x_m")
        if distance[0] * math.cos(math.radians(max(abs(bearing[0]), abs(bearing[1])))) <= aim_x:
            raise ConfigError(f"every launch point of '{s}.distance_m' and '{s}.bearing_deg' must lie ahead of '{p}.aim_x_m'")

    def punch(kind: str) -> tuple[float, ...] | None:
        """The punch's launch state (x0, y0, z0, vx, vy, vz) of a kind, None for empty: a straight flight."""
        if kind in _AIMS:
            d, b = _uniform(rng, distance), math.radians(_uniform(rng, bearing))
            key, side = _AIMS[kind]
            aim_y = _uniform(rng, aims[key]) * (side if side else _sign(rng))
            speed = _uniform(rng, punch_speed)
            start = np.array([d * math.cos(b), d * math.sin(b), _uniform(rng, punch_launch)])
            path = np.array([aim_x, aim_y, _uniform(rng, punch_aim)]) - start  # |path| > 0: x0 > aim_x
            return (*start, *(speed * path / np.linalg.norm(path)))
        if kind == "static":
            d, b = _uniform(rng, static), math.radians(_uniform(rng, bearing))
            return (d * math.cos(b), d * math.sin(b), _uniform(rng, punch_launch), 0.0, 0.0, 0.0)
        if kind == "receding":
            d, b = _uniform(rng, receding), math.radians(_uniform(rng, bearing))
            return (d * math.cos(b), d * math.sin(b), _uniform(rng, punch_aim), _uniform(rng, punch_speed), 0.0, 0.0)
        if kind == "background_motion":
            d, side = _uniform(rng, background), _sign(rng)
            speed = _uniform(rng, punch_speed)
            return (d, side * (d - eye[0]) * half_view, _uniform(rng, punch_launch), 0.0, -side * speed, 0.0)
        return None

    scenarios: list[dict[str, Any]] = []
    for index in range(int(n)):
        kind = _pick(rng, kinds)
        obj = objects[0] if len(objects) == 1 else objects[int(rng.integers(len(objects)))]
        ball: tuple[float, ...] | None = None
        if obj == "punch":
            ball = punch(kind)
        elif kind in _AIMS:
            d, b = _uniform(rng, distance), math.radians(_uniform(rng, bearing))
            x0, y0 = d * math.cos(b), d * math.sin(b)
            mode = _pick(rng, modes)
            key, side = _AIMS[kind]
            aim_y = _uniform(rng, aims[key]) * (side if side else _sign(rng))
            length = math.hypot(x0, aim_y - y0)  # > 0: x0 > 0
            if mode == "roll":
                speed, z0, vz = _uniform(rng, roll_speed), floor_z, 0.0
            else:
                speed, z0 = _uniform(rng, throw_speed), _uniform(rng, release)
                flight = length / speed  # time to the plane x = 0
                vz = (eye[2] - z0) / flight + 0.5 * gravity * flight
            ball = (x0, y0, z0, -speed * x0 / length, speed * (aim_y - y0) / length, vz)
        elif kind == "static":
            d, b = _uniform(rng, static), math.radians(_uniform(rng, bearing))
            ball = (d * math.cos(b), d * math.sin(b), floor_z, 0.0, 0.0, 0.0)
        elif kind == "receding":
            d, b = _uniform(rng, receding), math.radians(_uniform(rng, bearing))
            ball = (d * math.cos(b), d * math.sin(b), floor_z, _uniform(rng, roll_speed), 0.0, 0.0)
        elif kind == "background_motion":
            d, side = _uniform(rng, background), _sign(rng)
            speed = _uniform(rng, roll_speed)
            ball = (d, side * (d - eye[0]) * half_view, floor_z, 0.0, -side * speed, 0.0)
        situation: dict[str, float] = {} if ball is None else {k: float(v) for k, v in zip(ball_keys, ball)}
        situation["dog_forward_speed_m_s"] = 0.0
        situation["settle_s"] = settle_s
        visual = _draw_visual(rng, *visual_ranges)  # last: the geometry draws above keep their order
        scenarios.append({"scenario_id": index, "kind": kind, "object_kind": obj, "situation": situation, "visual": visual})
    return scenarios


# ----- head view -------------------------------------------------------------------------------------------------


class HeadView:
    """The head camera on Person B's side, until Go2World.render_frame(camera="head") exists.

    frame(world) renders the world's current state: a free camera at vision.head_camera.pos_m in the trunk
    (base_link) frame, looking along trunk +x tilted up by pitch_up_deg (trunk pitch and yaw kept; a free camera has
    no roll), vertical field of view fovy_deg, a square image of render_size pixels. Only world.model and
    world.policy.data are read, and neither is written except vis.global_.fovy, which is restored at once.

    The pose comes from a private MjData: world.policy.data's qpos copied in, then mj_kinematics, mj_comPos and
    mj_camlight, because right after reset the ball was placed by writing qpos only and after mj_step the derived
    positions lag qpos by one physics step (experiments/proofs/B2_research/head_view_proof.py). Shadows are off and
    the robot's visual meshes (geom group 2) hidden: 2.4 ms instead of 45-66 ms a render on this machine, with 7 of
    9216 pixels differing by at most 10 levels (render_cost_probe.out.txt). The renderer is rebuilt when world.model
    changes object (Go2World swaps scenes with and without the ball).

    A scenario's visual (frame(world, visual), Person B's stopgap until Person A's world draws the look): the thrown
    object (the geoms named "ball" and, when the model has one, "punch") gets rgba (object_grey, object_grey,
    object_grey, 1) and emission _OBJECT_EMISSION, written into this renderer's scene (mjvScene geoms) after
    update_scene, so model.geom_rgba, the physics and world.render_frame are untouched; and every sky pixel (no geom
    at that pixel in a segmentation render of the same scene, not a test for black) becomes the backdrop grey.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._eye, self._pitch_up_deg, self._fovy_deg = _camera(config)
        size = require(config, "vision.head_camera.render_size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ConfigError(f"config.yaml key 'vision.head_camera.render_size' must be an int >= 1; got {size!r}")
        self.size = size
        self._object_geoms: frozenset[int] = frozenset()
        self._option = mujoco.MjvOption()
        self._option.geomgroup[2] = 0  # robot visual meshes
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.distance = 1.0  # only the direction from the eye to lookat matters
        self._model: mujoco.MjModel | None = None
        self._renderer: mujoco.Renderer | None = None
        self._scratch: mujoco.MjData | None = None
        self._base = -1

    def frame(self, world: Any, visual: Mapping[str, Any] | None = None) -> np.ndarray:
        """(S, S, 3) uint8 RGB of ``world``'s current state, S = render_size. When world.render_frame accepts a
        ``camera`` argument, world.render_frame(camera="head") is returned instead.

        visual: a scenario's "visual" {"object_grey", "backdrop_grey"} (numbers in [0, 1]); None or empty draws the
        model's own colours on its black sky. Given, the object geoms are drawn in (g_o, g_o, g_o, 1) with emission
        _OBJECT_EMISSION, a segmentation render of the same scene finds the sky (pixels without a geom: object id -1),
        and those pixels become round(backdrop_grey * 255) in all three channels. Two renders instead of one.

        Raises ValueError for a visual without both greys in [0, 1]; NotImplementedError for a visual when Go2World
        takes a "visual" situation key (see _reset).
        """
        if visual:
            if _world_takes_visual():
                raise NotImplementedError(_VISUAL_UNMAPPED)
            grey, backdrop = _grey(visual, "object_grey"), _grey(visual, "backdrop_grey")
        render_frame = getattr(world, "render_frame", None)
        if render_frame is not None and "camera" in inspect.signature(render_frame).parameters:
            return render_frame(camera="head")
        model, data = world.model, world.policy.data
        if model is not self._model:
            self.close()
            self._renderer = mujoco.Renderer(model, self.size, self.size)
            self._scratch = mujoco.MjData(model)
            self._base = model.body("base_link").id
            ids = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in _OBJECT_GEOMS)
            self._object_geoms = frozenset(i for i in ids if i >= 0)
            self._model = model
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        mujoco.mj_camlight(model, scratch)
        rot = scratch.xmat[self._base].reshape(3, 3)  # body to world; column 0 is trunk +x
        eye = scratch.xpos[self._base] + rot @ self._eye
        forward = rot[:, 0] / np.linalg.norm(rot[:, 0])
        azimuth = math.atan2(forward[1], forward[0])
        elevation = math.asin(max(-1.0, min(1.0, float(forward[2])))) + math.radians(self._pitch_up_deg)
        camera = self._camera
        camera.azimuth, camera.elevation = math.degrees(azimuth), math.degrees(elevation)
        camera.lookat[:] = eye + camera.distance * np.array(
            [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)]
        )
        fovy = float(model.vis.global_.fovy)  # a free camera's field of view is the model's
        model.vis.global_.fovy = self._fovy_deg
        try:
            self._renderer.update_scene(scratch, camera=camera, scene_option=self._option)
        finally:
            model.vis.global_.fovy = fovy
        renderer = self._renderer
        scene = renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        if not visual:
            return renderer.render()
        if self._object_geoms:
            geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
            for index in range(scene.ngeom):
                geom = scene.geoms[index]
                if geom.objtype == geom_type and geom.objid in self._object_geoms:
                    geom.rgba[:] = (grey, grey, grey, 1.0)
                    geom.emission = _OBJECT_EMISSION
        renderer.enable_segmentation_rendering()
        try:
            segmentation = renderer.render()  # (S, S, 2) object id, object type; -1 where no geom is drawn
        finally:
            renderer.disable_segmentation_rendering()
        image = renderer.render()
        image[segmentation[:, :, 0] < 0] = np.uint8(round(backdrop * 255.0))
        return image

    def close(self) -> None:
        """Release the renderer (its OpenGL context); the next frame builds a new one."""
        if self._renderer is not None:
            self._renderer.close()
        self._model = self._renderer = self._scratch = None


# ----- episodes --------------------------------------------------------------------------------------------------


def frame_times(config: Mapping[str, Any]) -> np.ndarray:
    """(n_frames,) float64 frame times in s from the launch: frame k at the control period
    frame_control_periods gives for k / vision.frame_rate_hz, for every frame before the end of the episode
    (skills.dodge_vision.episode_duration_s). 30 Hz, 0.02 s, 3.0 s: 90 frames, 0.0, 0.04, 0.06, 0.1, ... 2.96."""
    timing = _timing(config)
    return timing.frame_periods * timing.period_s


class _EpisodeAugmentation:
    """vision.augment_window's variation with one draw per episode instead of one per window: gain, bias,
    noise_std and blur drawn once (rng.uniform, in that order), then each frame gets the same arithmetic as
    augment_window with fresh noise. augment_window cannot be reused frame by frame: it redraws the four values."""

    def __init__(self, rng: np.random.Generator, augmentation: Mapping[str, Any]) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"augment_rng must be a numpy Generator; got {type(rng).__name__}")
        ranges = {key: _range(augmentation, key) for key in ("exposure_gain", "exposure_bias", "noise_std", "blur_mix")}
        if ranges["noise_std"][0] < 0.0 or ranges["blur_mix"][0] < 0.0 or ranges["blur_mix"][1] > 1.0:
            raise ValueError(f"augmentation needs noise_std >= 0 and blur_mix in [0, 1]; got {dict(augmentation)}")
        self._rng = rng
        self.gain = rng.uniform(*ranges["exposure_gain"])
        self.bias = rng.uniform(*ranges["exposure_bias"])
        self.noise_std = rng.uniform(*ranges["noise_std"])
        self.blur = rng.uniform(*ranges["blur_mix"])

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """(N, N) frame on the 0-1 scale -> (N, N) float32 in [0, 1]."""
        x = frame.astype(np.float64)
        x = (1.0 - self.blur) * x + self.blur * _box_mean(x)
        x = self.gain * x + self.bias
        x = x + self._rng.normal(0.0, self.noise_std, size=x.shape)
        return np.clip(x, 0.0, 1.0).astype(np.float32)


def _check_episode_args(world: Any, view: Any, scenario: Any, attempt_seed: Any, settings: Any, ep: _Episode) -> None:
    if not isinstance(settings, VisionSettings):
        raise TypeError(f"settings must be VisionSettings; got {type(settings).__name__}")
    if not callable(getattr(view, "frame", None)):
        raise TypeError(f"view must have a frame(world) method; got {type(view).__name__}")
    if not isinstance(scenario, Mapping) or not isinstance(scenario.get("situation"), Mapping):
        raise ValueError(f"scenario must be a mapping with a 'situation' mapping; got {scenario!r}")
    if isinstance(attempt_seed, bool) or not isinstance(attempt_seed, (int, np.integer)) or attempt_seed < 0:
        raise ValueError(f"attempt_seed must be an int >= 0; got {attempt_seed!r}")
    if abs(world.policy.control_dt - ep.timing.period_s) > 1e-12:
        raise ValueError(
            f"the world's control period {world.policy.control_dt} s is not command_schema.update_period_s "
            f"{ep.timing.period_s} s"
        )


def _check_frame(name: str, frame: Any, settings: VisionSettings, ep: _Episode) -> int:
    n_frames = ep.timing.frame_periods.size
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
        raise TypeError(f"{name} must be an int; got {frame!r}")
    if not settings.frames - 1 <= frame < n_frames:
        raise ValueError(
            f"{name} must be in [{settings.frames - 1}, {n_frames}) so that its {settings.frames}-frame window is full "
            f"and inside the episode; got {frame}"
        )
    return int(frame)


def _frames_at(
    world: Any, view: Any, settings: VisionSettings, periods: Sequence[int], ep: _Episode, visual: Mapping[str, Any] | None
) -> list[np.ndarray]:
    """Preprocessed frames (view.frame(world, visual)) at the given ascending control periods after the launch
    (period 0 is the state reset returned), stepping the zero command in between; the world is left at the last."""
    zero = ep.actions.commands[ep.actions.no_op]
    frames, now = [], 0
    for period in periods:
        if period > now:
            world.step(zero, (period - now) * ep.timing.period_s)
            now = period
        frames.append(preprocess_frame(view.frame(world, visual), settings))
    return frames


def _measured(outcome: Mapping[str, Any], clip: float, travel: float) -> dict[str, float]:
    """The five outcomes by name from Go2World.finish_episode() and the command travel."""
    if outcome["sim_unstable"]:
        raise RuntimeError("MuJoCo flagged a bad state in this episode (sim_unstable 1); its outcome is no measurement")
    separation = outcome["min_separation_m"]
    return {
        "min_separation_m": clip if separation is None else min(float(separation), clip),
        "collision": float(outcome["collision"]),
        "fall": float(outcome["fall"]),
        "recovery_time_s": float(outcome["recovery_time_s"]),
        "command_travel_m": float(travel),
    }


def observe_and_act(
    world: Any,
    view: Any,
    scenario: Mapping[str, Any],
    attempt_seed: int,
    decision_frame: int,
    action_index: int,
    settings: VisionSettings,
    augment_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """One measured one-shot row on the real world.

    reset(scenario["situation"], attempt_seed) through _reset (a punch: Person A's punch body or the weightless
    ball); the zero command until the decision frame, rendering (``view``) and
    preprocessing the frames of its window; then the action's command for hold_s and the zero command until
    episode_duration_s; finish_episode(). Only the window's own frames are rendered: earlier frames would leave it.
    A hold that would end after episode_duration_s runs in full (frames in decision_time_range_s end in time).

    Args:
        world:          a Go2World; settings other than the window length come from world.config.
        view:           a HeadView (anything with frame(world, visual) -> (H, W, 3) uint8).
        scenario:       from sample_threat_scenarios; its "visual" goes to every view.frame call.
        attempt_seed:   int >= 0, passed to world.reset.
        decision_frame: index into frame_times(world.config), >= settings.frames - 1 (full window).
        action_index:   candidate index into skills.dodge_vision.actions.
        settings:       the image-window contract (window length settings.frames).
        augment_rng:    when given, the window is varied with vision.augment_window (one draw per window).

    Returns:
        (features (D,) float32 in [0, 1], flatten_window order; measured {outcome name: float}):
        min_separation_m = min(world value, separation_clip_m), or the clip when the world reports None;
        collision, fall, recovery_time_s from the world; command_travel_m = max |command field| x hold_s.

    Raises:
        TypeError / ValueError for a bad argument; RuntimeError when MuJoCo flagged a bad state.
    """
    ep = _episode(world.config)
    _check_episode_args(world, view, scenario, attempt_seed, settings, ep)
    frame = _check_frame("decision_frame", decision_frame, settings, ep)
    if isinstance(action_index, bool) or not isinstance(action_index, (int, np.integer)):
        raise TypeError(f"action_index must be an int; got {action_index!r}")
    if not 0 <= action_index < len(ep.actions.names):
        raise ValueError(f"action_index must be in [0, {len(ep.actions.names)}); got {action_index}")
    _reset(world, scenario, attempt_seed)
    periods = ep.timing.frame_periods[frame - settings.frames + 1 : frame + 1]
    window = np.stack(_frames_at(world, view, settings, periods.tolist(), ep, scenario.get("visual")))
    if augment_rng is not None:
        window = augment_window(window, augment_rng, ep.augmentation)
    rest = ep.timing.n_periods - int(periods[-1]) - ep.hold_periods
    world.step(ep.actions.commands[int(action_index)], ep.hold_s)
    if rest > 0:
        world.step(ep.actions.commands[ep.actions.no_op], rest * ep.timing.period_s)
    outcome = world.finish_episode()
    return flatten_window(window), _measured(outcome, ep.separation_clip_m, ep.travel_m[int(action_index)])


def render_windows(
    world: Any,
    view: Any,
    scenario: Mapping[str, Any],
    attempt_seed: int,
    decision_frames: Sequence[int],
    settings: VisionSettings,
    augment_rngs: Sequence[np.random.Generator] | None = None,
) -> np.ndarray:
    """(len(decision_frames), D) float32: the windows at several decision frames of one zero-command episode
    (dream situations: nothing is acted). The episode runs only to the last frame needed and is not finished; the
    next reset abandons it without a trace. augment_rngs, when given, has one Generator per decision frame and
    varies that window with vision.augment_window. Arguments as in observe_and_act."""
    ep = _episode(world.config)
    _check_episode_args(world, view, scenario, attempt_seed, settings, ep)
    if isinstance(decision_frames, (str, bytes)) or not isinstance(decision_frames, Sequence) or not decision_frames:
        raise ValueError(f"decision_frames must be a non-empty sequence of ints; got {decision_frames!r}")
    frames = [_check_frame("decision frame", frame, settings, ep) for frame in decision_frames]
    if augment_rngs is not None and len(augment_rngs) != len(frames):
        raise ValueError(f"augment_rngs must have one Generator per decision frame ({len(frames)}); got {len(augment_rngs)}")
    needed = sorted({k for frame in frames for k in range(frame - settings.frames + 1, frame + 1)})
    _reset(world, scenario, attempt_seed)
    periods = ep.timing.frame_periods[needed].tolist()
    rendered = dict(zip(needed, _frames_at(world, view, settings, periods, ep, scenario.get("visual"))))
    out = np.empty((len(frames), settings.n_features), dtype=np.float32)
    for row, frame in enumerate(frames):
        window = np.stack([rendered[k] for k in range(frame - settings.frames + 1, frame + 1)])
        if augment_rngs is not None:
            window = augment_window(window, augment_rngs[row], ep.augmentation)
        out[row] = flatten_window(window)
    return out


# ----- kinematic dream frames ------------------------------------------------------------------------------------
#
# In a dream episode the dog only stands on the zero command and a punch flies a straight line at constant velocity
# (gravcomp, no drag). Before the punch touches the dog nothing couples the two (the object is its own free body after
# the robot: its own tree, no constraint rows), the simulator is deterministic and the zero command makes the latency
# draw irrelevant, so the robot's qpos at each frame is the same in every zero-command episode of the punch scene.
# A scene without an object runs Go2World's scene without the object body, whose standing dog differs (measured
# 2026-11-21, world seed 0: up to 0.023 in qpos over the episode, already at the launch instant; up to 0.04 grey in a
# window), so it gets a stand of its own. Both identities were checked against render_windows: 0.0 difference.
# record_stand records both once; render_windows_kinematic puts the stand and the punch's line into a private MjData
# and paints the frame through HeadView, with no physics episode.


def _stand_scenario(config: Mapping[str, Any], empty: bool = False) -> dict[str, Any]:
    """The punch stand (the punch _STAND_OBJECT) or, with empty, the scene without an object."""
    situation: dict[str, Any] = {} if empty else {k: float(v) for k, v in zip(_ball_keys(), _STAND_OBJECT)}
    situation.update(dog_forward_speed_m_s=0.0, settle_s=_value(config, f"{_ROOT}.settle_s"))
    return {"scenario_id": -1, "kind": "empty" if empty else "stand", "object_kind": "punch", "situation": situation}


def _stand_qpos(world: Any, ep: _Episode, q: int) -> np.ndarray:
    """(n_frames, q) qpos[:q] at every camera frame of the episode just reset, stepping the zero command."""
    rows = np.empty((ep.timing.frame_periods.size, q), dtype=np.float64)
    zero, now = ep.actions.commands[ep.actions.no_op], 0
    for index, period in enumerate(ep.timing.frame_periods.tolist()):
        if period > now:
            world.step(zero, (period - now) * ep.timing.period_s)
            now = period
        rows[index] = world.policy.data.qpos[:q]
    return rows


def record_stand(world: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Two zero-command episodes, recording at every camera frame time the robot's part of qpos (all qpos before the
    object's free joint: the trunk free joint and the 12 leg joints; HeadView reads qpos only): one in the punch scene
    with the punch _STAND_OBJECT (50 m behind, flying away), with the trunk pose at the launch instant (the frame of a
    situation's object coordinates), and one in the scene without an object. The last episode is left open (the next
    reset abandons it).

    Returns {"frame_periods" (n_frames,) int64, "robot_qpos" (n_frames, q) float64 (punch scene), "empty_qpos"
    (n_frames, q) float64 (scene without an object), "launch_pos" (3,) float64, "launch_yaw" float, "object_qposadr"
    q, "world_seed", "world_preset", and in memory only "model" / "empty_model" (the two scenes' MjModels), "data" /
    "empty_data" (a private MjData of each) and the "probe_*" keys of _contact_probe}.
    """
    from learned_judgment.env.world import _yaw  # on use: see _ball_keys; the yaw Go2World._throw turns the launch by

    ep = _episode(config)
    if abs(world.policy.control_dt - ep.timing.period_s) > 1e-12:
        raise ValueError(f"the world's control period {world.policy.control_dt} s is not {ep.timing.period_s} s")
    _reset(world, _stand_scenario(config), 0)
    model, data = world.model, world.policy.data
    q = int(model.jnt_qposadr[model.body("ball").jntadr[0]])
    launch_pos, launch_yaw = data.qpos[0:3].copy(), float(_yaw(data.qpos[3:7]))
    robot = _stand_qpos(world, ep, q)
    _reset(world, _stand_scenario(config, empty=True), 0)
    empty_model = world.model
    if empty_model.nq != q:
        raise RuntimeError(f"the scene without an object has nq {empty_model.nq}, not the robot's {q}")
    empty = _stand_qpos(world, ep, q)
    return {
        "frame_periods": ep.timing.frame_periods.astype(np.int64),
        "robot_qpos": robot,
        "empty_qpos": empty,
        "launch_pos": launch_pos,
        "launch_yaw": launch_yaw,
        "object_qposadr": q,
        "world_seed": int(world.seed),
        "world_preset": str(world.preset),
        "model": model,
        "data": mujoco.MjData(model),
        "empty_model": empty_model,
        "empty_data": mujoco.MjData(empty_model),
    } | _contact_probe(model, robot, q)


def _save_stand(stand: Mapping[str, Any], path: Path) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        frame_periods=stand["frame_periods"],
        robot_qpos=stand["robot_qpos"],
        empty_qpos=stand["empty_qpos"],
        launch_pos=stand["launch_pos"],
        launch_yaw=np.float64(stand["launch_yaw"]),
        object_qposadr=np.int64(stand["object_qposadr"]),
        world_seed=np.int64(stand["world_seed"]),
        world_preset=np.str_(stand["world_preset"]),
    )
    atomic_write_bytes(path, buffer.getvalue())


def _load_stand(world: Any, config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """stand.npz for this world: a reset of each stand scenario gives that scene's model, and its launch-instant qpos
    must equal the file's bit for bit (else RuntimeError: the file belongs to another world or config)."""
    with np.load(path) as file:
        stand: dict[str, Any] = {key: file[key] for key in file.files}
    stand["launch_yaw"] = float(stand["launch_yaw"])
    stand["object_qposadr"] = q = int(stand["object_qposadr"])
    stand["world_seed"], stand["world_preset"] = int(stand["world_seed"]), str(stand["world_preset"])
    if stand["world_seed"] != world.seed or stand["world_preset"] != world.preset:
        raise RuntimeError(f"{path} is the stand of world seed {stand['world_seed']} {stand['world_preset']!r}")
    for empty, key in ((False, ""), (True, "empty_")):
        _reset(world, _stand_scenario(config, empty), 0)
        qpos = world.policy.data.qpos
        recorded = stand["empty_qpos" if empty else "robot_qpos"]
        if stand["frame_periods"][0] != 0 or not np.array_equal(qpos[:q], recorded[0]) or (
            not empty and not np.array_equal(qpos[0:3], stand["launch_pos"])
        ):
            raise RuntimeError(f"{path}: this world's stand differs at the launch instant; the file is stale")
        stand[f"{key}model"], stand[f"{key}data"] = world.model, mujoco.MjData(world.model)
    stand.update(_contact_probe(stand["model"], stand["robot_qpos"], q))
    return stand


def _contact_probe(model: Any, robot_qpos: np.ndarray, q: int) -> dict[str, Any]:
    """In-memory keys of a stand for _kinematic_limit_s: every geom of the punch scene the object can touch (collision
    masks set, not the object's: the robot's collision geoms and the floor) with its pose at each frame of the stand."""
    data = mujoco.MjData(model)
    body = model.body("ball").id
    geoms = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] != body
                      and (model.geom_contype[g] or model.geom_conaffinity[g])], dtype=np.int64)
    plane = model.geom_type[geoms] == int(mujoco.mjtGeom.mjGEOM_PLANE)  # rbound 0: no bounding sphere
    solid = geoms[~plane]
    xpos = np.empty((robot_qpos.shape[0], solid.size, 3))
    xmat = np.empty((robot_qpos.shape[0], solid.size, 9))
    for k, row in enumerate(robot_qpos):
        data.qpos[:q] = row
        mujoco.mj_kinematics(model, data)
        xpos[k], xmat[k] = data.geom_xpos[solid], data.geom_xmat[solid]
    # Each geom's surface move from frame k to k + 1 (the last frame repeats the largest): centre move plus rbound
    # times the rotation angle (trace(R_k^T R_k+1) = sum of the elementwise product).
    turn = np.arccos(np.clip(((xmat[:-1] * xmat[1:]).sum(axis=2) - 1.0) / 2.0, -1.0, 1.0))
    move = np.linalg.norm(np.diff(xpos, axis=0), axis=2) + model.geom_rbound[solid] * turn
    move = np.vstack([move, move.max(axis=0)])
    obj = model.geom("ball").id
    return {"probe": data, "object_geom": obj, "probe_planes": geoms[plane], "probe_geoms": solid,
            "probe_reach": model.geom_rbound[solid] + model.geom_rbound[obj], "probe_xpos": xpos, "probe_xmat": xmat,
            "probe_move": move}


def _launch_world(stand: Mapping[str, Any], situation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(position (3,), velocity (3,)) at the launch of a situation's object in the world frame, as Go2World._throw
    turns them with the stand's launch-instant trunk pose."""
    launch = [float(situation[key]) for key in _ball_keys()]
    yaw = float(stand["launch_yaw"])
    turn = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return stand["launch_pos"] + turn @ launch[:3], turn @ launch[3:]


def _kinematic_limit_s(config: Mapping[str, Any], scenario: Mapping[str, Any], stand: Mapping[str, Any]) -> float:
    """Decision times below this are drawn exactly by render_windows_kinematic: inf for a scene without an object,
    -inf for an object other than a punch, else the punch's predicted first contact minus
    skills.dodge_vision.kinematic_contact_margin_s (inf when no contact is predicted in the stand's frames).

    The contact, from the stand's real geometry: the first frame time t_k at which the punch on its straight line is
    closer than |velocity| x (t_(k+1) - t_k) + _SWAY_FACTOR x (the geom's move from frame k to k + 1) to a geom it can
    touch in the stand's pose of frame k (mj_geomDistance; the robot's collision geoms and the floor, which does not
    move). No contact can happen between two earlier frames, so t_k is at or before the real first contact. (The
    physics guess's trunk capsule missed real contacts with the legs and hips: 7 of 300 dream scenes, 2026-11-21.)"""
    situation = scenario["situation"]
    if not any(key in situation for key in _ball_keys()):
        return math.inf
    if scenario.get("object_kind", "ball") != "punch":
        return -math.inf
    margin = _value(config, f"{_ROOT}.kinematic_contact_margin_s")
    if margin < 0.0:
        raise ConfigError(f"config.yaml key '{_ROOT}.kinematic_contact_margin_s' must be >= 0; got {margin}")
    start, velocity = _launch_world(stand, situation)
    speed = float(np.linalg.norm(velocity))
    period_s = _positive(config, "robot.walking_policy.command_schema.update_period_s")
    times = stand["frame_periods"] * period_s
    gaps = np.diff(times, append=times[-1] + np.diff(times).max())
    model, data, obj = stand["model"], stand["probe"], stand["object_geom"]
    geoms, planes, reach = stand["probe_geoms"], stand["probe_planes"], stand["probe_reach"]
    data.geom_xmat[obj] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for k, t in enumerate(times):
        point = start + velocity * t
        travel = speed * gaps[k]
        thresholds = travel + _SWAY_FACTOR * stand["probe_move"][k]
        near = np.flatnonzero(np.linalg.norm(stand["probe_xpos"][k] - point, axis=1) - reach < thresholds)
        data.geom_xpos[geoms], data.geom_xmat[geoms] = stand["probe_xpos"][k], stand["probe_xmat"][k]
        data.geom_xpos[obj] = point
        pairs = [(int(geoms[i]), float(thresholds[i])) for i in near] + [(int(g), travel) for g in planes]
        for g, threshold in pairs:
            if mujoco.mj_geomDistance(model, data, obj, g, threshold, None) < threshold:
                return float(t) - margin
    return math.inf


def render_windows_kinematic(
    world: Any,
    view: Any,
    scenario: Mapping[str, Any],
    decision_frames: Sequence[int],
    settings: VisionSettings,
    stand: Mapping[str, Any],
    augment_rngs: Sequence[np.random.Generator] | None = None,
) -> np.ndarray:
    """(len(decision_frames), D) float32: render_windows of the same arguments without a physics episode.

    For each needed frame k: a private MjData (stand["data"], of the punch scene stand["model"]) gets the robot qpos
    the stand recorded at frame k and the object's pose launch position + velocity x t (t = frame k's time; the
    situation's launch state turned into the world frame as Go2World._throw does, from the stand's launch-instant
    trunk pose; orientation identity), and view.frame paints it (HeadView's kinematics pass, the scenario's visual).
    Preprocessing, windows and augmentation (one augment_rngs Generator per decision frame) are those of render_windows.
    A scene without an object is drawn from the stand of the scene without an object (stand["empty_*"]).

    stand: from record_stand or _load_stand, for this world (seed, preset) and config (frame periods).

    Raises ValueError for what it cannot draw exactly (the caller renders those with render_windows): an object other
    than a punch, any decision frame at or after _kinematic_limit_s (the predicted first contact, from the stand's
    collision geoms and the floor, less the margin), a world that renders the head camera itself; and for a bad
    argument or a stand of another world.
    """
    ep = _episode(world.config)
    if not isinstance(settings, VisionSettings):
        raise TypeError(f"settings must be VisionSettings; got {type(settings).__name__}")
    if not callable(getattr(view, "frame", None)):
        raise TypeError(f"view must have a frame(world) method; got {type(view).__name__}")
    if not isinstance(scenario, Mapping) or not isinstance(scenario.get("situation"), Mapping):
        raise ValueError(f"scenario must be a mapping with a 'situation' mapping; got {scenario!r}")
    if not isinstance(stand, Mapping) or "model" not in stand:
        raise ValueError("stand must come from record_stand or _load_stand")
    if stand["world_seed"] != world.seed or stand["world_preset"] != world.preset:
        raise ValueError(f"the stand is of world seed {stand['world_seed']} {stand['world_preset']!r}, not this world's")
    if not np.array_equal(stand["frame_periods"], ep.timing.frame_periods):
        raise ValueError("the stand's frame periods are not this config's")
    render_frame = getattr(world, "render_frame", None)
    if render_frame is not None and "camera" in inspect.signature(render_frame).parameters:
        raise ValueError("the world renders the head camera itself; its frames are drawn by render_windows")
    if isinstance(decision_frames, (str, bytes)) or not isinstance(decision_frames, Sequence) or not decision_frames:
        raise ValueError(f"decision_frames must be a non-empty sequence of ints; got {decision_frames!r}")
    frames = [_check_frame("decision frame", frame, settings, ep) for frame in decision_frames]
    if augment_rngs is not None and len(augment_rngs) != len(frames):
        raise ValueError(f"augment_rngs must have one Generator per decision frame ({len(frames)}); got {len(augment_rngs)}")
    situation = scenario["situation"]
    keys = _ball_keys()
    has_object = any(key in situation for key in keys)
    if has_object and scenario.get("object_kind", "ball") != "punch":
        raise ValueError(f"object_kind {scenario.get('object_kind', 'ball')!r} is not drawn kinematically (only a punch)")
    limit = _kinematic_limit_s(world.config, scenario, stand)
    times = ep.timing.frame_periods * ep.timing.period_s
    late = [frame for frame in frames if times[frame] >= limit]
    if late:
        raise ValueError(f"decision frame(s) {late} lie at or after {limit:.3f} s, the punch's predicted contact less the margin")

    q = int(stand["object_qposadr"])
    if has_object:
        start, velocity = _launch_world(stand, situation)
        model, data, robot = stand["model"], stand["data"], stand["robot_qpos"]
    else:
        model, data, robot = stand["empty_model"], stand["empty_data"], stand["empty_qpos"]
    proxy = SimpleNamespace(model=model, policy=SimpleNamespace(data=data))
    visual = scenario.get("visual")
    needed = sorted({k for frame in frames for k in range(frame - settings.frames + 1, frame + 1)})
    rendered = {}
    for k in needed:
        data.qpos[:q] = robot[k]
        if has_object:
            data.qpos[q : q + 3] = start + velocity * times[k]
            data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
        rendered[k] = preprocess_frame(view.frame(proxy, visual), settings)
    out = np.empty((len(frames), settings.n_features), dtype=np.float32)
    for row, frame in enumerate(frames):
        window = np.stack([rendered[k] for k in range(frame - settings.frames + 1, frame + 1)])
        if augment_rngs is not None:
            window = augment_window(window, augment_rngs[row], ep.augmentation)
        out[row] = flatten_window(window)
    return out


def closed_loop(
    world: Any,
    view: Any,
    scenario: Mapping[str, Any],
    attempt_seed: int,
    policy: Callable[[np.ndarray, float], Mapping[str, Any]],
    settings: VisionSettings,
    augment_rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """One closed-loop episode: the zero command until the policy says otherwise.

    The world is stepped one control period at a time. At every frame whose window is full,
    policy(window (frames, N, N) float32, t_s) must return a mapping with at least "command" (a mapping of the
    command fields, validated by the world) and "action" (str); that command is applied until the next frame.

    augment_rng, when given, makes one draw of vision.augmentation's four values for the whole episode and applies
    them to every frame as it is taken, with fresh noise per frame.

    Returns:
        {"measured": as observe_and_act, except command_travel_m = sum over control periods of
                     |(vel_x, vel_y)| x period,
         "trace":    per policy call {key: value of the returned mapping except numpy arrays, "t_s": t_s},
         "issued":   [(t_s, action)] for the outputs whose "issued" is True}

    Raises TypeError / ValueError for a bad argument or a policy output without "command" and "action", the world's
    CommandError for a command it does not accept, RuntimeError when MuJoCo flagged a bad state.
    """
    ep = _episode(world.config)
    _check_episode_args(world, view, scenario, attempt_seed, settings, ep)
    if not callable(policy):
        raise TypeError(f"policy must be callable; got {type(policy).__name__}")
    augment = None if augment_rng is None else _EpisodeAugmentation(augment_rng, ep.augmentation)
    frame_periods = set(ep.timing.frame_periods.tolist())
    period_s = ep.timing.period_s
    visual = scenario.get("visual")
    _reset(world, scenario, attempt_seed)
    window = FrameWindow(settings)
    command: dict[str, Any] = dict(ep.actions.commands[ep.actions.no_op])
    travel = 0.0
    trace: list[dict[str, Any]] = []
    issued: list[tuple[float, str]] = []
    for period in range(ep.timing.n_periods):
        if period in frame_periods:
            frame = preprocess_frame(view.frame(world, visual), settings)
            window.push(frame if augment is None else augment.apply(frame))
            if window.full:
                t_s = period * period_s
                out = policy(window.array(), t_s)
                if not isinstance(out, Mapping) or not isinstance(out.get("command"), Mapping) or "action" not in out:
                    raise ValueError(f"policy must return a mapping with 'command' (a mapping) and 'action'; got {out!r}")
                command = dict(out["command"])
                entry = {key: value for key, value in out.items() if not isinstance(value, np.ndarray)}
                entry["command"], entry["t_s"] = dict(command), t_s
                trace.append(entry)
                if out.get("issued") is True:
                    issued.append((t_s, str(out["action"])))
        world.step(command, period_s)
        travel += math.hypot(float(command[_PLANAR[0]]), float(command[_PLANAR[1]])) * period_s
    outcome = world.finish_episode()
    return {"measured": _measured(outcome, ep.separation_clip_m, travel), "trace": trace, "issued": issued}


# ----- physics guess and objective -------------------------------------------------------------------------------


def _time_grid(step_s: float, duration_s: float) -> np.ndarray:
    """(T,) 0, step, 2 step, ... up to and including duration_s."""
    count = int(math.floor(duration_s / step_s + _TOLERANCE))
    times = np.minimum(step_s * np.arange(count + 1, dtype=np.float64), duration_s)
    if duration_s - times[-1] > _TOLERANCE:
        times = np.append(times, duration_s)
    return times


def physics_guess(
    config: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    decision_times_s: Sequence[float] | np.ndarray,
    action_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Outcome guess (n, K) in OUTCOME_NAMES order from the scenario's truth (scorer side; never a student input).

    Args:
        config:           config.yaml (skills.dodge_vision.physics, ball.radius_m, the actions).
        scenarios:        n scenarios, one per row (from sample_threat_scenarios).
        decision_times_s: (n,) s after the launch at which the row's command is issued, >= 0.
        action_indices:   (n,) candidate indices into skills.dodge_vision.actions.

    Model on a grid of physics.time_step_s from 0 to episode_duration_s (frame: Go2World's at the launch, z from the
    trunk origin):
      * object, by scenario["object_kind"] (default "ball"):
        ball:  from its launch state, ballistic under gravity_m_s2 until its centre reaches the floor rest height
               ball.radius_m - base_height_m, then on the floor keeping its horizontal velocity (a ball starting at
               or below that height keeps its height: a rolling start).
        punch: from its launch state, a straight line at constant velocity for the whole episode (no gravity, no
               floor stop); radius ball.radius_m (the stand-in is the ball).
      * dog: at rest; from decision time + assumed_latency_s its planar velocity approaches the action's
        (vel_x, vel_y) first-order (response_time_s) while the command is held for hold_s, then decays the same
        way to 0. Heading fixed.
      * trunk: capsule along x of half length body_half_length_m and radius body_radius_m on the trunk origin.
      * clearance = distance(ball centre, capsule axis segment) - body_radius_m - ball radius.
    Outcomes: min_separation_m = clip(min clearance, 0, separation_clip_m); collision = logistic(-min clearance /
    collision_softness_m); fall 0; recovery_time_s 0; command_travel_m = max |command field| x hold_s (exact).
    A scenario without a ball: min_separation_m = separation_clip_m, collision 0.

    Raises ValueError for a bad argument; ConfigError for a missing key or an action with a non-planar field.
    """
    ep = _episode(config)
    n = len(scenarios)
    times = np.asarray(decision_times_s, dtype=np.float64)
    actions = np.asarray(action_indices)
    if times.shape != (n,) or not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError(f"decision_times_s must be ({n},) finite and >= 0; got shape {times.shape}")
    if actions.shape != (n,) or (n and not np.issubdtype(actions.dtype, np.integer)):
        raise ValueError(f"action_indices must be ({n},) ints; got {actions.shape} {actions.dtype}")
    if np.any(actions < 0) or np.any(actions >= len(ep.actions.names)):
        raise ValueError(f"action_indices must be in [0, {len(ep.actions.names)})")
    fields = ep.actions.fields
    planar = [fields.index(name) for name in _PLANAR]
    if np.any(np.delete(ep.actions.features, planar, axis=1)):
        raise ConfigError(f"the physics guess models planar commands only; an action of '{_ROOT}.actions' sets another field")
    p = f"{_ROOT}.physics"
    gravity, latency = _positive(config, f"{p}.gravity_m_s2"), _value(config, f"{p}.assumed_latency_s")
    tau, step_s = _positive(config, f"{p}.response_time_s"), _positive(config, f"{p}.time_step_s")
    softness, half = _positive(config, f"{p}.collision_softness_m"), _positive(config, f"{p}.body_half_length_m")
    radius = _positive(config, "ball.radius_m")
    reach = _positive(config, f"{p}.body_radius_m") + radius
    floor_z = radius - _positive(config, f"{p}.base_height_m")
    duration = _positive(config, f"{_ROOT}.episode_duration_s")

    ball_keys = _ball_keys()
    situations, objects = [], []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping) or not isinstance(scenario.get("situation"), Mapping):
            raise ValueError(f"scenarios[{index}] must be a mapping with a 'situation' mapping")
        objects.append(scenario.get("object_kind", "ball"))
        if objects[-1] not in _OBJECT_KINDS:
            raise ValueError(f"scenarios[{index}] has object_kind {objects[-1]!r}, not one of {list(_OBJECT_KINDS)}")
        situations.append(scenario["situation"])
    has_ball = np.array([all(k in s for k in ball_keys) for s in situations], dtype=bool)
    closest = np.full(n, np.inf)
    rows = np.flatnonzero(has_ball)
    if rows.size:
        launch = np.array([[float(situations[i][k]) for k in ball_keys] for i in rows], dtype=np.float64)
        x0, y0, z0, vx, vy, vz = launch.T
        command = ep.actions.features[actions[rows]][:, planar]
        punch = np.array([objects[i] == "punch" for i in rows], dtype=bool)
        height = z0 - floor_z  # ball centre above its floor rest height
        landing = np.where(height > 0.0, (vz + np.sqrt(vz * vz + 2.0 * gravity * np.maximum(height, 0.0))) / gravity, 0.0)
        landing[punch] = np.inf  # a punch: straight at constant velocity for the whole episode
        fall = np.where(punch, 0.0, gravity)
        start = times[rows] + latency  # the dog's velocity starts to change
        stop = start + ep.hold_s  # ... and starts to decay back to 0
        near = np.full(rows.size, np.inf)
        for t in _time_grid(step_s, duration):
            flight = np.minimum(t, landing)
            ball_x, ball_y = x0 + vx * t, y0 + vy * t
            ball_z = z0 + vz * flight - 0.5 * fall * flight * flight
            rise = np.clip(t - start, 0.0, ep.hold_s)  # time spent rising toward the commanded velocity
            decay = np.maximum(t - stop, 0.0)  # time since the held command ended
            left = np.exp(-rise / tau)
            travel = rise - tau * (1.0 - left) + (1.0 - left) * tau * (1.0 - np.exp(-decay / tau))  # per unit speed
            along = ball_x - command[:, 0] * travel
            off_axis = along - np.clip(along, -half, half)
            distance = np.sqrt(off_axis * off_axis + (ball_y - command[:, 1] * travel) ** 2 + ball_z * ball_z)
            np.minimum(near, distance - reach, out=near)
        closest[rows] = near
    guess = np.empty((n, len(OUTCOME_NAMES)), dtype=np.float64)
    guess[:, 0] = np.clip(closest, 0.0, ep.separation_clip_m)
    guess[:, 1] = 0.5 * (1.0 + np.tanh(-closest / (2.0 * softness)))  # logistic(-clearance / softness); 0 at inf
    guess[:, 2] = 0.0
    guess[:, 3] = 0.0
    guess[:, 4] = np.asarray(ep.travel_m, dtype=np.float64)[actions] if n else 0.0
    return guess


TRUTH_NAMES = ("present", "time_to_arrival_s", "aim_y_m", "aim_z_m", "speed_m_s")


# shortcut: straight flight only (the punch); a ball under gravity raises ValueError here, give it physics_guess's ballistic path before throwing balls again.
def truth_features(
    config: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]], decision_times_s: Sequence[float]
) -> np.ndarray:
    """Scene truth (n, 5) float64 in TRUTH_NAMES order, for the teacher's "truth" block (never a student input).

    From the scenario's launch state (the six _ball_keys, launch frame: x ahead, y left) moved on its straight line to
    the row's decision time: an object arrives when its x velocity is < 0 and it is still ahead (x > 0);
    time_to_arrival_s = x / closing speed clipped to [0, episode_duration_s], aim = (y, z) at that moment. Otherwise
    time_to_arrival_s = episode_duration_s and the aim is the current (y, z). speed_m_s = |velocity|. No object (no
    launch key, kind empty): [0, episode_duration_s, 0, 0, 0].
    """
    duration = _positive(config, f"{_ROOT}.episode_duration_s")
    times = np.asarray(decision_times_s, dtype=np.float64)
    n = len(scenarios)
    if times.shape != (n,) or not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError(f"decision_times_s must be ({n},) finite and >= 0; got shape {times.shape}")
    keys = _ball_keys()
    out = np.zeros((n, len(TRUTH_NAMES)), dtype=np.float64)
    out[:, 1] = duration
    for i, scenario in enumerate(scenarios):
        situation = scenario["situation"]
        if not any(k in situation for k in keys):
            continue
        if scenario.get("object_kind", "ball") != "punch":
            raise ValueError(f"truth_features models a straight flight (punch) only; scenario {scenario.get('scenario_id')} "
                             f"has object_kind {scenario.get('object_kind', 'ball')!r}")
        x0, y0, z0, vx, vy, vz = (float(situation[k]) for k in keys)
        t = times[i]
        x, y, z = x0 + vx * t, y0 + vy * t, z0 + vz * t
        arrival = duration
        if vx < 0.0 and x > 0.0:
            arrival = min(x / -vx, duration)
            y, z = y + vy * arrival, z + vz * arrival
        out[i] = (1.0, arrival, y, z, math.sqrt(vx * vx + vy * vy + vz * vz))
    return out


def objective(config: Mapping[str, Any], predicted: OutcomeDistribution) -> np.ndarray:
    """Risk-aware utility (n,), higher is better (weights: skills.dodge_vision.objective_weights)::

        - collision  * max(mean collision, P(min_separation_m <= contact_margin_m))
        - fall       * mean fall
        + separation * min(lowest-level quantile of min_separation_m, separation_cap_m)
        - motion     * mean command_travel_m
        - recovery   * mean recovery_time_s

    The formula of skills/dodge.py DodgeSpec.objective with command_travel_m for lateral_command_m. The lowest level
    is predicted.quantile_levels[0]. Raises TypeError / ValueError unless predicted is an OutcomeDistribution over
    OUTCOME_NAMES; ConfigError for a missing weight.
    """
    if not isinstance(predicted, OutcomeDistribution):
        raise TypeError(f"predicted must be an OutcomeDistribution; got {type(predicted).__name__}")
    if tuple(predicted.names) != OUTCOME_NAMES:
        raise ValueError(f"predicted outcomes {predicted.names} are not {OUTCOME_NAMES}")
    w = {key: _value(config, f"{_ROOT}.objective_weights.{key}") for key in (
        "collision", "fall", "separation", "separation_cap_m", "contact_margin_m", "motion", "recovery")}
    contact = np.maximum(
        predicted.mean_of("collision"), predicted.prob_below("min_separation_m", w["contact_margin_m"])
    )
    low_separation = predicted.quantile_of("min_separation_m", float(predicted.quantile_levels[0]))
    return (
        -w["collision"] * contact
        - w["fall"] * predicted.mean_of("fall")
        + w["separation"] * np.minimum(low_separation, w["separation_cap_m"])
        - w["motion"] * predicted.mean_of("command_travel_m")
        - w["recovery"] * predicted.mean_of("recovery_time_s")
    )


@dataclass(frozen=True)
class DodgeVisionSpec:
    """What learners and scorers need of dodge_vision (not a SkillSpec; the registry does not load it).

    name                  "dodge_vision"
    action_names          (C,) skills.dodge_vision.actions in config order
    action_features       (C, 3) float64, read-only: each action's command fields in action_feature_names order
    action_feature_names  the verified command fields by native index (vel_x, vel_y, yaw_rate)
    no_op_index           index of the zero command
    outcome_names         OUTCOME_NAMES; outcome_kinds OUTCOME_KINDS
    outcome_bounds        (K, 2) float64, read-only: [low, high] per outcome
    truth_names           TRUTH_NAMES: the columns of truth() (teacher side only; never a student input)
    """

    name: str
    action_names: tuple[str, ...]
    action_features: np.ndarray
    action_feature_names: tuple[str, ...]
    no_op_index: int
    outcome_names: tuple[str, ...]
    outcome_kinds: tuple[str, ...]
    outcome_bounds: np.ndarray
    config: Mapping[str, Any] = field(repr=False, compare=False)
    truth_names: tuple[str, ...] = TRUTH_NAMES

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        """(n,) objective(config, predicted)."""
        return objective(self.config, predicted)

    def truth(self, scenarios: Sequence[Mapping[str, Any]], decision_times_s: Sequence[float]) -> np.ndarray:
        """(n, 5) truth_features(config, scenarios, decision_times_s)."""
        return truth_features(self.config, scenarios, decision_times_s)


def build_observed_spec(config: Mapping[str, Any]) -> DodgeVisionSpec:
    """The DodgeVisionSpec of config.yaml. Raises ConfigError for a missing or invalid key."""
    ep = _episode(config)
    bounds = np.array(
        [
            [0.0, ep.separation_clip_m],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, _positive(config, f"{_ROOT}.episode_duration_s")],
            [0.0, max(ep.travel_m)],
        ],
        dtype=np.float64,
    )
    bounds.setflags(write=False)
    return DodgeVisionSpec(
        name=SKILL,
        action_names=ep.actions.names,
        action_features=ep.actions.features,
        action_feature_names=ep.actions.fields,
        no_op_index=ep.actions.no_op,
        outcome_names=OUTCOME_NAMES,
        outcome_kinds=OUTCOME_KINDS,
        outcome_bounds=bounds,
        config=config,
    )


# ----- collection ------------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _make_world(config: Mapping[str, Any]) -> Any:
    """A Go2World of observed.world_seed and observed.world_preset that writes no trace files."""
    from learned_judgment.env import Go2World  # on use: see _ball_keys

    config = copy.deepcopy(dict(config))
    config["world"]["save_traces"] = False
    return Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))


def _worker_init(config: Mapping[str, Any], frames: int, stand_path: str | None = None) -> None:
    """Per process: its own Go2World (_make_world), HeadView and, given stand_path, the stand (_load_stand)."""
    world = _make_world(config)
    _WORKER.update(world=world, view=HeadView(config), settings=VisionSettings.from_config(config, frames))
    _WORKER["stand"] = None if stand_path is None else _load_stand(world, world.config, Path(stand_path))


def _row_task(task: tuple) -> tuple[int, int, np.ndarray, dict[str, float]]:
    scenario, frame, action, augment = task
    w = _WORKER
    features, measured = observe_and_act(
        w["world"], w["view"], scenario, scenario["scenario_id"], frame, action, w["settings"], np.random.default_rng(augment)
    )
    return scenario["scenario_id"], action, features, measured


def _dream_task(task: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """(scenario_id, windows (len(frames), D), kinematic (len(frames),) bool). With a stand loaded, each decision frame
    before _kinematic_limit_s is drawn by render_windows_kinematic and the rest by render_windows; each window keeps
    its own augmentation Generator either way."""
    scenario, frames, augments = task
    w = _WORKER
    world, stand = w["world"], w["stand"]
    rngs = [np.random.default_rng(seed) for seed in augments]
    kinematic = np.zeros(len(frames), dtype=bool)
    if stand is not None:
        limit = _kinematic_limit_s(world.config, scenario, stand)
        kinematic[:] = frame_times(world.config)[frames] < limit
    windows = np.empty((len(frames), w["settings"].n_features), dtype=np.float32)
    for flag in (True, False):
        rows = np.flatnonzero(kinematic == flag)
        if rows.size == 0:
            continue
        chosen, chosen_rngs = [frames[i] for i in rows], [rngs[i] for i in rows]
        if flag:
            windows[rows] = render_windows_kinematic(world, w["view"], scenario, chosen, w["settings"], stand, chosen_rngs)
        else:
            windows[rows] = render_windows(world, w["view"], scenario, scenario["scenario_id"], chosen, w["settings"], chosen_rngs)
    return scenario["scenario_id"], windows, kinematic


def _run(
    config: Mapping[str, Any], frames: int, function: Callable[[tuple], Any], tasks: list[tuple], stand_path: str | None = None
) -> list[Any]:
    """function over tasks in observed.workers spawned processes, results in task order, sparse progress lines."""
    workers = require(config, "observed.workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"config.yaml key 'observed.workers' must be an int >= 1; got {workers!r}")
    every = max(1, len(tasks) // 10)
    results = []
    context = multiprocessing.get_context("spawn")
    initargs = (config, frames, stand_path)
    with context.Pool(min(workers, len(tasks)), initializer=_worker_init, initargs=initargs) as pool:
        for result in pool.imap(function, tasks):
            results.append(result)
            if len(results) % every == 0 or len(results) == len(tasks):
                print(f"{len(results)}/{len(tasks)}", flush=True)
    return results


def _split_plan(config: Mapping[str, Any], split: str, n: int) -> tuple[list[dict[str, Any]], list[np.random.SeedSequence]]:
    """The split's scenarios and, per scenario, its SeedSequence (see the module docstring)."""
    seed = require(config, "runtime.seed")
    code = SPLIT_CODES[split]
    scenarios = sample_threat_scenarios(config, np.random.default_rng(np.random.SeedSequence([seed, code])), n)
    seeds = [np.random.SeedSequence([seed, code], spawn_key=(s["scenario_id"],)) for s in scenarios]
    return scenarios, seeds


def _eligible_frames(config: Mapping[str, Any], frames: int) -> np.ndarray:
    """Frame indices whose time lies in skills.dodge_vision.decision_time_range_s and whose window is full."""
    low, high = _pair(config, f"{_ROOT}.decision_time_range_s", minimum=0.0)
    times = frame_times(config)
    index = np.arange(times.size)
    eligible = index[(times >= low - _TOLERANCE) & (times <= high + _TOLERANCE) & (index >= frames - 1)]
    if eligible.size == 0:
        raise ConfigError(f"no frame with a full {frames}-frame window lies in '{_ROOT}.decision_time_range_s'")
    return eligible


def arrival_time_s(config: Mapping[str, Any], scenario: Mapping[str, Any]) -> float | None:
    """t_s at which the scenario's object, on its straight line from the launch state, reaches the plane
    x = punch.aim_x_m (the head) of the dog's launch frame: (x0 - aim_x_m) / -vx; None for a scene without an
    object or with one that does not approach (x velocity >= -1e-9). Scorer-side arithmetic, as
    _frames_before_arrival; dodge_eval uses it as the deadline of a near approach that makes no contact."""
    situation = scenario["situation"]
    if "ball_vel_x_m_s" not in situation or float(situation["ball_vel_x_m_s"]) >= -1e-9:
        return None
    aim_x = float(require(config, f"{_ROOT}.punch.aim_x_m"))
    return (float(situation["ball_rel_x_m"]) - aim_x) / -float(situation["ball_vel_x_m_s"])


def _frames_before_arrival(config: Mapping[str, Any], scenario: Mapping[str, Any], eligible: np.ndarray) -> np.ndarray:
    """The eligible decision frames of one scenario that lie before its object reaches the dog.

    An object moving toward the dog (x velocity < 0 in the launch frame) reaches the plane x = punch.aim_x_m at
    t = (x0 - aim_x_m) / -vx on its straight line (scorer-side arithmetic from the scenario's launch state; a ball
    under gravity is treated the same way, as an upper bound on how long the approach lasts). Frames later than
    that minus skills.dodge_vision.pre_contact_margin_s are dropped: after the object has hit or passed, the image
    shows a bounce or nothing, and development run v2 had half of its incoming windows there. Scenarios without an
    object, or with one that does not approach, keep every eligible frame. If nothing is left (an arrival before
    the first full window), the earliest eligible frame is kept.
    """
    arrival = arrival_time_s(config, scenario)
    if arrival is None:
        return eligible
    latest = arrival - float(require(config, f"{_ROOT}.pre_contact_margin_s"))
    kept = eligible[frame_times(config)[eligible] <= latest + _TOLERANCE]
    return kept if kept.size else eligible[:1]


def _meta(config: Mapping[str, Any], split: str, frames: int, scenarios: list[dict[str, Any]], n_situations: int) -> dict[str, Any]:
    spec = build_observed_spec(config)
    kinds = Counter(s["kind"] for s in scenarios)
    return {
        "skill": SKILL,
        "split": split,
        "action_names": list(spec.action_names),
        "outcome_names": list(spec.outcome_names),
        "outcome_kinds": list(spec.outcome_kinds),
        "outcome_bounds": spec.outcome_bounds.tolist(),
        "frames": frames,
        "n_situations": n_situations,
        "seeds": {
            "runtime_seed": require(config, "runtime.seed"),
            "split_code": SPLIT_CODES[split],
            "scenarios": "np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code]))",
            "per_scenario": "np.random.SeedSequence([runtime_seed, split_code], spawn_key=(scenario_id,)).spawn(2): "
            "decision frame(s), augmentation",
            "attempt_seed": "scenario_id",
        },
        "world_seed": require(config, "observed.world_seed"),
        "world_preset": require(config, "observed.world_preset"),
        "config_digest": config_digest(config),
        "kind_counts": {kind: kinds.get(kind, 0) for kind in require(config, f"{_ROOT}.kind_weights")},
        # True when a punch with an object ran as the stand-in (the ball with its weight cancelled, _reset).
        "punch_stand_in": _punch_runs_as_stand_in() and any(
            s.get("object_kind") == "punch" and any(k in s["situation"] for k in _ball_keys()) for s in scenarios
        ),
        "scenarios": scenarios,
    }


def _write(out_dir: Path, stem: str, meta: Mapping[str, Any], **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(out_dir / f"{stem}.npz", buffer.getvalue())
    atomic_write_text(out_dir / f"{stem}.json", json.dumps(meta, indent=1, allow_nan=False) + "\n")


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return value


def collect_rows(config: Mapping[str, Any], split: str, out_dir: str | Path, n: int | None = None, frames: int | None = None) -> dict[str, Any]:
    """Measure the one-shot rows of a split and write <out_dir>/rows.npz and rows.json; returns the meta.

    n scenarios (default observed.dodge_vision.<split>_scenarios) from the split's seed, one decision frame each
    drawn uniformly among the eligible frames, every action run (attempt_seed = scenario_id) with one augmentation
    draw per scenario shared by its C rows. Raises RuntimeError when the C windows of a scenario differ.
    """
    if split not in ("context", "dev", "test"):
        raise ValueError(f"split must be context, dev or test; got {split!r}")
    n = _count("n", require(config, f"observed.dodge_vision.{split}_scenarios") if n is None else n)
    settings = VisionSettings.from_config(config, frames)
    ep = _episode(config)
    eligible = _eligible_frames(config, settings.frames)
    scenarios, seeds = _split_plan(config, split, n)
    tasks = []
    for scenario, seed in zip(scenarios, seeds):
        frame_seed, augment_seed = seed.spawn(2)
        own = _frames_before_arrival(config, scenario, eligible)
        frame = int(own[np.random.default_rng(frame_seed).integers(own.size)])
        tasks.extend((scenario, frame, action, augment_seed) for action in range(len(ep.actions.names)))
    c = len(ep.actions.names)
    print(f"rows: split {split}, {n} scenarios x {c} actions = {len(tasks)} episodes", flush=True)
    start = time.perf_counter()
    results = _run(config, settings.frames, _row_task, tasks)
    wall_s = time.perf_counter() - start

    scenario_index = np.array([r[0] for r in results], dtype=np.int64)
    action = np.array([r[1] for r in results], dtype=np.int64)
    features = np.stack([r[2] for r in results]).astype(np.float32, copy=False)
    measured = np.array([[r[3][name] for name in OUTCOME_NAMES] for r in results], dtype=np.float64)
    decision_frame = np.array([task[1] for task in tasks], dtype=np.int64)
    decision_time_s = frame_times(config)[decision_frame]
    worst_spread = 0.0
    for sid in range(n):
        rows = np.flatnonzero(scenario_index == sid)
        if sorted(action[rows].tolist()) != list(range(c)):
            raise RuntimeError(f"scenario {sid} has actions {sorted(action[rows].tolist())}, not all {c}")
        spread = float(np.abs(features[rows] - features[rows[0]]).max())
        worst_spread = max(worst_spread, spread)
        # The render is not always bit-identical between episodes (seen: 0.001-0.002 on a 0-1 grey scale, rarely);
        # every row of a situation gets the window of its first row, and a larger difference still stops the run.
        features[rows] = features[rows[0]]
        if spread > float(require(config, "skills.dodge_vision.window_repeat_tolerance")):
            raise RuntimeError(
                f"scenario {sid}: the {c} action episodes saw different windows before acting (max |difference| "
                f"{spread}); the episodes are not deterministic"
            )
    physics = physics_guess(config, [scenarios[i] for i in scenario_index], decision_time_s, action)
    meta = _meta(config, split, settings.frames, scenarios, n)
    _write(Path(out_dir), "rows", meta, scenario_index=scenario_index, decision_time_s=decision_time_s,
           action=action, features=features, physics=physics, measured=measured)

    kinds = [scenarios[i]["kind"] for i in scenario_index]
    no_op = action == ep.actions.no_op
    print("kind counts: " + ", ".join(f"{k} {v}" for k, v in meta["kind_counts"].items()))
    print("collisions under NO_OP: " + ", ".join(
        f"{k} {int(sum(measured[r, 1] for r in np.flatnonzero(no_op) if kinds[r] == k))}/{v}"
        for k, v in meta["kind_counts"].items() if v))
    print(f"wall time {wall_s:.1f} s for {len(tasks)} episodes: {wall_s / len(tasks):.3f} s per episode")
    print(f"wrote {Path(out_dir) / 'rows.npz'} and rows.json", flush=True)
    return meta


def collect_dream(
    config: Mapping[str, Any], out_dir: str | Path, n: int | None = None, render: str | None = None
) -> dict[str, Any]:
    """Render the dream situations and write <out_dir>/dream.npz and dream.json; returns the meta.

    n zero-command scenes (default observed.dodge_vision.dream_episodes) from the dream seed, each giving
    windows_per_dream_episode windows at distinct eligible decision frames (each window its own augmentation draw),
    and the physics guess of every action for each window.

    render (default skills.dodge_vision.dream_render): "physics" runs a zero-command episode per scene
    (render_windows); "kinematic" records the stand once (record_stand, written to <out_dir>/stand.npz and loaded by
    every worker) and draws each window before _kinematic_limit_s with render_windows_kinematic, the rest with
    render_windows. dream.json's render counts both.
    """
    render = require(config, f"{_ROOT}.dream_render") if render is None else render
    if render not in _DREAM_RENDERS:
        raise ValueError(f"render must be one of {list(_DREAM_RENDERS)}; got {render!r}")
    n = _count("n", require(config, "observed.dodge_vision.dream_episodes") if n is None else n)
    per_episode = _count(f"{_ROOT}.windows_per_dream_episode", require(config, f"{_ROOT}.windows_per_dream_episode"))
    settings = VisionSettings.from_config(config)
    c = len(_actions(config).names)
    eligible = _eligible_frames(config, settings.frames)
    if per_episode > eligible.size:
        raise ConfigError(f"'{_ROOT}.windows_per_dream_episode' {per_episode} exceeds the {eligible.size} eligible frames")
    scenarios, seeds = _split_plan(config, "dream", n)
    tasks = []
    for scenario, seed in zip(scenarios, seeds):
        frame_seed, augment_seed = seed.spawn(2)
        own = _frames_before_arrival(config, scenario, eligible)
        chosen = np.sort(np.random.default_rng(frame_seed).choice(own, size=per_episode, replace=bool(own.size < per_episode)))
        tasks.append((scenario, [int(f) for f in chosen], augment_seed.spawn(per_episode)))
    print(f"dream: {n} episodes x {per_episode} windows = {n * per_episode} situations, render {render}", flush=True)
    start = time.perf_counter()
    stand_path = None
    if render == "kinematic":
        stand_path = Path(out_dir) / "stand.npz"
        _save_stand(record_stand(_make_world(config), config), stand_path)
    results = _run(config, settings.frames, _dream_task, tasks, None if stand_path is None else str(stand_path))
    wall_s = time.perf_counter() - start

    scenario_index = np.repeat(np.array([r[0] for r in results], dtype=np.int64), per_episode)
    features = np.concatenate([r[1] for r in results]).astype(np.float32, copy=False)
    kinematic = np.concatenate([r[2] for r in results])
    decision_frame = np.array([f for task in tasks for f in task[1]], dtype=np.int64)
    decision_time_s = frame_times(config)[decision_frame]
    m = scenario_index.size
    physics = physics_guess(
        config, [scenarios[i] for i in np.repeat(scenario_index, c)], np.repeat(decision_time_s, c), np.tile(np.arange(c), m)
    ).reshape(m, c, len(OUTCOME_NAMES))
    meta = _meta(config, "dream", settings.frames, scenarios, m)
    meta["n_episodes"], meta["windows_per_dream_episode"] = n, per_episode
    meta["render"] = {
        "mode": render,
        "kinematic_windows": int(kinematic.sum()),
        "physics_windows": int(m - kinematic.sum()),
        "wall_s": round(wall_s, 3),
    }
    _write(Path(out_dir), "dream", meta, scenario_index=scenario_index, decision_time_s=decision_time_s,
           features=features, physics=physics, kinematic=kinematic)

    no_op = build_observed_spec(config).no_op_index
    kinds = [scenarios[i]["kind"] for i in scenario_index]
    print("kind counts (episodes): " + ", ".join(f"{k} {v}" for k, v in meta["kind_counts"].items()))
    print("physics-guess NO_OP collision p > 0.5 (windows): " + ", ".join(
        f"{k} {int(sum(physics[r, no_op, 1] > 0.5 for r in range(m) if kinds[r] == k))}/{v * per_episode}"
        for k, v in meta["kind_counts"].items() if v))
    print(f"render {render}: {meta['render']['kinematic_windows']} kinematic, {meta['render']['physics_windows']} physics windows")
    print(f"wall time {wall_s:.1f} s for {n} scenes ({m} windows): {wall_s / n:.3f} s per scene, {m / wall_s:.1f} windows/s")
    print(f"wrote {Path(out_dir) / 'dream.npz'} and dream.json", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (see the module docstring). Returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.skills.dodge_vision", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rows = commands.add_parser("rows", help="measured one-shot rows of a split")
    rows.add_argument("--split", required=True, choices=("context", "dev", "test"))
    rows.add_argument("--out", required=True, help="output directory")
    rows.add_argument("--n", type=int, help="scenarios (default observed.dodge_vision.<split>_scenarios)")
    rows.add_argument("--frames", type=int, help="window length (vision.frames or a vision.frame_variants member)")
    dream = commands.add_parser("dream", help="rendered dream situations")
    dream.add_argument("--out", required=True, help="output directory")
    dream.add_argument("--n", type=int, help="episodes (default observed.dodge_vision.dream_episodes)")
    dream.add_argument("--render", choices=_DREAM_RENDERS, help="default skills.dodge_vision.dream_render")
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "rows":
        collect_rows(config, args.split, args.out, args.n, args.frames)
    else:
        collect_dream(config, args.out, args.n, args.render)
    return 0


if __name__ == "__main__":
    # Through the importable module, so spawned workers find _worker_init and the tasks by their module name.
    from learned_judgment.skills import dodge_vision as _module

    raise SystemExit(_module.main())
