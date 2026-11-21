"""DODGE from the head camera (Person B): threat scenarios, the head view, episodes on the real Go2 world, the
physics guess, the objective and the collection command line.

Lane contract (Person B, dodge_vision)
--------------------------------------
The dog stands on the zero command; at t_s 0 (the launch) an object may move in the scene. A student sees only
image windows from the head camera and picks one of C commands; the scorer side (physics guess, objective) may use
the scenario's truth, which never becomes a student input. Shared arrays:

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
                seeds, world_seed, world_preset, config_digest, kind_counts, scenarios (index = scenario_index)
    dream.npz   scenario_index (m,) int64, decision_time_s (m,) float64, features (m, D) float32,
                physics (m, C, K) float64
    dream.json  the keys of rows.json plus n_episodes and windows_per_dream_episode

    python -m learned_judgment.skills.dodge_vision rows --split {context|dev|test} --out DIR [--n N] [--frames F]
    python -m learned_judgment.skills.dodge_vision dream --out DIR [--n N]

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
    "DodgeVisionSpec",
    "HeadView",
    "build_observed_spec",
    "closed_loop",
    "collect_dream",
    "collect_rows",
    "frame_times",
    "main",
    "objective",
    "observe_and_act",
    "physics_guess",
    "render_windows",
    "sample_threat_scenarios",
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
_PLANAR = ("vel_x", "vel_y")  # verified command fields of the dog's planar velocity (x ahead, y left)
_ROOT = "skills.dodge_vision"
_TOLERANCE = 1e-9  # s or m: times are sums of 0.02 s periods


# ----- config ----------------------------------------------------------------------------------------------------


def _number(key: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ConfigError(f"config.yaml key {key!r} must be a finite number; got {value!r}")
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


def sample_threat_scenarios(config: Mapping[str, Any], rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    """``n`` scenarios drawn from config.yaml skills.dodge_vision; no world needed.

    Each is {"scenario_id": i (0 .. n - 1), "kind": a key of kind_weights, "object_kind": "ball", "situation": a
    dict Go2World.reset accepts, "visual": {}}. The situation always has dog_forward_speed_m_s 0.0 and settle_s =
    skills.dodge_vision.settle_s; kind "empty" has no ball key, every other kind all six BALL_KEYS (Go2World's frame
    at the launch: origin at the trunk origin, x ahead, y left, z up; velocity over the ground). Values are floats.

    Draws, row by row in this order (so row i does not depend on n), each uniform over its scenarios range:
      kind                 by kind_weights
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

    scenarios: list[dict[str, Any]] = []
    for index in range(int(n)):
        kind = _pick(rng, kinds)
        ball: tuple[float, ...] | None = None
        if kind in _AIMS:
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
        scenarios.append({"scenario_id": index, "kind": kind, "object_kind": "ball", "situation": situation, "visual": {}})
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
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._eye, self._pitch_up_deg, self._fovy_deg = _camera(config)
        size = require(config, "vision.head_camera.render_size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ConfigError(f"config.yaml key 'vision.head_camera.render_size' must be an int >= 1; got {size!r}")
        self.size = size
        self._option = mujoco.MjvOption()
        self._option.geomgroup[2] = 0  # robot visual meshes
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.distance = 1.0  # only the direction from the eye to lookat matters
        self._model: mujoco.MjModel | None = None
        self._renderer: mujoco.Renderer | None = None
        self._scratch: mujoco.MjData | None = None
        self._base = -1

    def frame(self, world: Any) -> np.ndarray:
        """(S, S, 3) uint8 RGB of ``world``'s current state, S = render_size. When world.render_frame accepts a
        ``camera`` argument, world.render_frame(camera="head") is returned instead."""
        render_frame = getattr(world, "render_frame", None)
        if render_frame is not None and "camera" in inspect.signature(render_frame).parameters:
            return render_frame(camera="head")
        model, data = world.model, world.policy.data
        if model is not self._model:
            self.close()
            self._renderer = mujoco.Renderer(model, self.size, self.size)
            self._scratch = mujoco.MjData(model)
            self._base = model.body("base_link").id
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
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return self._renderer.render()

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


def _frames_at(world: Any, view: Any, settings: VisionSettings, periods: Sequence[int], ep: _Episode) -> list[np.ndarray]:
    """Preprocessed frames at the given ascending control periods after the launch (period 0 is the state reset
    returned), stepping the zero command in between; the world is left at the last of them."""
    zero = ep.actions.commands[ep.actions.no_op]
    frames, now = [], 0
    for period in periods:
        if period > now:
            world.step(zero, (period - now) * ep.timing.period_s)
            now = period
        frames.append(preprocess_frame(view.frame(world), settings))
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

    reset(scenario["situation"], attempt_seed); the zero command until the decision frame, rendering (``view``) and
    preprocessing the frames of its window; then the action's command for hold_s and the zero command until
    episode_duration_s; finish_episode(). Only the window's own frames are rendered: earlier frames would leave it.
    A hold that would end after episode_duration_s runs in full (frames in decision_time_range_s end in time).

    Args:
        world:          a Go2World; settings other than the window length come from world.config.
        view:           a HeadView (anything with frame(world) -> (H, W, 3) uint8).
        scenario:       from sample_threat_scenarios.
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
    world.reset(dict(scenario["situation"]), attempt_seed=int(attempt_seed))
    periods = ep.timing.frame_periods[frame - settings.frames + 1 : frame + 1]
    window = np.stack(_frames_at(world, view, settings, periods.tolist(), ep))
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
    world.reset(dict(scenario["situation"]), attempt_seed=int(attempt_seed))
    rendered = dict(zip(needed, _frames_at(world, view, settings, ep.timing.frame_periods[needed].tolist(), ep)))
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
    world.reset(dict(scenario["situation"]), attempt_seed=int(attempt_seed))
    window = FrameWindow(settings)
    command: dict[str, Any] = dict(ep.actions.commands[ep.actions.no_op])
    travel = 0.0
    trace: list[dict[str, Any]] = []
    issued: list[tuple[float, str]] = []
    for period in range(ep.timing.n_periods):
        if period in frame_periods:
            frame = preprocess_frame(view.frame(world), settings)
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
      * ball: from its launch state, ballistic under gravity_m_s2 until its centre reaches the floor rest height
        ball.radius_m - base_height_m, then on the floor keeping its horizontal velocity (a ball starting at or
        below that height keeps its height: a rolling start).
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
    situations = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping) or not isinstance(scenario.get("situation"), Mapping):
            raise ValueError(f"scenarios[{index}] must be a mapping with a 'situation' mapping")
        situations.append(scenario["situation"])
    has_ball = np.array([all(k in s for k in ball_keys) for s in situations], dtype=bool)
    closest = np.full(n, np.inf)
    rows = np.flatnonzero(has_ball)
    if rows.size:
        launch = np.array([[float(situations[i][k]) for k in ball_keys] for i in rows], dtype=np.float64)
        x0, y0, z0, vx, vy, vz = launch.T
        command = ep.actions.features[actions[rows]][:, planar]
        height = z0 - floor_z  # ball centre above its floor rest height
        landing = np.where(height > 0.0, (vz + np.sqrt(vz * vz + 2.0 * gravity * np.maximum(height, 0.0))) / gravity, 0.0)
        start = times[rows] + latency  # the dog's velocity starts to change
        stop = start + ep.hold_s  # ... and starts to decay back to 0
        near = np.full(rows.size, np.inf)
        for t in _time_grid(step_s, duration):
            flight = np.minimum(t, landing)
            ball_x, ball_y = x0 + vx * t, y0 + vy * t
            ball_z = z0 + vz * flight - 0.5 * gravity * flight * flight
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

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        """(n,) objective(config, predicted)."""
        return objective(self.config, predicted)


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


def _worker_init(config: Mapping[str, Any], frames: int) -> None:
    """Per process: its own Go2World (observed.world_seed, observed.world_preset; no trace files) and HeadView."""
    from learned_judgment.env import Go2World  # on use: see _ball_keys

    config = copy.deepcopy(dict(config))
    config["world"]["save_traces"] = False
    world = Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    _WORKER.update(world=world, view=HeadView(config), settings=VisionSettings.from_config(config, frames))


def _row_task(task: tuple) -> tuple[int, int, np.ndarray, dict[str, float]]:
    scenario, frame, action, augment = task
    w = _WORKER
    features, measured = observe_and_act(
        w["world"], w["view"], scenario, scenario["scenario_id"], frame, action, w["settings"], np.random.default_rng(augment)
    )
    return scenario["scenario_id"], action, features, measured


def _dream_task(task: tuple) -> tuple[int, np.ndarray]:
    scenario, frames, augments = task
    w = _WORKER
    rngs = [np.random.default_rng(seed) for seed in augments]
    windows = render_windows(w["world"], w["view"], scenario, scenario["scenario_id"], frames, w["settings"], rngs)
    return scenario["scenario_id"], windows


def _run(config: Mapping[str, Any], frames: int, function: Callable[[tuple], Any], tasks: list[tuple]) -> list[Any]:
    """function over tasks in observed.workers spawned processes, results in task order, sparse progress lines."""
    workers = require(config, "observed.workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"config.yaml key 'observed.workers' must be an int >= 1; got {workers!r}")
    every = max(1, len(tasks) // 10)
    results = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(workers, len(tasks)), initializer=_worker_init, initargs=(config, frames)) as pool:
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
        frame = int(eligible[np.random.default_rng(frame_seed).integers(eligible.size)])
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
    for sid in range(n):
        rows = np.flatnonzero(scenario_index == sid)
        if sorted(action[rows].tolist()) != list(range(c)):
            raise RuntimeError(f"scenario {sid} has actions {sorted(action[rows].tolist())}, not all {c}")
        spread = float(np.abs(features[rows] - features[rows[0]]).max())
        if spread != 0.0:
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


def collect_dream(config: Mapping[str, Any], out_dir: str | Path, n: int | None = None) -> dict[str, Any]:
    """Render the dream situations and write <out_dir>/dream.npz and dream.json; returns the meta.

    n zero-command episodes (default observed.dodge_vision.dream_episodes) from the dream seed, each giving
    windows_per_dream_episode windows at distinct eligible decision frames (each window its own augmentation draw),
    and the physics guess of every action for each window.
    """
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
        chosen = np.sort(np.random.default_rng(frame_seed).choice(eligible, size=per_episode, replace=False))
        tasks.append((scenario, [int(f) for f in chosen], augment_seed.spawn(per_episode)))
    print(f"dream: {n} episodes x {per_episode} windows = {n * per_episode} situations", flush=True)
    start = time.perf_counter()
    results = _run(config, settings.frames, _dream_task, tasks)
    wall_s = time.perf_counter() - start

    scenario_index = np.repeat(np.array([r[0] for r in results], dtype=np.int64), per_episode)
    features = np.concatenate([r[1] for r in results]).astype(np.float32, copy=False)
    decision_frame = np.array([f for task in tasks for f in task[1]], dtype=np.int64)
    decision_time_s = frame_times(config)[decision_frame]
    m = scenario_index.size
    physics = physics_guess(
        config, [scenarios[i] for i in np.repeat(scenario_index, c)], np.repeat(decision_time_s, c), np.tile(np.arange(c), m)
    ).reshape(m, c, len(OUTCOME_NAMES))
    meta = _meta(config, "dream", settings.frames, scenarios, m)
    meta["n_episodes"], meta["windows_per_dream_episode"] = n, per_episode
    _write(Path(out_dir), "dream", meta, scenario_index=scenario_index, decision_time_s=decision_time_s,
           features=features, physics=physics)

    no_op = build_observed_spec(config).no_op_index
    kinds = [scenarios[i]["kind"] for i in scenario_index]
    print("kind counts (episodes): " + ", ".join(f"{k} {v}" for k, v in meta["kind_counts"].items()))
    print("physics-guess NO_OP collision p > 0.5 (windows): " + ", ".join(
        f"{k} {int(sum(physics[r, no_op, 1] > 0.5 for r in range(m) if kinds[r] == k))}/{v * per_episode}"
        for k, v in meta["kind_counts"].items() if v))
    print(f"wall time {wall_s:.1f} s for {n} episodes: {wall_s / n:.3f} s per episode")
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
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "rows":
        collect_rows(config, args.split, args.out, args.n, args.frames)
    else:
        collect_dream(config, args.out, args.n)
    return 0


if __name__ == "__main__":
    # Through the importable module, so spawned workers find _worker_init and the tasks by their module name.
    from learned_judgment.skills import dodge_vision as _module

    raise SystemExit(_module.main())
