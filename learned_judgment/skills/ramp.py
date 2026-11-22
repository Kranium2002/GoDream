"""RAMP from the trunk's pitch (Person B): the slope detector, episodes on the real Go2 world, the physics guess, the
objective and the collection / evaluation command line.

Lane contract (Person B, ramp)
------------------------------
The walking policy is frozen. On flat ground the nominal walking command (vel_x = the scenario's nominal speed,
vel_y 0, yaw_rate 0) passes through UNCHANGED. Only the trunk pitch read from base_quat_wxyz (what an IMU gives) may
trigger the learned correction: a SlopeDetector watches the pitch every control period; on its first NORMAL -> SLOPE
switch the nominal command is kept for skills.ramp.observe_s more, then the policy is called ONCE with the four
observed features and returns a candidate index; that candidate's command is sent while the detector stays in SLOPE
(a later re-entry reuses it) and the nominal command is sent again in NORMAL. The incline angle, the ramp's position
and the scenario kind are hidden truth: they shape the scenario and the scoring, never a policy input.

    PITCH       nose_up_deg(q) = degrees(asin(clip(2 (x z - w y), -1, 1))) for q = (w, x, y, z), body to world: the
                world-z component of the trunk's +x axis (row 2, column 0 of the rotation matrix), so nose up is
                positive whatever the heading.
    CANDIDATES  C = len(climb_speeds_m_s) x len(climb_yaw_rates_rad_s) of skills.ramp.candidates, speed-major
                (index = i_speed * n_yaw + i_yaw); command {vel_x: climb speed, vel_y: 0, yaw_rate}, each checked
                against the verified command schema. No zero command, no threat head: no_op_index is None.
    FEATURES    D = 4 float32, FEATURE_NAMES order, each mapped to [0, 1] by skills.ramp.feature_ranges and clipped:
                  filtered_pitch_deg  the detector's filtered pitch at the decision
                  raw_pitch_deg       the pitch of the decision period's observation
                  pitch_rate_deg_s    (filtered pitch at the decision - filtered pitch at the entry) / observe_s
                  nominal_speed_m_s   the nominal walking speed
    OUTCOMES    K = 5 in OUTCOME_NAMES order (finish_episode keys of Go2World):
                  progress_m   regression  final_forward_m
                  reached_top  binary      final_forward_m >= ramp_end_m and no fall (0 without a reported ramp)
                  fall         binary      fall
                  sideways_m   regression  |final_sideways_m|
                  slip_m       regression  foot_slip_m; 0.0 when the world does not report it (rows.json "slip")
    TIME        whole control periods of robot.walking_policy.command_schema.update_period_s from the decision
                instant of Go2World.reset (t_s 0, after the world's settle trot at the nominal speed).
    SCORING     ramp_start_m, ramp_end_m and foot_slip_m of finish_episode are read after the episode only. The trunk's
                distance along the initial heading comes from the observation of every control period
                (Go2World.step's return value), not from a trace file.

World contract (Person A's traversable incline): the situation key "incline_deg" (absent or 0 = flat) builds a ramp
then a level plateau ahead of the dog; finish_episode adds ramp_start_m / ramp_end_m for it. A Go2World whose
SITUATION_KEYS lacks "incline_deg" makes every slope scenario raise RuntimeError before any episode runs.

Files of the command line, one directory per call (the observed.py array contract, features (n, D = 4)):

    rows.npz    scenario_index (n,) int64, decision_time_s (n,) float64, action (n,) int64,
                features (n, D) float32, physics (n, K) float64, measured (n, K) float64
    rows.json   skill, split, action_names, outcome_names, outcome_kinds, outcome_bounds, feature_names,
                feature_ranges, frames (null), n_situations (scenarios with rows), n_scenarios, seeds, world_seed,
                world_preset, config_digest, kind_counts, detection, slip, scenarios (index = scenario_index; each
                with its detection record)
    dream.npz   scenario_index (m,) int64, decision_time_s (m,) float64, features (m, D) float32,
                physics (m, C, K) float64
    dream.json  the keys of rows.json plus sampling (mode: skills.ramp.dream_sampling; recorded adds the rows
                directory, the number of scenarios drawn from and the noise)
    run.json    records (per scenario: the nominal pass-through and the student) and summary per method

    python -m learned_judgment.skills.ramp rows --split {context|dev|test} --out DIR [--n N]
    python -m learned_judgment.skills.ramp attempts --out DIR [--n N]
    python -m learned_judgment.skills.ramp dream --out DIR [--n N] [--from-rows DIR]

dream (no simulator), by skills.ramp.dream_sampling: uniform draws each feature independently over its range;
recorded draws the measured feature vectors of the context rows in --from-rows (one per scenario, with its decision
time) plus noise with the measured vectors' covariance times skills.ramp.dream_noise squared (recorded_situations).

attempts (the fast run's context, MISSION.md's 10-40 attempts) runs ONE episode per context scenario, N default
observed.ramp.fast_attempts: with probability engine.attempt_exploration_fraction a uniform candidate, else the physics
rule at the decision (collect_attempts); a scenario without a decision gives no row. rows.json adds "collection":
"attempts", n_attempts, exploration_fraction, attempt_sources and per scenario "attempt".
    python -m learned_judgment.skills.ramp run --student DIR --split {context|dev|test} --out DIR [--n N] [--run RUN]

run also appends "evaluate" events to <observed.runs_dir>/<RUN>/events.jsonl and merges into its result.json the UI
keys run_id, skill, seeds, scenario_ids, metrics, paired (methods "no_op" = the nominal pass-through, "snn" = the
student) and a section run_<split>; RUN defaults to the student's provenance.run_id.

Seeds. The scenarios of a split come from np.random.default_rng(np.random.SeedSequence([runtime.seed, code]))
(code: SPLIT_CODES); scenario i's episodes use attempt_seed i, so every candidate and both run methods face the
same noise. Rows: candidate 0 runs first for every scenario; the other C - 1 run only where it reached a decision.

This module defines no SPEC_CLASS, so the SkillSpec registry (learned_judgment.skills) does not load it.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import multiprocessing
import numbers
import sys
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from learned_judgment.config import ConfigError, config_digest, load_config, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.fileio import atomic_write_bytes, atomic_write_text

__all__ = [
    "FEATURE_NAMES",
    "INCLINE_KEY",
    "KINDS",
    "NORMAL",
    "OUTCOME_KINDS",
    "OUTCOME_NAMES",
    "SKILL",
    "SLOPE",
    "SPLIT_CODES",
    "RampSpec",
    "SlopeDetector",
    "build_observed_spec",
    "collect_attempts",
    "collect_dream",
    "collect_rows",
    "features_in_units",
    "main",
    "nominal_command",
    "nominal_index",
    "normalise_features",
    "nose_up_deg",
    "objective",
    "physics_choice",
    "physics_guess",
    "recorded_situations",
    "run_episode",
    "run_student",
    "sample_ramp_scenarios",
]

SKILL = "ramp"
NORMAL, SLOPE = "NORMAL", "SLOPE"
INCLINE_KEY = "incline_deg"
KINDS = ("flat", "slope")
FEATURE_NAMES = ("filtered_pitch_deg", "raw_pitch_deg", "pitch_rate_deg_s", "nominal_speed_m_s")
OUTCOME_NAMES = ("progress_m", "reached_top", "fall", "sideways_m", "slip_m")
OUTCOME_KINDS = ("regression", "binary", "binary", "regression", "regression")
SPLIT_CODES = {"context": 1, "dev": 2, "test": 3, "dream": 4}
_FIELDS = ("vel_x", "vel_y", "yaw_rate")  # the verified command fields by native index
_OBJECTIVE_KEYS = ("progress", "reached_top", "fall", "sideways", "slip", "spread")
_ROOT = "skills.ramp"
_OBSERVED = "observed.ramp"
_TOLERANCE = 1e-9  # s or m: times are sums of 0.02 s periods
_SCORING_SOURCE = "per-period observations (Go2World.step's observation of every control period)"


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


def _non_negative(config: Mapping[str, Any], key: str) -> float:
    number = _value(config, key)
    if number < 0.0:
        raise ConfigError(f"config.yaml key {key!r} must be >= 0; got {number}")
    return number


def _pair(config: Mapping[str, Any], key: str, minimum: float | None = None, strict: bool = False) -> tuple[float, float]:
    """[low, high] of finite numbers, low <= high (low < high when strict; low >= minimum when given)."""
    pair = require(config, key)
    if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
        raise ConfigError(f"config.yaml key {key!r} must be [low, high]; got {pair!r}")
    low, high = _number(key, pair[0]), _number(key, pair[1])
    if low > high or (strict and low == high) or (minimum is not None and low < minimum):
        rule = "low < high" if strict else "low <= high"
        raise ConfigError(f"config.yaml key {key!r} must have {rule}" + (f" and low >= {minimum}" if minimum is not None
                                                                          else "") + f"; got {list(pair)}")
    return low, high


def _numbers(config: Mapping[str, Any], key: str) -> tuple[float, ...]:
    values = require(config, key)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ConfigError(f"config.yaml key {key!r} must be a non-empty list of numbers; got {values!r}")
    out = tuple(_number(key, value) for value in values)
    if len(set(out)) != len(out):
        raise ConfigError(f"config.yaml key {key!r} repeats a value: {list(out)}")
    return out


def _whole_periods(key: str, seconds: float, period_s: float) -> int:
    count = round(seconds / period_s)
    if count < 1 or abs(count * period_s - seconds) > _TOLERANCE:
        raise ConfigError(f"config.yaml key {key!r} ({seconds} s) must be a whole number >= 1 of {period_s} s periods")
    return count


@dataclass(frozen=True)
class _Candidates:
    names: tuple[str, ...]
    commands: tuple[dict[str, float], ...]
    features: np.ndarray  # (C, 3) float64, read-only, _FIELDS order


def _candidates(config: Mapping[str, Any]) -> _Candidates:
    """skills.ramp.candidates, speed-major, each checked against the verified command schema."""
    from learned_judgment.env.commands import validate_command  # on use: learned_judgment.env loads the policy runtime

    schema = require(config, "robot.walking_policy.command_schema")
    fields = tuple(sorted(schema["fields"], key=lambda name: schema["fields"][name]["native_index"]))
    if fields != _FIELDS:
        raise ConfigError(f"the ramp candidates need the command fields {list(_FIELDS)}; the schema has {list(fields)}")
    speeds = _numbers(config, f"{_ROOT}.candidates.climb_speeds_m_s")
    yaw_rates = _numbers(config, f"{_ROOT}.candidates.climb_yaw_rates_rad_s")
    names, commands = [], []
    for speed in speeds:
        for yaw_rate in yaw_rates:
            command = {"vel_x": speed, "vel_y": 0.0, "yaw_rate": yaw_rate}
            try:
                validate_command(dict(command), schema)
            except ValueError as error:
                raise ConfigError(f"candidate {command} of '{_ROOT}.candidates' is not a verified command: {error}") from error
            names.append(f"vx{speed:+.2f}_yaw{yaw_rate:+.2f}")
            commands.append(command)
    if len(set(names)) != len(names):
        raise ConfigError(f"'{_ROOT}.candidates' give candidate names that collide at 2 decimals: {names}")
    features = np.array([[command[f] for f in _FIELDS] for command in commands], dtype=np.float64)
    features.setflags(write=False)
    return _Candidates(tuple(names), tuple(commands), features)


def _feature_ranges(config: Mapping[str, Any]) -> np.ndarray:
    """(D, 2) [low, high] per feature in FEATURE_NAMES order (skills.ramp.feature_ranges)."""
    key = f"{_ROOT}.feature_ranges"
    ranges = require(config, key)
    if not isinstance(ranges, Mapping) or set(ranges) != set(FEATURE_NAMES):
        raise ConfigError(f"config.yaml key {key!r} must map exactly {list(FEATURE_NAMES)} to [low, high]; got {ranges!r}")
    return np.array([_pair(config, f"{key}.{name}", strict=True) for name in FEATURE_NAMES], dtype=np.float64)


def _outcome_bounds(config: Mapping[str, Any]) -> np.ndarray:
    key = f"{_ROOT}.outcome_bounds"
    bounds = np.array(
        [
            _pair(config, f"{key}.progress_m", strict=True),
            (0.0, 1.0),
            (0.0, 1.0),
            _pair(config, f"{key}.sideways_m", minimum=0.0, strict=True),
            _pair(config, f"{key}.slip_m", minimum=0.0, strict=True),
        ],
        dtype=np.float64,
    )
    bounds.setflags(write=False)
    return bounds


@dataclass(frozen=True)
class _Timing:
    period_s: float
    n_periods: int  # control periods per episode after the decision instant
    observe_s: float


def _timing(config: Mapping[str, Any]) -> _Timing:
    period_s = _positive(config, "robot.walking_policy.command_schema.update_period_s")
    duration = _positive(config, f"{_ROOT}.episode_duration_s")
    n_periods = _whole_periods(f"{_ROOT}.episode_duration_s", duration, period_s)
    observe_s = _positive(config, f"{_ROOT}.observe_s")
    _whole_periods(f"{_ROOT}.observe_s", observe_s, period_s)
    return _Timing(period_s, n_periods, observe_s)


def _objective_weights(config: Mapping[str, Any]) -> dict[str, float]:
    key = f"{_ROOT}.objective_weights"
    weights = require(config, key)
    if not isinstance(weights, Mapping) or set(weights) != set(_OBJECTIVE_KEYS):
        raise ConfigError(f"config.yaml key {key!r} must map exactly {list(_OBJECTIVE_KEYS)} to weights; got {weights!r}")
    return {name: _non_negative(config, f"{key}.{name}") for name in _OBJECTIVE_KEYS}


# ----- pitch, detector, features -----------------------------------------------------------------------------------


def nose_up_deg(quat_wxyz: Sequence[float]) -> float:
    """Trunk pitch in degrees, nose up positive: degrees(asin(clip(2 (x z - w y), -1, 1))) of the body-to-world
    quaternion (w, x, y, z), the world-z component of the trunk's +x axis. Independent of the heading."""
    w, x, y, z = (float(value) for value in quat_wxyz)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y)))))


def _tilt_deg(quat_wxyz: Sequence[float]) -> float:
    """Angle between trunk z and world z, degrees (Go2World's recovery and fall tilt)."""
    _, x, y, _ = (float(value) for value in quat_wxyz)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _yaw(quat_wxyz: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SlopeDetector:
    """NORMAL / SLOPE from the trunk pitch, one sample per control period (pure logic, no I/O).

    filtered = mean of the samples whose time lies in (t_s - window_s, t_s]. NORMAL -> SLOPE once filtered >= on_deg
    at every sample for hold_s (from the first such sample to the current one); SLOPE -> NORMAL once filtered <= off_deg
    for hold_s. hold_s 0 switches on the first such sample. Attributes: mode, filtered (None before the first
    sample), entered_at / exited_at (t_s of the latest switch into / out of SLOPE, None before one), changes
    [(t_s, mode)] of every switch.
    """

    def __init__(self, window_s: float, on_deg: float, off_deg: float, hold_s: float) -> None:
        values = {"window_s": window_s, "on_deg": on_deg, "off_deg": off_deg, "hold_s": hold_s}
        for name, value in values.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number; got {value!r}")
        if window_s <= 0.0 or hold_s < 0.0 or off_deg >= on_deg:
            raise ValueError(f"the detector needs window_s > 0, hold_s >= 0 and off_deg < on_deg; got {values}")
        self.window_s, self.on_deg, self.off_deg, self.hold_s = float(window_s), float(on_deg), float(off_deg), float(hold_s)
        self.reset()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SlopeDetector":
        """skills.ramp.detector.{window_s, on_deg, off_deg, hold_s}."""
        key = f"{_ROOT}.detector"
        values = (_positive(config, f"{key}.window_s"), _value(config, f"{key}.on_deg"),
                  _value(config, f"{key}.off_deg"), _non_negative(config, f"{key}.hold_s"))
        try:
            return cls(*values)
        except ValueError as error:
            raise ConfigError(f"config.yaml key {key!r}: {error}") from error

    def reset(self) -> None:
        self.mode = NORMAL
        self.filtered: float | None = None
        self.entered_at: float | None = None
        self.exited_at: float | None = None
        self.changes: list[tuple[float, str]] = []
        self._samples: deque[tuple[float, float]] = deque()
        self._since: float | None = None  # first sample of the current run meeting the switch condition
        self._last_t: float | None = None

    def update(self, nose_up_deg: float, t_s: float) -> str:
        """Add one sample; return the mode after it. ValueError for a non-finite value or a time going back."""
        pitch, t = float(nose_up_deg), float(t_s)
        if not (math.isfinite(pitch) and math.isfinite(t)):
            raise ValueError(f"pitch and time must be finite; got {nose_up_deg!r}, {t_s!r}")
        if self._last_t is not None and t < self._last_t:
            raise ValueError(f"t_s went back from {self._last_t} to {t}")
        self._last_t = t
        self._samples.append((t, pitch))
        while self._samples[0][0] <= t - self.window_s + _TOLERANCE:
            self._samples.popleft()
        self.filtered = sum(value for _, value in self._samples) / len(self._samples)
        met = self.filtered >= self.on_deg if self.mode == NORMAL else self.filtered <= self.off_deg
        if not met:
            self._since = None
            return self.mode
        if self._since is None:
            self._since = t
        if t - self._since >= self.hold_s - _TOLERANCE:
            self.mode = SLOPE if self.mode == NORMAL else NORMAL
            if self.mode == SLOPE:
                self.entered_at = t
            else:
                self.exited_at = t
            self.changes.append((t, self.mode))
            self._since = None
        return self.mode


def normalise_features(config: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    """(..., D) values in units (FEATURE_NAMES order) -> float32 in [0, 1] by skills.ramp.feature_ranges, clipped."""
    ranges = _feature_ranges(config)
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (len(FEATURE_NAMES),) or not np.all(np.isfinite(array)):
        raise ValueError(f"values must be finite with a last axis of {len(FEATURE_NAMES)}; got {array.shape}")
    return np.clip((array - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]), 0.0, 1.0).astype(np.float32)


def features_in_units(config: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """(..., D) features in [0, 1] -> float64 values in units (the inverse map of normalise_features)."""
    ranges = _feature_ranges(config)
    array = np.asarray(features, dtype=np.float64)
    if array.shape[-1:] != (len(FEATURE_NAMES),) or not np.all(np.isfinite(array)):
        raise ValueError(f"features must be finite with a last axis of {len(FEATURE_NAMES)}; got {array.shape}")
    return ranges[:, 0] + array * (ranges[:, 1] - ranges[:, 0])


def nominal_command(speed_m_s: float) -> dict[str, float]:
    """The nominal walking command: {vel_x: speed, vel_y: 0, yaw_rate: 0}."""
    return {"vel_x": float(speed_m_s), "vel_y": 0.0, "yaw_rate": 0.0}


def nominal_index(config: Mapping[str, Any], speed_m_s: float) -> int | None:
    """Index of the candidate equal to the nominal command at ``speed_m_s`` (yaw 0), or None when none is."""
    for index, command in enumerate(_candidates(config).commands):
        if abs(command["vel_x"] - float(speed_m_s)) <= _TOLERANCE and command["yaw_rate"] == 0.0:
            return index
    return None


# ----- scenarios -------------------------------------------------------------------------------------------------


def sample_ramp_scenarios(config: Mapping[str, Any], rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    """``n`` scenarios from skills.ramp.scenarios; no world needed.

    Each is {"scenario_id": i, "kind": "flat" | "slope", "situation": {"incline_deg" (slope only),
    "dog_forward_speed_m_s"}}. Per row, in this order: kind (flat when rng.random() < flat_share), an incline uniform
    in incline_deg (drawn for every row, kept for slope rows), a nominal speed uniform in nominal_speed_m_s.
    Raises TypeError for a non-Generator rng, ValueError for a bad n, ConfigError for a missing or invalid key.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator; got {type(rng).__name__}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be an int >= 0; got {n!r}")
    key = f"{_ROOT}.scenarios"
    flat_share = _value(config, f"{key}.flat_share")
    if not 0.0 <= flat_share <= 1.0:
        raise ConfigError(f"config.yaml key '{key}.flat_share' must lie in [0, 1]; got {flat_share}")
    incline = _pair(config, f"{key}.incline_deg", minimum=_TOLERANCE)
    speed = _pair(config, f"{key}.nominal_speed_m_s", minimum=_TOLERANCE)
    scenarios = []
    for index in range(int(n)):
        flat = bool(rng.random() < flat_share)
        angle = float(rng.uniform(*incline))
        situation: dict[str, float] = {} if flat else {INCLINE_KEY: angle}
        situation["dog_forward_speed_m_s"] = float(rng.uniform(*speed))
        scenarios.append({"scenario_id": index, "kind": "flat" if flat else "slope", "situation": situation})
    return scenarios


# ----- episodes --------------------------------------------------------------------------------------------------


def _require_incline(world: Any | None = None) -> None:
    """RuntimeError unless the Go2World (module) accepts the situation key incline_deg."""
    if world is None:
        from learned_judgment.env import world as module  # on use: see _candidates
    else:
        module = sys.modules.get(type(world).__module__)
    keys = getattr(module, "SITUATION_KEYS", None)
    if keys is not None and INCLINE_KEY not in keys:
        raise RuntimeError(
            f"Go2World in this checkout does not accept the situation key {INCLINE_KEY!r} (Person A's traversable "
            f"incline is not merged; it accepts {list(keys)}): slope scenarios cannot run"
        )


def _ramp_marks(outcome: Mapping[str, Any], slope: bool) -> tuple[float | None, float | None]:
    start, end = outcome.get("ramp_start_m"), outcome.get("ramp_end_m")
    if slope and (start is None or end is None):
        raise RuntimeError(
            "finish_episode reported no ramp_start_m / ramp_end_m for a slope scenario; the incline contract needs both "
            f"(got {start!r}, {end!r})"
        )
    if start is None or end is None:
        return None, None
    return float(start), float(end)


def _measured(outcome: Mapping[str, Any], ramp_end: float | None) -> tuple[dict[str, float], bool]:
    """The five outcomes by name and whether the world reported foot_slip_m."""
    if outcome["sim_unstable"]:
        raise RuntimeError("MuJoCo flagged a bad state in this episode (sim_unstable 1); its outcome is no measurement")
    progress, fall = float(outcome["final_forward_m"]), float(outcome["fall"])
    slip = outcome.get("foot_slip_m")
    measured = {
        "progress_m": progress,
        "reached_top": float(ramp_end is not None and progress >= ramp_end and not fall),
        "fall": fall,
        "sideways_m": abs(float(outcome["final_sideways_m"])),
        "slip_m": 0.0 if slip is None else float(slip),
    }
    return measured, slip is not None


def _first_time(t_s: np.ndarray, mask: np.ndarray) -> float | None:
    hits = np.flatnonzero(mask)
    return None if hits.size == 0 else float(t_s[hits[0]])


def _recovery_after(t_s: np.ndarray, within: np.ndarray, start_s: float, hold_s: float) -> float | None:
    """Time from start_s to the first sample at or after it from which ``within`` holds for hold_s; None if never."""
    run_start = None
    for t, ok in zip(t_s, within):
        if t < start_s - _TOLERANCE:
            continue
        if not ok:
            run_start = None
            continue
        run_start = t if run_start is None else run_start
        if t - run_start >= hold_s - _TOLERANCE:
            return float(run_start - start_s)
    return None


def run_episode(
    world: Any, scenario: Mapping[str, Any], attempt_seed: int, policy: Callable[[np.ndarray], int] | None
) -> dict[str, Any]:
    """One complete episode on the real world with the detector in the loop (see the module docstring).

    Args:
        world:        a Go2World (settings from world.config).
        scenario:     from sample_ramp_scenarios.
        attempt_seed: int >= 0, passed to world.reset.
        policy:       policy(features (D,) float32 in [0, 1]) -> candidate index, called once at the decision; None
                      is the nominal pass-through (the detector still runs and the features are still built).

    Returns:
        {"scenario_id", "kind", "features" (D,) float32 or None (no decision reached), "decision_time_s" (None
        likewise), "action" (int or None), "measured" {outcome name: float}, "slip_reported" bool,
        "modes" [(t_s, mode)] switches, "entered" bool, "entry_time_s", "entry_latency_s" (entry time minus the time
        the trunk origin crossed ramp_start_m; None when either is unknown), "commands_changed_on_flat" (control
        periods sending a non-nominal command while the trunk origin, at the period's start, was not within
        [ramp_start_m, ramp_end_m]; every such period without a reported ramp), "scoring_source",
        "ramp_start_m", "ramp_end_m", "crossed_ramp_start_s", "reached_ramp_end_s", "recovery_s" (from reaching
        ramp_end_m until trunk tilt and tilt rate hold within robot.recovery for its hold_s; None if not reached or
        never held), "elapsed_s", "final_yaw_change_rad", "final_speed_m_s", "trace" {"t_s", "forward_m",
        "nose_up_deg", "filtered_deg", "tilt_deg", "tilt_rate_rad_s"} per sample (the reset's and one per period)
        and "sent_nominal" per period}.

    Raises:
        TypeError / ValueError for a bad argument or policy output; RuntimeError for a slope scenario the world cannot
        build, a missing ramp report, or a state MuJoCo flagged as bad; ConfigError for a missing key.
    """
    config = world.config
    timing, candidates = _timing(config), _candidates(config)
    detector = SlopeDetector.from_config(config)
    _feature_ranges(config)
    if not isinstance(scenario, Mapping) or not isinstance(scenario.get("situation"), Mapping):
        raise ValueError(f"scenario must be a mapping with a 'situation' mapping; got {scenario!r}")
    if isinstance(attempt_seed, bool) or not isinstance(attempt_seed, (int, np.integer)) or attempt_seed < 0:
        raise ValueError(f"attempt_seed must be an int >= 0; got {attempt_seed!r}")
    if policy is not None and not callable(policy):
        raise TypeError(f"policy must be None or callable; got {type(policy).__name__}")
    if abs(world.policy.control_dt - timing.period_s) > 1e-12:
        raise ValueError(f"the world's control period {world.policy.control_dt} s is not update_period_s {timing.period_s} s")
    situation = dict(scenario["situation"])
    speed = situation.get("dog_forward_speed_m_s")
    if isinstance(speed, bool) or not isinstance(speed, numbers.Real) or not math.isfinite(speed):
        raise ValueError(f"the situation needs a finite dog_forward_speed_m_s (the nominal speed); got {speed!r}")
    slope = float(situation.get(INCLINE_KEY, 0.0)) != 0.0
    if INCLINE_KEY in situation:
        _require_incline(world)
    nominal = nominal_command(float(speed))
    c = len(candidates.names)

    obs = world.reset(situation, attempt_seed=int(attempt_seed))
    origin = np.asarray(obs["base_pos_m"][:2], dtype=np.float64)
    yaw0 = _yaw(obs["base_quat_wxyz"])
    heading = np.array([math.cos(yaw0), math.sin(yaw0)])
    trace: dict[str, list[float]] = {k: [] for k in ("t_s", "forward_m", "nose_up_deg", "filtered_deg", "tilt_deg",
                                                    "tilt_rate_rad_s")}
    state: dict[str, Any] = {"entry": None, "entry_filtered": None, "decided": False, "features": None,
                             "decision_t": None, "action": None}

    def observe(observation: Mapping[str, Any]) -> str:
        t = float(observation["t_s"])
        quat = observation["base_quat_wxyz"]
        pitch = nose_up_deg(quat)
        before = detector.mode
        mode = detector.update(pitch, t)
        ang_vel = observation["base_ang_vel_rad_s"]
        trace["t_s"].append(t)
        trace["forward_m"].append(float((np.asarray(observation["base_pos_m"][:2]) - origin) @ heading))
        trace["nose_up_deg"].append(pitch)
        trace["filtered_deg"].append(float(detector.filtered))
        trace["tilt_deg"].append(_tilt_deg(quat))
        trace["tilt_rate_rad_s"].append(float(math.hypot(ang_vel[0], ang_vel[1])))
        if mode == SLOPE and before == NORMAL and state["entry"] is None:
            state["entry"], state["entry_filtered"] = t, float(detector.filtered)
        if state["entry"] is not None and not state["decided"] and t >= state["entry"] + timing.observe_s - _TOLERANCE:
            rate = (float(detector.filtered) - state["entry_filtered"]) / timing.observe_s
            features = normalise_features(config, [float(detector.filtered), pitch, rate, float(speed)])
            state.update(decided=True, features=features, decision_t=t)
            if policy is not None:
                # A policy with takes_decision_time (the attempts' physics rule) also gets the decision time.
                takes_time = getattr(policy, "takes_decision_time", False) is True
                action = policy(features.copy(), t) if takes_time else policy(features.copy())
                if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < c:
                    raise ValueError(f"policy must return a candidate index in [0, {c}); got {action!r}")
                state["action"] = int(action)
        return mode

    mode = observe(obs)
    sent_nominal: list[bool] = []
    sent_at: list[float] = []
    for _ in range(timing.n_periods):
        active = state["action"] is not None and mode == SLOPE
        command = candidates.commands[state["action"]] if active else nominal
        sent_nominal.append(command == nominal)
        sent_at.append(trace["forward_m"][-1])
        obs = world.step(command, timing.period_s)
        mode = observe(obs)
    outcome = world.finish_episode()

    ramp_start, ramp_end = _ramp_marks(outcome, slope)
    measured, slip_reported = _measured(outcome, ramp_end)
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in trace.items()}
    changed = ~np.asarray(sent_nominal, dtype=bool)
    if ramp_start is None:
        crossed_start = reached_end = recovery_s = None
        changed_on_flat = int(changed.sum())
    else:
        at = np.asarray(sent_at, dtype=np.float64)
        changed_on_flat = int(np.sum(changed & ~((at >= ramp_start) & (at <= ramp_end))))
        crossed_start = _first_time(arrays["t_s"], arrays["forward_m"] >= ramp_start)
        reached_end = _first_time(arrays["t_s"], arrays["forward_m"] >= ramp_end)
        recovery_s = None
        if reached_end is not None:
            limits = "robot.recovery"
            within = (arrays["tilt_deg"] <= _positive(config, f"{limits}.max_tilt_deg")) & (
                arrays["tilt_rate_rad_s"] <= _positive(config, f"{limits}.max_tilt_rate_rad_s"))
            recovery_s = _recovery_after(arrays["t_s"], within, reached_end, _positive(config, f"{limits}.hold_s"))
    entry = state["entry"]
    return {
        "scenario_id": scenario.get("scenario_id"),
        "kind": scenario.get("kind"),
        "features": state["features"],
        "decision_time_s": state["decision_t"],
        "action": state["action"],
        "measured": measured,
        "slip_reported": slip_reported,
        "modes": list(detector.changes),
        "entered": entry is not None,
        "entry_time_s": entry,
        "entry_latency_s": None if entry is None or crossed_start is None else float(entry - crossed_start),
        "commands_changed_on_flat": changed_on_flat,
        "scoring_source": _SCORING_SOURCE,
        "ramp_start_m": ramp_start,
        "ramp_end_m": ramp_end,
        "crossed_ramp_start_s": crossed_start,
        "reached_ramp_end_s": reached_end,
        "recovery_s": recovery_s,
        "elapsed_s": float(outcome["elapsed_s"]),
        "final_yaw_change_rad": float(outcome["final_yaw_change_rad"]),
        "final_speed_m_s": float(outcome["final_speed_m_s"]),
        "trace": {**arrays, "sent_nominal": np.asarray(sent_nominal, dtype=bool)},
    }


# ----- physics guess and objective -------------------------------------------------------------------------------


def physics_guess(
    config: Mapping[str, Any],
    features: np.ndarray,
    action_indices: Sequence[int] | np.ndarray,
    remaining_s: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Outcome guess (n, K) in OUTCOME_NAMES order from the observed features only.

    Args:
        features:       (n, D) in [0, 1]; mapped back to units (features_in_units).
        action_indices: (n,) candidate indices.
        remaining_s:    (n,) s from the decision to the end of the episode, >= 0.

    Model (constants skills.ramp.physics): slope theta = the filtered pitch; climb speed v = max(0, k_track x v_cmd -
    k_slope_s x gravity_m_s2 x sin theta) for the candidate's vel_x; progress_m = v x remaining (the distance already
    walked is not observed); reached_top = logistic((progress - typical_ramp_length_m) / softness_m); fall 0;
    sideways_m = |yaw_rate + yaw_drift_rad_s| x v x remaining^2 / 2; slip_m 0. Each clipped to its outcome bounds.
    """
    candidates = _candidates(config)
    units = features_in_units(config, features)
    n = units.shape[0] if units.ndim == 2 else -1
    actions = np.asarray(action_indices)
    remaining = np.asarray(remaining_s, dtype=np.float64)
    if units.ndim != 2 or actions.shape != (n,) or remaining.shape != (n,):
        raise ValueError(f"features (n, D), action_indices (n,) and remaining_s (n,) must agree; got {units.shape}, "
                         f"{actions.shape}, {remaining.shape}")
    if n and (not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions >= len(candidates.names))):
        raise ValueError(f"action_indices must be ints in [0, {len(candidates.names)})")
    if not np.all(np.isfinite(remaining)) or np.any(remaining < 0.0):
        raise ValueError("remaining_s must be finite and >= 0")
    p = f"{_ROOT}.physics"
    k_track, k_slope = _positive(config, f"{p}.k_track"), _non_negative(config, f"{p}.k_slope_s")
    gravity, length = _positive(config, f"{p}.gravity_m_s2"), _positive(config, f"{p}.typical_ramp_length_m")
    softness, drift = _positive(config, f"{p}.softness_m"), _value(config, f"{p}.yaw_drift_rad_s")
    bounds = _outcome_bounds(config)
    theta = np.radians(units[:, 0])
    command = candidates.features[actions.astype(np.int64)] if n else np.zeros((0, 3))
    speed = np.maximum(0.0, k_track * command[:, 0] - k_slope * gravity * np.sin(theta))
    progress = speed * remaining
    guess = np.zeros((n, len(OUTCOME_NAMES)), dtype=np.float64)
    guess[:, 0] = progress
    guess[:, 1] = 0.5 * (1.0 + np.tanh((progress - length) / (2.0 * softness)))  # logistic((progress - L) / softness)
    guess[:, 3] = np.abs(command[:, 2] + drift) * speed * remaining * remaining / 2.0
    return np.clip(guess, bounds[:, 0], bounds[:, 1])


def objective(config: Mapping[str, Any], predicted: OutcomeDistribution) -> np.ndarray:
    """Utility (n,), higher is better (weights skills.ramp.objective_weights, all >= 0)::

        + progress * mean progress_m + reached_top * mean reached_top - fall * mean fall
        - sideways * mean sideways_m - slip * mean slip_m - spread * spread of progress_m

    Raises TypeError / ValueError unless predicted is an OutcomeDistribution over OUTCOME_NAMES; ConfigError for a
    missing weight."""
    if not isinstance(predicted, OutcomeDistribution):
        raise TypeError(f"predicted must be an OutcomeDistribution; got {type(predicted).__name__}")
    if tuple(predicted.names) != OUTCOME_NAMES:
        raise ValueError(f"predicted outcomes {predicted.names} are not {OUTCOME_NAMES}")
    w = _objective_weights(config)
    return (
        w["progress"] * predicted.mean_of("progress_m")
        + w["reached_top"] * predicted.mean_of("reached_top")
        - w["fall"] * predicted.mean_of("fall")
        - w["sideways"] * predicted.mean_of("sideways_m")
        - w["slip"] * predicted.mean_of("slip_m")
        - w["spread"] * predicted.spread_of("progress_m")
    )


def physics_choice(config: Mapping[str, Any], features: np.ndarray, remaining_s: Sequence[float] | np.ndarray) -> np.ndarray:
    """(n,) the physics rule: argmax over the candidates of the objective of the physics guess as a point mass."""
    array = np.asarray(features)
    remaining = np.asarray(remaining_s, dtype=np.float64)
    n, c = array.shape[0], len(_candidates(config).names)
    physics = physics_guess(config, np.repeat(array, c, axis=0), np.tile(np.arange(c), n), np.repeat(remaining, c))
    levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
    return np.argmax(objective(config, OutcomeDistribution.point(OUTCOME_NAMES, physics, levels)).reshape(n, c), axis=1)


@dataclass(frozen=True)
class RampSpec:
    """What learners and scorers need of ramp (not a SkillSpec; the registry does not load it).

    name "ramp"; action_names (C,) candidate names, speed-major; action_features (C, 3) float64 read-only in
    action_feature_names order (vel_x, vel_y, yaw_rate); no_op_index None (no threat head); outcome_names /
    outcome_kinds; outcome_bounds (K, 2) float64 read-only; feature_names FEATURE_NAMES.
    """

    name: str
    action_names: tuple[str, ...]
    action_features: np.ndarray
    action_feature_names: tuple[str, ...]
    no_op_index: None
    outcome_names: tuple[str, ...]
    outcome_kinds: tuple[str, ...]
    outcome_bounds: np.ndarray
    feature_names: tuple[str, ...]
    config: Mapping[str, Any] = field(repr=False, compare=False)

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        """(n,) objective(config, predicted)."""
        return objective(self.config, predicted)

    def nominal_index(self, speed_m_s: float) -> int | None:
        """nominal_index(config, speed_m_s)."""
        return nominal_index(self.config, speed_m_s)


def build_observed_spec(config: Mapping[str, Any]) -> RampSpec:
    """The RampSpec of config.yaml. Raises ConfigError for a missing or invalid key."""
    candidates = _candidates(config)
    _feature_ranges(config)
    _objective_weights(config)
    return RampSpec(
        name=SKILL,
        action_names=candidates.names,
        action_features=candidates.features,
        action_feature_names=_FIELDS,
        no_op_index=None,
        outcome_names=OUTCOME_NAMES,
        outcome_kinds=OUTCOME_KINDS,
        outcome_bounds=_outcome_bounds(config),
        feature_names=FEATURE_NAMES,
        config=config,
    )


# ----- student policy --------------------------------------------------------------------------------------------


class _StudentPolicy:
    """The exported ramp student (observed.save_student with frames None): argmax of the action head."""

    def __init__(self, directory: str | Path, config: Mapping[str, Any]) -> None:
        from learned_judgment.observed import INPUT_ENCODING, STUDENT_FORMAT, student_readout
        from learned_judgment.students.snn_core import load_network

        folder = Path(directory)
        student = json.loads((folder / "student.json").read_text(encoding="utf-8"))
        spec = build_observed_spec(config)
        c, d = len(spec.action_names), len(FEATURE_NAMES)
        expected = {
            "format": STUDENT_FORMAT, "skill": SKILL, "action_names": list(spec.action_names),
            "heads": {"action": [0, c]}, "input": {"features": d, "encoding": INPUT_ENCODING},
        }
        for key, value in expected.items():
            if student.get(key) != value:
                raise ValueError(f"{folder / 'student.json'} {key} is {student.get(key)!r}; the ramp spec needs {value!r}")
        self.network, _ = load_network(folder / "network")
        if self.network.n_inputs != d or self.network.n_outputs != c:
            raise ValueError(f"the network is {self.network.layer_sizes}; the ramp student needs {d} inputs, {c} outputs")
        self.heads = expected["heads"]
        self.digest = student.get("network", {}).get("sha256")
        self._readout = student_readout

    def __call__(self, features: np.ndarray) -> int:
        _, scores = self._readout(self.network, np.asarray(features)[None, :], self.heads)
        return int(np.argmax(scores[0]))


# ----- collection ------------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(config: Mapping[str, Any], student_dir: str | None) -> None:
    """Per process: its own Go2World (observed.world_seed, observed.world_preset; no trace files) and the student."""
    from learned_judgment.env import Go2World  # on use: see _candidates

    config = copy.deepcopy(dict(config))
    config["world"]["save_traces"] = False
    _WORKER["world"] = Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    _WORKER["student"] = None if student_dir is None else _StudentPolicy(student_dir, config)


class _AttemptPolicy:
    """A context attempt's choice at the decision: the drawn candidate when exploring (explore_index), else the physics
    rule (physics_choice with remaining time episode_duration_s - decision time)."""

    takes_decision_time = True

    def __init__(self, config: Mapping[str, Any], explore_index: int | None) -> None:
        self.config, self.explore_index = config, explore_index
        self.duration = _positive(config, f"{_ROOT}.episode_duration_s")

    def __call__(self, features: np.ndarray, decision_time_s: float) -> int:
        if self.explore_index is not None:
            return int(self.explore_index)
        remaining = max(self.duration - float(decision_time_s), 0.0)
        return int(physics_choice(self.config, np.asarray(features)[None, :], [remaining])[0])


def _episode_task(task: tuple) -> dict[str, Any]:
    """(scenario, policy) with policy an int (constant candidate), "student", "nominal" or ("attempt", explore index
    or None); the trace is dropped."""
    scenario, which = task
    if isinstance(which, tuple):
        result = run_episode(_WORKER["world"], scenario, scenario["scenario_id"],
                             _AttemptPolicy(_WORKER["world"].config, which[1]))
        result.pop("trace")
        result["policy"] = "attempt"
        return result
    if which == "nominal":
        policy = None
    elif which == "student":
        policy = _WORKER["student"]
    else:
        policy = lambda _features, index=int(which): index  # noqa: E731 - a constant candidate
    result = run_episode(_WORKER["world"], scenario, scenario["scenario_id"], policy)
    result.pop("trace")
    result["policy"] = which
    return result


@contextmanager
def _pool(config: Mapping[str, Any], student_dir: str | None = None) -> Iterator[Any]:
    workers = require(config, "observed.workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"config.yaml key 'observed.workers' must be an int >= 1; got {workers!r}")
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_worker_init, initargs=(config, student_dir)) as pool:
        yield pool


def _map(pool: Any, tasks: list[tuple], label: str, on_progress: Callable[[int], None] | None = None) -> list[dict[str, Any]]:
    """_episode_task over tasks, results in task order, sparse progress lines (and on_progress(done) with each)."""
    every = max(1, len(tasks) // 10)
    results = []
    for result in pool.imap(_episode_task, tasks):
        results.append(result)
        if len(results) % every == 0 or len(results) == len(tasks):
            print(f"{label} {len(results)}/{len(tasks)}", flush=True)
            if on_progress is not None:
                on_progress(len(results))
    return results


def _split_plan(config: Mapping[str, Any], split: str, n: int) -> list[dict[str, Any]]:
    seed = require(config, "runtime.seed")
    return sample_ramp_scenarios(config, np.random.default_rng(np.random.SeedSequence([seed, SPLIT_CODES[split]])), n)


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    return value


def _meta(config: Mapping[str, Any], split: str, scenarios: list[dict[str, Any]], n_situations: int) -> dict[str, Any]:
    spec = build_observed_spec(config)
    kinds = Counter(s["kind"] for s in scenarios)
    return {
        "skill": SKILL,
        "split": split,
        "action_names": list(spec.action_names),
        "outcome_names": list(spec.outcome_names),
        "outcome_kinds": list(spec.outcome_kinds),
        "outcome_bounds": spec.outcome_bounds.tolist(),
        "feature_names": list(FEATURE_NAMES),
        "feature_ranges": _feature_ranges(config).tolist(),
        "frames": None,
        "n_situations": n_situations,
        "n_scenarios": len(scenarios),
        "seeds": {
            "runtime_seed": require(config, "runtime.seed"),
            "split_code": SPLIT_CODES[split],
            "scenarios": "np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code]))",
            "attempt_seed": "scenario_id",
        },
        "world_seed": require(config, "observed.world_seed"),
        "world_preset": require(config, "observed.world_preset"),
        "config_digest": config_digest(config),
        "kind_counts": {kind: kinds.get(kind, 0) for kind in KINDS},
        "scenarios": scenarios,
    }


def _write(out_dir: Path, stem: str, meta: Mapping[str, Any], **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(out_dir / f"{stem}.npz", buffer.getvalue())
    atomic_write_text(out_dir / f"{stem}.json", json.dumps(_jsonable(meta), indent=1, allow_nan=False) + "\n")


def _detection(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    flat = [r for r in results if r["kind"] == "flat"]
    slope = [r for r in results if r["kind"] == "slope"]
    return {
        "flat_scenarios": len(flat),
        "flat_false_activations": sum(r["entered"] for r in flat),
        "slope_scenarios": len(slope),
        "missed_slope_entries": sum(not r["entered"] for r in slope),
        "entered_without_decision": sum(r["entered"] and r["features"] is None for r in results),
    }


def collect_rows(config: Mapping[str, Any], split: str, out_dir: str | Path, n: int | None = None) -> dict[str, Any]:
    """Measure the one-shot rows of a split and write <out_dir>/rows.npz and rows.json; returns the meta.

    n scenarios (default observed.ramp.<split>_scenarios) from the split's seed. Candidate 0 runs first on every
    scenario; where it reached a decision the other C - 1 candidates run too (attempt_seed = scenario_id). A scenario
    without a decision gives no rows but stays in rows.json with its detection record. Raises RuntimeError when the
    candidates of a scenario saw different features, decision times or detector switches before acting.
    """
    if split not in ("context", "dev", "test"):
        raise ValueError(f"split must be context, dev or test; got {split!r}")
    n = _count("n", require(config, f"{_OBSERVED}.{split}_scenarios") if n is None else n)
    spec = build_observed_spec(config)
    c = len(spec.action_names)
    duration = _positive(config, f"{_ROOT}.episode_duration_s")
    scenarios = _split_plan(config, split, n)
    if any(INCLINE_KEY in s["situation"] for s in scenarios):
        _require_incline()
    print(f"rows: split {split}, {n} scenarios, {c} candidates", flush=True)
    start = time.perf_counter()
    with _pool(config) as pool:
        first = _map(pool, [(s, 0) for s in scenarios], "candidate 0:")
        decided = [r for r in first if r["features"] is not None]
        rest = _map(pool, [(scenarios[r["scenario_id"]], a) for r in decided for a in range(1, c)], "other candidates:") \
            if decided and c > 1 else []
    wall_s = time.perf_counter() - start

    by_scenario: dict[int, list[dict[str, Any]]] = {r["scenario_id"]: [r] for r in decided}
    for r in rest:
        by_scenario[r["scenario_id"]].append(r)
    for sid, results in by_scenario.items():
        reference = results[0]
        for r in results[1:]:
            if not (np.array_equal(r["features"], reference["features"]) and r["decision_time_s"] == reference["decision_time_s"]
                    and r["modes"][:1] == reference["modes"][:1]):
                raise RuntimeError(f"scenario {sid}: candidate {r['action']} saw other features or decision time than "
                                   f"candidate 0 before acting; the episodes are not deterministic")
    ordered = [r for sid in sorted(by_scenario) for r in sorted(by_scenario[sid], key=lambda r: r["action"])]
    scenario_index = np.array([r["scenario_id"] for r in ordered], dtype=np.int64)
    action = np.array([r["action"] for r in ordered], dtype=np.int64)
    features = np.array([r["features"] for r in ordered], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    decision_time_s = np.array([r["decision_time_s"] for r in ordered], dtype=np.float64)
    measured = np.array([[r["measured"][name] for name in OUTCOME_NAMES] for r in ordered], dtype=np.float64)
    measured = measured.reshape(-1, len(OUTCOME_NAMES))
    physics = physics_guess(config, features, action, np.maximum(duration - decision_time_s, 0.0))

    for scenario, r in zip(scenarios, first):
        scenario["detection"] = {key: r[key] for key in (
            "entered", "entry_time_s", "decision_time_s", "entry_latency_s", "modes", "ramp_start_m", "ramp_end_m",
            "commands_changed_on_flat")}
        scenario["detection"]["decided"] = r["features"] is not None
    meta = _meta(config, split, scenarios, len(by_scenario))
    meta["detection"] = _detection(first)
    reported = [r["slip_reported"] for r in first + rest]
    meta["slip"] = {"reported_by_world": bool(reported) and all(reported),
                    "note": "slip_m is foot_slip_m of finish_episode; 0.0 in every episode where the world does not "
                            f"report it ({reported.count(False)} of {len(reported)} episodes)"}
    meta["scoring_source"] = _SCORING_SOURCE
    _write(Path(out_dir), "rows", meta, scenario_index=scenario_index, decision_time_s=decision_time_s, action=action,
           features=features, physics=physics, measured=measured)

    episodes = len(first) + len(rest)
    print("detection: " + json.dumps(meta["detection"]))
    print(f"rows {len(ordered)} from {len(by_scenario)} situations; slip reported by the world: {meta['slip']['reported_by_world']}")
    print(f"wall time {wall_s:.1f} s for {episodes} episodes: {wall_s / max(episodes, 1):.3f} s per episode")
    print(f"wrote {Path(out_dir) / 'rows.npz'} and rows.json", flush=True)
    return meta


def collect_attempts(config: Mapping[str, Any], out_dir: str | Path, n: int | None = None) -> dict[str, Any]:
    """Attempts-style context (MISSION.md: about 10-40 attempts per skill): n scenarios of the context split (default
    observed.ramp.fast_attempts), ONE episode each; writes <out_dir>/rows.npz and rows.json in the rows contract (one
    row per scenario that reached a decision; a scenario without one gives no row, as in collect_rows) and returns the
    meta.

    Scenario i's candidate, the mix of engine.collect.exploration_actions: u = np.random.default_rng(
    np.random.SeedSequence([runtime.seed, SPLIT_CODES["context"], i])).random(2); u[0] <
    engine.attempt_exploration_fraction -> candidate min(floor(u[1] x C), C - 1) (uniform), else the physics rule
    (physics_choice on the features observed at the decision, remaining time episode_duration_s - decision time).
    """
    n = _count("n", require(config, f"{_OBSERVED}.fast_attempts") if n is None else n)
    key = "engine.attempt_exploration_fraction"
    fraction = _value(config, key)
    if not 0.0 <= fraction <= 1.0:
        raise ConfigError(f"config.yaml key {key!r} is a probability and must lie in [0, 1]; got {fraction}")
    spec = build_observed_spec(config)
    c = len(spec.action_names)
    duration = _positive(config, f"{_ROOT}.episode_duration_s")
    seed = int(require(config, "runtime.seed"))
    scenarios = _split_plan(config, "context", n)
    if any(INCLINE_KEY in s["situation"] for s in scenarios):
        _require_incline()
    tasks = []
    for scenario in scenarios:
        u = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_CODES["context"], scenario["scenario_id"]])).random(2)
        explore = bool(u[0] < fraction)
        scenario["attempt"] = {"source": "explore" if explore else "physics"}
        tasks.append((scenario, ("attempt", min(int(math.floor(u[1] * c)), c - 1) if explore else None)))
    print(f"attempts: {n} context scenarios, one episode each (exploration fraction {fraction})", flush=True)
    start = time.perf_counter()
    with _pool(config) as pool:
        results = _map(pool, tasks, "attempts:")
    wall_s = time.perf_counter() - start

    decided = sorted((r for r in results if r["features"] is not None), key=lambda r: r["scenario_id"])
    scenario_index = np.array([r["scenario_id"] for r in decided], dtype=np.int64)
    action = np.array([r["action"] for r in decided], dtype=np.int64)
    features = np.array([r["features"] for r in decided], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    decision_time_s = np.array([r["decision_time_s"] for r in decided], dtype=np.float64)
    measured = np.array([[r["measured"][name] for name in OUTCOME_NAMES] for r in decided], dtype=np.float64)
    measured = measured.reshape(-1, len(OUTCOME_NAMES))
    physics = physics_guess(config, features, action, np.maximum(duration - decision_time_s, 0.0))
    for scenario, r in zip(scenarios, results):
        scenario["detection"] = {key: r[key] for key in (
            "entered", "entry_time_s", "decision_time_s", "entry_latency_s", "modes", "ramp_start_m", "ramp_end_m",
            "commands_changed_on_flat")}
        scenario["detection"]["decided"] = r["features"] is not None
        scenario["attempt"]["action"] = r["action"]
    meta = _meta(config, "context", scenarios, len(decided))
    meta["detection"] = _detection(results)
    reported = [r["slip_reported"] for r in results]
    meta["slip"] = {"reported_by_world": bool(reported) and all(reported),
                    "note": "slip_m is foot_slip_m of finish_episode; 0.0 in every episode where the world does not "
                            f"report it ({reported.count(False)} of {len(reported)} episodes)"}
    meta["scoring_source"] = _SCORING_SOURCE
    sources = Counter(s["attempt"]["source"] for s in scenarios)
    meta.update(collection="attempts", n_attempts=n, exploration_fraction=fraction, wall_s=wall_s,
                attempt_sources={"explore": sources.get("explore", 0), "physics": sources.get("physics", 0)})
    meta["seeds"]["attempt_choice"] = ("np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code, "
                                       "scenario_id])).random(2): u[0] < exploration_fraction -> candidate "
                                       "min(floor(u[1] * C), C - 1), else physics_choice at the decision")
    _write(Path(out_dir), "rows", meta, scenario_index=scenario_index, decision_time_s=decision_time_s, action=action,
           features=features, physics=physics, measured=measured)
    print("detection: " + json.dumps(meta["detection"]))
    print(f"rows {len(decided)} from {n} attempts; sources {meta['attempt_sources']}")
    print(f"wall time {wall_s:.1f} s for {len(results)} episodes (pool start-up included)")
    print(f"wrote {Path(out_dir) / 'rows.npz'} and rows.json", flush=True)
    return meta


def recorded_situations(
    features: np.ndarray, decision_time_s: np.ndarray, scenario_index: np.ndarray, n: int, noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """n dream situations drawn from measured ones: (n, D) float32 features in [0, 1] and (n,) float64 decision times.

    One feature vector and decision time per scenario (its first row; a full rows run repeats them once per
    candidate), scenarios picked uniformly with replacement (rng.integers), then a smoothed bootstrap: noise drawn from
    a normal with mean 0 and covariance noise^2 x cov(V) of the per-scenario vectors V (rng.multivariate_normal,
    method eigh, which tolerates a singular covariance), so features that move together stay together and a feature
    constant across the scenarios gets none; no noise with fewer than two scenarios or noise 0. Clipped to [0, 1].
    Each situation keeps its scenario's decision time.
    """
    features = np.asarray(features, dtype=np.float32)
    times = np.asarray(decision_time_s, dtype=np.float64)
    index = np.asarray(scenario_index)
    r = features.shape[0] if features.ndim == 2 else -1
    if r < 1 or times.shape != (r,) or index.shape != (r,):
        raise ValueError(f"features (r >= 1, D), decision_time_s (r,) and scenario_index (r,) must agree; got "
                         f"{features.shape}, {times.shape}, {index.shape}")
    n = _count("n", n)
    if not (math.isfinite(noise) and noise >= 0.0):
        raise ValueError(f"noise must be finite and >= 0; got {noise!r}")
    _, first = np.unique(index, return_index=True)
    rows = first[rng.integers(first.size, size=n)]
    drawn = features[rows].astype(np.float64)
    if first.size >= 2 and noise > 0.0:
        vectors = features[first]
        offsets = rng.multivariate_normal(np.zeros(vectors.shape[1]), noise**2 * np.cov(vectors, rowvar=False),
                                          size=n, method="eigh")
        offsets[:, np.ptp(vectors, axis=0) == 0.0] = 0.0  # exactly none, not eigh's rounding residue
        drawn += offsets
    return np.clip(drawn, 0.0, 1.0).astype(np.float32), times[rows]


def collect_dream(
    config: Mapping[str, Any], out_dir: str | Path, n: int | None = None, from_rows: str | Path | None = None
) -> dict[str, Any]:
    """Write <out_dir>/dream.npz and dream.json without a simulator; returns the meta.

    n situations (default observed.ramp.dream_situations) from the dream seed, by skills.ramp.dream_sampling:
      uniform   features uniform in [0, 1) per feature (uniform over the declared feature ranges), then decision times
                uniform in skills.ramp.dream_decision_time_range_s (the behaviour before the key).
      recorded  recorded_situations over <from_rows>/rows.npz (ramp rows --split context or ramp attempts) with noise
                skills.ramp.dream_noise; ValueError without from_rows.
    Then the physics guess of every candidate with remaining time episode_duration_s - decision time.
    """
    n = _count("n", require(config, f"{_OBSERVED}.dream_situations") if n is None else n)
    spec = build_observed_spec(config)
    c, k = len(spec.action_names), len(OUTCOME_NAMES)
    duration = _positive(config, f"{_ROOT}.episode_duration_s")
    times_key = f"{_ROOT}.dream_decision_time_range_s"
    low, high = _pair(config, times_key, minimum=0.0)
    if high > duration:
        raise ConfigError(f"config.yaml key {times_key!r} must end by episode_duration_s {duration}; got {[low, high]}")
    mode = require(config, f"{_ROOT}.dream_sampling")
    if mode not in ("uniform", "recorded"):
        raise ConfigError(f"config.yaml key '{_ROOT}.dream_sampling' must be uniform or recorded; got {mode!r}")
    rng = np.random.default_rng(np.random.SeedSequence([require(config, "runtime.seed"), SPLIT_CODES["dream"]]))
    if mode == "uniform":
        features = rng.random((n, len(FEATURE_NAMES))).astype(np.float32)
        decision_time_s = rng.uniform(low, high, size=n)
        sampling = {
            "mode": "uniform",
            "features": "sampled uniformly over the declared feature ranges, not rendered",
            "decision_time_s": f"uniform in {times_key} {[low, high]}",
            "draw_order": "rng.random((n, D)) then rng.uniform(low, high, n)",
        }
    else:
        if from_rows is None:
            raise ValueError(f"{_ROOT}.dream_sampling is recorded: pass the context rows directory (--from-rows DIR)")
        folder = Path(from_rows)
        rows_meta = json.loads((folder / "rows.json").read_text(encoding="utf-8"))
        if rows_meta.get("feature_names") != list(FEATURE_NAMES) or not np.allclose(
                rows_meta.get("feature_ranges"), _feature_ranges(config)):
            raise ValueError(f"{folder / 'rows.json'} was measured with other features or feature ranges than config.yaml")
        noise = _non_negative(config, f"{_ROOT}.dream_noise")
        with np.load(folder / "rows.npz") as rows:
            index = rows["scenario_index"]
            features, decision_time_s = recorded_situations(rows["features"], rows["decision_time_s"], index, n, noise, rng)
        sampling = {
            "mode": "recorded",
            "rows": folder.as_posix(),
            "n_scenarios_drawn_from": int(np.unique(index).size),
            "dream_noise": noise,
            "noise": f"noise covariance = dream_noise squared x covariance of the measured vectors (dream_noise {noise})",
            "features": "the measured feature vector of a context scenario (its first row), scenarios drawn uniformly "
                        "with replacement, plus that normal noise (none for a feature constant across the scenarios, "
                        "none with fewer than two scenarios), clipped to [0, 1]",
            "decision_time_s": "the drawn scenario's measured decision time",
            "draw_order": "rng.integers(scenarios, n) then rng.multivariate_normal(0, dream_noise^2 cov(V), n, "
                          "method='eigh')",
        }
    physics = physics_guess(config, np.repeat(features, c, axis=0), np.tile(np.arange(c), n),
                            np.repeat(duration - decision_time_s, c)).reshape(n, c, k)
    meta = _meta(config, "dream", [], n)
    meta["sampling"] = sampling
    _write(Path(out_dir), "dream", meta, scenario_index=np.arange(n, dtype=np.int64), decision_time_s=decision_time_s,
           features=features, physics=physics)
    best = np.argmax(objective(config, OutcomeDistribution.point(
        OUTCOME_NAMES, physics.reshape(n * c, k), np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
    )).reshape(n, c), axis=1)
    counts = Counter(spec.action_names[i] for i in best)
    print("physics-best candidate counts: " + ", ".join(f"{name} {counts.get(name, 0)}" for name in spec.action_names))
    print(f"wrote {Path(out_dir) / 'dream.npz'} and dream.json", flush=True)
    return meta


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    known = np.array([v for v in values if v is not None], dtype=np.float64)
    if known.size == 0:
        return {"n": 0}
    return {"n": int(known.size), "mean": float(known.mean()), "median": float(np.median(known)),
            "min": float(known.min()), "max": float(known.max())}


def _summary(results: Sequence[Mapping[str, Any]], action_names: Sequence[str]) -> dict[str, Any]:
    slope = [r for r in results if r["kind"] == "slope"]
    by_kind = {kind: [r for r in results if r["kind"] == kind] for kind in KINDS}
    reached = [r for r in slope if r["measured"]["reached_top"]]
    return {
        **_detection(results),
        "entry_latency_s": _stats([r["entry_latency_s"] for r in slope]),
        "ascent_success": {"count": len(reached), "of_slope": len(slope),
                           "rate": len(reached) / len(slope) if slope else None},
        "falls": {"count": int(sum(r["measured"]["fall"] for r in results)), "of": len(results)},
        "slip_m": {kind: _stats([r["measured"]["slip_m"] for r in rows]) for kind, rows in by_kind.items()},
        "progress_m": {kind: _stats([r["measured"]["progress_m"] for r in rows]) for kind, rows in by_kind.items()},
        "sideways_m": {kind: _stats([r["measured"]["sideways_m"] for r in rows]) for kind, rows in by_kind.items()},
        "time_to_top_s": _stats([r["reached_ramp_end_s"] for r in slope]),
        "elapsed_s": _stats([r["elapsed_s"] for r in results]),
        "recovery_s": {**_stats([r["recovery_s"] for r in slope if r["reached_ramp_end_s"] is not None]),
                       "not_recovered": sum(r["reached_ramp_end_s"] is not None and r["recovery_s"] is None for r in slope)},
        "commands_changed_on_flat": {"total": int(sum(r["commands_changed_on_flat"] for r in results)),
                                     "episodes_with_any": sum(r["commands_changed_on_flat"] > 0 for r in results)},
        "action_counts": {name: sum(r["action"] == i for r in results) for i, name in enumerate(action_names)},
    }


_UI_METHODS = {"no_op": "nominal", "snn": "student"}  # result.json method name -> run.json method
_METRIC_NOTES = {
    "ascent_success_rate": "slope scenarios whose episode ended with reached_top 1 (final_forward_m >= ramp_end_m, no "
                           "fall); Wilson interval",
    "fall_rate": "every scenario (flat and slope) with fall 1; Wilson interval",
    "mean_progress_m": "mean final_forward_m over the slope scenarios",
    "mean_slip_m": "mean foot_slip_m over the slope scenarios (the whole episode, approach included)",
    "flat_false_activations": "flat scenarios in which the detector switched to SLOPE at least once (a count; n = flat "
                              "scenarios)",
    "no_op": "the nominal walking command throughout (RAMP has no NO_OP; the detector observes and never acts)",
    "snn": "the exported spiking student's argmax candidate, sent while the detector is in SLOPE",
}


def _ui_metrics(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]], split: str) -> tuple[list, list, str]:
    """result.json UI keys "metrics" and "paired" (ascent success snn - no_op, percentile bootstrap over the slope
    scenarios at evaluation.bootstrap_confidence_level with evaluation.bootstrap_resamples) and the bootstrap stream."""
    from learned_judgment.dodge_eval import paired_bootstrap, wilson_interval
    from learned_judgment.engine.streams import stream_rng

    level = _value(config, "evaluation.bootstrap_confidence_level")
    resamples = _count("evaluation.bootstrap_resamples", require(config, "evaluation.bootstrap_resamples"))
    slope = [r for r in records if r["kind"] == "slope"]
    flat = [r for r in records if r["kind"] == "flat"]

    def rate(k: int, n: int) -> dict[str, Any]:
        low, high = wilson_interval(k, n, level)
        return {"value": k / n if n else None, "ci_low": low, "ci_high": high, "n": n}

    def mean(values: list[float]) -> dict[str, Any]:
        return {"value": float(np.mean(values)) if values else None, "ci_low": None, "ci_high": None, "n": len(values)}

    def per_method(make: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        return {ui: make(method) for ui, method in _UI_METHODS.items()}

    metrics = [
        {"name": "ascent_success_rate", "unit": "fraction", "higher_is_better": True,
         "values": per_method(lambda m: rate(sum(int(r[m]["measured"]["reached_top"]) for r in slope), len(slope)))},
        {"name": "fall_rate", "unit": "fraction", "higher_is_better": False,
         "values": per_method(lambda m: rate(sum(int(r[m]["measured"]["fall"]) for r in records), len(records)))},
        {"name": "mean_progress_m", "unit": "m", "higher_is_better": True,
         "values": per_method(lambda m: mean([r[m]["measured"]["progress_m"] for r in slope]))},
        {"name": "mean_slip_m", "unit": "m", "higher_is_better": False,
         "values": per_method(lambda m: mean([r[m]["measured"]["slip_m"] for r in slope]))},
        {"name": "flat_false_activations", "unit": "count", "higher_is_better": False,
         "values": per_method(lambda m: {"value": sum(int(r[m]["entered"]) for r in flat), "ci_low": None,
                                         "ci_high": None, "n": len(flat)})},
    ]
    stream = f"{SKILL}/run/{split}/paired_bootstrap"
    differences = [r["student"]["measured"]["reached_top"] - r["nominal"]["measured"]["reached_top"] for r in slope]
    p = paired_bootstrap(differences, resamples, level, stream_rng(int(require(config, "runtime.seed")), stream, 0))
    paired = [{"name": "ascent_success_rate", "a": "snn", "b": "no_op", "diff": p["diff"], "ci_low": p["ci_low"],
               "ci_high": p["ci_high"], "n": p["n"]}]
    return metrics, paired, f"{stream}[0]"


def _merge_result(run_dir: Path, run_id: str, updates: Mapping[str, Any]) -> Path:
    """Merge top-level keys into <run_dir>/result.json (observed's file: every other key is kept)."""
    from learned_judgment.observed import _dumps, _jsonable as _observed_jsonable, _utc_now  # Paths become strings

    path = run_dir / "result.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": run_id, "skill": SKILL}
    if result.get("run_id") != run_id or result.get("skill") != SKILL:
        raise ValueError(f"{path} belongs to run {result.get('run_id')!r} of skill {result.get('skill')!r}")
    result.update(_observed_jsonable(dict(updates)))
    result["updated_utc"] = _utc_now().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _dumps(result, indent=1) + "\n")
    return path


def _student_run_id(student_dir: str | Path) -> str:
    """provenance.run_id of <student_dir>/student.json (written by observed train)."""
    path = Path(student_dir) / "student.json"
    run_id = json.loads(path.read_text(encoding="utf-8")).get("provenance", {}).get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{path} has no provenance.run_id; pass --run")
    return run_id


def run_student(
    config: Mapping[str, Any], student_dir: str | Path, split: str, out_dir: str | Path, n: int | None = None,
    run_id: str | None = None, argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Complete flat -> slope -> flat episodes of a split with the trained student deciding at the entry, next to the
    nominal pass-through on the same scenarios and attempt seeds; writes <out_dir>/run.json and returns it.

    With run_id, the run's events.jsonl gets "evaluate" events and its result.json the UI keys (run_id, skill, seeds,
    scenario_ids, metrics, paired: methods "no_op" = the nominal pass-through and "snn" = the student) and a section
    "run_<split>"; every other key of result.json is kept."""
    if split not in ("context", "dev", "test"):
        raise ValueError(f"split must be context, dev or test; got {split!r}")
    n = _count("n", require(config, f"{_OBSERVED}.{split}_scenarios") if n is None else n)
    spec = build_observed_spec(config)
    student = _StudentPolicy(student_dir, config)  # checked here before any worker starts
    scenarios = _split_plan(config, split, n)
    if any(INCLINE_KEY in s["situation"] for s in scenarios):
        _require_incline()
    methods = ("nominal", "student")
    folder = Path(out_dir)
    events = run_dir = None
    section = f"run_{split}"
    common = {"argv": list(argv or []), "split": split, "student": Path(student_dir).as_posix(),
              "network_sha256": student.digest, "config_digest": config_digest(config)}
    if run_id is not None:
        from learned_judgment.observed import EventLog, _run_dir

        run_dir = _run_dir(config, run_id)
        events = EventLog(run_dir / "events.jsonl", run_id, SKILL)
        _merge_result(run_dir, run_id, {section: {"state": "running", **common}})
        events.emit("evaluate", "running", done=0, total=n * len(methods), data={"split": split, **common})
    print(f"run: split {split}, {n} scenarios x {methods}", flush=True)
    try:
        start = time.perf_counter()
        progress = None if events is None else (
            lambda done: events.emit("evaluate", "running", done=done, total=n * len(methods)))
        with _pool(config, str(student_dir)) as pool:
            results = _map(pool, [(s, method) for s in scenarios for method in methods], "episodes:", progress)
        wall_s = time.perf_counter() - start
        records = []
        for scenario in scenarios:
            own = {r["policy"]: r for r in results if r["scenario_id"] == scenario["scenario_id"]}
            records.append({**scenario, **{method: own[method] for method in methods}})
        out = {
            **_meta(config, split, scenarios, n),
            "student": {"dir": Path(student_dir).as_posix(), "network_sha256": student.digest},
            "methods": {"nominal": "the nominal command throughout (detector observes, never acts)",
                        "student": "the student's argmax candidate (observed.student_readout, action head) at the entry"},
            "scoring_source": _SCORING_SOURCE,
            "records": records,
            "summary": {method: _summary([record[method] for record in records], spec.action_names) for method in methods},
            "wall_s": wall_s,
        }
        out.pop("scenarios")
        atomic_write_text(folder / "run.json", json.dumps(_jsonable(out), indent=1, allow_nan=False) + "\n")
        if run_id is not None:
            metrics, paired, stream = _ui_metrics(config, records, split)
            seeds = {**out["seeds"], "split": split, "world_seed": out["world_seed"], "world_preset": out["world_preset"],
                     "bootstrap": stream}
            scenario_ids = [record["scenario_id"] for record in records]
            _merge_result(run_dir, run_id, {
                section: {"state": "done", **common, "records": folder / "run.json", "summary": out["summary"],
                          "metric_notes": _METRIC_NOTES, "wall_s": wall_s, "seeds": seeds,
                          "bootstrap": {"method": "percentile bootstrap over slope scenarios",
                                        "resamples": require(config, "evaluation.bootstrap_resamples"),
                                        "level": require(config, "evaluation.bootstrap_confidence_level")},
                          "where": "simulation (MuJoCo Go2World); student inference on this computer's CPU (numpy)"},
                "seeds": seeds, "scenario_ids": scenario_ids, "metrics": metrics, "paired": paired,
            })
    except Exception as error:
        if events is not None:
            events.emit("evaluate", "failed", data={"type": type(error).__name__, "message": str(error)})
            _merge_result(run_dir, run_id, {section: {"state": "failed", **common, "error": {
                "type": type(error).__name__, "message": str(error)}}})
        raise
    for method in methods:
        s = out["summary"][method]
        print(f"{method}: false activations {s['flat_false_activations']}/{s['flat_scenarios']}, missed entries "
              f"{s['missed_slope_entries']}/{s['slope_scenarios']}, ascents {s['ascent_success']['count']}/"
              f"{s['ascent_success']['of_slope']}, falls {s['falls']['count']}, commands changed on flat "
              f"{s['commands_changed_on_flat']['total']}")
    if events is not None:
        events.emit("evaluate", "done", done=n * len(methods), total=n * len(methods), data={
            "split": split, "records": folder / "run.json", "metrics": metrics, "paired": paired})
    print(f"wall time {wall_s:.1f} s; wrote {folder / 'run.json'}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (see the module docstring). Returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.skills.ramp", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rows = commands.add_parser("rows", help="measured one-shot rows of a split")
    rows.add_argument("--split", required=True, choices=("context", "dev", "test"))
    rows.add_argument("--out", required=True, help="output directory")
    rows.add_argument("--n", type=int, help="scenarios (default observed.ramp.<split>_scenarios)")
    attempts = commands.add_parser("attempts", help="context split, ONE episode per scenario (explore or physics rule)")
    attempts.add_argument("--out", required=True, help="output directory")
    attempts.add_argument("--n", type=int, help="attempts (default observed.ramp.fast_attempts)")
    dream = commands.add_parser("dream", help="dream situations by skills.ramp.dream_sampling (no simulator)")
    dream.add_argument("--out", required=True, help="output directory")
    dream.add_argument("--n", type=int, help="situations (default observed.ramp.dream_situations)")
    dream.add_argument("--from-rows", help="context rows directory (required when dream_sampling is recorded)")
    run = commands.add_parser("run", help="closed-loop episodes: the student next to the nominal pass-through")
    run.add_argument("--student", required=True, help="exported student directory (observed train)")
    run.add_argument("--split", required=True, choices=("context", "dev", "test"))
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--n", type=int, help="scenarios (default observed.ramp.<split>_scenarios)")
    run.add_argument("--run", help="run id for events.jsonl / result.json (default: student.json provenance.run_id)")
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "rows":
        collect_rows(config, args.split, args.out, args.n)
    elif args.command == "attempts":
        collect_attempts(config, args.out, args.n)
    elif args.command == "dream":
        collect_dream(config, args.out, args.n, args.from_rows)
    else:
        run_id = args.run if args.run else _student_run_id(args.student)
        run_student(config, args.student, args.split, args.out, args.n, run_id=run_id,
                    argv=list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    # Through the importable module, so spawned workers find _worker_init and the tasks by their module name.
    from learned_judgment.skills import ramp as _module

    raise SystemExit(_module.main())
