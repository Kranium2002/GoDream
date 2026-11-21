"""MOVE (Person B): move the standing dog to a target and stop. One axis per request: forward/backward distance
(vel_x), sideways distance (vel_y) or a turn angle (yaw_rate). The walking policy is frozen; the dog learns how hard
(command level) and how long (hold) to command.

Lane contract (Person B, move)
------------------------------
One decision at the decision instant (t_s 0 of Go2World.reset, after the dog stood on the zero command for
skills.move.settle_s): the policy is called ONCE with the five features and returns a candidate index; that candidate's
command is held for its hold_s, then the zero command for skills.move.settle_tail_s, then finish_episode. The episode
lasts hold_s + settle_tail_s; drift on the zero command after that is not counted.

    KINDS       forward (vel_x, m), sideways (vel_y, m), turn (yaw_rate, rad); + forward, + left, + anticlockwise.
    SCENARIO    {"scenario_id", "kind", "target" (signed, m or rad), "situation": {"dog_forward_speed_m_s": 0.0,
                "settle_s": skills.move.settle_s}}. Per scenario three draws u = rng.random(3): the kind by
                skills.move.kind_weights (u[0]), |target| uniform in skills.move.targets.<kind> (u[1]), the sign
                (+ when u[2] < 0.5).
    SCALE       per kind, the high end of its targets range (forward_m, sideways_m: m; turn_rad: rad).
    FEATURES    D = 5 float32 in [0, 1], FEATURE_NAMES order: one-hot of the kind (3), |target| / scale, 1.0 for a
                positive target else 0.0. They come from the request, not from a sensor.
    CANDIDATES  C = len(command_levels) x len(hold_s) of skills.move.candidates, level-major (index = i_level * n_hold +
                i_hold): the kind's field at level x the verified range end on the target's side (max for +, min for
                -), the other two fields 0, held for hold_s; every one checked against the verified command schema.
                action_features (C, 2) = (command_level, hold_s). no_op_index None.
    OUTCOMES    K = 3 in OUTCOME_NAMES order:
                  error    regression  |final value on the kind's axis - target| / scale. DIMENSIONLESS: the objective
                                       receives only the predicted outcomes (observed.py calls spec.objective(predicted)),
                                       so the per-row scale is folded into the outcome. The error in m or rad is
                                       error_units (rows.npz, rows.json, run.json).
                  drift_m  regression  forward: |final_sideways_m|; sideways: |final_forward_m|; turn:
                                       hypot(final_forward_m, final_sideways_m). m.
                  fall     binary      finish_episode fall.
                Final values are finish_episode's (heading frame of the decision instant). The turn's final value is the
                yaw change summed per control period (unwrapped), checked against final_yaw_change_rad modulo 2 pi.
    PHYSICS     travel = track[kind] x level x max(0, hold_s - latency_s) toward the target; error = |travel - |target||
                / scale; drift 0; fall 0 (skills.move.physics). Clipped to the outcome bounds.
    OBJECTIVE   -(w.error x mean error + w.drift x mean drift_m + w.fall x mean fall + w.spread x spread of error).

Files of the command line, one directory per call (the observed.py array contract):

    rows.npz    scenario_index (n,) int64, decision_time_s (n,) float64 (0.0), action (n,) int64, features (n, 5)
                float32, physics (n, 3) float64, measured (n, 3) float64; and, outside the contract, target (n,),
                error_units (n,) (m or rad), final (n, 3) [final_forward_m, final_sideways_m, yaw change unwrapped]
    rows.json   skill, split, action_names, outcome_names, outcome_kinds, outcome_bounds, feature_names, frames (null),
                n_situations, n_scenarios, seeds, world_seed, world_preset, config_digest, kind_counts, scales, units,
                error_note, scenarios (index = scenario_index; each with error_units per candidate)
    dream.npz   scenario_index (m,) int64, decision_time_s (m,) float64 (0.0), features (m, 5) float32,
                physics (m, C, 3) float64
    dream.json  the keys of rows.json plus sampling (no simulator)
    run.json    records (per scenario: the physics rule and the student) and summary per method and kind

    python -m learned_judgment.skills.move rows --split {context|dev|test} --out DIR [--n N]
    python -m learned_judgment.skills.move dream --out DIR [--n N]
    python -m learned_judgment.skills.move run --student DIR --split {context|dev|test} --out DIR [--n N] [--run RUN]

run with --run RUN also merges the UI keys (seeds, scenario_ids, metrics, paired) and a section "closed_loop_<split>"
into <observed.runs_dir>/RUN/result.json and appends "evaluate" events to its events.jsonl.

Seeds. The scenarios of a split come from np.random.default_rng(np.random.SeedSequence([runtime.seed, code]))
(code: SPLIT_CODES); scenario i's episodes use attempt_seed i, so every candidate and both run methods face the same
noise. The bootstrap of run uses engine.streams.stream_rng(runtime.seed, "move/run/<split>/paired_bootstrap").

This module defines no SPEC_CLASS, so the SkillSpec registry (learned_judgment.skills) does not load it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import numbers
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from learned_judgment.config import ConfigError, config_digest, load_config, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.fileio import atomic_write_text
from learned_judgment.skills.ramp import (  # the generic config / file helpers of the template skill
    _count,
    _jsonable,
    _non_negative,
    _numbers,
    _pair,
    _positive,
    _value,
    _whole_periods,
    _write,
    _yaw,
)

__all__ = [
    "ACTION_FEATURE_NAMES",
    "FEATURE_NAMES",
    "KINDS",
    "OUTCOME_KINDS",
    "OUTCOME_NAMES",
    "SKILL",
    "SPLIT_CODES",
    "MoveSpec",
    "build_observed_spec",
    "candidate_command",
    "collect_dream",
    "collect_rows",
    "decode_features",
    "main",
    "objective",
    "physics_choice",
    "physics_guess",
    "run_episode",
    "run_student",
    "sample_move_scenarios",
    "scenario_features",
]

SKILL = "move"
KINDS = ("forward", "sideways", "turn")
FIELD_OF_KIND = {"forward": "vel_x", "sideways": "vel_y", "turn": "yaw_rate"}
TARGET_KEYS = {"forward": "forward_m", "sideways": "sideways_m", "turn": "turn_rad"}
UNITS = {"forward": "m", "sideways": "m", "turn": "rad"}
DISPLAY = {"forward": ("cm", 100.0), "sideways": ("cm", 100.0), "turn": ("deg", 180.0 / math.pi)}
FEATURE_NAMES = ("kind_forward", "kind_sideways", "kind_turn", "target_magnitude", "target_positive")
ACTION_FEATURE_NAMES = ("command_level", "hold_s")
OUTCOME_NAMES = ("error", "drift_m", "fall")
OUTCOME_KINDS = ("regression", "regression", "binary")
SPLIT_CODES = {"context": 1, "dev": 2, "test": 3, "dream": 4}
METHODS = ("physics_only", "snn")
_FIELDS = ("vel_x", "vel_y", "yaw_rate")  # the verified command fields by native index
_OBJECTIVE_KEYS = ("error", "drift", "fall", "spread")
_ROOT = "skills.move"
_OBSERVED = "observed.move"
_TOLERANCE = 1e-9
_YAW_CHECK_RAD = 1e-6
_ERROR_NOTE = ("error is |final value on the kind's axis - target| / scale (dimensionless), because spec.objective "
               "receives only the predicted outcomes; error_units is the same in m (forward, sideways) or rad (turn)")


# ----- config ----------------------------------------------------------------------------------------------------


def _kind_index(kind: Any) -> int:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}")
    return KINDS.index(kind)


def _period(config: Mapping[str, Any]) -> float:
    return _positive(config, "robot.walking_policy.command_schema.update_period_s")


def _schema(config: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = require(config, "robot.walking_policy.command_schema")
    fields = tuple(sorted(schema["fields"], key=lambda name: schema["fields"][name]["native_index"]))
    if fields != _FIELDS:
        raise ConfigError(f"move needs the command fields {list(_FIELDS)}; the schema has {list(fields)}")
    return schema


@dataclass(frozen=True)
class _Grid:
    names: tuple[str, ...]
    features: np.ndarray  # (C, 2) float64 read-only: command_level, hold_s


def _grid(config: Mapping[str, Any]) -> _Grid:
    """skills.move.candidates, level-major; every level checked on both range ends of every field."""
    from learned_judgment.env.commands import validate_command  # on use: learned_judgment.env loads the policy runtime

    schema = _schema(config)
    period = _period(config)
    levels = _numbers(config, f"{_ROOT}.candidates.command_levels")
    holds = _numbers(config, f"{_ROOT}.candidates.hold_s")
    for level in levels:
        if not 0.0 < level <= 1.0:
            raise ConfigError(f"'{_ROOT}.candidates.command_levels' must lie in (0, 1]; got {level}")
        for name in _FIELDS:
            for end in ("min", "max"):
                command = {f: 0.0 for f in _FIELDS} | {name: level * float(schema["fields"][name][end])}
                try:
                    validate_command(dict(command), schema)
                except ValueError as error:
                    raise ConfigError(f"level {level} gives {command}, not a verified command: {error}") from error
    for hold in holds:
        _whole_periods(f"{_ROOT}.candidates.hold_s", hold, period)
    names = tuple(f"L{level:.2f}_H{hold:.2f}" for level in levels for hold in holds)
    if len(set(names)) != len(names):
        raise ConfigError(f"'{_ROOT}.candidates' give candidate names that collide at 2 decimals: {names}")
    features = np.array([[level, hold] for level in levels for hold in holds], dtype=np.float64)
    features.setflags(write=False)
    return _Grid(names, features)


def _targets(config: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    key = f"{_ROOT}.targets"
    raw = require(config, key)
    if not isinstance(raw, Mapping) or set(raw) != set(TARGET_KEYS.values()):
        raise ConfigError(f"config.yaml key {key!r} must map exactly {list(TARGET_KEYS.values())}; got {raw!r}")
    return {kind: _pair(config, f"{key}.{TARGET_KEYS[kind]}", minimum=_TOLERANCE, strict=True) for kind in KINDS}


def _scales(config: Mapping[str, Any]) -> np.ndarray:
    """(3,) the high end of each kind's targets range, KINDS order."""
    targets = _targets(config)
    return np.array([targets[kind][1] for kind in KINDS], dtype=np.float64)


def _kind_weights(config: Mapping[str, Any]) -> np.ndarray:
    key = f"{_ROOT}.kind_weights"
    raw = require(config, key)
    if not isinstance(raw, Mapping) or set(raw) != set(KINDS):
        raise ConfigError(f"config.yaml key {key!r} must map exactly {list(KINDS)} to weights; got {raw!r}")
    weights = np.array([_non_negative(config, f"{key}.{kind}") for kind in KINDS], dtype=np.float64)
    if weights.sum() <= 0.0:
        raise ConfigError(f"config.yaml key {key!r} needs a positive weight")
    return weights / weights.sum()


def _outcome_bounds(config: Mapping[str, Any]) -> np.ndarray:
    key = f"{_ROOT}.outcome_bounds"
    bounds = np.array(
        [_pair(config, f"{key}.error", minimum=0.0, strict=True), _pair(config, f"{key}.drift_m", minimum=0.0, strict=True),
         (0.0, 1.0)],
        dtype=np.float64,
    )
    bounds.setflags(write=False)
    return bounds


def _physics(config: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    """(latency_s, track (3,) per unit command, KINDS order)."""
    key = f"{_ROOT}.physics"
    track = require(config, f"{key}.track")
    if not isinstance(track, Mapping) or set(track) != set(KINDS):
        raise ConfigError(f"config.yaml key '{key}.track' must map exactly {list(KINDS)}; got {track!r}")
    return _non_negative(config, f"{key}.latency_s"), np.array([_positive(config, f"{key}.track.{k}") for k in KINDS])


def _objective_weights(config: Mapping[str, Any]) -> dict[str, float]:
    key = f"{_ROOT}.objective_weights"
    weights = require(config, key)
    if not isinstance(weights, Mapping) or set(weights) != set(_OBJECTIVE_KEYS):
        raise ConfigError(f"config.yaml key {key!r} must map exactly {list(_OBJECTIVE_KEYS)} to weights; got {weights!r}")
    return {name: _non_negative(config, f"{key}.{name}") for name in _OBJECTIVE_KEYS}


def _tail_periods(config: Mapping[str, Any]) -> tuple[float, float]:
    """(settle_s, settle_tail_s), the tail a whole number of control periods."""
    settle = _non_negative(config, f"{_ROOT}.settle_s")
    tail = _positive(config, f"{_ROOT}.settle_tail_s")
    _whole_periods(f"{_ROOT}.settle_tail_s", tail, _period(config))
    return settle, tail


# ----- scenarios and features ------------------------------------------------------------------------------------


def sample_move_scenarios(config: Mapping[str, Any], rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    """``n`` scenarios (see the module docstring); no world needed."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator; got {type(rng).__name__}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be an int >= 0; got {n!r}")
    cumulative = np.cumsum(_kind_weights(config))
    targets = _targets(config)
    settle, _ = _tail_periods(config)
    scenarios = []
    for index in range(int(n)):
        u_kind, u_magnitude, u_sign = (float(u) for u in rng.random(3))
        kind = KINDS[min(int(np.searchsorted(cumulative, u_kind, side="right")), len(KINDS) - 1)]
        low, high = targets[kind]
        target = (1.0 if u_sign < 0.5 else -1.0) * (low + u_magnitude * (high - low))
        scenarios.append({"scenario_id": index, "kind": kind, "target": target,
                          "situation": {"dog_forward_speed_m_s": 0.0, "settle_s": settle}})
    return scenarios


def scenario_features(config: Mapping[str, Any], kind: str, target: float) -> np.ndarray:
    """(D,) float32 in [0, 1]: one-hot of the kind, |target| / scale (clipped to 1), 1.0 for target > 0."""
    k = _kind_index(kind)
    if isinstance(target, bool) or not isinstance(target, numbers.Real) or not math.isfinite(target) or target == 0.0:
        raise ValueError(f"target must be a finite non-zero number; got {target!r}")
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    features[k] = 1.0
    features[3] = min(1.0, abs(float(target)) / float(_scales(config)[k]))
    features[4] = 1.0 if target > 0.0 else 0.0
    return features


def decode_features(config: Mapping[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kind index (n,), signed target in m or rad (n,), scale (n,)) of features (n, D) in [0, 1]."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(array)):
        raise ValueError(f"features must be finite with shape (n, {len(FEATURE_NAMES)}); got {array.shape}")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("features must lie in [0, 1]")
    kind = np.argmax(array[:, :3], axis=1)
    scale = _scales(config)[kind]
    sign = np.where(array[:, 4] >= 0.5, 1.0, -1.0)
    return kind, sign * array[:, 3] * scale, scale


def candidate_command(config: Mapping[str, Any], kind: str, target: float, action: int) -> dict[str, float]:
    """The command of candidate ``action`` for a target: the kind's field at level x the range end on its side."""
    grid, schema = _grid(config), _schema(config)
    level = float(grid.features[int(action), 0])
    name = FIELD_OF_KIND[KINDS[_kind_index(kind)]]
    end = float(schema["fields"][name]["max" if target > 0.0 else "min"])
    return {f: 0.0 for f in _FIELDS} | {name: level * end}


# ----- episodes --------------------------------------------------------------------------------------------------


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def run_episode(
    world: Any, scenario: Mapping[str, Any], attempt_seed: int, policy: Callable[[np.ndarray], int]
) -> dict[str, Any]:
    """One complete episode on the real world: reset, one decision at t_s 0, the candidate's command for its hold_s,
    the zero command for settle_tail_s, finish_episode.

    Returns {"scenario_id", "kind", "target", "features" (D,) float32, "action", "command", "hold_s", "measured"
    {outcome: float}, "error_units" (m or rad), "final" {"forward_m", "sideways_m", "yaw_change_rad" (unwrapped),
    "yaw_change_wrapped_rad" (finish_episode's)}, "start" {"base_pos_m", "yaw_rad"} (the reset's observation),
    "elapsed_s", "final_speed_m_s", "fall_time_s"}.

    Raises ValueError for a bad argument or policy output, RuntimeError for a state MuJoCo flagged as bad or an
    unwrapped yaw that disagrees with finish_episode, ConfigError for a missing key.
    """
    config = world.config
    grid = _grid(config)
    _, tail = _tail_periods(config)
    period = _period(config)
    if abs(world.policy.control_dt - period) > 1e-12:
        raise ValueError(f"the world's control period {world.policy.control_dt} s is not update_period_s {period} s")
    if not isinstance(scenario, Mapping) or not isinstance(scenario.get("situation"), Mapping):
        raise ValueError(f"scenario must be a mapping with a 'situation' mapping; got {scenario!r}")
    if isinstance(attempt_seed, bool) or not isinstance(attempt_seed, (int, np.integer)) or attempt_seed < 0:
        raise ValueError(f"attempt_seed must be an int >= 0; got {attempt_seed!r}")
    if not callable(policy):
        raise TypeError(f"policy must be callable; got {type(policy).__name__}")
    kind, target = scenario["kind"], float(scenario["target"])
    k = _kind_index(kind)
    features = scenario_features(config, kind, target)
    c = len(grid.names)

    obs = world.reset(dict(scenario["situation"]), attempt_seed=int(attempt_seed))
    start = {"base_pos_m": np.asarray(obs["base_pos_m"], dtype=np.float64).tolist(), "yaw_rad": _yaw(obs["base_quat_wxyz"])}
    action = policy(features.copy())
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < c:
        raise ValueError(f"policy must return a candidate index in [0, {c}); got {action!r}")
    action = int(action)
    command = candidate_command(config, kind, target, action)
    hold = float(grid.features[action, 1])
    yaw = {"last": start["yaw_rad"], "total": 0.0}

    def on_period(observation: Mapping[str, Any]) -> None:
        now = _yaw(observation["base_quat_wxyz"])
        yaw["total"] += _wrap(now - yaw["last"])
        yaw["last"] = now

    world.step(command, hold, on_period)
    world.step({f: 0.0 for f in _FIELDS}, tail, on_period)
    outcome = world.finish_episode()
    if outcome["sim_unstable"]:
        raise RuntimeError("MuJoCo flagged a bad state in this episode (sim_unstable 1); its outcome is no measurement")
    wrapped = float(outcome["final_yaw_change_rad"])
    if abs(_wrap(yaw["total"] - wrapped)) > _YAW_CHECK_RAD:
        raise RuntimeError(f"the unwrapped yaw change {yaw['total']} rad disagrees with finish_episode's {wrapped} rad")
    forward, sideways = float(outcome["final_forward_m"]), float(outcome["final_sideways_m"])
    value = (forward, sideways, yaw["total"])[k]
    drift = (abs(sideways), abs(forward), math.hypot(forward, sideways))[k]
    error_units = abs(value - target)
    scale = float(_scales(config)[k])
    return {
        "scenario_id": scenario.get("scenario_id"),
        "kind": kind,
        "target": target,
        "features": features,
        "action": action,
        "command": command,
        "hold_s": hold,
        "measured": {"error": error_units / scale, "drift_m": drift, "fall": float(outcome["fall"])},
        "error_units": error_units,
        "final": {"forward_m": forward, "sideways_m": sideways, "yaw_change_rad": yaw["total"],
                  "yaw_change_wrapped_rad": wrapped},
        "start": start,
        "elapsed_s": float(outcome["elapsed_s"]),
        "final_speed_m_s": float(outcome["final_speed_m_s"]),
        "fall_time_s": outcome["fall_time_s"],
    }


# ----- physics guess and objective -------------------------------------------------------------------------------


def physics_guess(config: Mapping[str, Any], features: np.ndarray, action_indices: Sequence[int] | np.ndarray) -> np.ndarray:
    """Outcome guess (n, K) in OUTCOME_NAMES order from the features (n, D) and candidate indices (n,).

    travel = track[kind] x level x max(0, hold_s - latency_s) toward the target (skills.move.physics); error =
    |travel - |target|| / scale; drift_m 0; fall 0. Clipped to the outcome bounds.
    """
    grid = _grid(config)
    kind, target, scale = decode_features(config, features)
    n = kind.shape[0]
    actions = np.asarray(action_indices)
    if actions.shape != (n,):
        raise ValueError(f"features (n, D) and action_indices (n,) must agree; got {kind.shape}, {actions.shape}")
    if n and (not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions >= len(grid.names))):
        raise ValueError(f"action_indices must be ints in [0, {len(grid.names)})")
    latency, track = _physics(config)
    bounds = _outcome_bounds(config)
    command = grid.features[actions.astype(np.int64)] if n else np.zeros((0, 2))
    travel = track[kind] * command[:, 0] * np.maximum(0.0, command[:, 1] - latency)
    guess = np.zeros((n, len(OUTCOME_NAMES)), dtype=np.float64)
    guess[:, 0] = np.abs(travel - np.abs(target)) / scale
    return np.clip(guess, bounds[:, 0], bounds[:, 1])


def objective(config: Mapping[str, Any], predicted: OutcomeDistribution) -> np.ndarray:
    """Utility (n,), higher is better (weights skills.move.objective_weights, all >= 0)::

        -(error * mean error + drift * mean drift_m + fall * mean fall + spread * spread of error)

    error is already divided by the kind's scale (see the module docstring)."""
    if not isinstance(predicted, OutcomeDistribution):
        raise TypeError(f"predicted must be an OutcomeDistribution; got {type(predicted).__name__}")
    if tuple(predicted.names) != OUTCOME_NAMES:
        raise ValueError(f"predicted outcomes {predicted.names} are not {OUTCOME_NAMES}")
    w = _objective_weights(config)
    return -(
        w["error"] * predicted.mean_of("error")
        + w["drift"] * predicted.mean_of("drift_m")
        + w["fall"] * predicted.mean_of("fall")
        + w["spread"] * predicted.spread_of("error")
    )


def physics_choice(config: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """(n,) the physics rule: argmax over the candidates of the objective of the physics guess as a point mass."""
    array = np.asarray(features)
    n, c = array.shape[0], len(_grid(config).names)
    physics = physics_guess(config, np.repeat(array, c, axis=0), np.tile(np.arange(c), n))
    levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
    scores = objective(config, OutcomeDistribution.point(OUTCOME_NAMES, physics, levels)).reshape(n, c)
    return np.argmax(scores, axis=1)


@dataclass(frozen=True)
class MoveSpec:
    """What learners and scorers need of move (not a SkillSpec; the registry does not load it).

    name "move"; action_names (C,) level-major; action_features (C, 2) float64 read-only (command_level, hold_s);
    no_op_index None; outcome_names / outcome_kinds; outcome_bounds (K, 2) float64 read-only; feature_names.
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


def build_observed_spec(config: Mapping[str, Any]) -> MoveSpec:
    """The MoveSpec of config.yaml. Raises ConfigError for a missing or invalid key."""
    grid = _grid(config)
    _targets(config)
    _kind_weights(config)
    _physics(config)
    _objective_weights(config)
    _tail_periods(config)
    return MoveSpec(
        name=SKILL,
        action_names=grid.names,
        action_features=grid.features,
        action_feature_names=ACTION_FEATURE_NAMES,
        no_op_index=None,
        outcome_names=OUTCOME_NAMES,
        outcome_kinds=OUTCOME_KINDS,
        outcome_bounds=_outcome_bounds(config),
        feature_names=FEATURE_NAMES,
        config=config,
    )


# ----- student policy --------------------------------------------------------------------------------------------


class _StudentPolicy:
    """The exported move student (observed.save_student with frames None): argmax of the action head."""

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
                raise ValueError(f"{folder / 'student.json'} {key} is {student.get(key)!r}; the move spec needs {value!r}")
        self.network, _ = load_network(folder / "network")
        if self.network.n_inputs != d or self.network.n_outputs != c:
            raise ValueError(f"the network is {self.network.layer_sizes}; the move student needs {d} inputs, {c} outputs")
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
    from learned_judgment.env import Go2World  # on use: see _grid

    config = copy.deepcopy(dict(config))
    config["world"]["save_traces"] = False
    _WORKER["world"] = Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    _WORKER["student"] = None if student_dir is None else _StudentPolicy(student_dir, config)


def _episode_task(task: tuple) -> dict[str, Any]:
    """(scenario, label, index): index an int (constant candidate) or None (the student)."""
    scenario, label, index = task
    policy = _WORKER["student"] if index is None else (lambda _features, a=int(index): a)
    result = run_episode(_WORKER["world"], scenario, scenario["scenario_id"], policy)
    result["policy"] = label
    return result


@contextmanager
def _pool(config: Mapping[str, Any], student_dir: str | None = None) -> Iterator[Any]:
    workers = require(config, "observed.workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"config.yaml key 'observed.workers' must be an int >= 1; got {workers!r}")
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_worker_init, initargs=(config, student_dir)) as pool:
        yield pool


def _map(pool: Any, tasks: list[tuple], label: str, on_result: Callable[[int], None] | None = None) -> list[dict[str, Any]]:
    """_episode_task over tasks, results in task order, sparse progress lines."""
    every = max(1, len(tasks) // 10)
    results = []
    for result in pool.imap(_episode_task, tasks):
        results.append(result)
        if len(results) % every == 0 or len(results) == len(tasks):
            print(f"{label} {len(results)}/{len(tasks)}", flush=True)
            if on_result is not None:
                on_result(len(results))
    return results


def _split_plan(config: Mapping[str, Any], split: str, n: int) -> list[dict[str, Any]]:
    seed = require(config, "runtime.seed")
    return sample_move_scenarios(config, np.random.default_rng(np.random.SeedSequence([seed, SPLIT_CODES[split]])), n)


def _seeds(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "runtime_seed": require(config, "runtime.seed"),
        "split": split,
        "split_code": SPLIT_CODES[split],
        "scenarios": "np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code])), rng.random(3) per scenario",
        "attempt_seed": "scenario_id",
        "world_seed": require(config, "observed.world_seed"),
        "world_preset": require(config, "observed.world_preset"),
    }


def _meta(config: Mapping[str, Any], split: str, scenarios: list[dict[str, Any]], n_situations: int) -> dict[str, Any]:
    spec = build_observed_spec(config)
    kinds = Counter(s["kind"] for s in scenarios)
    return {
        "skill": SKILL,
        "split": split,
        "action_names": list(spec.action_names),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "outcome_names": list(spec.outcome_names),
        "outcome_kinds": list(spec.outcome_kinds),
        "outcome_bounds": spec.outcome_bounds.tolist(),
        "feature_names": list(FEATURE_NAMES),
        "frames": None,
        "n_situations": n_situations,
        "n_scenarios": len(scenarios),
        "seeds": _seeds(config, split),
        "world_seed": require(config, "observed.world_seed"),
        "world_preset": require(config, "observed.world_preset"),
        "config_digest": config_digest(config),
        "kind_counts": {kind: kinds.get(kind, 0) for kind in KINDS},
        "scales": {kind: float(scale) for kind, scale in zip(KINDS, _scales(config))},
        "units": dict(UNITS),
        "error_note": _ERROR_NOTE,
        "where": "simulation (MuJoCo Go2World, frozen walking policy) on this computer's CPU",
        "scenarios": scenarios,
    }


def collect_rows(config: Mapping[str, Any], split: str, out_dir: str | Path, n: int | None = None) -> dict[str, Any]:
    """Measure every candidate of every scenario of a split; write <out_dir>/rows.npz and rows.json; return the meta.

    n scenarios (default observed.move.<split>_scenarios). Raises RuntimeError when the candidates of a scenario saw
    another reset observation (the episodes are not deterministic).
    """
    if split not in ("context", "dev", "test"):
        raise ValueError(f"split must be context, dev or test; got {split!r}")
    n = _count("n", require(config, f"{_OBSERVED}.{split}_scenarios") if n is None else n)
    spec = build_observed_spec(config)
    c = len(spec.action_names)
    scenarios = _split_plan(config, split, n)
    print(f"rows: split {split}, {n} scenarios, {c} candidates: {n * c} episodes", flush=True)
    start = time.perf_counter()
    with _pool(config) as pool:
        results = _map(pool, [(s, "candidate", a) for s in scenarios for a in range(c)], "episodes:")
    wall_s = time.perf_counter() - start

    by_scenario: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        by_scenario.setdefault(r["scenario_id"], []).append(r)
    for sid, rows in by_scenario.items():
        if any(r["start"] != rows[0]["start"] for r in rows[1:]):
            raise RuntimeError(f"scenario {sid}: the candidates saw different reset observations; not deterministic")
    ordered = [r for sid in sorted(by_scenario) for r in sorted(by_scenario[sid], key=lambda r: r["action"])]
    scenario_index = np.array([r["scenario_id"] for r in ordered], dtype=np.int64)
    action = np.array([r["action"] for r in ordered], dtype=np.int64)
    features = np.array([r["features"] for r in ordered], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    measured = np.array([[r["measured"][k] for k in OUTCOME_NAMES] for r in ordered], dtype=np.float64)
    physics = physics_guess(config, features, action)
    for scenario in scenarios:
        own = sorted(by_scenario[scenario["scenario_id"]], key=lambda r: r["action"])
        scenario["error_units"] = [r["error_units"] for r in own]
        scenario["falls"] = int(sum(r["measured"]["fall"] for r in own))
    meta = _meta(config, split, scenarios, len(by_scenario))
    meta["wall_s"] = wall_s
    _write(Path(out_dir), "rows", meta, scenario_index=scenario_index, decision_time_s=np.zeros(len(ordered)),
           action=action, features=features, physics=physics, measured=measured,
           target=np.array([r["target"] for r in ordered], dtype=np.float64),
           error_units=np.array([r["error_units"] for r in ordered], dtype=np.float64),
           final=np.array([[r["final"][k] for k in ("forward_m", "sideways_m", "yaw_change_rad")] for r in ordered],
                          dtype=np.float64).reshape(-1, 3))
    episodes = len(results)
    print(f"rows {len(ordered)} from {len(by_scenario)} situations; kinds {meta['kind_counts']}; "
          f"falls {int(measured[:, 2].sum())}")
    print(f"wall time {wall_s:.1f} s for {episodes} episodes: {wall_s / max(episodes, 1):.3f} s per episode")
    print(f"wrote {Path(out_dir) / 'rows.npz'} and rows.json", flush=True)
    return meta


def collect_dream(config: Mapping[str, Any], out_dir: str | Path, n: int | None = None) -> dict[str, Any]:
    """Write <out_dir>/dream.npz and dream.json without a simulator; return the meta.

    n situations (default observed.move.dream_situations) drawn like the scenarios of a split (sample_move_scenarios)
    from the dream seed; the physics guess of every candidate.
    """
    n = _count("n", require(config, f"{_OBSERVED}.dream_situations") if n is None else n)
    spec = build_observed_spec(config)
    c, k = len(spec.action_names), len(OUTCOME_NAMES)
    scenarios = _split_plan(config, "dream", n)
    features = np.stack([scenario_features(config, s["kind"], s["target"]) for s in scenarios]).astype(np.float32)
    physics = physics_guess(config, np.repeat(features, c, axis=0), np.tile(np.arange(c), n)).reshape(n, c, k)
    meta = _meta(config, "dream", [], n)
    meta["kind_counts"] = {kind: sum(s["kind"] == kind for s in scenarios) for kind in KINDS}
    meta["sampling"] = {
        "simulator": "none: features from sampled requests, physics from physics_guess",
        "features": "requests sampled like a split's scenarios (sample_move_scenarios): kind by skills.move.kind_weights, "
                    "|target| uniform in skills.move.targets.<kind>, sign uniform; then scenario_features",
        "decision_time_s": "0.0 (one decision at the decision instant)",
    }
    _write(Path(out_dir), "dream", meta, scenario_index=np.arange(n, dtype=np.int64), decision_time_s=np.zeros(n),
           features=features, physics=physics)
    counts = Counter(spec.action_names[i] for i in physics_choice(config, features))
    print("physics-best candidate counts: " + ", ".join(f"{name} {counts[name]}" for name in spec.action_names if counts[name]))
    print(f"kinds {meta['kind_counts']}; wrote {Path(out_dir) / 'dream.npz'} and dream.json", flush=True)
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


def _paired_bootstrap(differences: Sequence[float], resamples: int, level: float, rng: np.random.Generator) -> dict[str, Any]:
    """Mean of per-scenario differences and its percentile bootstrap interval (resampling scenarios)."""
    d = np.asarray(differences, dtype=np.float64)
    if d.size == 0:
        return {"diff": None, "ci_low": None, "ci_high": None, "n": 0}
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        means[start:stop] = d[rng.integers(0, d.size, size=(stop - start, d.size))].mean(axis=1)
    low, high = np.quantile(means, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return {"diff": float(d.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(d.size)}


def _kind_summary(results: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    unit, factor = DISPLAY[kind]
    if not results:
        return {"n": 0, "unit": unit}
    error = np.array([r["error_units"] for r in results]) * factor
    drift = np.array([r["measured"]["drift_m"] for r in results]) * 100.0
    return {"n": len(results), "unit": unit, "mean_error": float(error.mean()), "max_error": float(error.max()),
            "median_error": float(np.median(error)), "mean_drift_cm": float(drift.mean()),
            "max_drift_cm": float(drift.max()), "falls": int(sum(r["measured"]["fall"] for r in results))}


def _summary(results: Sequence[Mapping[str, Any]], action_names: Sequence[str]) -> dict[str, Any]:
    return {
        "n": len(results),
        "mean_error_normalised": float(np.mean([r["measured"]["error"] for r in results])) if results else None,
        "falls": int(sum(r["measured"]["fall"] for r in results)),
        "by_kind": {kind: _kind_summary([r for r in results if r["kind"] == kind], kind) for kind in KINDS},
        "action_counts": {name: sum(r["action"] == i for r in results) for i, name in enumerate(action_names)},
    }


def _ui(summary: Mapping[str, Mapping[str, Any]], paired: Mapping[str, Any], level: float) -> tuple[list, list]:
    """The result.json UI keys "metrics" and "paired" (dodge_eval's layout)."""
    metrics = []
    for kind, name in (("forward", "mean_error_forward_cm"), ("sideways", "mean_error_sideways_cm"),
                       ("turn", "mean_error_turn_deg")):
        values = {}
        for method in METHODS:
            s = summary[method]["by_kind"][kind]
            values[method] = {"value": s.get("mean_error"), "ci_low": None, "ci_high": None, "n": s["n"]}
        metrics.append({"name": name, "unit": DISPLAY[kind][0], "higher_is_better": False, "values": values})
    values = {}
    for method in METHODS:
        n, falls = summary[method]["n"], summary[method]["falls"]
        low, high = _wilson(falls, n, level)
        values[method] = {"value": falls / n if n else None, "ci_low": low, "ci_high": high, "n": n}
    metrics.append({"name": "fall_rate", "unit": "fraction", "higher_is_better": False, "values": values})
    return metrics, [{"name": "error_normalised", "a": "snn", "b": "physics_only", "diff": paired["diff"],
                      "ci_low": paired["ci_low"], "ci_high": paired["ci_high"], "n": paired["n"]}]


def _merge_result(run_dir: Path, run_id: str, updates: Mapping[str, Any]) -> Path:
    """Merge top-level keys into <run_dir>/result.json (observed's file: other keys are kept)."""
    from learned_judgment.observed import _dumps, _utc_now

    path = run_dir / "result.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": run_id, "skill": SKILL}
    if result.get("run_id") != run_id or result.get("skill") != SKILL:
        raise ValueError(f"{path} belongs to run {result.get('run_id')!r} of skill {result.get('skill')!r}")
    result.update(_jsonable(dict(updates)))
    result["updated_utc"] = _utc_now().isoformat()
    atomic_write_text(path, _dumps(result, indent=1) + "\n")
    return path


def run_student(
    config: Mapping[str, Any], student_dir: str | Path, split: str, out_dir: str | Path, n: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Episodes of a split with the student's choice next to the physics rule's, same scenarios and attempt seeds;
    writes <out_dir>/run.json and returns it. With run_id, also result.json UI keys and "evaluate" events."""
    from learned_judgment.engine.streams import stream_rng
    from learned_judgment.observed import EventLog, _run_dir

    if split not in ("context", "dev", "test"):
        raise ValueError(f"split must be context, dev or test; got {split!r}")
    n = _count("n", require(config, f"{_OBSERVED}.{split}_scenarios") if n is None else n)
    spec = build_observed_spec(config)
    student = _StudentPolicy(student_dir, config)  # checked here before any worker starts
    level = float(require(config, "evaluation.bootstrap_confidence_level"))
    resamples = _count("evaluation.bootstrap_resamples", require(config, "evaluation.bootstrap_resamples"))
    seed = int(require(config, "runtime.seed"))
    scenarios = _split_plan(config, split, n)
    features = np.stack([scenario_features(config, s["kind"], s["target"]) for s in scenarios])
    rule = physics_choice(config, features)
    run_dir = None if run_id is None else _run_dir(config, run_id)
    events = None if run_dir is None else EventLog(run_dir / "events.jsonl", run_id, SKILL)
    section = f"closed_loop_{split}"
    common = {"student": Path(student_dir).resolve().as_posix(), "network_sha256": student.digest, "split": split, "n": n}
    if events is not None:
        _merge_result(run_dir, run_id, {section: {"state": "running", **common}})
        events.emit("evaluate", "running", done=0, total=2 * n, data={"split": split, "student": common["student"]})
    print(f"run: split {split}, {n} scenarios x {METHODS}", flush=True)
    try:
        start = time.perf_counter()
        tasks = [task for s, a in zip(scenarios, rule) for task in ((s, "physics_only", int(a)), (s, "snn", None))]
        with _pool(config, str(student_dir)) as pool:
            progress = None if events is None else (lambda done: events.emit("evaluate", "running", done=done, total=2 * n))
            results = _map(pool, tasks, "episodes:", progress)
        wall_s = time.perf_counter() - start
        records = []
        for scenario in scenarios:
            own = {r["policy"]: r for r in results if r["scenario_id"] == scenario["scenario_id"]}
            records.append({**scenario, **{m: {key: own[m][key] for key in (
                "action", "command", "hold_s", "measured", "error_units", "final", "elapsed_s", "final_speed_m_s")}
                for m in METHODS}})
            for m in METHODS:
                records[-1][m]["action_name"] = spec.action_names[own[m]["action"]]
        summary = {m: _summary([{**r[m], "kind": r["kind"]} for r in records], spec.action_names) for m in METHODS}
        stream = f"{SKILL}/run/{split}/paired_bootstrap"
        d = [r["snn"]["measured"]["error"] - r["physics_only"]["measured"]["error"] for r in records]
        paired = {"a": "snn", "b": "physics_only", "outcome": "error (normalised)",
                  **_paired_bootstrap(d, resamples, level, stream_rng(seed, stream, 0)),
                  "method": "percentile bootstrap over scenarios", "resamples": resamples, "level": level,
                  "stream": f"{stream}[0]"}
        seeds = {**_seeds(config, split), "bootstrap": paired["stream"]}
        out = {
            "skill": SKILL, "split": split, **common, "seeds": seeds, "config_digest": config_digest(config),
            "methods": {"physics_only": "argmax of the objective of the physics guess (point mass) per scenario",
                        "snn": "the student's argmax candidate (observed.student_readout, action head)"},
            "scales": {kind: float(scale) for kind, scale in zip(KINDS, _scales(config))}, "error_note": _ERROR_NOTE,
            "where": "simulation (MuJoCo Go2World); student inference on this computer's CPU (numpy)",
            "records": records, "summary": summary, "paired_error": paired, "wall_s": wall_s,
        }
        folder = Path(out_dir)
        atomic_write_text(folder / "run.json", json.dumps(_jsonable(out), indent=1, allow_nan=False) + "\n")
        if events is not None:
            metrics, ui_paired = _ui(summary, paired, level)
            _merge_result(run_dir, run_id, {
                section: {"state": "done", **common, "records": (folder / "run.json").as_posix(), "summary": summary,
                          "paired_error": paired, "wall_s": wall_s, "seeds": seeds, "where": out["where"]},
                "seeds": seeds, "scenario_ids": [s["scenario_id"] for s in scenarios], "metrics": metrics,
                "paired": ui_paired,
            })
    except Exception as error:
        if events is not None:
            events.emit("evaluate", "failed", data={"type": type(error).__name__, "message": str(error)})
            _merge_result(run_dir, run_id, {section: {"state": "failed", **common,
                                                      "error": {"type": type(error).__name__, "message": str(error)}}})
        raise
    headline = {m: {kind: {key: s["by_kind"][kind].get(key) for key in ("n", "unit", "mean_error", "max_error",
                                                                     "mean_drift_cm", "falls")}
                       for kind in KINDS} for m, s in summary.items()}
    if events is not None:
        events.emit("evaluate", "done", done=2 * n, total=2 * n, data={"split": split, "summary": headline,
                                                                      "paired_error": paired,
                                                                      "records": (folder / "run.json").as_posix()})
    for m in METHODS:
        print(f"{m}: " + "; ".join(
            f"{kind} n {h['n']} mean {h['mean_error']:.1f} max {h['max_error']:.1f} {h['unit']}, drift "
            f"{h['mean_drift_cm']:.1f} cm, falls {h['falls']}" for kind, h in headline[m].items() if h["n"]))
    print(f"paired error (snn - physics_only, normalised): {paired['diff']} [{paired['ci_low']}, {paired['ci_high']}]")
    print(f"wall time {wall_s:.1f} s; wrote {folder / 'run.json'}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (see the module docstring). Returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.skills.move", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rows = commands.add_parser("rows", help="every candidate of every scenario of a split, measured")
    rows.add_argument("--split", required=True, choices=("context", "dev", "test"))
    rows.add_argument("--out", required=True, help="output directory")
    rows.add_argument("--n", type=int, help="scenarios (default observed.move.<split>_scenarios)")
    dream = commands.add_parser("dream", help="dream situations sampled like the scenarios (no simulator)")
    dream.add_argument("--out", required=True, help="output directory")
    dream.add_argument("--n", type=int, help="situations (default observed.move.dream_situations)")
    run = commands.add_parser("run", help="episodes: the student next to the physics rule")
    run.add_argument("--student", required=True, help="exported student directory (observed train)")
    run.add_argument("--split", required=True, choices=("context", "dev", "test"))
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--n", type=int, help="scenarios (default observed.move.<split>_scenarios)")
    run.add_argument("--run", help="run id: also merge the UI keys into <observed.runs_dir>/RUN/result.json")
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "rows":
        collect_rows(config, args.split, args.out, args.n)
    elif args.command == "dream":
        collect_dream(config, args.out, args.n)
    else:
        run_student(config, args.student, args.split, args.out, args.n, args.run)
    return 0


if __name__ == "__main__":
    # Through the importable module, so spawned workers find _worker_init and the tasks by their module name.
    from learned_judgment.skills import move as _module

    raise SystemExit(_module.main())
