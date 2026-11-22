"""The shared attempt row and its on-disk format (engine lane, Person B).

Lane contract
-------------
Every actual Go2 MuJoCo attempt (``source == "real"``) and every teacher-generated decision
(``source == "dream"``) is stored as one :class:`AttemptRow` with the MISSION.md fields::

    run_id, seed, skill, world_preset, source, situation, action,
    physics_guess, measured_outcome, residual, timestamp_utc   (+ teacher, dream rows only)

    residual = measured_outcome - physics_guess, outcome by outcome

A dream is never a measurement: its ``measured_outcome`` and ``residual`` are ``None`` and its
``teacher`` summary says which method chose the action and what it predicted. Rows are
validated when built and when read. Nothing in this module runs the simulator or a model.

On-disk format: JSON Lines
--------------------------
One row per line, UTF-8, ``\\n`` line ends. Each line is the canonical JSON of one object::

    json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

with exactly these keys:

    run_id            str          run identifier
    seed              int >= 0     real rows: the attempt seed of the episode, i.e. the second argument
                                   of ``world.reset(situation, attempt_seed)``. The world's own seed
                                   (the hidden world of the whole run) is recorded by the run, not
                                   per row. Dream rows: the seed of the dream run.
    skill             str          SkillSpec name
    world_preset      str          declared preset name (config.yaml world.presets)
    source            str          "real" | "dream"
    situation         object       {spec.situation name: float | level string}
    action            object       {spec.actions name: float | level string}
    physics_guess     object       {spec.outcomes name: float}
    measured_outcome  object|null  {spec.outcomes name: float}; null exactly for "dream"
    residual          object|null  {spec.outcomes name: measured - physics}; null exactly for "dream"
    timestamp_utc     str          ISO 8601 with a zero UTC offset, supplied by the caller
    teacher           object|null  dream rows: the keys of TEACHER_KEYS; null for real rows

Attempt identity: for ``source == "real"`` the pair ``(run_id, seed)`` names one attempt and is
unique. :func:`append_rows` refuses a batch that repeats a pair or collides with a pair already
in the file, and :func:`rows_to_arrays` refuses rows that repeat a pair. Dream rows carry no
such identity (one dream run writes many rows with the same run_id and seed).

Values are keyed by name, so the sorted on-disk key order carries no meaning: the fixed
outcome order of a skill is ``spec.outcomes`` and :func:`rows_to_arrays` restores it.

The stored ``physics_guess`` is the physics of the spec that wrote the row. Rows carry no spec
version, so :func:`rows_to_arrays` recomputes ``spec.physics_guess`` for every real row and,
when the formula or one of its constants has changed since, either refuses the rows
(:class:`StalePhysicsError`) or recomputes physics and residuals from the stored measured
outcome, as the caller chooses. The file itself is never rewritten.

Categorical values are stored as level strings. Floats use Python's shortest round-trip
repr, so reading returns the identical float64 bits; non-finite values are rejected on
write and on read (proof: experiments/proofs/B_students_eval_contract/p10_persist and
p10b_jsonl_pandas -- read with ``json.loads`` per line, never ``pandas.read_json`` defaults).

A file has a single writer (the process that runs the episodes); concurrent appends from
several processes are not safe on Windows (p10_persist). Each append re-validates every row
from the exact bytes it is about to write, then flushes and fsyncs; appending to a file whose
last line is incomplete is refused.

Teacher summary of a dream row (``teacher`` object)
---------------------------------------------------
    method            str         label of what produced the scores (e.g. "teacher")
    request_hash      str|null    cache key of the last residual-model request behind the
                                  decision: the last entry of ``request_hashes``, null when empty
    request_hashes    list[str]   cache keys of every residual-model request behind the decision,
                                  in request order (several when the model sends one request per
                                  outcome or splits a large query set); empty when none was made
    n_candidates      int         C, the number of candidates scored for this situation
    chosen_index      int         index of the chosen candidate, in [0, C)
    physics_index     int         index of the candidate physics alone would choose
    chosen_score      float       objective of the chosen candidate
    physics_score     float       objective of the physics candidate under the same prediction
    physics_action    object      {action dim: value} of the physics candidate
    predicted_mean    object      {outcome: float} at the chosen candidate (binary: probability)
    predicted_median  object      {outcome: float} at the chosen candidate
    predicted_spread  object      {outcome: float >= 0} at the chosen candidate
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from learned_judgment.engine.types import Decision
from learned_judgment.skills.base import SkillSpec

__all__ = [
    "PHYSICS_MISMATCH_POLICIES",
    "SOURCES",
    "TEACHER_KEYS",
    "AttemptRow",
    "RowArrays",
    "StalePhysicsError",
    "append_rows",
    "build_dream_row",
    "build_real_row",
    "check_new_attempts",
    "decision_request_group",
    "read_rows",
    "rows_to_arrays",
    "validate_row",
]

SOURCES = ("real", "dream")
PHYSICS_MISMATCH_POLICIES = ("error", "recompute")
TEACHER_KEYS = (
    "method",
    "request_hash",
    "request_hashes",
    "n_candidates",
    "chosen_index",
    "physics_index",
    "chosen_score",
    "physics_score",
    "physics_action",
    "predicted_mean",
    "predicted_median",
    "predicted_spread",
)


class StalePhysicsError(ValueError):
    """A stored row's ``physics_guess`` differs from what the current SkillSpec computes for the
    same situation and action: the physics formula or one of its constants changed after the
    row was written, so its stored residual is relative to another physics."""


# --------------------------------------------------------------------------------------
# Value checks (structural only: types, finiteness, key sets)
# --------------------------------------------------------------------------------------


def _text(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string; got {value!r}")
    return value


def _count(field: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an int; got {type(value).__name__}")
    if int(value) < minimum:
        raise ValueError(f"{field} must be >= {minimum}; got {value}")
    return int(value)


def _finite(field: str, value: Any) -> float:
    # float covers np.float64 (a float subclass); ints and bools are rejected so that a
    # number written as an integer cannot pass for a float64 value.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, float):
        raise ValueError(f"{field} must be a float; got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite; got {number}")
    return number


def _value_map(field: str, values: Any, *, allow_levels: bool) -> dict[str, float | str]:
    """Copy of a non-empty {name: float} mapping (level strings allowed when ``allow_levels``)."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field} must be a non-empty mapping")
    copy: dict[str, float | str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field} has a key that is not a non-empty string: {key!r}")
        if allow_levels and isinstance(value, str):
            copy[key] = value
        else:
            copy[key] = _finite(f"{field}[{key!r}]", value)
    return copy


def _same_keys(field: str, values: Mapping[str, Any], expected: Sequence[str]) -> None:
    if set(values) != set(expected):
        missing = sorted(set(expected) - set(values))
        extra = sorted(set(values) - set(expected))
        raise ValueError(f"{field} keys do not match; missing={missing} extra={extra}")


def _utc_timestamp(value: Any) -> str:
    text = _text("timestamp_utc", value)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"timestamp_utc {text!r} is not ISO 8601") from None
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp_utc {text!r} must carry a zero UTC offset (e.g. '+00:00' or 'Z')")
    return text


def _request_hashes(field: str, values: Any) -> list[str]:
    """Copy of a list / tuple of unique non-empty strings (possibly empty)."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a list of request hashes; got {type(values).__name__}")
    hashes = [_text(f"{field}[{position}]", value) for position, value in enumerate(values)]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{field} repeats a request hash: {hashes}")
    return hashes


def _teacher_summary(teacher: Any, outcome_keys: Sequence[str], action_keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(teacher, Mapping):
        raise ValueError("teacher must be a mapping for a dream row")
    _same_keys("teacher", teacher, TEACHER_KEYS)
    n_candidates = _count("teacher.n_candidates", teacher["n_candidates"], minimum=1)
    hashes = _request_hashes("teacher.request_hashes", teacher["request_hashes"])
    last = hashes[-1] if hashes else None
    if teacher["request_hash"] != last:
        raise ValueError(
            f"teacher.request_hash {teacher['request_hash']!r} must be the last of teacher.request_hashes "
            f"({last!r}; null when no request was made)"
        )
    summary: dict[str, Any] = {
        "method": _text("teacher.method", teacher["method"]),
        "request_hash": last,
        "request_hashes": hashes,
        "n_candidates": n_candidates,
    }
    for key in ("chosen_index", "physics_index"):
        index = _count(f"teacher.{key}", teacher[key], minimum=0)
        if index >= n_candidates:
            raise ValueError(f"teacher.{key}={index} is outside [0, {n_candidates})")
        summary[key] = index
    for key in ("chosen_score", "physics_score"):
        summary[key] = _finite(f"teacher.{key}", teacher[key])
    physics_action = _value_map("teacher.physics_action", teacher["physics_action"], allow_levels=True)
    _same_keys("teacher.physics_action", physics_action, action_keys)
    summary["physics_action"] = physics_action
    for key in ("predicted_mean", "predicted_median", "predicted_spread"):
        values = _value_map(f"teacher.{key}", teacher[key], allow_levels=False)
        _same_keys(f"teacher.{key}", values, outcome_keys)
        summary[key] = values
    negative = sorted(name for name, value in summary["predicted_spread"].items() if value < 0.0)
    if negative:
        raise ValueError(f"teacher.predicted_spread is negative for {negative}")
    return summary


# --------------------------------------------------------------------------------------
# The row
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptRow:
    """One shared attempt row. See the module docstring for field meanings and the format.

    Construction checks everything that does not need the SkillSpec: types, finiteness,
    consistent key sets, the source / measured / residual / teacher rules, and
    ``residual == measured_outcome - physics_guess`` exactly. :func:`validate_row` adds the
    SkillSpec checks (names, levels, bounds).
    """

    run_id: str
    seed: int
    skill: str
    world_preset: str
    source: str
    situation: dict[str, float | str]
    action: dict[str, float | str]
    physics_guess: dict[str, float]
    measured_outcome: dict[str, float] | None
    residual: dict[str, float] | None
    timestamp_utc: str
    teacher: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "run_id", _text("run_id", self.run_id))
        set_(self, "seed", _count("seed", self.seed, minimum=0))
        set_(self, "skill", _text("skill", self.skill))
        set_(self, "world_preset", _text("world_preset", self.world_preset))
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}; got {self.source!r}")
        set_(self, "situation", _value_map("situation", self.situation, allow_levels=True))
        set_(self, "action", _value_map("action", self.action, allow_levels=True))
        physics = _value_map("physics_guess", self.physics_guess, allow_levels=False)
        set_(self, "physics_guess", physics)
        set_(self, "timestamp_utc", _utc_timestamp(self.timestamp_utc))

        if self.source == "real":
            if self.measured_outcome is None or self.residual is None:
                raise ValueError("a real row needs measured_outcome and residual")
            if self.teacher is not None:
                raise ValueError("a real row must not carry a teacher summary")
            measured = _value_map("measured_outcome", self.measured_outcome, allow_levels=False)
            residual = _value_map("residual", self.residual, allow_levels=False)
            _same_keys("measured_outcome", measured, tuple(physics))
            _same_keys("residual", residual, tuple(physics))
            wrong = sorted(name for name in physics if residual[name] != measured[name] - physics[name])
            if wrong:
                raise ValueError(f"residual != measured_outcome - physics_guess for {wrong}")
            set_(self, "measured_outcome", measured)
            set_(self, "residual", residual)
        else:
            if self.measured_outcome is not None or self.residual is not None:
                raise ValueError("a dream row is not a measurement: measured_outcome and residual must be None")
            if self.teacher is None:
                raise ValueError("a dream row needs its teacher summary")
            set_(self, "teacher", _teacher_summary(self.teacher, tuple(physics), tuple(self.action)))

    def to_json_line(self) -> bytes:
        """Canonical JSON of the row plus ``\\n``, UTF-8 encoded."""
        record = {item.name: getattr(self, item.name) for item in fields(self)}
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return (text + "\n").encode("utf-8")


_ROW_KEYS = frozenset(item.name for item in fields(AttemptRow))


@dataclass(frozen=True)
class _Encoded:
    situation: np.ndarray  # (S,)
    action: np.ndarray  # (A,)
    physics: np.ndarray  # (K,)
    measured: np.ndarray | None  # (K,)
    residual: np.ndarray | None  # (K,)


def _encode(spec: SkillSpec, row: AttemptRow) -> _Encoded:
    if row.skill != spec.name:
        raise ValueError(f"row belongs to skill {row.skill!r}, not {spec.name!r}")
    situation = spec.encode_situation(row.situation)
    action = spec.encode_action(row.action)
    _same_keys("physics_guess", row.physics_guess, spec.outcome_names)
    physics = np.array([row.physics_guess[name] for name in spec.outcome_names], dtype=np.float64)
    outside = [
        name
        for name, value, low, high in zip(spec.outcome_names, physics, spec.outcome_low, spec.outcome_high)
        if not low <= value <= high
    ]
    if outside:
        raise ValueError(f"physics_guess leaves the outcome bounds for {outside}")
    if row.source == "real":
        measured = spec.encode_outcome(row.measured_outcome)
        residual = np.array([row.residual[name] for name in spec.outcome_names], dtype=np.float64)
        return _Encoded(situation, action, physics, measured, residual)
    spec.encode_action(row.teacher["physics_action"])
    return _Encoded(situation, action, physics, None, None)


def validate_row(spec: SkillSpec, row: AttemptRow) -> None:
    """Raise ValueError unless ``row`` matches ``spec`` exactly: skill name; situation, action
    and outcome names (no missing, no extra); categorical levels; situation and action
    bounds; physics guess inside the outcome bounds; a real row's measured outcome
    (binary outcomes 0/1); a dream row's physics action.

    It does not compare the stored physics guess with the current ``spec.physics_guess``, so
    a file written under an older physics stays readable; :func:`rows_to_arrays` does that
    comparison before any row can become a residual target."""
    _encode(spec, row)


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def _vector(field: str, values: np.ndarray, width: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (width,):
        raise ValueError(f"{field} must have shape ({width},); got {vector.shape}")
    return vector


def _outcome_map(spec: SkillSpec, values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(spec.outcome_names, values)}


def _physics_at(spec: SkillSpec, situation: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(K,) ``spec.physics_guess`` of one encoded situation (S,) and action (A,), evaluated as a
    one-row batch. Both :func:`build_real_row` and :func:`rows_to_arrays` use this exact call,
    so an unchanged spec reproduces a stored physics guess bit for bit."""
    physics = np.asarray(spec.physics_guess(situation[None, :], action[None, :]), dtype=np.float64)
    if physics.shape != (1, spec.n_outcome):
        raise ValueError(f"physics_guess must return (1, {spec.n_outcome}); got {physics.shape}")
    return physics[0]


def decision_request_group(decision: Decision) -> tuple[str, ...]:
    """Every provider request key behind ``decision``, in request order.

    ``decision.request_hashes`` when it is set; otherwise ``(decision.request_hash,)`` when a
    single hash was recorded; ``()`` when no request was made. Raises ValueError when the
    decision names a ``request_hash`` that is not among its ``request_hashes``.
    """
    group = tuple(decision.request_hashes)
    if not group and decision.request_hash is not None:
        group = (decision.request_hash,)
    if decision.request_hash is not None and decision.request_hash not in group:
        raise ValueError(
            f"decision.request_hash {decision.request_hash!r} is not one of its request_hashes {list(group)}"
        )
    return group


def build_real_row(
    spec: SkillSpec,
    *,
    run_id: str,
    seed: int,
    world_preset: str,
    situation: np.ndarray,
    action: np.ndarray,
    measured_outcome: Mapping[str, float],
    timestamp_utc: str,
) -> AttemptRow:
    """Row of one actual episode.

    Args:
        situation:        (S,) encoded situation the episode was run in.
        action:           (A,) encoded action that was executed; must be inside the spec bounds.
        measured_outcome: what ``spec.episode`` returned, keyed by outcome name. Encoded with
                          ``spec.encode_outcome``: a missing or extra key is rejected.

    ``physics_guess`` is ``spec.physics_guess`` of this (situation, action) -- a pure function
    of the spec -- and ``residual = measured - physics`` in the spec's outcome order. ``seed``
    is the attempt seed the episode was reset with.
    """
    situation = _vector("situation", situation, spec.n_situation)
    action = _vector("action", action, spec.n_action)
    if not bool(spec.action_in_bounds(action[None, :])[0]):
        raise ValueError(f"action {action.tolist()} is outside the declared bounds of skill {spec.name!r}")
    physics = _physics_at(spec, situation, action)
    measured = spec.encode_outcome(measured_outcome)
    residual = measured - physics
    row = AttemptRow(
        run_id=run_id,
        seed=seed,
        skill=spec.name,
        world_preset=world_preset,
        source="real",
        situation=spec.decode_situation(situation),
        action=spec.decode_action(action),
        physics_guess=_outcome_map(spec, physics),
        measured_outcome=_outcome_map(spec, measured),
        residual=_outcome_map(spec, residual),
        timestamp_utc=timestamp_utc,
    )
    validate_row(spec, row)
    return row


def _dream_row(
    spec: SkillSpec,
    *,
    run_id: str,
    seed: int,
    world_preset: str,
    timestamp_utc: str,
    method: str,
    request_hashes: Sequence[str],
    situation: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    chosen_index: int,
    physics_index: int,
    chosen_physics: np.ndarray,
    chosen_mean: np.ndarray,
    chosen_median: np.ndarray,
    chosen_spread: np.ndarray,
) -> AttemptRow:
    """Dream row from arrays: situation (S,), candidates (C, A), scores (C,), and the
    physics guess / predicted mean / median / spread (K,) at the chosen candidate.
    ``request_hashes`` are every request key behind the decision, in request order.
    Shared by :func:`build_dream_row` and ``dream.dream_rows`` so both write one format."""
    candidates = np.asarray(candidates, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != spec.n_action or scores.shape != (candidates.shape[0],):
        raise ValueError(f"candidates (C, {spec.n_action}) and scores (C,) disagree: {candidates.shape}, {scores.shape}")
    hashes = list(request_hashes)
    teacher = {
        "method": method,
        "request_hash": hashes[-1] if hashes else None,
        "request_hashes": hashes,
        "n_candidates": int(candidates.shape[0]),
        "chosen_index": int(chosen_index),
        "physics_index": int(physics_index),
        "chosen_score": float(scores[chosen_index]),
        "physics_score": float(scores[physics_index]),
        "physics_action": spec.decode_action(candidates[physics_index]),
        "predicted_mean": _outcome_map(spec, _vector("chosen_mean", chosen_mean, spec.n_outcome)),
        "predicted_median": _outcome_map(spec, _vector("chosen_median", chosen_median, spec.n_outcome)),
        "predicted_spread": _outcome_map(spec, _vector("chosen_spread", chosen_spread, spec.n_outcome)),
    }
    row = AttemptRow(
        run_id=run_id,
        seed=seed,
        skill=spec.name,
        world_preset=world_preset,
        source="dream",
        situation=spec.decode_situation(_vector("situation", situation, spec.n_situation)),
        action=spec.decode_action(candidates[chosen_index]),
        physics_guess=_outcome_map(spec, _vector("chosen_physics", chosen_physics, spec.n_outcome)),
        measured_outcome=None,
        residual=None,
        timestamp_utc=timestamp_utc,
        teacher=teacher,
    )
    validate_row(spec, row)
    return row


def build_dream_row(
    spec: SkillSpec,
    *,
    run_id: str,
    seed: int,
    world_preset: str,
    decision: Decision,
    timestamp_utc: str,
) -> AttemptRow:
    """Row of one teacher decision (``source == "dream"``): the chosen action, the physics
    guess at that action, and the teacher summary, which lists every provider request behind
    the decision (:func:`decision_request_group`). It carries no measured outcome."""
    if decision.skill != spec.name:
        raise ValueError(f"decision belongs to skill {decision.skill!r}, not {spec.name!r}")
    if decision.predicted.names != spec.outcome_names:
        raise ValueError(f"decision outcomes {decision.predicted.names} differ from {spec.outcome_names}")
    chosen = decision.chosen_index
    return _dream_row(
        spec,
        run_id=run_id,
        seed=seed,
        world_preset=world_preset,
        timestamp_utc=timestamp_utc,
        method=decision.method,
        request_hashes=decision_request_group(decision),
        situation=decision.situation,
        candidates=decision.candidates,
        scores=decision.scores,
        chosen_index=chosen,
        physics_index=decision.physics_index,
        chosen_physics=decision.predicted.physics[chosen],
        chosen_mean=decision.predicted.mean[chosen],
        chosen_median=decision.predicted.median[chosen],
        chosen_spread=decision.predicted.spread[chosen],
    )


# --------------------------------------------------------------------------------------
# JSON Lines persistence
# --------------------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate key {key!r}")
        record[key] = value
    return record


def _parse_line(line: bytes) -> AttemptRow:
    """One row from the bytes of one line (without its ``\\n``), validated as on read."""
    record = json.loads(line.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates)
    if not isinstance(record, dict):
        raise ValueError("a row must be a JSON object")
    if set(record) != _ROW_KEYS:
        missing = sorted(_ROW_KEYS - set(record))
        extra = sorted(set(record) - _ROW_KEYS)
        raise ValueError(f"row keys do not match; missing={missing} extra={extra}")
    return AttemptRow(**record)


def _check_complete_last_line(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                raise ValueError(f"{path} ends with an incomplete line; repair it before appending")


def _recorded_attempts(path: Path) -> set[tuple[str, int]]:
    """(run_id, seed) of every real row already in ``path``; empty when the file does not exist.
    The whole file is read and validated, so a damaged file is never appended to."""
    if not path.exists():
        return set()
    _check_complete_last_line(path)
    return {(row.run_id, row.seed) for row in read_rows(path) if row.source == "real"}


def _clash_message(path: Path, clash: Sequence[tuple[str, int]]) -> str:
    shown = ", ".join(f"(run_id={run_id!r}, seed={seed})" for run_id, seed in clash[:5])
    more = f" and {len(clash) - 5} more" if len(clash) > 5 else ""
    return (
        f"{path} already holds the real attempts {shown}{more}: a (run_id, seed) pair names one attempt, "
        "so a rerun needs a new run_id (or new attempt seeds)"
    )


def check_new_attempts(
    path: Path, *, run_id: Any, world_preset: Any, seeds: Sequence[Any], timestamp_utc: Any
) -> None:
    """Validate the identity of real attempts that are about to be run, BEFORE the first episode.

    Applies the checks :class:`AttemptRow` would apply after each episode, so that an invalid
    value stops a run before the world is stepped: ``run_id`` and ``world_preset`` non-empty
    strings, every attempt seed an int >= 0, ``timestamp_utc`` (one sample of the run's clock)
    ISO 8601 with a zero UTC offset. Also refuses repeated seeds, and -- when ``path`` exists --
    a file whose last line is incomplete, a file that does not read back as valid rows, and any
    ``(run_id, seed)`` already recorded in it. Raises ValueError naming the offending value.
    """
    _text("run_id", run_id)
    _text("world_preset", world_preset)
    _utc_timestamp(timestamp_utc)
    first: dict[int, int] = {}
    for position, seed in enumerate(seeds):
        value = _count(f"attempt seed [{position}]", seed, minimum=0)
        if value in first:
            raise ValueError(
                f"attempt seed [{position}] = {value} repeats attempt seed [{first[value]}]: "
                "(run_id, seed) must name exactly one attempt"
            )
        first[value] = position
    clash = sorted({(run_id, seed) for seed in first} & _recorded_attempts(Path(path)))
    if clash:
        raise ValueError(_clash_message(Path(path), clash))


def append_rows(path: Path, rows: Sequence[AttemptRow], *, spec: SkillSpec | None) -> None:
    """Append ``rows`` to the JSON Lines file ``path`` (created with its parent directories).

    Args:
        spec: the SkillSpec every row must match (:func:`validate_row`), or ``None`` for a
              file that holds rows of several skills (structural checks only). Required, so a
              caller decides explicitly.

    Immediately before writing, every row is validated again from the exact bytes that would
    be written: the line is parsed back and rebuilt as an :class:`AttemptRow` (so a row whose
    dicts were changed after it was built is refused), it must re-serialize to the same bytes,
    and it is checked against ``spec`` when one is given. A real row's ``(run_id, seed)`` must
    be unique: a batch that repeats a pair, or that holds a pair already in the file, is
    refused. Any refusal raises ValueError and writes nothing. Appending to a file whose last
    line is incomplete is refused. The bytes are flushed and fsynced. Single writer per file.
    """
    path = Path(path)
    if spec is not None and not isinstance(spec, SkillSpec):
        raise TypeError(f"spec must be a SkillSpec or None; got {type(spec).__name__}")
    lines: list[bytes] = []
    batch: dict[tuple[str, int], int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, AttemptRow):
            raise TypeError(f"rows[{position}] is {type(row).__name__}, not AttemptRow")
        try:
            line = row.to_json_line()
            copy = _parse_line(line[:-1])
            if copy.to_json_line() != line:
                raise ValueError("the row does not re-serialize to the same bytes")
            if spec is not None:
                validate_row(spec, copy)
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            raise ValueError(f"rows[{position}] fails validation immediately before writing: {error}") from error
        if copy.source == "real":
            key = (copy.run_id, copy.seed)
            if key in batch:
                raise ValueError(
                    f"rows[{position}] repeats the attempt (run_id={key[0]!r}, seed={key[1]}) of rows[{batch[key]}]"
                )
            batch[key] = position
        lines.append(line)
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    clash = sorted(set(batch) & _recorded_attempts(path))
    if clash:
        raise ValueError(_clash_message(path, clash))
    with open(path, "ab") as handle:
        handle.write(b"".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path, *, spec: SkillSpec | None = None) -> list[AttemptRow]:
    """Read every row of the JSON Lines file ``path`` with ``json.loads`` per line.

    Each line must hold exactly the row keys and pass :class:`AttemptRow` validation; NaN,
    Infinity, duplicate keys, empty lines and an incomplete last line are rejected with the
    line number. When ``spec`` is given every row is also checked with :func:`validate_row`
    (a file holding rows of several skills is read without it and validated per skill).
    Repeated attempts are not refused here (a damaged file stays inspectable);
    :func:`rows_to_arrays` refuses them before they can become residual targets.
    """
    path = Path(path)
    data = path.read_bytes()
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise ValueError(f"{path}: the last line is incomplete (no trailing newline)")
    rows: list[AttemptRow] = []
    for number, line in enumerate(data.split(b"\n")[:-1], start=1):
        where = f"{path}:{number}"
        if not line.strip():
            raise ValueError(f"{where}: empty line")
        try:
            row = _parse_line(line)
            if spec is not None:
                validate_row(spec, row)
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            raise ValueError(f"{where}: {error}") from error
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Rows -> arrays
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RowArrays:
    """Arrays of ``n`` real rows in row order, in the spec's column orders (float64).

    Attributes:
        situations:         (n, S)
        actions:            (n, A)
        physics:            (n, K) physics guess of the CURRENT spec (equal to the stored one,
                            except at the rows listed in ``stale_physics_rows``)
        measured:           (n, K) measured outcome, as stored
        residuals:          (n, K) measured - physics
        world_presets:      n declared world preset names
        stale_physics_rows: positions (in the given rows) whose stored physics guess differed
                            from the current spec and was recomputed (policy "recompute");
                            always empty under policy "error"
    """

    situations: np.ndarray
    actions: np.ndarray
    physics: np.ndarray
    measured: np.ndarray
    residuals: np.ndarray
    world_presets: list[str]
    stale_physics_rows: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return int(self.situations.shape[0])

    @property
    def n_stale_physics(self) -> int:
        """How many rows had a stored physics guess that differs from the current spec."""
        return len(self.stale_physics_rows)


def rows_to_arrays(spec: SkillSpec, rows: Sequence[AttemptRow], *, on_physics_mismatch: str) -> RowArrays:
    """Validate ``rows`` against ``spec`` and stack them; real rows only.

    For every row, ``spec.physics_guess`` is recomputed exactly as :func:`build_real_row`
    computed it (a one-row batch of the stored situation and action) and compared with the
    stored ``physics_guess``, bit for bit. ``on_physics_mismatch`` (required) says what a
    difference means:

        "error"      raise :class:`StalePhysicsError` naming the first row whose stored physics
                     differs: its residual was computed against another physics.
        "recompute"  use the current physics and ``residual = stored measured - current
                     physics`` for that row, and list it in ``RowArrays.stale_physics_rows``.
                     The file is never rewritten.

    Raises ValueError for a dream row (a dream is not a measurement and never becomes a
    residual target), for two rows with the same ``(run_id, seed)`` (one attempt counted
    twice), for any row that fails :func:`validate_row`, and when a recomputed physics guess
    is not finite or leaves the outcome bounds. Zero rows give arrays with zero rows and the
    spec's widths.
    """
    if not isinstance(on_physics_mismatch, str) or on_physics_mismatch not in PHYSICS_MISMATCH_POLICIES:
        raise ValueError(
            f"on_physics_mismatch must be one of {PHYSICS_MISMATCH_POLICIES}; got {on_physics_mismatch!r}"
        )
    encoded: list[_Encoded] = []
    presets: list[str] = []
    stale: list[int] = []
    seen: dict[tuple[str, int], int] = {}
    low, high = spec.outcome_low, spec.outcome_high
    for position, row in enumerate(rows):
        if not isinstance(row, AttemptRow):
            raise TypeError(f"rows[{position}] is {type(row).__name__}, not AttemptRow")
        if row.source != "real":
            raise ValueError(f"rows[{position}] has source {row.source!r}; only real rows become arrays")
        key = (row.run_id, row.seed)
        if key in seen:
            raise ValueError(
                f"rows[{position}] repeats the attempt (run_id={row.run_id!r}, seed={row.seed}) of rows[{seen[key]}]: "
                "one attempt would be counted twice"
            )
        seen[key] = position
        try:
            item = _encode(spec, row)
            current = _physics_at(spec, item.situation, item.action)
        except ValueError as error:
            raise ValueError(f"rows[{position}]: {error}") from error
        if not np.array_equal(current, item.physics):
            where = f"rows[{position}] (run_id={row.run_id!r}, seed={row.seed})"
            if on_physics_mismatch == "error":
                raise StalePhysicsError(
                    f"{where}: stored physics_guess {item.physics.tolist()} differs from the current "
                    f"{spec.name}.physics_guess {current.tolist()}; its residual was computed against another "
                    "physics. Re-collect, or pass on_physics_mismatch='recompute' to recompute physics and "
                    "residuals from the stored measured outcome"
                )
            if not np.all(np.isfinite(current)) or np.any(current < low) or np.any(current > high):
                raise ValueError(f"{where}: the current physics_guess {current.tolist()} is not finite or leaves the outcome bounds")
            item = _Encoded(item.situation, item.action, current, item.measured, item.measured - current)
            stale.append(position)
        encoded.append(item)
        presets.append(row.world_preset)

    def stack(attribute: str, width: int) -> np.ndarray:
        if not encoded:
            return np.empty((0, width), dtype=np.float64)
        return np.stack([getattr(item, attribute) for item in encoded]).astype(np.float64, copy=False)

    return RowArrays(
        situations=stack("situation", spec.n_situation),
        actions=stack("action", spec.n_action),
        physics=stack("physics", spec.n_outcome),
        measured=stack("measured", spec.n_outcome),
        residuals=stack("residual", spec.n_outcome),
        world_presets=presets,
        stale_physics_rows=tuple(stale),
    )
