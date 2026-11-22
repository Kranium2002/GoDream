"""Per-skill spiking student, trained inside the dream, and the deployed decision object.

Lane contract (Person B, per-skill spiking student)
---------------------------------------------------
The physics guess corrected by the PFN is a dream world. A :class:`~learned_judgment.engine.dream.DreamSet`
holds, for every dreamed situation, the dreamed objective score of every candidate command of
the skill's candidate grid. :meth:`SNNStudent.train_in_dream` trains one spiking network inside
that dream to choose the best candidate: one output per candidate, soft targets built from the
dreamed scores (:mod:`learned_judgment.students.targets`). The student never learns the PFN's
outcome predictions, and every number it reports about itself is measured inside the dream,
not in the simulator. The in-dream numbers stored by training come from the early-stopping
rows: the optimiser never trained on them, but they chose the restored weights, so they are
not held out.

Deciding uses only numpy (:class:`~learned_judgment.students.snn_core.SpikingNetwork`); importing
this module, loading a saved student and deciding never import torch. Training imports torch
lazily through :mod:`learned_judgment.students.snn_train`.

:class:`DeployedStudent` is the object that decides: one skill's network, its input encoding,
its candidate grid and the records that guard them. :meth:`SNNStudent.deployed` returns the
per-skill student's; a shared multi-skill network builds one per skill with its own encoding.
Before it decides with a spec it compares the spec's action dimensions with the ones it was
trained on and recomputes a digest of the candidate grid over stored probe situations; any
difference raises :class:`GridMismatchError`, because output k would then mean a different
command. Situation features outside their declared ranges are refused or clipped and counted,
as ``students.snn.out_of_range`` says (:class:`OutOfRangeError`, :class:`OutOfRangeLog`).

Shapes (float64 unless noted)::

    n situations, S situation features (spec order), D encoded inputs, C candidates, A action dims
    decide_index      (n, S) -> (n,) int64 candidate index         (batched network path)
    decide_index_one  (S,)   -> int candidate index                (per-decision path)
    decide            (n, S) -> (n, A) = spec.candidate_actions(situation, points_per_dim)[index]
    decide_one        (S,)   -> (A,)  through the per-decision path

On disk a student is the network's ``<stem>.npz`` + ``<stem>.json`` (snn_core.save_network,
no pickle); the JSON carries everything needed to rebuild the encoder, the candidate grid
record and the digests of the situations training saw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.engine.dream import DreamSet
from learned_judgment.engine.streams import stream_rng, stream_seed
from learned_judgment.skills.base import ActionDim, SkillSpec, Variable
from learned_judgment.students.encoding import InputEncoder
from learned_judgment.students.snn_core import (
    OperationCounts,
    SpikingNetwork,
    count_operations,
    load_network,
    save_network,
)
from learned_judgment.students.snn_train import OUT_OF_RANGE_POLICIES, SNNSettings
from learned_judgment.students.targets import dream_regret, flat_rows, row_spreads, score_spread, soft_targets

__all__ = [
    "ACTION_RECORD_KEYS",
    "GRID_PROBE_MAX_ROWS",
    "STUDENT_KIND",
    "STUDENT_RECORD_FORMAT",
    "BoundStudent",
    "DeployedStudent",
    "GridMismatchError",
    "OutOfRangeError",
    "OutOfRangeLog",
    "SNNStudent",
    "action_record",
    "candidate_grid_sha256",
    "check_dreams",
    "check_points_per_dim",
    "dream_content_sha256",
    "situation_digests",
    "situation_row_digests",
    "split_rows_by_situation",
    "student_record",
    "variable_from_record",
    "variable_record",
]

STUDENT_KIND = "per_skill_snn"
STUDENT_RECORD_FORMAT = "learned-judgment/student-record/v2"
# Structural (lead decision S1): the candidate-grid digest is computed over the first distinct
# training situations, at most this many, which are stored with it.
GRID_PROBE_MAX_ROWS = 16
ACTION_RECORD_KEYS = ("name", "kind", "unit", "low", "high", "levels", "role", "command_field", "command_values")
_GRID_DIGEST_TAG = b"learned-judgment/candidate-grid/v1"
_SITUATION_DIGEST_TAG = b"learned-judgment/situation/v1"
_SITUATION_DIGEST_DEFINITION = (
    "sha256 over b'learned-judgment/situation/v1', b'float64:<S>' and the S float64 values of the situation "
    "(-0.0 written as 0.0); identical situations have identical digests"
)
_GRID_DIGEST_DEFINITION = (
    "sha256 over b'learned-judgment/candidate-grid/v1', points_per_dim, and for each probe situation the shape "
    "and float64 bytes of spec.candidate_actions(situation, points_per_dim) (-0.0 written as 0.0)"
)
_RECORD_KEYS = frozenset(
    (
        "format",
        "skill",
        "points_per_dim",
        "n_candidates",
        "action_names",
        "action_records",
        "situation_variables",
        "encoder",
        "candidate_grid",
        "situations_seen",
        "out_of_range",
    )
)


class GridMismatchError(ValueError):
    """The spec's action dimensions or candidate grid differ from the ones the student was
    trained on, so network output k would map to a different command. The student refuses."""


class OutOfRangeError(ValueError):
    """A situation feature lies outside its declared range and the student's out_of_range
    policy is "error": the student refuses to decide."""


# --------------------------------------------------------------------------------------
# Records and digests (public; the shared multi-skill network uses the same ones)
# --------------------------------------------------------------------------------------


def check_points_per_dim(points_per_dim: Any, n_action: int) -> tuple[int, ...]:
    """``points_per_dim`` as a tuple of ``n_action`` integers >= 1 (ValueError otherwise)."""
    if isinstance(points_per_dim, (str, bytes)) or not isinstance(points_per_dim, Sequence):
        raise ValueError(f"points_per_dim must be a list of {n_action} integers; got {points_per_dim!r}")
    points = tuple(points_per_dim)
    if len(points) != n_action or any(
        isinstance(p, bool) or not isinstance(p, (int, np.integer)) or int(p) < 1 for p in points
    ):
        raise ValueError(f"points_per_dim must be {n_action} integers >= 1; got {list(points)}")
    return tuple(int(p) for p in points)


def variable_record(variable: Variable) -> dict[str, Any]:
    """JSON record of one situation variable (inverse: :func:`variable_from_record`)."""
    return {
        "name": variable.name,
        "kind": variable.kind,
        "unit": variable.unit,
        "description": variable.description,
        "low": variable.low,
        "high": variable.high,
        "levels": list(variable.levels),
    }


def variable_from_record(record: Mapping[str, Any]) -> Variable:
    """Rebuild a :class:`Variable` from :func:`variable_record`."""
    return Variable(
        name=str(record["name"]),
        kind=record["kind"],
        unit=str(record["unit"]),
        description=str(record["description"]),
        low=None if record["low"] is None else float(record["low"]),
        high=None if record["high"] is None else float(record["high"]),
        levels=tuple(str(level) for level in record["levels"]),
    )


def action_record(dim: ActionDim) -> dict[str, Any]:
    """Canonical JSON record of one action dimension: every field that decides which command a
    grid index stands for (name, kind, unit, low, high, levels, role, command_field,
    command_values). Two dimensions with equal records give the same commands."""
    if not isinstance(dim, ActionDim):
        raise TypeError(f"an action dimension must be an ActionDim; got {type(dim).__name__}")
    return {
        "name": str(dim.name),
        "kind": str(dim.kind),
        "unit": str(dim.unit),
        "low": None if dim.low is None else float(dim.low),
        "high": None if dim.high is None else float(dim.high),
        "levels": [str(level) for level in dim.levels],
        "role": str(dim.role),
        "command_field": None if dim.command_field is None else str(dim.command_field),
        "command_values": [float(value) for value in dim.command_values],
    }


def _canonical_action_record(record: Any, position: int) -> dict[str, Any]:
    """Validate a stored action record and return it in the form :func:`action_record` gives."""
    if not isinstance(record, Mapping) or set(record) != set(ACTION_RECORD_KEYS):
        keys = sorted(record) if isinstance(record, Mapping) else type(record).__name__
        raise ValueError(f"action record {position} must have exactly the keys {list(ACTION_RECORD_KEYS)}; got {keys}")

    def number(key: str) -> float | None:
        value = record[key]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"action record {position}: {key} must be a number or null; got {value!r}")
        return float(value)

    for key in ("name", "kind", "unit", "role"):
        if not isinstance(record[key], str):
            raise ValueError(f"action record {position}: {key} must be a string; got {record[key]!r}")
    if record["command_field"] is not None and not isinstance(record["command_field"], str):
        raise ValueError(f"action record {position}: command_field must be a string or null")
    if not isinstance(record["levels"], Sequence) or isinstance(record["levels"], str):
        raise ValueError(f"action record {position}: levels must be a list")
    if not isinstance(record["command_values"], Sequence) or isinstance(record["command_values"], str):
        raise ValueError(f"action record {position}: command_values must be a list")
    return {
        "name": record["name"],
        "kind": record["kind"],
        "unit": record["unit"],
        "low": number("low"),
        "high": number("high"),
        "levels": [str(level) for level in record["levels"]],
        "role": record["role"],
        "command_field": record["command_field"],
        "command_values": [float(value) for value in record["command_values"]],
    }


def check_dreams(spec: SkillSpec, dreams: DreamSet, points: tuple[int, ...]) -> None:
    """Refuse a DreamSet that does not belong to ``spec`` and to the grid of ``points``."""
    if not isinstance(dreams, DreamSet):
        raise TypeError(f"dreams must be a DreamSet; got {type(dreams).__name__}")
    for label, have, want in (
        ("skill", dreams.skill, spec.name),
        ("situation_names", dreams.situation_names, spec.situation_names),
        ("action_names", dreams.action_names, spec.action_names),
        ("outcome_names", dreams.outcome_names, spec.outcome_names),
        ("primary_outcome", dreams.primary_outcome, spec.primary_outcome),
    ):
        if have != want:
            raise ValueError(f"DreamSet.{label} {have!r} does not match the spec's {want!r}")
    for row in range(dreams.n_rows):
        grid = np.asarray(spec.candidate_actions(dreams.situations[row], points), dtype=np.float64)
        if grid.shape != dreams.candidates.shape[1:] or not np.array_equal(grid, dreams.candidates[row]):
            raise ValueError(
                f"dream row {row}: the dream's {dreams.n_candidates} candidates are not the grid "
                f"spec.candidate_actions(situation, points_per_dim={list(points)}) gives ({grid.shape[0]} candidates); "
                "train with the points_per_dim the dream was made with"
            )


def dream_content_sha256(dreams: DreamSet, rows: np.ndarray) -> str:
    """sha256 over the dream rows the student used (names, situations, candidates, scores, choice)."""
    hasher = hashlib.sha256(b"learned-judgment/dream-rows/v1")
    for name in (dreams.skill, dreams.method, *dreams.situation_names, *dreams.action_names):
        hasher.update(name.encode("utf-8") + b"\0")
    for array in (dreams.situations[rows], dreams.candidates[rows], dreams.scores[rows], dreams.chosen[rows]):
        array = np.ascontiguousarray(array)
        hasher.update(f"{array.dtype.str}{array.shape}".encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _situation_matrix(situations: Any) -> np.ndarray:
    array = np.asarray(situations, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"situations must have shape (n, S >= 1); got {array.shape}")
    if not np.all(np.isfinite(array)):
        row, column = np.argwhere(~np.isfinite(array))[0]
        raise ValueError(f"situations contain non-finite values (first at row {row}, column {column})")
    return np.ascontiguousarray(array + 0.0)  # + 0.0 turns -0.0 into 0.0


def situation_row_digests(situations: np.ndarray) -> list[str]:
    """One sha256 hex digest per row of ``situations`` (n, S), in row order.

    Identical situations have identical digests (-0.0 counts as 0.0); the definition is stored
    with every student under ``situations_seen``.
    """
    array = _situation_matrix(situations)
    prefix = _SITUATION_DIGEST_TAG + f"float64:{array.shape[1]}".encode("utf-8")
    return [hashlib.sha256(prefix + row.tobytes()).hexdigest() for row in array]


def situation_digests(situations: np.ndarray) -> set[str]:
    """The set of digests of the situations (n, S): compare it with a student's
    ``training_situation_digests`` to find held-out rows that training saw."""
    return set(situation_row_digests(situations))


def split_rows_by_situation(
    situations: np.ndarray, *, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into (kept rows, held rows) by DISTINCT situation.

    Rows with identical situations (same digest) always land on the same side. The distinct
    situations are taken in order of first appearance and permuted with ``rng``; the first
    ``max(1, round(fraction x u))`` of the u distinct situations are held. When every situation
    is distinct this is the row permutation ``rng.permutation(n)``.

    Returns:
        (kept rows, held rows): sorted int64 row indices.

    Raises:
        ValueError when the held situations leave no kept row.
    """
    digests = situation_row_digests(situations)
    distinct = list(dict.fromkeys(digests))
    n_held = max(1, int(round(float(fraction) * len(distinct))))
    if n_held >= len(distinct):
        raise ValueError(
            f"{len(digests)} rows hold {len(distinct)} distinct situation(s): holding out {n_held} of them "
            f"(fraction {fraction}) leaves no training rows"
        )
    order = rng.permutation(len(distinct))
    held = {distinct[index] for index in order[:n_held]}
    held_mask = np.array([digest in held for digest in digests], dtype=bool)
    return np.flatnonzero(~held_mask).astype(np.int64), np.flatnonzero(held_mask).astype(np.int64)


def candidate_grid_sha256(spec: SkillSpec, probe_situations: np.ndarray, points_per_dim: Sequence[int]) -> str:
    """sha256 of ``spec``'s candidate grids over ``probe_situations`` (k, S) at ``points_per_dim``.

    It changes when the grid changes in any way the probe can see: bounds, levels, grid
    resolution, the order of candidates, or a skill's own ``candidate_actions``.
    """
    points = check_points_per_dim(points_per_dim, spec.n_action)
    probe = _situation_matrix(probe_situations)
    hasher = hashlib.sha256(_GRID_DIGEST_TAG)
    hasher.update(json.dumps(list(points)).encode("utf-8"))
    for situation in probe:
        grid = np.ascontiguousarray(np.asarray(spec.candidate_actions(situation, points), dtype=np.float64) + 0.0)
        hasher.update(f"{grid.shape}".encode("utf-8"))
        hasher.update(grid.tobytes())
    return hasher.hexdigest()


def student_record(
    skill: str,
    encoder: InputEncoder,
    points: Sequence[int],
    n_candidates: int,
    actions: Sequence[ActionDim],
    *,
    grid_probe: np.ndarray,
    grid_sha256: str,
    optimiser_situations: Sequence[str],
    early_stopping_situations: Sequence[str],
    out_of_range: str,
) -> dict[str, Any]:
    """The JSON record a deployed student is rebuilt and checked from (stored as meta["student"]).

    Args:
        skill, encoder, points, n_candidates: the student's skill, its own situation encoder,
                        grid resolution and number of network outputs C.
        actions:        the spec's ActionDims at training time.
        grid_probe:     (k, S) probe situations; grid_sha256 = candidate_grid_sha256 over them.
        optimiser_situations, early_stopping_situations: digests of every situation the
                        optimiser trained on and every situation early stopping evaluated.
        out_of_range:   the deployed policy, one of OUT_OF_RANGE_POLICIES.
    """
    points = tuple(int(p) for p in points)
    probe = _situation_matrix(grid_probe)
    return {
        "format": STUDENT_RECORD_FORMAT,
        "skill": skill,
        "points_per_dim": list(points),
        "n_candidates": int(n_candidates),
        "action_names": [dim.name for dim in actions],
        "action_records": [action_record(dim) for dim in actions],
        "situation_variables": [variable_record(variable) for variable in encoder.variables],
        "encoder": encoder.describe(),
        "candidate_grid": {
            "definition": _GRID_DIGEST_DEFINITION,
            "probe_situations": probe.tolist(),
            "sha256": str(grid_sha256),
        },
        "situations_seen": {
            "definition": _SITUATION_DIGEST_DEFINITION,
            "optimiser_rows": sorted(set(optimiser_situations)),
            "early_stopping_rows": sorted(set(early_stopping_situations)),
        },
        "out_of_range": out_of_range,
    }


# Old private names, kept so that modules written against them keep importing.
_points_per_dim = check_points_per_dim
_variable_record = variable_record
_variable_from_record = variable_from_record
_check_dreams = check_dreams
_dream_content_sha256 = dream_content_sha256


# --------------------------------------------------------------------------------------
# The deployed student
# --------------------------------------------------------------------------------------


class OutOfRangeLog:
    """Running count of what a deployed student decided on, kept by the student itself.

    Attributes:
        policy:                   "error" or "clip" (students.snn.out_of_range at training time).
        decisions:                situations decided on (each row of a batch counts once).
        decisions_with_clipping:  of those, situations with at least one clipped value.
        values_clipped:           clipped feature values in total.
        per_feature:              {feature name: clipped values}.
    """

    def __init__(self, policy: str, feature_names: Sequence[str]) -> None:
        self.policy = policy
        self.feature_names = tuple(feature_names)
        self.decisions = 0
        self.decisions_with_clipping = 0
        self.values_clipped = 0
        self.per_feature = {name: 0 for name in self.feature_names}

    def record(self, clipped: np.ndarray) -> None:
        """Add one call's decisions; ``clipped`` is bool (n, S): True where a value was clipped."""
        clipped = np.asarray(clipped, dtype=bool)
        self.decisions += int(clipped.shape[0])
        self.decisions_with_clipping += int(np.sum(clipped.any(axis=1)))
        self.values_clipped += int(clipped.sum())
        for name, count in zip(self.feature_names, clipped.sum(axis=0)):
            self.per_feature[name] += int(count)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the counts."""
        return {
            "policy": self.policy,
            "decisions": self.decisions,
            "decisions_with_clipping": self.decisions_with_clipping,
            "values_clipped": self.values_clipped,
            "per_feature": dict(self.per_feature),
        }


@dataclass(frozen=True, eq=False)
class DeployedStudent:
    """One skill's deployed spiking decision maker (numpy only).

    Attributes:
        skill:               SkillSpec name.
        network:             numpy network; output k is candidate k of the skill's grid.
        encode:              situations (n, S) -> encoded inputs (n, D) of ``network``.
        encode_one:          situation (S,) -> encoded input (D,), bitwise equal to the row of encode.
        situation_variables: the S situation variables the student was trained with (ranges for
                             the out_of_range policy, and the encoding a spec must match).
        points_per_dim:      candidate grid resolution.
        n_candidates:        C, number of network outputs.
        action_records:      action_record() of every action dimension at training time.
        grid_probe:          (k, S) probe situations of the candidate-grid digest.
        grid_sha256:         candidate_grid_sha256 of the training spec over ``grid_probe``.
        training_situation_digests: digests of every situation the optimiser or early stopping saw.
        out_of_range:        "error" or "clip".
        meta:                provenance (JSON-serializable).
        out_of_range_log:    (set on construction) the running :class:`OutOfRangeLog`.
    """

    skill: str
    network: SpikingNetwork
    encode: Callable[[np.ndarray], np.ndarray]
    encode_one: Callable[[np.ndarray], np.ndarray]
    situation_variables: tuple[Variable, ...]
    points_per_dim: tuple[int, ...]
    n_candidates: int
    action_records: tuple[Mapping[str, Any], ...]
    grid_probe: np.ndarray
    grid_sha256: str
    training_situation_digests: frozenset[str]
    out_of_range: str
    meta: Mapping[str, Any]
    out_of_range_log: OutOfRangeLog = field(init=False, repr=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if not isinstance(self.skill, str) or not self.skill:
            raise ValueError(f"skill must be a non-empty string; got {self.skill!r}")
        if not isinstance(self.network, SpikingNetwork):
            raise TypeError(f"network must be a SpikingNetwork; got {type(self.network).__name__}")
        if not callable(self.encode) or not callable(self.encode_one):
            raise TypeError("encode and encode_one must be callables")
        variables = tuple(self.situation_variables)
        if not variables or not all(isinstance(variable, Variable) for variable in variables):
            raise ValueError("situation_variables must be a non-empty sequence of Variable")
        problems = [problem for variable in variables for problem in variable.problems()]
        if problems:
            raise ValueError("; ".join(problems))
        records = tuple(_canonical_action_record(record, i) for i, record in enumerate(self.action_records))
        if not records:
            raise ValueError("action_records is empty: the student's candidate grid is unknown")
        points = check_points_per_dim(self.points_per_dim, len(records))
        if isinstance(self.n_candidates, bool) or int(self.n_candidates) != self.network.n_outputs:
            raise ValueError(f"n_candidates {self.n_candidates} differs from the network's {self.network.n_outputs} outputs")
        probe = _situation_matrix(self.grid_probe)
        if probe.shape[1] != len(variables) or not 1 <= probe.shape[0] <= GRID_PROBE_MAX_ROWS:
            raise ValueError(
                f"grid_probe must have shape (1..{GRID_PROBE_MAX_ROWS}, {len(variables)}); got {probe.shape}"
            )
        if not isinstance(self.grid_sha256, str) or len(self.grid_sha256) != 64:
            raise ValueError(f"grid_sha256 must be a sha256 hex digest; got {self.grid_sha256!r}")
        digests = frozenset(self.training_situation_digests)
        if not digests or not all(isinstance(digest, str) and len(digest) == 64 for digest in digests):
            raise ValueError("training_situation_digests must be a non-empty set of sha256 hex digests")
        if self.out_of_range not in OUT_OF_RANGE_POLICIES:
            raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE_POLICIES}; got {self.out_of_range!r}")
        if not isinstance(self.meta, Mapping):
            raise TypeError(f"meta must be a mapping; got {type(self.meta).__name__}")
        set_(self, "situation_variables", variables)
        set_(self, "action_records", records)
        set_(self, "points_per_dim", points)
        set_(self, "n_candidates", int(self.n_candidates))
        set_(self, "grid_probe", probe)
        set_(self, "training_situation_digests", digests)
        # Ranges of the out_of_range policy: (S,) low / high, unbounded for categorical features.
        set_(self, "_low", np.array([-np.inf if v.kind == "categorical" else v.low for v in variables], dtype=np.float64))
        set_(self, "_high", np.array([np.inf if v.kind == "categorical" else v.high for v in variables], dtype=np.float64))
        set_(self, "_encoding", InputEncoder.from_variables(variables).describe())
        encoded = np.asarray(self.encode(probe[:1]), dtype=np.float64)
        encoded_one = np.asarray(self.encode_one(probe[0]), dtype=np.float64)
        if encoded.shape != (1, self.network.n_inputs) or encoded_one.shape != (self.network.n_inputs,):
            raise ValueError(
                f"encode gives {encoded.shape} and encode_one {encoded_one.shape}; the network takes "
                f"{self.network.n_inputs} inputs"
            )
        if not np.array_equal(encoded[0], encoded_one):
            raise ValueError("encode_one(situation) differs from encode(situation[None, :])[0]")
        set_(self, "out_of_range_log", OutOfRangeLog(self.out_of_range, [variable.name for variable in variables]))

    # ----- structure ---------------------------------------------------------------------

    @property
    def digest(self) -> str:
        """sha256 identity of the network (snn_core.SpikingNetwork.digest)."""
        return self.network.digest()

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(record["name"] for record in self.action_records)

    # ----- guards ------------------------------------------------------------------------

    def check_spec(self, spec: SkillSpec) -> None:
        """Refuse a spec this student cannot decide for.

        Raises:
            TypeError for a non-SkillSpec; ValueError for another skill name or a different
            situation encoding (variables, ranges, levels); :class:`GridMismatchError` when an
            action dimension differs from its training record in any field, or when the
            candidate grid over the stored probe situations has a different digest.
        """
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if spec.name != self.skill:
            raise ValueError(f"this student was trained for skill {self.skill!r}, not {spec.name!r}")
        if InputEncoder.from_variables(spec.situation).describe() != self._encoding:
            raise ValueError(
                f"spec {spec.name!r} encodes its situation differently from the encoder this student was trained "
                "with (situation variables, ranges or levels changed)"
            )
        current = tuple(action_record(dim) for dim in spec.actions)
        if current != self.action_records:
            raise GridMismatchError(
                f"spec {spec.name!r}: {self._record_difference(current)}; output k of the student would stand for a "
                "different command, so it refuses to decide (retrain it inside a dream of the current spec)"
            )
        digest = candidate_grid_sha256(spec, self.grid_probe, self.points_per_dim)
        if digest != self.grid_sha256:
            raise GridMismatchError(
                f"spec {spec.name!r}: candidate_actions over the {self.grid_probe.shape[0]} stored probe situations "
                f"at points_per_dim {list(self.points_per_dim)} differs from the grid the student was trained on "
                f"(digest {digest[:16]}... instead of {self.grid_sha256[:16]}...); the student refuses to decide"
            )

    def _record_difference(self, current: tuple[Mapping[str, Any], ...]) -> str:
        if len(current) != len(self.action_records):
            return f"{len(current)} action dimensions instead of the trained {len(self.action_records)}"
        for position, (have, want) in enumerate(zip(current, self.action_records)):
            for key in ACTION_RECORD_KEYS:
                if have[key] != want[key]:
                    return (
                        f"action dimension {position} ({want['name']!r}) has {key} {have[key]!r}; the student was "
                        f"trained with {want[key]!r}"
                    )
        return "the action dimensions differ"

    def check_situations(self, situations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the input checks and the out_of_range policy to situations (n, S).

        Returns:
            (situations to decide on (n, S) float64, clipped (n, S) bool: True where a value was
            clipped into its declared range; all False under "error").

        Raises:
            ValueError for a wrong shape, a non-finite value or an invalid categorical level
            index (naming the row and feature); :class:`OutOfRangeError` under "error" when a
            float feature lies outside [low, high] (naming the feature, value, row and range).
        """
        array = np.asarray(situations, dtype=np.float64)
        width = len(self.situation_variables)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"situations must have shape (n, {width}); got {array.shape}")
        if not np.all(np.isfinite(array)):
            row, column = np.argwhere(~np.isfinite(array))[0]
            raise ValueError(
                f"situations contain non-finite values (first at row {row}, feature "
                f"{self.situation_variables[column].name!r}); the student refuses to decide on them"
            )
        for column, variable in enumerate(self.situation_variables):
            if variable.kind == "categorical":
                values = array[:, column]
                valid = (values == np.floor(values)) & (values >= 0.0) & (values < len(variable.levels))
                if not np.all(valid):
                    row = int(np.argwhere(~valid)[0][0])
                    raise ValueError(
                        f"situation feature {variable.name!r} = {float(values[row])!r} at row {row} is not a level index "
                        f"of {variable.levels}"
                    )
        outside = (array < self._low) | (array > self._high)
        if not outside.any():
            return array, outside
        if self.out_of_range == "error":
            row, column = (int(value) for value in np.argwhere(outside)[0])
            variable = self.situation_variables[column]
            raise OutOfRangeError(
                f"situation feature {variable.name!r} = {float(array[row, column])!r} at row {row} lies outside its declared "
                f"range [{variable.low}, {variable.high}] ({int(outside.sum())} value(s) out of range in total); "
                "students.snn.out_of_range is 'error', so the student refuses to decide"
            )
        return np.clip(array, self._low, self._high), outside

    def seen_in_training(self, situations: np.ndarray) -> np.ndarray:
        """bool (n,): True for each situation (n, S) whose digest is one the optimiser or early
        stopping saw, i.e. a row that cannot count as held out for this student."""
        return np.array(
            [digest in self.training_situation_digests for digest in situation_row_digests(situations)], dtype=bool
        )

    # ----- deciding ----------------------------------------------------------------------

    def decide_index(self, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n,) int64 index of the chosen candidate (batched numpy path)."""
        array, clipped = self.check_situations(situations)
        index = self.network.decide(self.encode(array)).astype(np.int64)
        self.out_of_range_log.record(clipped)
        return index

    def decide_index_one(self, situation: np.ndarray) -> int:
        """(S,) situation -> index of the chosen candidate through the per-decision path
        (``encode_one`` + ``SpikingNetwork.decide_one``)."""
        array, clipped = self.check_situations(self._one(situation))
        index = self.network.decide_one(self.encode_one(array[0]))
        self.out_of_range_log.record(clipped)
        return index

    def _one(self, situation: np.ndarray) -> np.ndarray:
        vector = np.asarray(situation, dtype=np.float64)
        if vector.shape != (len(self.situation_variables),):
            raise ValueError(f"situation must have shape ({len(self.situation_variables)},); got {vector.shape}")
        return vector[None, :]

    def _action(self, spec: SkillSpec, situation: np.ndarray, index: int, label: str) -> np.ndarray:
        grid = np.asarray(spec.candidate_actions(situation, self.points_per_dim), dtype=np.float64)
        if grid.shape != (self.n_candidates, spec.n_action):
            raise GridMismatchError(
                f"{label}: spec.candidate_actions gives {grid.shape}; the student was trained on "
                f"({self.n_candidates}, {spec.n_action})"
            )
        return grid[index]

    def decide(self, spec: SkillSpec, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n, A) chosen actions: ``spec.candidate_actions(situation,
        points_per_dim)[index]`` for each row (on the situation as decided on, i.e. clipped under
        "clip"). Runs :meth:`check_spec` first."""
        self.check_spec(spec)
        array, clipped = self.check_situations(situations)
        index = self.network.decide(self.encode(array)).astype(np.int64)
        actions = np.empty((array.shape[0], spec.n_action), dtype=np.float64)
        for row in range(array.shape[0]):
            actions[row] = self._action(spec, array[row], int(index[row]), f"row {row}")
        if not np.all(spec.action_in_bounds(actions)):
            raise ValueError(f"skill {spec.name!r} produced a chosen action outside its declared bounds")
        self.out_of_range_log.record(clipped)
        return actions

    def decide_one(self, spec: SkillSpec, situation: np.ndarray) -> np.ndarray:
        """(S,) situation -> (A,) chosen action through the per-decision path. Runs
        :meth:`check_spec` first."""
        self.check_spec(spec)
        array, clipped = self.check_situations(self._one(situation))
        index = self.network.decide_one(self.encode_one(array[0]))
        action = self._action(spec, array[0], index, "situation")
        if not np.all(spec.action_in_bounds(action[None, :])):
            raise ValueError(f"skill {spec.name!r} produced a chosen action outside its declared bounds")
        self.out_of_range_log.record(clipped)
        return action

    def operation_counts(self, situations: np.ndarray) -> OperationCounts:
        """Operations per decision measured on ``situations`` (n, S) (inputs to the energy
        estimate). Applies the same input checks and out_of_range policy as deciding; not
        counted in the out_of_range log, because nothing is decided."""
        array, _ = self.check_situations(situations)
        return count_operations(self.network, self.encode(array))

    def bind(self, spec: SkillSpec) -> "BoundStudent":
        """This student bound to ``spec``: :meth:`check_spec` runs ONCE here, and the returned
        :class:`BoundStudent` decides without repeating it (see there)."""
        return BoundStudent(self, spec)


@dataclass(frozen=True, eq=False)
class BoundStudent:
    """A :class:`DeployedStudent` bound to one SkillSpec whose grid was checked once.

    Construction (normally through :meth:`DeployedStudent.bind`) runs
    ``deployed.check_spec(spec)`` once: the action records and the candidate-grid digest over
    the stored probe situations. The decide methods do not repeat that grid check; each call
    only verifies that ``spec.actions`` and ``spec.situation`` are still the objects checked at
    binding (an identity test) and raises :class:`GridMismatchError` if either was replaced.
    A spec whose ``candidate_actions`` behaves differently after binding is not detected: bind
    again after changing a spec. :meth:`DeployedStudent.decide_one` keeps its per-call check.

    Per-decision paths (numpy only; the out_of_range policy and log as in DeployedStudent)::

        decide_one        (S,) -> (A,)  input checks -> encode_one -> SpikingNetwork.decide_one
                                        -> spec.candidate_actions(situation, points_per_dim)[index]
                                        -> action_in_bounds        (situation in, command values out)
        decide_index_one  (S,) -> int   input checks -> encode_one -> SpikingNetwork.decide_one
                                        (no candidate lookup)

    Attributes:
        deployed: the student.
        spec:     the SkillSpec checked at binding.
    """

    deployed: DeployedStudent
    spec: SkillSpec

    def __post_init__(self) -> None:
        if not isinstance(self.deployed, DeployedStudent):
            raise TypeError(f"deployed must be a DeployedStudent; got {type(self.deployed).__name__}")
        self.deployed.check_spec(self.spec)
        object.__setattr__(self, "_actions", self.spec.actions)
        object.__setattr__(self, "_situation", self.spec.situation)

    @property
    def skill(self) -> str:
        return self.deployed.skill

    @property
    def digest(self) -> str:
        """sha256 identity of the network (snn_core.SpikingNetwork.digest)."""
        return self.deployed.digest

    def _check_unchanged(self) -> None:
        if self.spec.actions is not self._actions or self.spec.situation is not self._situation:
            raise GridMismatchError(
                f"spec {self.spec.name!r}: its actions or situation were replaced after binding; bind the student again "
                "so that the grid check runs on the current spec"
            )

    def decide_index_one(self, situation: np.ndarray) -> int:
        """(S,) situation -> index of the chosen candidate (no candidate lookup)."""
        self._check_unchanged()
        return self.deployed.decide_index_one(situation)

    def decide_one(self, situation: np.ndarray) -> np.ndarray:
        """(S,) situation -> (A,) chosen action: the full deployed path, without the grid check
        that ran at binding."""
        self._check_unchanged()
        deployed = self.deployed
        array, clipped = deployed.check_situations(deployed._one(situation))
        index = deployed.network.decide_one(deployed.encode_one(array[0]))
        action = deployed._action(self.spec, array[0], index, "situation")
        if not np.all(self.spec.action_in_bounds(action[None, :])):
            raise ValueError(f"skill {self.spec.name!r} produced a chosen action outside its declared bounds")
        deployed.out_of_range_log.record(clipped)
        return action


# --------------------------------------------------------------------------------------
# The per-skill student
# --------------------------------------------------------------------------------------


def _record_problem(record: Mapping[str, Any], key: str) -> str:
    return f"meta['student'] has no valid {key!r}; this student cannot be checked against a spec and is refused"


@dataclass(frozen=True)
class SNNStudent:
    """A dream-trained spiking network that picks one candidate command per situation.

    Attributes:
        skill:          SkillSpec name the student belongs to.
        network:        the deployed numpy network; output k is candidate k of the grid.
        encoder:        situation (n, S) -> encoded inputs (n, D).
        points_per_dim: candidate grid resolution the dream and the student use.
        n_candidates:   C, number of network outputs.
        meta:           JSON-serializable provenance: settings, encoder, dream, training report,
                        parity report, in-dream numbers on the early-stopping rows, and under
                        "student" the record (:func:`student_record`) the deployed student is
                        built and checked from.

    Construction refuses a meta without a complete student record: a student whose candidate
    grid cannot be checked never decides.
    """

    skill: str
    network: SpikingNetwork
    encoder: InputEncoder
    points_per_dim: tuple[int, ...]
    n_candidates: int
    meta: dict
    _deployed: DeployedStudent = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.skill, str) or not self.skill:
            raise ValueError(f"skill must be a non-empty string; got {self.skill!r}")
        if not isinstance(self.network, SpikingNetwork):
            raise TypeError(f"network must be a SpikingNetwork; got {type(self.network).__name__}")
        if not isinstance(self.encoder, InputEncoder):
            raise TypeError(f"encoder must be an InputEncoder; got {type(self.encoder).__name__}")
        if self.network.n_inputs != self.encoder.width:
            raise ValueError(f"the network takes {self.network.n_inputs} inputs but the encoder gives {self.encoder.width}")
        if int(self.n_candidates) != self.network.n_outputs:
            raise ValueError(f"n_candidates {self.n_candidates} differs from the network's {self.network.n_outputs} outputs")
        points = check_points_per_dim(self.points_per_dim, len(self.points_per_dim))
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be a dict; got {type(self.meta).__name__}")
        object.__setattr__(self, "points_per_dim", points)
        object.__setattr__(self, "n_candidates", int(self.n_candidates))
        object.__setattr__(self, "_deployed", self._build_deployed())

    def _build_deployed(self) -> DeployedStudent:
        record = self.meta.get("student")
        if not isinstance(record, Mapping):
            raise ValueError("meta['student'] is missing; this student cannot be checked against a spec and is refused")
        missing = sorted(_RECORD_KEYS - set(record))
        if missing or record.get("format") != STUDENT_RECORD_FORMAT:
            raise ValueError(
                f"meta['student'] is not a {STUDENT_RECORD_FORMAT} record (missing {missing}, format "
                f"{record.get('format')!r}); a student without its action records and candidate-grid digest cannot be "
                "checked against a spec and is refused (retrain it)"
            )
        for key, have, want in (
            ("skill", record["skill"], self.skill),
            ("points_per_dim", list(record["points_per_dim"]), list(self.points_per_dim)),
            ("n_candidates", record["n_candidates"], self.n_candidates),
        ):
            if have != want:
                raise ValueError(f"meta['student'][{key!r}] is {have!r} but the student has {want!r}")
        if not self.encoder.matches(record["encoder"]):
            raise ValueError("meta['student']['encoder'] does not describe this student's encoder")
        records = record["action_records"]
        if not isinstance(records, Sequence) or [item.get("name") if isinstance(item, Mapping) else None for item in records] != list(
            record["action_names"]
        ):
            raise ValueError(_record_problem(record, "action_records"))
        grid = record["candidate_grid"]
        seen = record["situations_seen"]
        if not isinstance(grid, Mapping) or not {"probe_situations", "sha256"} <= set(grid):
            raise ValueError(_record_problem(record, "candidate_grid"))
        if not isinstance(seen, Mapping) or not {"optimiser_rows", "early_stopping_rows"} <= set(seen):
            raise ValueError(_record_problem(record, "situations_seen"))
        return DeployedStudent(
            skill=self.skill,
            network=self.network,
            encode=self.encoder.encode,
            encode_one=self.encoder.encode_one,
            situation_variables=self.encoder.variables,
            points_per_dim=self.points_per_dim,
            n_candidates=self.n_candidates,
            action_records=tuple(records),
            grid_probe=np.asarray(grid["probe_situations"], dtype=np.float64),
            grid_sha256=grid["sha256"],
            training_situation_digests=frozenset((*seen["optimiser_rows"], *seen["early_stopping_rows"])),
            out_of_range=record["out_of_range"],
            meta=self.meta,
        )

    # ----- training ---------------------------------------------------------------------

    @classmethod
    def train_in_dream(
        cls,
        spec: SkillSpec,
        dreams: DreamSet,
        *,
        settings: SNNSettings,
        points_per_dim: Sequence[int],
        base_seed: int,
    ) -> "SNNStudent":
        """Train a spiking student for ``spec`` inside ``dreams``.

        Steps: validate the dream against ``spec`` and the grid of ``points_per_dim``; refuse
        dream situations outside the declared ranges; split the rows into training rows and
        early-stopping rows by DISTINCT situation (:func:`split_rows_by_situation`, stream
        ``"<skill>/snn/validation_split"``; identical situations stay on one side); encode
        situations with ``InputEncoder.from_variables(spec.situation)``; build soft targets
        (``settings.target_spread``: one scale from the training rows, or each row's own);
        train one head named ``spec.name`` (initialisation from stream ``"<skill>/snn/init"``,
        batch order from ``"<skill>/snn/train"``); export to numpy and check parity on every
        training and early-stopping row; record the action dimensions, the candidate-grid digest
        over the first distinct training situations (at most :data:`GRID_PROBE_MAX_ROWS`) and the
        digests of every situation training saw.

        Raises ValueError for a mismatched dream, a wrong grid, out-of-range situations, too few
        distinct situations or nothing to learn;
        :class:`~learned_judgment.students.snn_train.TrainingError` when training does not beat
        initialisation and the best constant prediction, or leaves a hidden layer silent;
        :class:`~learned_judgment.students.snn_train.ParityError` when the numpy network does
        not reproduce the torch model.
        """
        from learned_judgment.students import snn_train

        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if not isinstance(settings, SNNSettings):
            raise TypeError(f"settings must be SNNSettings; got {type(settings).__name__}")
        if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
            raise ValueError(f"base_seed must be a non-negative integer; got {base_seed!r}")
        base_seed = int(base_seed)
        points = check_points_per_dim(points_per_dim, spec.n_action)
        check_dreams(spec, dreams, points)
        for column, variable in enumerate(spec.situation):
            if variable.kind != "float":
                continue
            values = dreams.situations[:, column]
            outside = (values < variable.low) | (values > variable.high)
            if np.any(outside):
                row = int(np.argwhere(outside)[0][0])
                raise ValueError(
                    f"dream row {row}: situation feature {variable.name!r} = {float(values[row])!r} lies outside its declared "
                    f"range [{variable.low}, {variable.high}]; the student is trained on the declared situation space only"
                )

        n = dreams.n_rows
        split_stream = f"{spec.name}/snn/validation_split"
        train_rows, validation_rows = split_rows_by_situation(
            dreams.situations, fraction=settings.validation_fraction, rng=stream_rng(base_seed, split_stream)
        )

        encoder = InputEncoder.from_variables(spec.situation)
        encoded = encoder.encode(dreams.situations)
        train_scores = dreams.scores[train_rows]
        train_row_spreads = row_spreads(train_scores)
        train_flat = flat_rows(train_scores)
        if settings.target_spread == "global_median":
            spread: float | None = score_spread(train_scores)
        else:
            spread = None
            if np.all(train_flat):
                raise ValueError(
                    f"all {train_rows.size} training rows score every candidate identically; the dream gives nothing "
                    "to learn"
                )
        targets = soft_targets(
            dreams.scores, temperature=settings.target_temperature, target_spread=settings.target_spread, spread=spread
        )

        init_stream, train_stream = f"{spec.name}/snn/init", f"{spec.name}/snn/train"
        init_seed, train_seed = stream_seed(base_seed, init_stream), stream_seed(base_seed, train_stream)
        model = snn_train.build_model(encoder.width, {spec.name: dreams.n_candidates}, settings, seed=init_seed)
        report = snn_train.fit(
            model,
            {spec.name: (encoded[train_rows], targets[train_rows])},
            validation={spec.name: (encoded[validation_rows], targets[validation_rows])},
            sampling={spec.name: 1.0},
            settings=settings,
            seed=train_seed,
        )
        network = snn_train.export_network(model, spec.name)
        parity = snn_train.check_parity(
            model, spec.name, network, encoded, logit_tolerance=settings.parity_logit_tolerance
        )

        row_digests = situation_row_digests(dreams.situations)
        probe_rows: list[int] = []
        probe_seen: set[str] = set()
        for row in train_rows:
            if row_digests[row] not in probe_seen:
                probe_seen.add(row_digests[row])
                probe_rows.append(int(row))
                if len(probe_rows) == GRID_PROBE_MAX_ROWS:
                    break
        grid_probe = dreams.situations[probe_rows]
        record = student_record(
            spec.name,
            encoder,
            points,
            dreams.n_candidates,
            spec.actions,
            grid_probe=grid_probe,
            grid_sha256=candidate_grid_sha256(spec, grid_probe, points),
            optimiser_situations=[row_digests[row] for row in train_rows],
            early_stopping_situations=[row_digests[row] for row in validation_rows],
            out_of_range=settings.out_of_range,
        )

        decisions = network.decide(encoded[validation_rows])
        regret = dream_regret(dreams.scores[validation_rows], decisions)
        median_row_spread = float(np.median(train_row_spreads))
        meta = {
            "kind": STUDENT_KIND,
            "label": (
                "dream-trained spiking student: trained to choose the candidate the dream scores best; every number "
                "here is measured inside the dream, not in the simulator; the numbers under "
                "'early_stopping_rows_in_dream' come from the rows that chose the restored weights, so they are not "
                "held-out numbers"
            ),
            "decision_runtime": "numpy (learned_judgment.students.snn_core); torch was used for training only",
            "student": record,
            "settings": settings.describe(),
            "encoder": encoder.describe(),
            "dream": {
                "method": dreams.method,
                "request_hashes": list(dreams.request_hashes),
                "n_rows_used": int(n),
                "n_distinct_situations": len(set(row_digests)),
                "n_rows_train": int(train_rows.size),
                "n_rows_early_stopping": int(validation_rows.size),
                "split": (
                    "by distinct situation (learned_judgment.students.snn.split_rows_by_situation): rows with "
                    "identical situations are all training rows or all early-stopping rows"
                ),
                "n_candidates": int(dreams.n_candidates),
                "target_spread": settings.target_spread,
                "score_spread": spread,
                "score_spread_rows": "training rows (null under target_spread 'per_row': each row uses its own)",
                "training_row_spread": {
                    "definition": "population std of each training row's dreamed scores across its candidates",
                    "min": float(np.min(train_row_spreads)),
                    "median": median_row_spread,
                    "max": float(np.max(train_row_spreads)),
                    "n_flat_rows": int(np.sum(train_flat)),
                },
                "rows_sha256": dream_content_sha256(dreams, np.arange(n)),
            },
            "seeds": {
                "base_seed": base_seed,
                "validation_split_stream": split_stream,
                "init_stream": init_stream,
                "init_seed": int(init_seed),
                "train_stream": train_stream,
                "train_seed": int(train_seed),
            },
            "training": report.to_json(),
            "parity": {**parity, "rows": "every training and early-stopping row of the dream"},
            "early_stopping_rows_in_dream": {
                "label": (
                    "inside the dream, on the early-stopping rows: the optimiser never trained on them, but they chose "
                    "the restored weights, so these numbers are optimistic and NOT held out"
                ),
                "n_rows": int(validation_rows.size),
                "agreement_with_dream_choice": float(np.mean(decisions == dreams.chosen[validation_rows])),
                "fraction_with_zero_dream_regret": float(np.mean(regret == 0.0)),
                "mean_dream_regret": float(np.mean(regret)),
                "mean_dream_regret_over_spread": (
                    float(np.mean(regret) / median_row_spread) if median_row_spread > 0.0 else None
                ),
                "spread_definition": "median population std of the training rows' scores (dream.training_row_spread)",
            },
        }
        return cls(
            skill=spec.name,
            network=network,
            encoder=encoder,
            points_per_dim=points,
            n_candidates=dreams.n_candidates,
            meta=meta,
        )

    # Old private name of :func:`student_record`, kept so that modules written against it keep importing.
    _student_record = staticmethod(student_record)

    # ----- deciding (numpy only, through the deployed student) ----------------------------

    def deployed(self) -> DeployedStudent:
        """The :class:`DeployedStudent` every decision of this student goes through (one
        object per student, so its out_of_range log accumulates)."""
        return self._deployed

    @property
    def digest(self) -> str:
        """sha256 identity of the trained network (snn_core.SpikingNetwork.digest)."""
        return self.network.digest()

    @property
    def out_of_range_log(self) -> OutOfRangeLog:
        return self._deployed.out_of_range_log

    @property
    def training_situation_digests(self) -> frozenset[str]:
        """Digests of every situation the optimiser or early stopping saw."""
        return self._deployed.training_situation_digests

    def check_spec(self, spec: SkillSpec) -> None:
        """See :meth:`DeployedStudent.check_spec`."""
        self._deployed.check_spec(spec)

    def decide_index(self, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n,) int64 index of the chosen candidate (batched numpy path)."""
        return self._deployed.decide_index(situations)

    def decide_index_one(self, situation: np.ndarray) -> int:
        """(S,) situation -> chosen candidate index through encoder.encode_one + network.decide_one."""
        return self._deployed.decide_index_one(situation)

    def decide(self, spec: SkillSpec, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n, A) chosen actions; refuses a spec whose name, situation
        encoding, action dimensions or candidate grid differ from the trained ones."""
        return self._deployed.decide(spec, situations)

    def decide_one(self, spec: SkillSpec, situation: np.ndarray) -> np.ndarray:
        """(S,) situation -> (A,) chosen action through the per-decision path; same spec checks."""
        return self._deployed.decide_one(spec, situation)

    def operation_counts(self, situations: np.ndarray) -> OperationCounts:
        """Operations per decision measured on ``situations`` (n, S) (inputs to the energy estimate)."""
        return self._deployed.operation_counts(situations)

    # ----- persistence --------------------------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Write ``<stem>.npz`` + ``<stem>.json`` (no pickle); return the network digest."""
        meta = dict(self.meta)
        meta["kind"] = STUDENT_KIND
        try:
            return save_network(path, self.network, meta)
        except (TypeError, ValueError) as error:
            raise ValueError(f"student meta is not JSON-serializable with finite numbers: {error}") from error

    @classmethod
    def load(cls, path: str | Path) -> "SNNStudent":
        """Inverse of :meth:`save`; verifies the network digest, the encoder description and the
        student record (a file without action records and grid digest is refused)."""
        network, meta = load_network(path)
        if meta.get("kind") != STUDENT_KIND:
            raise ValueError(f"{path} is not a {STUDENT_KIND} file (kind {meta.get('kind')!r})")
        record = meta.get("student")
        if not isinstance(record, Mapping) or "situation_variables" not in record or "encoder" not in record:
            raise ValueError(f"{path}: meta['student'] with situation variables and encoder is missing")
        encoder = InputEncoder.from_variables([variable_from_record(item) for item in record["situation_variables"]])
        if not encoder.matches(record["encoder"]):
            raise ValueError(f"{path}: the stored situation variables do not rebuild the stored encoder")
        return cls(
            skill=str(record["skill"]),
            network=network,
            encoder=encoder,
            points_per_dim=tuple(int(p) for p in record["points_per_dim"]),
            n_candidates=int(record["n_candidates"]),
            meta=meta,
        )
