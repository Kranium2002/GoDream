"""Shared multi-skill spiking student with reserved skill slots, trained in the dream with dream rehearsal.

Lane contract (Person B, lane B: shared multi-skill spiking student)
--------------------------------------------------------------------
The physics guess corrected by the PFN is a dream world. A
:class:`~learned_judgment.engine.dream.DreamSet` holds, for every dreamed situation, the dreamed
objective score of every candidate command of one skill. :class:`SharedSNNStudent` is ONE spiking
network for all skills: a spiking trunk shared by every skill, a skill-slot input and one readout
head per skill (one output per candidate of that skill's grid). Each head is trained inside its
skill's dream to choose the candidate the dream scores best (soft targets from the dreamed scores,
:mod:`learned_judgment.students.targets`). The network never learns the PFN's outcome predictions,
and every number it reports about itself is measured inside the dream, not in the simulator.

Skill slots. The network is created with ``students.snn.shared_skill_slots`` slots and
``students.snn.shared_feature_width`` feature columns, both fixed for its whole life; there is no
list of skill names. A skill takes the next free slot when it is added and keeps it
(:meth:`~learned_judgment.students.encoding.SharedInputEncoder.with_skill`). A network holding one
skill, with every other slot empty, trains, decides and is measured like any other.

Skills are added in sequence (DODGE -> BRAKE -> TURN -> RAMP in the project). Each
:meth:`SharedSNNStudent.add_skill` continues from the current network (the trunk is initialised
once, at the first add; Adam starts fresh at every add) and adds the new skill's head:

    rehearsal=True   every batch mixes rows of the new skill with TRAINING rows of the stored dream
                     sets of the skills added earlier; the earlier skills together get
                     ``students.snn.rehearsal_fraction`` of each batch, split evenly among them.
                     Rehearsal rows are stored dreams: never real attempts, never a new provider call.
    rehearsal=False  only the new skill's rows (the no-rehearsal control that shows forgetting).

Both arms use the same seed streams, splits, settings and step limits; they differ in batch
composition and, through it, in what early stopping evaluates (see below).

Three parts of every skill's dream, fixed when the skill is added and kept for the life of the
student; all three split by DISTINCT situation
(:func:`~learned_judgment.students.snn.split_rows_by_situation`), so identical situations always
fall in one part::

    retention rows       students.snn.shared_retention_fraction of the distinct situations
                         (stream "<skill>/shared_snn/retention_split"). Never in a batch and never
                         passed to early stopping, at any add. The retention matrix and backward
                         transfer are computed on these rows only.
    early-stopping rows  students.snn.validation_fraction of the remaining distinct situations
                         (stream "<skill>/shared_snn/early_stopping_split"). Passed to fit as
                         validation rows at the skill's own add and, with rehearsal, at later adds:
                         they pick the restored weights, so numbers on them are NOT held out.
    training rows        the rest: the optimiser's rows, and the only rows rehearsal draws.

Early stopping is :func:`~learned_judgment.students.snn_train.fit`'s sampling-weighted validation
loss over the heads trained in that add: with rehearsal it includes the earlier skills'
early-stopping rows, without rehearsal only the new skill's.

Stages. Stage ``i`` is the network right after the ``i``-th add (0-based). The numpy view (trunk +
head) of every skill held at every stage is kept in memory and on disk, because hypothesis H7
compares an earlier skill before and after later skills were added. :meth:`deployed` returns a
:class:`~learned_judgment.students.snn.DeployedStudent` for any skill and stage, so the same
measurement code serves per-skill and shared students.

Input layout (:class:`~learned_judgment.students.encoding.SharedInputEncoder`)::

    [ skill's own encoding zero-padded to feature_width | one-hot over n_slots slots ]
    W = feature_width + n_slots, the network input width, fixed when the student is created

Shapes (float64 unless noted)::

    n situations, S situation features of the skill (spec order), A action dims, C_k candidates
    decide_index      (skill, (n, S)) -> (n,) int64 candidate index
    decide_index_one  (skill, (S,))   -> int candidate index (per-decision path)
    decide            (spec, (n, S))  -> (n, A) = spec.candidate_actions(situation, points_per_dim)[index]
    decide_one        (spec, (S,))    -> (A,)

Every decision goes through the skill's DeployedStudent (latest stage unless a stage is named):
it refuses a spec whose action records or candidate grid changed (GridMismatchError), and a
situation outside the declared ranges as ``students.snn.out_of_range`` says (OutOfRangeError or
clip-and-count). Deciding, loading and importing this module use numpy only; torch and snnTorch
are imported lazily by :mod:`learned_judgment.students.snn_train` when a skill is added.

On disk (:meth:`save`, no pickle anywhere)::

    <directory>/manifest.json                       settings, slots, splits, seeds, records,
                                                    stage index with every view's sha256,
                                                    sha256 of the model state
    <directory>/model_state.npz                     torch state dict as named arrays (weights only)
    <directory>/networks/stage_<i>/<skill>.npz|.json  view of <skill> after the i-th add
    <directory>/dreams/<skill>.npz|.json            stored dream set (engine.dream.save_dreams)
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from learned_judgment.config import require
from learned_judgment.engine.dream import DreamSet, load_dreams, save_dreams
from learned_judgment.engine.streams import stream_rng, stream_seed
from learned_judgment.fileio import atomic_write_bytes
from learned_judgment.skills.base import SkillSpec
from learned_judgment.students.encoding import InputEncoder, SharedInputEncoder
from learned_judgment.students.snn import (
    GRID_PROBE_MAX_ROWS,
    STUDENT_RECORD_FORMAT,
    DeployedStudent,
    candidate_grid_sha256,
    check_dreams,
    check_points_per_dim,
    dream_content_sha256,
    situation_row_digests,
    split_rows_by_situation,
    student_record,
    variable_from_record,
)
from learned_judgment.students.snn_core import OperationCounts, SpikingNetwork, load_network, save_network
from learned_judgment.students.snn_train import SNNSettings
from learned_judgment.students.targets import dream_regret, flat_rows, row_spreads, score_spread, soft_targets

__all__ = ["MANIFEST_FORMAT", "STUDENT_KIND", "SharedSNNStudent"]

STUDENT_KIND = "shared_snn"
MANIFEST_FORMAT = "learned-judgment/shared-spiking-student/v2"
# Earlier manifest formats load() refuses, with the reason.
_RETIRED_MANIFEST_FORMATS = {
    "learned-judgment/shared-spiking-student/v1": (
        "it fixed a list of skill names instead of reserving skill slots, kept only the latest network views (the "
        "views before later adds were lost), and its held-out rows were also the early-stopping rows; retrain the "
        "shared student"
    ),
}
_VIEW_KIND = "shared_snn_view"
_DREAMS_KIND = "shared_snn_stored_dreams"
_MANIFEST_FILE = "manifest.json"
_STATE_FILE = "model_state.npz"
_NETWORKS_DIR = "networks"
_DREAMS_DIR = "dreams"
_TRUNK_INIT_STREAM = "shared_snn/trunk_init"
_DIGEST_LIST_TAG = b"learned-judgment/situation-digest-list/v1"
_DECISION_RUNTIME = "numpy (learned_judgment.students.snn_core); torch was used for training only"
_IN_DREAM_KEYS = (
    "agreement_with_dream_choice",
    "fraction_with_zero_dream_regret",
    "mean_dream_regret",
    "mean_dream_regret_over_spread",
)
_LABEL = (
    "dream-trained shared spiking student: one spiking trunk for all skills, a reserved skill-slot input and one "
    "readout head per skill, each head trained to choose the candidate its skill's dream scores best; every "
    "agreement and regret number is measured inside the dream, not in the simulator; retention numbers come from "
    "retention rows that no optimiser step and no early-stopping evaluation ever saw"
)
_RETENTION_LABEL = (
    "inside the dream, on the skill's retention rows: never in a training batch and never passed to early stopping, "
    "at any add (held out)"
)
_EARLY_STOPPING_LABEL = (
    "inside the dream, on the skill's early-stopping rows: the optimiser never trained on them, but they chose the "
    "restored weights at the skill's own add and, with rehearsal, at later adds, so these numbers are optimistic and "
    "NOT held out"
)
_SPLIT_DEFINITION = (
    "by distinct situation (learned_judgment.students.snn.split_rows_by_situation): first the retention rows "
    "(students.snn.shared_retention_fraction of the distinct situations), then the early-stopping rows "
    "(students.snn.validation_fraction of the remaining distinct situations); the rest are training rows. Rows "
    "with identical situations always fall in one part"
)
_ROW_USES = {
    "training": "optimiser batches at the skill's own add and, with rehearsal, at every later add",
    "early_stopping": "early stopping at the skill's own add and, with rehearsal, at every later add (not held out)",
    "retention": "never passed to training or early stopping; used only for the retention numbers",
}
_ROW_PARTS = ("training", "early_stopping", "retention")


def _json_text(value: Any, label: str, *, sort_keys: bool) -> str:
    try:
        return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not JSON-serializable with finite numbers: {error}") from error


def _canonical_json(value: Any, label: str) -> bytes:
    """UTF-8 JSON with sorted keys and a final newline (the on-disk form)."""
    return (_json_text(value, label, sort_keys=True) + "\n").encode("utf-8")


def _json_copy(value: Any, label: str) -> Any:
    """Deep copy through JSON (key order kept); raises ValueError for non-finite numbers."""
    return json.loads(_json_text(value, label, sort_keys=False))


def _state_bytes(state: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **state)
    return buffer.getvalue()


def _digest_list_sha256(digests: Iterable[str]) -> str:
    """sha256 of a set of situation digests (sorted, one per line): compares row sets cheaply."""
    return hashlib.sha256(_DIGEST_LIST_TAG + "\n".join(sorted(set(digests))).encode("utf-8")).hexdigest()


def _check_integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _check_fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number; got {value!r}")
    fraction = float(value)
    if not (math.isfinite(fraction) and 0.0 < fraction < 1.0):
        raise ValueError(f"{name} must lie in (0, 1); got {fraction}")
    return fraction


def _check_dream_ranges(spec: SkillSpec, dreams: DreamSet) -> None:
    """Refuse dream situations outside the spec's declared float ranges (the student is trained
    on the declared situation space only; categorical levels are checked by the encoder)."""
    for column, variable in enumerate(spec.situation):
        if variable.kind != "float":
            continue
        values = dreams.situations[:, column]
        outside = (values < variable.low) | (values > variable.high)
        if np.any(outside):
            row = int(np.argwhere(outside)[0][0])
            raise ValueError(
                f"dream row {row}: situation feature {variable.name!r} = {float(values[row])!r} lies outside its declared "
                f"range [{variable.low}, {variable.high}]; the shared student is trained on the declared situation "
                "space only"
            )


def _three_way_split(
    skill: str,
    situations: np.ndarray,
    *,
    retention_fraction: float,
    early_stopping_fraction: float,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(training rows, early-stopping rows, retention rows, description): sorted int64 indices
    into the dream's rows (n,), a partition of range(n) by distinct situation."""
    retention_stream = f"{skill}/shared_snn/retention_split"
    early_stream = f"{skill}/shared_snn/early_stopping_split"
    try:
        rest, retention = split_rows_by_situation(
            situations, fraction=retention_fraction, rng=stream_rng(base_seed, retention_stream)
        )
    except ValueError as error:
        raise ValueError(
            f"skill {skill!r}: cannot hold out retention situations (students.snn.shared_retention_fraction "
            f"{retention_fraction}): {error}"
        ) from error
    try:
        train_local, early_local = split_rows_by_situation(
            situations[rest], fraction=early_stopping_fraction, rng=stream_rng(base_seed, early_stream)
        )
    except ValueError as error:
        raise ValueError(
            f"skill {skill!r}: after the {retention.size} retention rows, cannot hold out early-stopping situations "
            f"(students.snn.validation_fraction {early_stopping_fraction}): {error}"
        ) from error
    train, early = rest[train_local], rest[early_local]
    description = {
        "definition": _SPLIT_DEFINITION,
        "retention_fraction": retention_fraction,
        "early_stopping_fraction": early_stopping_fraction,
        "retention_stream": retention_stream,
        "early_stopping_stream": early_stream,
        "n_rows": int(situations.shape[0]),
        "n_distinct_situations": len(set(situation_row_digests(situations))),
        "n_rows_training": int(train.size),
        "n_rows_early_stopping": int(early.size),
        "n_rows_retention": int(retention.size),
        "uses": dict(_ROW_USES),
    }
    return train.astype(np.int64), early.astype(np.int64), retention.astype(np.int64), description


def _row_array(directory: Path, skill: str, part: str, values: Any, n: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or rows.size < 1 or np.any(np.diff(rows) <= 0) or rows[0] < 0 or rows[-1] >= n:
        raise ValueError(f"{directory}: the {part} rows of {skill!r} are not sorted unique rows in [0, {n})")
    return rows


@dataclass(frozen=True, eq=False)
class _HeldSkill:
    """One skill held by the shared network (internal).

    Attributes:
        name, position, slot: skill name; its place in the add sequence (0-based, = the stage at
                             which it was added); its input slot (fixed for the network's life).
        encoder:             the skill's own situation encoder, (n, S) -> (n, D_k), D_k <= feature width.
        shared_encoder:      the shared encoder right after this skill took its slot; encodes this
                             skill's situations into the shared layout (n, W).
        points_per_dim:      candidate grid resolution the dream and the head use.
        n_candidates:        C_k, outputs of the skill's head.
        action_names:        the spec's action names at training time.
        dreams:              the stored dream set (rehearsal rows, early-stopping and retention rows).
        train_rows, early_stopping_rows, retention_rows: sorted int64 row indices, a partition of
                             the dream's rows by distinct situation, fixed when the skill was added.
        spread:              soft-target scale under target_spread "global_median" (score spread of
                             the training rows); None under "per_row".
        regret_scale:        median population std of the training rows' scores (divides the mean
                             dream regret); None when that median is 0.
        rows_sha256:         dream_content_sha256 over every row of the stored dream set.
        seeds:               streams and seeds used when the skill was added.
        record:              snn.student_record(...) of the skill (action records, grid probe and
                             digest, digests of every situation the optimiser or early stopping saw).
        retention_digests:   sorted digests of the retention situations.
    """

    name: str
    position: int
    slot: int
    encoder: InputEncoder
    shared_encoder: SharedInputEncoder
    points_per_dim: tuple[int, ...]
    n_candidates: int
    action_names: tuple[str, ...]
    dreams: DreamSet
    train_rows: np.ndarray
    early_stopping_rows: np.ndarray
    retention_rows: np.ndarray
    spread: float | None
    regret_scale: float | None
    rows_sha256: str
    seeds: dict[str, Any]
    record: dict[str, Any]
    retention_digests: tuple[str, ...]

    def rows(self, part: str) -> np.ndarray:
        """Row indices of ``part`` ("training", "early_stopping" or "retention")."""
        return {
            "training": self.train_rows,
            "early_stopping": self.early_stopping_rows,
            "retention": self.retention_rows,
        }[part]


class SharedSNNStudent:
    """One spiking network for several skills: a shared trunk, a skill-slot input and one readout
    head per skill.

    Construct with :meth:`create` (all values from config.yaml) or explicitly with every value
    given; nothing is defaulted. The student starts empty and grows with :meth:`add_skill`.
    """

    def __init__(
        self,
        *,
        n_slots: int,
        feature_width: int,
        settings: SNNSettings,
        rehearsal_fraction: float,
        retention_fraction: float,
    ) -> None:
        """
        Args:
            n_slots:            skill slots reserved in the input, fixed for the network's life
                                (students.snn.shared_skill_slots), >= 1.
            feature_width:      columns reserved for a skill's own encoding
                                (students.snn.shared_feature_width), >= 1.
            settings:           hyperparameters of the network and of every training run.
            rehearsal_fraction: share of each batch drawn from earlier skills when rehearsing,
                                in (0, 1) (students.snn.rehearsal_fraction).
            retention_fraction: share of each skill's distinct dreamed situations kept only for
                                retention, in (0, 1) (students.snn.shared_retention_fraction).
        """
        n_slots = _check_integer("students.snn.shared_skill_slots", n_slots, 1)
        feature_width = _check_integer("students.snn.shared_feature_width", feature_width, 1)
        if not isinstance(settings, SNNSettings):
            raise TypeError(f"settings must be SNNSettings; got {type(settings).__name__}")
        self._rehearsal_fraction = _check_fraction("students.snn.rehearsal_fraction", rehearsal_fraction)
        self._retention_fraction = _check_fraction("students.snn.shared_retention_fraction", retention_fraction)
        self._encoder = SharedInputEncoder(n_slots=n_slots, feature_width=feature_width)
        self._settings = settings
        self._held: dict[str, _HeldSkill] = {}
        self._stages: list[dict[str, SpikingNetwork]] = []  # stage i: {skill: view} after the i-th add
        self._records: list[dict[str, Any]] = []
        self._state: dict[str, np.ndarray] | None = None
        self._trunk_seed: int | None = None
        self._model: Any = None  # torch model of the committed state; rebuilt from _state when None
        self._deployed: dict[tuple[str, int], DeployedStudent] = {}

    @classmethod
    def create(cls, config: Mapping[str, Any]) -> "SharedSNNStudent":
        """An empty shared student from config.yaml.

        Reads ``students.kind`` (must be ``"snn"``), ``students.shared_snn`` (must be true),
        ``students.snn.shared_skill_slots``, ``students.snn.shared_feature_width``,
        ``students.snn.rehearsal_fraction``, ``students.snn.shared_retention_fraction`` and every
        ``students.snn`` setting (:meth:`SNNSettings.from_config`). A missing or null key raises
        ``ConfigError`` naming it.
        """
        kind = require(config, "students.kind")
        if kind != "snn":
            raise ValueError(f"config.yaml students.kind is {kind!r}; the shared student is a spiking network ('snn')")
        if require(config, "students.shared_snn") is not True:
            raise ValueError("config.yaml students.shared_snn is not true: no shared spiking student is configured")
        return cls(
            n_slots=require(config, "students.snn.shared_skill_slots"),
            feature_width=require(config, "students.snn.shared_feature_width"),
            settings=SNNSettings.from_config(config),
            rehearsal_fraction=require(config, "students.snn.rehearsal_fraction"),
            retention_fraction=require(config, "students.snn.shared_retention_fraction"),
        )

    # ----- structure ----------------------------------------------------------------------

    @property
    def n_slots(self) -> int:
        """Skill slots of the input (fixed at creation)."""
        return self._encoder.n_slots

    @property
    def feature_width(self) -> int:
        return self._encoder.feature_width

    @property
    def input_width(self) -> int:
        """W = feature_width + n_slots: the network's input width, fixed at creation."""
        return self._encoder.width

    @property
    def settings(self) -> SNNSettings:
        return self._settings

    @property
    def rehearsal_fraction(self) -> float:
        return self._rehearsal_fraction

    @property
    def retention_fraction(self) -> float:
        return self._retention_fraction

    @property
    def skills_held(self) -> tuple[str, ...]:
        """Skills added so far, in the order they were added (= slot order)."""
        return tuple(self._held)

    @property
    def slots(self) -> dict[str, int]:
        """{skill: slot} of every skill held."""
        return {name: held.slot for name, held in self._held.items()}

    @property
    def free_slots(self) -> int:
        """Slots not taken by a skill yet."""
        return self._encoder.free_slots

    @property
    def n_stages(self) -> int:
        """Number of stages kept (= skills added): stage i is the network after the i-th add."""
        return len(self._stages)

    def slot_of(self, skill: str) -> int:
        """The input slot ``skill`` holds."""
        return self._held_skill(skill).slot

    def stage_skills(self, stage: int) -> tuple[str, ...]:
        """Skills held at ``stage`` (0-based), in add order."""
        return tuple(self._stages[self._stage_index(stage)])

    def network_view(self, skill: str, stage: int | None = None) -> SpikingNetwork:
        """The numpy network of ``skill`` at ``stage`` (latest when None): shared trunk + that
        skill's head, layer sizes (W, H_1, ..., H_L, C_skill)."""
        held = self._held_skill(skill)
        index = self._stage_index(stage)
        self._check_held_at(held, index)
        return self._stages[index][held.name]

    def network_digests(self, stage: int | None = None) -> dict[str, str]:
        """{skill: sha256 of its view} at ``stage`` (latest when None), in add order."""
        index = self._stage_index(stage)
        return {skill: network.digest() for skill, network in self._stages[index].items()}

    def training_rows(self, skill: str) -> np.ndarray:
        """(n_train,) int64 sorted indices into ``skill``'s stored dream set: the optimiser's rows."""
        return self._held_skill(skill).train_rows.copy()

    def early_stopping_rows(self, skill: str) -> np.ndarray:
        """(n_es,) int64 sorted indices: the skill's early-stopping rows (not held out)."""
        return self._held_skill(skill).early_stopping_rows.copy()

    def retention_rows(self, skill: str) -> np.ndarray:
        """(n_ret,) int64 sorted indices: the skill's retention rows, never trained on and never
        used for early stopping."""
        return self._held_skill(skill).retention_rows.copy()

    def retention_situation_digests(self, skill: str) -> frozenset[str]:
        """Digests (snn.situation_row_digests) of the skill's retention situations."""
        return frozenset(self._held_skill(skill).retention_digests)

    def stored_dreams(self, skill: str) -> DreamSet:
        """The skill's stored dream set (rows indexed by the three row parts)."""
        return self._held_skill(skill).dreams

    @property
    def records(self) -> list[dict[str, Any]]:
        """A copy of every :meth:`add_skill` record, in add order (record i describes stage i)."""
        return _json_copy(self._records, "add_skill records")

    # ----- training -----------------------------------------------------------------------

    def add_skill(
        self,
        spec: SkillSpec,
        dreams: DreamSet,
        *,
        points_per_dim: Sequence[int],
        base_seed: int,
        rehearsal: bool,
    ) -> dict[str, Any]:
        """Give ``spec`` the next free slot, add its head and train the shared network in the dream.

        Args:
            spec:           the skill's SkillSpec; its name (an identifier) must not be held yet.
            dreams:         DreamSet of ``spec``: n rows, C candidates from the grid of ``points_per_dim``,
                            every situation inside the declared ranges.
            points_per_dim: the grid the dream was made with (A integers).
            base_seed:      run seed. Streams under it: ``"<skill>/shared_snn/retention_split"``,
                            ``"<skill>/shared_snn/early_stopping_split"``, ``"<skill>/shared_snn/head_init"``,
                            ``"<skill>/shared_snn/train"`` and, for the first skill only,
                            ``"shared_snn/trunk_init"``.
            rehearsal:      True mixes the earlier skills' stored training rows into every batch;
                            False trains on the new skill's rows only (no-rehearsal control).

        Returns:
            A JSON-serializable record of stage ``position``: slot assignment, settings, split,
            seeds, sampling shares, situation sets used per trained skill, steps run, wall time,
            versions, the training report, a parity report per skill held (on every row of its
            stored dream), the network digests, ``retention_in_dream[skill]`` for EVERY skill
            held (on its retention rows) and ``early_stopping_rows_in_dream[skill]`` (labelled not
            held out).

        Raises:
            ValueError for a skill already held, no free slot, a name that is not an identifier,
            an encoding wider than :attr:`feature_width`, a dream that does not match ``spec`` or
            the grid, dream situations outside the declared ranges, too few distinct situations
            or nothing to learn; TrainingError / ParityError from snn_train. Every refusal before
            training happens without importing torch. On any error the student is left exactly
            as it was before the call.
        """
        from learned_judgment.students import snn_train

        started = time.perf_counter()
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if not isinstance(rehearsal, (bool, np.bool_)):
            raise ValueError(f"rehearsal must be True or False; got {rehearsal!r}")
        if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
            raise ValueError(f"base_seed must be a non-negative integer; got {base_seed!r}")
        base_seed = int(base_seed)
        name = spec.name
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"skill name {name!r} must be a Python identifier: it names the skill's head and its files")
        if name in self._held:
            held = self._held[name]
            raise ValueError(
                f"skill {name!r} already holds slot {held.slot} (added at stage {held.position}; skills held "
                f"{list(self._held)}); each skill is added once"
            )
        if self._encoder.free_slots == 0:
            raise ValueError(
                f"no free skill slot: all {self.n_slots} slots (students.snn.shared_skill_slots) are held by "
                f"{self.slots}; skill {name!r} cannot be added to this network"
            )
        shared_encoder = self._encoder.with_skill(name)
        slot = shared_encoder.slot_of(name)
        points = check_points_per_dim(points_per_dim, spec.n_action)
        encoder = InputEncoder.from_variables(spec.situation)
        if encoder.width > self.feature_width:
            raise ValueError(
                f"skill {name!r} encodes its situation into {encoder.width} columns but "
                f"students.snn.shared_feature_width is {self.feature_width}"
            )
        check_dreams(spec, dreams, points)
        _check_dream_ranges(spec, dreams)
        shared_encoder.encode(name, encoder, dreams.situations)  # refuses invalid categorical levels now

        settings = self._settings
        n = dreams.n_rows
        train_rows, early_rows, retention_rows, split = _three_way_split(
            name,
            dreams.situations,
            retention_fraction=self._retention_fraction,
            early_stopping_fraction=settings.validation_fraction,
            base_seed=base_seed,
        )
        row_digests = situation_row_digests(dreams.situations)
        train_scores = dreams.scores[train_rows]
        train_row_spreads = row_spreads(train_scores)
        train_flat = flat_rows(train_scores)
        if settings.target_spread == "global_median":
            spread: float | None = score_spread(train_scores)
        else:
            spread = None
            if np.all(train_flat):
                raise ValueError(
                    f"skill {name!r}: all {train_rows.size} training rows score every candidate identically; the dream "
                    "gives nothing to learn"
                )
        median_row_spread = float(np.median(train_row_spreads))

        probe_rows: list[int] = []
        probe_seen: set[str] = set()
        for row in train_rows:
            if row_digests[row] not in probe_seen:
                probe_seen.add(row_digests[row])
                probe_rows.append(int(row))
                if len(probe_rows) == GRID_PROBE_MAX_ROWS:
                    break
        grid_probe = dreams.situations[probe_rows]
        optimiser_digests = [row_digests[row] for row in train_rows]
        early_digests = [row_digests[row] for row in early_rows]
        retention_digests = tuple(sorted({row_digests[row] for row in retention_rows}))
        overlap = set(retention_digests) & (set(optimiser_digests) | set(early_digests))
        if overlap:
            raise ValueError(
                f"skill {name!r}: {len(overlap)} retention situation(s) are also training or early-stopping situations; "
                "the split by distinct situation was violated"
            )
        record_student = student_record(
            name,
            encoder,
            points,
            dreams.n_candidates,
            spec.actions,
            grid_probe=grid_probe,
            grid_sha256=candidate_grid_sha256(spec, grid_probe, points),
            optimiser_situations=optimiser_digests,
            early_stopping_situations=early_digests,
            out_of_range=settings.out_of_range,
        )

        head_stream, train_stream = f"{name}/shared_snn/head_init", f"{name}/shared_snn/train"
        trunk_initialised_here = self._trunk_seed is None
        trunk_seed = stream_seed(base_seed, _TRUNK_INIT_STREAM) if trunk_initialised_here else self._trunk_seed
        seeds = {
            "base_seed": base_seed,
            "retention_split_stream": split["retention_stream"],
            "early_stopping_split_stream": split["early_stopping_stream"],
            "head_init_stream": head_stream,
            "head_init_seed": int(stream_seed(base_seed, head_stream)),
            "train_stream": train_stream,
            "train_seed": int(stream_seed(base_seed, train_stream)),
            "trunk_init_stream": _TRUNK_INIT_STREAM,
            "trunk_init_seed": int(trunk_seed),
            "trunk_initialised_here": trunk_initialised_here,
        }
        new = _HeldSkill(
            name=name,
            position=len(self._held),
            slot=slot,
            encoder=encoder,
            shared_encoder=shared_encoder,
            points_per_dim=points,
            n_candidates=dreams.n_candidates,
            action_names=spec.action_names,
            dreams=dreams,
            train_rows=train_rows,
            early_stopping_rows=early_rows,
            retention_rows=retention_rows,
            spread=None if spread is None else float(spread),
            regret_scale=median_row_spread if median_row_spread > 0.0 else None,
            rows_sha256=dream_content_sha256(dreams, np.arange(n)),
            seeds=seeds,
            record=_json_copy(record_student, f"student record of {name!r}"),
            retention_digests=retention_digests,
        )
        earlier = tuple(self._held.values())
        rehearse = bool(rehearsal) and bool(earlier)
        if rehearse:
            share = self._rehearsal_fraction / len(earlier)
            sampling = {name: 1.0 - self._rehearsal_fraction, **{held.name: share for held in earlier}}
        else:
            sampling = {name: 1.0}
        tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        validation: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        used: dict[str, dict[str, Any]] = {}
        for held in (new, *(earlier if rehearse else ())):
            tasks[held.name], validation[held.name] = self._training_pairs(held)
            used[held.name] = self._situations_used(held)

        try:
            model = self._torch_model(trunk_seed)
            model.add_head(name, new.n_candidates, seed=seeds["head_init_seed"])
            report = snn_train.fit(
                model, tasks, validation=validation, sampling=sampling, settings=settings, seed=seeds["train_seed"]
            )
            stage_views: dict[str, SpikingNetwork] = {}
            parity: dict[str, dict[str, Any]] = {}
            retention: dict[str, dict[str, Any]] = {}
            early_numbers: dict[str, dict[str, Any]] = {}
            for held in (*earlier, new):
                network = snn_train.export_network(model, held.name)
                encoded = self._encode_rows(held, np.arange(held.dreams.n_rows))
                parity[held.name] = {
                    **snn_train.check_parity(
                        model, held.name, network, encoded, logit_tolerance=settings.parity_logit_tolerance
                    ),
                    "rows": "every row of the skill's stored dream set (training, early-stopping and retention rows)",
                }
                retention[held.name] = self._in_dream_numbers(held, network, encoded, held.retention_rows)
                early_numbers[held.name] = self._in_dream_numbers(held, network, encoded, held.early_stopping_rows)
                stage_views[held.name] = network
            state = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
            slots_after = {**self.slots, name: slot}
            record = _json_copy(
                {
                    "kind": "shared_snn_add_skill",
                    "label": _LABEL,
                    "decision_runtime": _DECISION_RUNTIME,
                    "stage": new.position,
                    "position": new.position,
                    "skill": name,
                    "slot": slot,
                    "n_slots": self.n_slots,
                    "slots": slots_after,
                    "free_slots": shared_encoder.free_slots,
                    "input_width": self.input_width,
                    "skills_held": list(slots_after),
                    "rehearsal_requested": bool(rehearsal),
                    "rehearsal_applied": rehearse,
                    "rehearsal_source": (
                        "stored dream sets of the earlier skills (their training rows only); no real attempts, no "
                        "provider calls"
                        if rehearse
                        else None
                    ),
                    "rehearsal_fraction": self._rehearsal_fraction,
                    "sampling": dict(report.sampling),
                    "heads_trained": list(report.heads),
                    "early_stopping": (
                        "snn_train.fit's sampling-weighted validation loss over the heads trained in this add "
                        f"({list(report.heads)}), each on its early-stopping rows; no retention row is passed to fit"
                    ),
                    "settings": settings.describe(),
                    "encoder": {"shared": shared_encoder.describe(), "skill": encoder.describe()},
                    "points_per_dim": list(points),
                    "dream": {
                        "method": dreams.method,
                        "request_hashes": list(dreams.request_hashes),
                        "request_groups": [list(group) for group in dreams.request_groups],
                        "n_rows": int(n),
                        "n_candidates": int(dreams.n_candidates),
                        "rows_sha256": new.rows_sha256,
                        "target_spread": settings.target_spread,
                        "score_spread": new.spread,
                        "score_spread_rows": "training rows (null under target_spread 'per_row': each row uses its own)",
                        "training_row_spread": {
                            "definition": "population std of each training row's dreamed scores across its candidates",
                            "min": float(np.min(train_row_spreads)),
                            "median": median_row_spread,
                            "max": float(np.max(train_row_spreads)),
                            "n_flat_rows": int(np.sum(train_flat)),
                        },
                    },
                    "split": split,
                    "seeds": seeds,
                    "situations_used": used,
                    "steps_run": report.steps_run,
                    "wall_seconds": float(time.perf_counter() - started),
                    "versions": dict(report.versions),
                    "training": report.to_json(),
                    "parity": parity,
                    "network_digests": {skill: network.digest() for skill, network in stage_views.items()},
                    "retention_in_dream_label": _RETENTION_LABEL,
                    "retention_in_dream": retention,
                    "early_stopping_rows_in_dream_label": _EARLY_STOPPING_LABEL,
                    "early_stopping_rows_in_dream": early_numbers,
                },
                f"add_skill record of {name!r}",
            )
        except BaseException:
            self._model = None  # it may hold the new head or half-trained weights; rebuilt from the committed state
            raise

        self._encoder = shared_encoder
        self._held[name] = new
        self._stages.append(stage_views)
        self._state = state
        self._trunk_seed = int(trunk_seed)
        self._model = model
        self._records.append(record)
        return _json_copy(record, f"add_skill record of {name!r}")

    def _encode_rows(self, held: _HeldSkill, rows: np.ndarray) -> np.ndarray:
        """Encoded inputs (len(rows), W) of the given rows of the skill's stored dream set."""
        return held.shared_encoder.encode(held.name, held.encoder, held.dreams.situations[rows])

    def _training_pairs(
        self, held: _HeldSkill
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """((training inputs (n_tr, W), targets (n_tr, C)), (early-stopping inputs (n_es, W), targets)).
        Retention rows are never encoded here."""
        settings = self._settings
        pairs = []
        for rows in (held.train_rows, held.early_stopping_rows):
            targets = soft_targets(
                held.dreams.scores[rows],
                temperature=settings.target_temperature,
                target_spread=settings.target_spread,
                spread=held.spread,
            )
            pairs.append((self._encode_rows(held, rows), targets))
        return pairs[0], pairs[1]

    def _situations_used(self, held: _HeldSkill) -> dict[str, Any]:
        """Which of ``held``'s situations one add passes to fit, as counts and digest-set sha256s."""
        seen = held.record["situations_seen"]
        passed = set(seen["optimiser_rows"]) | set(seen["early_stopping_rows"])
        return {
            "optimiser_rows": int(held.train_rows.size),
            "optimiser_situations_sha256": _digest_list_sha256(seen["optimiser_rows"]),
            "early_stopping_rows": int(held.early_stopping_rows.size),
            "early_stopping_situations_sha256": _digest_list_sha256(seen["early_stopping_rows"]),
            "retention_situations_passed_to_fit": len(passed & set(held.retention_digests)),
        }

    def _in_dream_numbers(
        self, held: _HeldSkill, network: SpikingNetwork, encoded: np.ndarray, rows: np.ndarray
    ) -> dict[str, Any]:
        """Agreement and dream regret of ``network`` on ``rows`` of the skill's dream;
        ``encoded`` holds the encoded inputs (n, W) of EVERY row of that dream."""
        decisions = network.decide(encoded[rows]).astype(np.int64)
        regret = dream_regret(held.dreams.scores[rows], decisions)
        mean = float(np.mean(regret))
        return {
            "n_rows": int(rows.size),
            "n_distinct_situations": len(set(situation_row_digests(held.dreams.situations[rows]))),
            "agreement_with_dream_choice": float(np.mean(decisions == held.dreams.chosen[rows])),
            "fraction_with_zero_dream_regret": float(np.mean(regret == 0.0)),
            "mean_dream_regret": mean,
            "mean_dream_regret_over_spread": None if held.regret_scale is None else mean / held.regret_scale,
            "spread_definition": "median population std of the skill's training rows' scores across candidates",
        }

    def _torch_model(self, trunk_seed: int) -> Any:
        """The torch model of the committed state: the cached one, or one rebuilt from the saved
        state dict (after :meth:`load` or a failed add), or a new trunk for the first skill."""
        if self._model is not None:
            return self._model
        import torch

        from learned_judgment.students import snn_train

        model = snn_train.build_model(self.input_width, {}, self._settings, seed=int(trunk_seed))
        if self._state is None:
            return model
        for held in self._held.values():
            model.add_head(held.name, held.n_candidates, seed=held.seeds["head_init_seed"])
        model.load_state_dict({key: torch.as_tensor(np.array(value)) for key, value in self._state.items()}, strict=True)
        model.eval()
        for skill, network in self._stages[-1].items():
            if snn_train.export_network(model, skill).digest() != network.digest():
                raise ValueError(
                    f"the stored model state does not reproduce the latest network view of skill {skill!r}; "
                    "the state file and the networks belong to different students"
                )
        return model

    # ----- deployed views (numpy only) ------------------------------------------------------

    def _held_skill(self, skill: str) -> _HeldSkill:
        if not isinstance(skill, str):
            raise ValueError(f"skill must be a skill name; got {skill!r}")
        if skill not in self._held:
            raise ValueError(
                f"skill {skill!r} has not been added to this shared student; skills held: {list(self._held)} "
                f"({self.free_slots} of {self.n_slots} slots free)"
            )
        return self._held[skill]

    def _stage_index(self, stage: int | None) -> int:
        if not self._stages:
            raise ValueError("this shared student holds no skill yet; it has no stage")
        if stage is None:
            return len(self._stages) - 1
        if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)) or not 0 <= int(stage) < len(self._stages):
            raise ValueError(
                f"stage must be an integer in [0, {len(self._stages)}) (stage i = after the i-th add, 0-based) or None "
                f"for the latest; got {stage!r}"
            )
        return int(stage)

    def _check_held_at(self, held: _HeldSkill, stage: int) -> None:
        if held.position > stage:
            raise ValueError(
                f"skill {held.name!r} was added at stage {held.position}; stage {stage} holds only "
                f"{list(self._stages[stage])}"
            )

    def deployed(self, skill: str, stage: int | None = None) -> DeployedStudent:
        """The :class:`~learned_judgment.students.snn.DeployedStudent` of ``skill`` at ``stage``.

        Args:
            skill: a skill held by this student.
            stage: 0-based stage (the network right after the ``stage``-th add); None = latest.
                   The skill must have been added at or before that stage.

        Returns:
            One object per (skill, stage), so its out_of_range log accumulates: the stage's view
            (trunk + head, input width W), ``encode`` / ``encode_one`` = the shared layout with
            the skill's slot, the skill's situation variables, grid record and digests of every
            situation the optimiser or early stopping saw, and a JSON-serializable meta (kind
            "shared_snn_view", stage, slot, student record, dream provenance, retention and
            early-stopping numbers of that stage).
        """
        held = self._held_skill(skill)
        index = self._stage_index(stage)
        self._check_held_at(held, index)
        key = (held.name, index)
        deployed = self._deployed.get(key)
        if deployed is None:
            deployed = self._build_deployed(held, index)
            self._deployed[key] = deployed
        return deployed

    def _build_deployed(self, held: _HeldSkill, stage: int) -> DeployedStudent:
        network = self._stages[stage][held.name]
        record = held.record
        seen = record["situations_seen"]
        own = self._records[held.position]
        at_stage = self._records[stage]
        meta = _json_copy(
            {
                "kind": _VIEW_KIND,
                "label": _LABEL,
                "decision_runtime": _DECISION_RUNTIME,
                "skill": held.name,
                "stage": stage,
                "stage_definition": "stage i = the shared network right after the i-th add (0-based)",
                "added_at_stage": held.position,
                "slot": held.slot,
                "n_slots": self.n_slots,
                "feature_width": self.feature_width,
                "input_width": self.input_width,
                "skills_held_at_stage": list(self._stages[stage]),
                "slots_at_stage": {name: self._held[name].slot for name in self._stages[stage]},
                "shared_encoder": held.shared_encoder.describe(),
                "network_sha256": network.digest(),
                "student": record,
                "settings": self._settings.describe(),
                "dream": own["dream"],
                "split": own["split"],
                "seeds": own["seeds"],
                "retention_in_dream_label": _RETENTION_LABEL,
                "retention_in_dream": at_stage["retention_in_dream"][held.name],
                "early_stopping_rows_in_dream_label": _EARLY_STOPPING_LABEL,
                "early_stopping_rows_in_dream": at_stage["early_stopping_rows_in_dream"][held.name],
            },
            f"view meta of {held.name!r} at stage {stage}",
        )
        return DeployedStudent(
            skill=held.name,
            network=network,
            encode=functools.partial(held.shared_encoder.encode, held.name, held.encoder),
            encode_one=functools.partial(held.shared_encoder.encode_one, held.name, held.encoder),
            situation_variables=held.encoder.variables,
            points_per_dim=held.points_per_dim,
            n_candidates=held.n_candidates,
            action_records=tuple(record["action_records"]),
            grid_probe=np.asarray(record["candidate_grid"]["probe_situations"], dtype=np.float64),
            grid_sha256=record["candidate_grid"]["sha256"],
            training_situation_digests=frozenset((*seen["optimiser_rows"], *seen["early_stopping_rows"])),
            out_of_range=record["out_of_range"],
            meta=meta,
        )

    # ----- deciding (numpy only, through the latest stage's deployed view) -----------------

    def check_spec(self, spec: SkillSpec) -> None:
        """Refuse a spec the shared student cannot decide for (see DeployedStudent.check_spec):
        ValueError for a skill not held or a changed situation encoding, GridMismatchError for
        changed action records or a changed candidate grid."""
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        self.deployed(spec.name).check_spec(spec)

    def decide_index(self, skill: str, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations of ``skill`` -> (n,) int64 index into its candidate grid (batched path)."""
        return self.deployed(skill).decide_index(situations)

    def decide_index_one(self, skill: str, situation: np.ndarray) -> int:
        """(S,) situation of ``skill`` -> chosen candidate index through the per-decision path."""
        return self.deployed(skill).decide_index_one(situation)

    def decide(self, spec: SkillSpec, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n, A) chosen actions ``spec.candidate_actions(situation,
        points_per_dim)[index]``; runs the spec and situation checks of DeployedStudent."""
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        return self.deployed(spec.name).decide(spec, situations)

    def decide_one(self, spec: SkillSpec, situation: np.ndarray) -> np.ndarray:
        """(S,) situation -> (A,) chosen action through the per-decision path; same checks."""
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        return self.deployed(spec.name).decide_one(spec, situation)

    def operation_counts(self, skill: str, situations: np.ndarray) -> OperationCounts:
        """Operations per decision of ``skill``'s latest view measured on ``situations`` (n, S); the
        first layer counts the full shared input width W (padding and every slot included)."""
        return self.deployed(skill).operation_counts(situations)

    # ----- retention ----------------------------------------------------------------------

    def retention_matrix(self) -> dict[str, Any]:
        """Retention of every skill as later skills were added, from the add_skill records only.

        Returns a JSON-serializable dict. With m skills added (``skills`` in add order), each
        matrix is m x m: ``R[i][j]`` is measured on skill j's RETENTION rows (never trained on,
        never used for early stopping) with the network of stage i, for j <= i, and is ``None``
        for j > i. Matrices: ``agreement_with_dream_choice``, ``fraction_with_zero_dream_regret``,
        ``mean_dream_regret`` and ``mean_dream_regret_over_spread``. ``backward_transfer[key]`` =
        mean over j < m-1 of (R[m-1][j] - R[j][j]); ``None`` when m < 2 or a needed value is
        ``None``: negative for agreement and positive for regret mean the earlier skills got worse.
        """
        skills = [record["skill"] for record in self._records]
        m = len(skills)
        matrices = {
            key: [
                [self._records[i]["retention_in_dream"][skills[j]][key] if j <= i else None for j in range(m)]
                for i in range(m)
            ]
            for key in _IN_DREAM_KEYS
        }
        backward: dict[str, float | None] = {}
        for key in _IN_DREAM_KEYS:
            pairs = [(matrices[key][m - 1][j], matrices[key][j][j]) for j in range(m - 1)]
            if m < 2 or any(after is None or before is None for after, before in pairs):
                backward[key] = None
            else:
                backward[key] = float(np.mean([after - before for after, before in pairs]))
        return _json_copy(
            {
                "label": _LABEL,
                "rows": _RETENTION_LABEL,
                "skills": skills,
                "slots": {record["skill"]: record["slot"] for record in self._records},
                "rehearsal_requested": [record["rehearsal_requested"] for record in self._records],
                "rehearsal_applied": [record["rehearsal_applied"] for record in self._records],
                "n_retention_rows": {
                    record["skill"]: record["retention_in_dream"][record["skill"]]["n_rows"] for record in self._records
                },
                "definition": (
                    "R[i][j]: measured on skill j's retention dream rows with the network of stage i (right after the "
                    "add at position i; rows = stage, columns = skills in add order); null for j > i. "
                    "backward_transfer = mean over j < last of (R[last][j] - R[j][j])"
                ),
                **matrices,
                "backward_transfer": backward,
            },
            "retention matrix",
        )

    # ----- persistence --------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write the student to ``directory`` (see the module docstring for the layout); no pickle.

        Every stage's network views (snn_core.save_network), every stored dream set
        (engine.dream.save_dreams) and the model state dict (named arrays) are written first;
        the manifest, which records their digests, is written last.
        """
        if not self._held or self._state is None:
            raise ValueError("this shared student holds no skill yet; there is nothing to save")
        directory = Path(directory)
        settings = self._settings.describe()
        stages = []
        for stage, views in enumerate(self._stages):
            entries = {}
            for skill, network in views.items():
                held = self._held[skill]
                relative = f"{_NETWORKS_DIR}/stage_{stage}/{skill}"
                digest = save_network(
                    directory / relative,
                    network,
                    {
                        "kind": _VIEW_KIND,
                        "label": _LABEL,
                        "skill": skill,
                        "stage": stage,
                        "slot": held.slot,
                        "n_slots": self.n_slots,
                        "feature_width": self.feature_width,
                        "student": held.record,
                        "settings": settings,
                    },
                )
                entries[skill] = {"file": relative, "network_sha256": digest}
            stages.append(
                {
                    "stage": stage,
                    "skill_added": self._records[stage]["skill"],
                    "skills_held": list(views),
                    "views": entries,
                }
            )
        held_entries = []
        for skill, held in self._held.items():
            save_dreams(
                directory / _DREAMS_DIR / f"{skill}.npz",
                held.dreams,
                {
                    "kind": _DREAMS_KIND,
                    "skill": skill,
                    "use": "training rows (rehearsal), early-stopping rows and retention rows of the shared spiking student",
                    "rows_sha256": held.rows_sha256,
                },
            )
            held_entries.append(
                {
                    "skill": skill,
                    "position": held.position,
                    "slot": held.slot,
                    "points_per_dim": list(held.points_per_dim),
                    "n_candidates": held.n_candidates,
                    "action_names": list(held.action_names),
                    "student": held.record,
                    "dreams_rows_sha256": held.rows_sha256,
                    "n_dream_rows": held.dreams.n_rows,
                    "rows": {part: [int(row) for row in held.rows(part)] for part in _ROW_PARTS},
                    "retention_situations": list(held.retention_digests),
                    "score_spread": held.spread,
                    "regret_scale": held.regret_scale,
                    "seeds": held.seeds,
                }
            )
        state = _state_bytes(self._state)
        atomic_write_bytes(directory / _STATE_FILE, state)
        manifest = {
            "format": MANIFEST_FORMAT,
            "kind": STUDENT_KIND,
            "label": _LABEL,
            "decision_runtime": _DECISION_RUNTIME,
            "n_slots": self.n_slots,
            "feature_width": self.feature_width,
            "input_width": self.input_width,
            "slots": self.slots,
            "free_slots": self.free_slots,
            "shared_encoder": self._encoder.describe(),
            "rehearsal_fraction": self._rehearsal_fraction,
            "retention_fraction": self._retention_fraction,
            "settings": settings,
            "trunk_init_seed": self._trunk_seed,
            "held": held_entries,
            "stages": stages,
            "state_file": _STATE_FILE,
            "state_sha256": hashlib.sha256(state).hexdigest(),
            "state_keys": list(self._state),
            "records": self._records,
            "layout": {
                "networks": f"{_NETWORKS_DIR}/stage_<i>/<skill>.npz + .json (view of <skill> after the i-th add)",
                "dreams": f"{_DREAMS_DIR}/<skill>.npz + .json",
                "state": _STATE_FILE,
            },
        }
        atomic_write_bytes(directory / _MANIFEST_FILE, _canonical_json(manifest, "shared student manifest"))

    @classmethod
    def load(cls, directory: str | Path) -> "SharedSNNStudent":
        """Inverse of :meth:`save`, numpy only (adding a skill to the loaded student imports torch then).

        Verifies the manifest format (refusing a retired v1 manifest with the reason), the slot
        assignment, every stored dream set's digests, every skill's encoder and student record,
        that the three row parts partition each dream and match the recorded situation digests,
        every stage's view digests and widths against the manifest and the add records, and the
        model-state sha256.
        """
        directory = Path(directory)
        manifest = json.loads((directory / _MANIFEST_FILE).read_text(encoding="utf-8"))
        found = manifest.get("format") if isinstance(manifest, dict) else None
        if isinstance(found, str) and found in _RETIRED_MANIFEST_FORMATS:
            raise ValueError(
                f"{directory / _MANIFEST_FILE} is an old {found} manifest, which is no longer read: "
                f"{_RETIRED_MANIFEST_FORMATS[found]}"
            )
        if found != MANIFEST_FORMAT:
            raise ValueError(f"{directory / _MANIFEST_FILE} is not a {MANIFEST_FORMAT} manifest (format {found!r})")
        try:
            settings = SNNSettings(**manifest["settings"])
        except TypeError as error:
            raise ValueError(f"{directory}: the stored settings do not match SNNSettings: {error}") from error
        student = cls(
            n_slots=manifest["n_slots"],
            feature_width=manifest["feature_width"],
            settings=settings,
            rehearsal_fraction=manifest["rehearsal_fraction"],
            retention_fraction=manifest["retention_fraction"],
        )
        shared = student._encoder
        for position, entry in enumerate(manifest["held"]):
            skill = entry["skill"]
            if not isinstance(skill, str) or not skill.isidentifier() or skill in student._held or entry["position"] != position:
                raise ValueError(f"{directory}: manifest entry {position} ({skill!r}) is not a new skill in add order")
            shared = shared.with_skill(skill)
            if entry["slot"] != shared.slot_of(skill):
                raise ValueError(f"{directory}: skill {skill!r} records slot {entry['slot']}; the next free slot was {shared.slot_of(skill)}")
            dreams, _ = load_dreams(directory / _DREAMS_DIR / f"{skill}.npz")
            n = dreams.n_rows
            if dreams.skill != skill or n != entry["n_dream_rows"] or dreams.n_candidates != int(entry["n_candidates"]):
                raise ValueError(f"{directory}: dreams/{skill}.npz is not the stored dream set of {skill!r}")
            if dream_content_sha256(dreams, np.arange(n)) != entry["dreams_rows_sha256"]:
                raise ValueError(f"{directory}: dreams/{skill}.npz rows do not match the digest in the manifest")
            record = entry["student"]
            if (
                not isinstance(record, dict)
                or record.get("format") != STUDENT_RECORD_FORMAT
                or record.get("skill") != skill
                or list(record.get("points_per_dim", [])) != list(entry["points_per_dim"])
                or record.get("n_candidates") != int(entry["n_candidates"])
            ):
                raise ValueError(f"{directory}: the student record of {skill!r} is not a matching {STUDENT_RECORD_FORMAT} record")
            encoder = InputEncoder.from_variables([variable_from_record(item) for item in record["situation_variables"]])
            if not encoder.matches(record["encoder"]) or encoder.width > student.feature_width:
                raise ValueError(f"{directory}: the stored situation variables of {skill!r} do not rebuild its encoder")
            rows = {part: _row_array(directory, skill, part, entry["rows"][part], n) for part in _ROW_PARTS}
            joined = np.sort(np.concatenate([rows[part] for part in _ROW_PARTS]))
            if not np.array_equal(joined, np.arange(n)):
                raise ValueError(f"{directory}: the training, early-stopping and retention rows of {skill!r} do not partition its dream")
            digests = situation_row_digests(dreams.situations)
            seen = record["situations_seen"]
            retention_digests = sorted({digests[row] for row in rows["retention"]})
            if (
                sorted({digests[row] for row in rows["training"]}) != list(seen["optimiser_rows"])
                or sorted({digests[row] for row in rows["early_stopping"]}) != list(seen["early_stopping_rows"])
                or retention_digests != list(entry["retention_situations"])
                or set(retention_digests) & (set(seen["optimiser_rows"]) | set(seen["early_stopping_rows"]))
            ):
                raise ValueError(
                    f"{directory}: the stored rows of {skill!r} do not match the recorded situation digests, or a "
                    "retention situation is also a training or early-stopping situation"
                )
            student._held[skill] = _HeldSkill(
                name=skill,
                position=position,
                slot=int(entry["slot"]),
                encoder=encoder,
                shared_encoder=shared,
                points_per_dim=tuple(int(p) for p in entry["points_per_dim"]),
                n_candidates=int(entry["n_candidates"]),
                action_names=tuple(entry["action_names"]),
                dreams=dreams,
                train_rows=rows["training"],
                early_stopping_rows=rows["early_stopping"],
                retention_rows=rows["retention"],
                spread=None if entry["score_spread"] is None else float(entry["score_spread"]),
                regret_scale=None if entry["regret_scale"] is None else float(entry["regret_scale"]),
                rows_sha256=entry["dreams_rows_sha256"],
                seeds=dict(entry["seeds"]),
                record=_json_copy(record, f"student record of {skill!r}"),
                retention_digests=tuple(retention_digests),
            )
        if not student._held:
            raise ValueError(f"{directory}: the manifest holds no skill")
        if shared.describe() != manifest["shared_encoder"] or manifest["slots"] != {k: v.slot for k, v in student._held.items()}:
            raise ValueError(f"{directory}: the slot assignment does not rebuild the stored shared encoder")
        student._encoder = shared

        records = manifest["records"]
        names = list(student._held)
        if [record["skill"] for record in records] != names or [record["stage"] for record in records] != list(range(len(names))):
            raise ValueError(f"{directory}: the add_skill records do not match the held skills in add order")
        if len(manifest["stages"]) != len(names):
            raise ValueError(f"{directory}: {len(manifest['stages'])} stages for {len(names)} skills added")
        for stage, entry in enumerate(manifest["stages"]):
            at_stage = names[: stage + 1]
            if (
                entry["stage"] != stage
                or entry["skill_added"] != names[stage]
                or entry["skills_held"] != at_stage
                or sorted(entry["views"]) != sorted(at_stage)
            ):
                raise ValueError(f"{directory}: stage {stage} of the manifest does not hold {at_stage}")
            views: dict[str, SpikingNetwork] = {}
            for skill in at_stage:
                view = entry["views"][skill]
                relative = f"{_NETWORKS_DIR}/stage_{stage}/{skill}"
                if view["file"] != relative:
                    raise ValueError(f"{directory}: stage {stage} names {view['file']!r} for {skill!r}, not {relative!r}")
                network, view_meta = load_network(directory / relative)
                if view_meta.get("kind") != _VIEW_KIND or view_meta.get("skill") != skill or view_meta.get("stage") != stage:
                    raise ValueError(f"{directory}: {relative} is not the view of {skill!r} at stage {stage}")
                if network.digest() != view["network_sha256"] or network.digest() != records[stage]["network_digests"][skill]:
                    raise ValueError(f"{directory}: {relative} does not match the digest in the manifest and the add record")
                if network.n_inputs != student.input_width or network.n_outputs != student._held[skill].n_candidates:
                    raise ValueError(
                        f"{directory}: {relative} has layers {network.layer_sizes}; expected input width "
                        f"{student.input_width} and {student._held[skill].n_candidates} outputs"
                    )
                views[skill] = network
            student._stages.append(views)

        if manifest["state_file"] != _STATE_FILE:
            raise ValueError(f"{directory}: the manifest names state file {manifest['state_file']!r}, not {_STATE_FILE!r}")
        state_bytes = (directory / _STATE_FILE).read_bytes()
        if hashlib.sha256(state_bytes).hexdigest() != manifest["state_sha256"]:
            raise ValueError(f"{directory}: {_STATE_FILE} does not match the sha256 in the manifest")
        with np.load(io.BytesIO(state_bytes), allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(manifest["state_keys"]):
                raise ValueError(f"{directory}: {_STATE_FILE} arrays differ from the manifest's state keys")
            state = {key: archive[key] for key in manifest["state_keys"]}
        student._state = state
        student._records = _json_copy(records, "add_skill records")
        student._trunk_seed = _check_integer("trunk_init_seed", manifest["trunk_init_seed"], 0)
        return student
