"""Dreaming: teacher decisions for many sampled situations (engine lane, Person B).

Lane contract
-------------
:func:`dream` samples situations with ``spec.sample_situations`` and decides all of them
with ONE :func:`~learned_judgment.engine.decide.decide_batch` call, i.e. one residual-model
predict call for every situation x candidate. The model may serve that call with several
cached provider requests (one per outcome, or a large query set split into parts); every
request key behind every decision is recorded (``request_groups`` / ``request_group``). The
result is a :class:`DreamSet`: the complete, self-describing training set of the spiking
student (written by another lane). Dreams are teacher decisions, never measurements;
whether they hold in the true simulator is measured separately (dream fidelity).

:func:`keep_confident` keeps decisions whose predictive spread of ``spec.primary_outcome``
at the chosen candidate is at most a threshold that the caller selected on dev data.

Arrays (float64 unless noted)::

    n   dreamed situations            C   candidates per situation (equal for every row)
    S   situation features            A   action dimensions          K   outcomes

On disk a DreamSet is two files written atomically (tmp file in the same directory, fsync,
``os.replace``): ``<name>.npz`` with the arrays (loaded with ``allow_pickle=False``) and
``<name>.json`` with the names, provenance, the caller's metadata and the sha256 of the
``.npz`` bytes, so a mismatched pair is detected on load. Format ``v2`` records every request
key per decision; a ``v1`` file (which kept only the last key of each decision) is refused.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from learned_judgment.fileio import atomic_write_bytes
from learned_judgment.engine.decide import decide_batch
from learned_judgment.engine.features import FeatureLayout
from learned_judgment.engine.rows import AttemptRow, _dream_row, decision_request_group
from learned_judgment.engine.types import Decision, ResidualModel
from learned_judgment.skills.base import SkillSpec

__all__ = [
    "DREAM_FORMAT",
    "DreamSet",
    "dream",
    "dream_rows",
    "dreams_from_decisions",
    "keep_confident",
    "load_dreams",
    "save_dreams",
]

DREAM_FORMAT = "learned-judgment-dreamset-v2"
# Earlier formats load_dreams refuses, with the reason.
_RETIRED_FORMATS = {
    "learned-judgment-dreamset-v1": (
        "it records only the last provider request of each decision (request_index), so the other "
        "requests behind a decision are lost; regenerate the dreams with engine.dream.dream"
    ),
}

_FLOAT_ARRAYS = (
    "situations",
    "candidates",
    "scores",
    "chosen_spread",
    "chosen_physics",
    "chosen_mean",
    "chosen_median",
    "chosen_outcome_spread",
)
_INT_ARRAYS = ("chosen", "physics_choice", "request_group")
_HEADER_KEYS = frozenset(
    (
        "format",
        "skill",
        "method",
        "situation_names",
        "action_names",
        "outcome_names",
        "primary_outcome",
        "spread_levels",
        "request_hashes",
        "request_groups",
        "n_rows",
        "n_candidates",
        "arrays_file",
        "arrays_sha256",
        "meta",
    )
)


@dataclass(frozen=True, eq=False)
class DreamSet:
    """Teacher decisions for ``n`` dreamed situations.

    Attributes:
        skill, method:          SkillSpec name; label of the scoring method.
        situations:             (n, S) dreamed situations.
        candidates:             (n, C, A) every situation's full candidate grid.
        scores:                 (n, C) objective per candidate (higher is better).
        chosen:                 (n,) int64 index of the chosen candidate.
        physics_choice:         (n,) int64 index physics alone would choose.
        chosen_spread:          (n,) predictive spread of ``primary_outcome`` at the chosen
                                candidate (the confidence used by :func:`keep_confident`).
        request_hashes:         distinct cache keys of the residual-model requests behind
                                this set, in first-use order (empty when none was made): the
                                hashes of ``request_groups`` in order, each once.
        request_groups:         distinct groups of request keys; a group lists every request
                                behind one decision, in request order (several when the model
                                sends one request per outcome or splits a large query set).
        request_group:          (n,) int64 index into ``request_groups``; -1 = no request.
        situation_names, action_names, outcome_names: column names in spec order.
        primary_outcome:        member of ``outcome_names``.
        spread_levels:          (low, high) quantile levels the spreads were computed from.
        chosen_physics:         (n, K) physics guess at the chosen candidate.
        chosen_mean:            (n, K) predicted outcome mean at the chosen candidate
                                (a binary outcome's mean is its probability).
        chosen_median:          (n, K) predicted outcome median at the chosen candidate.
        chosen_outcome_spread:  (n, K) predictive spread of every outcome at the chosen candidate.
    """

    skill: str
    method: str
    situations: np.ndarray
    candidates: np.ndarray
    scores: np.ndarray
    chosen: np.ndarray
    physics_choice: np.ndarray
    chosen_spread: np.ndarray
    request_hashes: tuple[str, ...]
    request_groups: tuple[tuple[str, ...], ...]
    request_group: np.ndarray
    situation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    primary_outcome: str
    spread_levels: tuple[float, float]
    chosen_physics: np.ndarray
    chosen_mean: np.ndarray
    chosen_median: np.ndarray
    chosen_outcome_spread: np.ndarray

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        for name in ("skill", "method", "primary_outcome"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"DreamSet.{name} must be a non-empty string; got {value!r}")
        for name in ("situation_names", "action_names", "outcome_names", "request_hashes"):
            value = tuple(getattr(self, name))
            if not all(isinstance(item, str) and item for item in value) or len(set(value)) != len(value):
                raise ValueError(f"DreamSet.{name} must hold unique non-empty strings; got {value}")
            set_(self, name, value)
        if self.primary_outcome not in self.outcome_names:
            raise ValueError(f"primary_outcome {self.primary_outcome!r} is not in {self.outcome_names}")
        if isinstance(self.request_groups, (str, bytes)):
            raise ValueError("DreamSet.request_groups must be a sequence of groups of request hashes")
        groups = []
        for position, group in enumerate(self.request_groups):
            if isinstance(group, (str, bytes)):
                raise ValueError(f"DreamSet.request_groups[{position}] must be a sequence of hashes, not one string")
            group = tuple(group)
            if not group or not all(isinstance(item, str) and item for item in group) or len(set(group)) != len(group):
                raise ValueError(
                    f"DreamSet.request_groups[{position}] must hold one or more unique non-empty strings; got {group}"
                )
            groups.append(group)
        if len(set(groups)) != len(groups):
            raise ValueError("DreamSet.request_groups must not repeat a group")
        set_(self, "request_groups", tuple(groups))
        flat = tuple(dict.fromkeys(item for group in groups for item in group))
        if self.request_hashes != flat:
            raise ValueError(
                f"DreamSet.request_hashes {self.request_hashes} must be the distinct hashes of request_groups "
                f"in first-use order {flat}"
            )
        low, high = (float(level) for level in self.spread_levels)
        if not 0.0 < low < high < 1.0:
            raise ValueError(f"spread_levels must satisfy 0 < low < high < 1; got {self.spread_levels}")
        set_(self, "spread_levels", (low, high))

        for name in _FLOAT_ARRAYS:
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"DreamSet.{name} contains non-finite values")
            set_(self, name, array)
        for name in _INT_ARRAYS:
            raw = np.asarray(getattr(self, name))
            if raw.size and raw.dtype.kind not in "iu":
                raise ValueError(f"DreamSet.{name} must hold integers; got dtype {raw.dtype}")
            set_(self, name, raw.astype(np.int64))

        s, a, k = len(self.situation_names), len(self.action_names), len(self.outcome_names)
        if self.candidates.ndim != 3 or self.candidates.shape[2] != a or self.candidates.shape[1] < 1:
            raise ValueError(f"candidates must have shape (n, C>=1, {a}); got {self.candidates.shape}")
        n, c = self.candidates.shape[:2]
        expected = {
            "situations": (n, s),
            "scores": (n, c),
            "chosen": (n,),
            "physics_choice": (n,),
            "chosen_spread": (n,),
            "request_group": (n,),
            "chosen_physics": (n, k),
            "chosen_mean": (n, k),
            "chosen_median": (n, k),
            "chosen_outcome_spread": (n, k),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"DreamSet.{name} must have shape {shape}; got {getattr(self, name).shape}")
        for name in ("chosen", "physics_choice"):
            index = getattr(self, name)
            if np.any(index < 0) or np.any(index >= c):
                raise ValueError(f"DreamSet.{name} holds an index outside [0, {c})")
        if np.any(self.request_group < -1) or np.any(self.request_group >= len(self.request_groups)):
            raise ValueError(f"DreamSet.request_group must lie in [-1, {len(self.request_groups)})")
        if np.any(self.chosen_outcome_spread < 0.0):
            raise ValueError("DreamSet.chosen_outcome_spread must be non-negative")
        primary = self.outcome_names.index(self.primary_outcome)
        if not np.array_equal(self.chosen_spread, self.chosen_outcome_spread[:, primary]):
            raise ValueError("chosen_spread must equal chosen_outcome_spread of the primary outcome")

    @property
    def n_rows(self) -> int:
        return int(self.situations.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def chosen_actions(self) -> np.ndarray:
        """(n, A) the teacher's chosen actions."""
        return self.candidates[np.arange(self.n_rows), self.chosen]

    @property
    def physics_actions(self) -> np.ndarray:
        """(n, A) the actions physics alone would choose."""
        return self.candidates[np.arange(self.n_rows), self.physics_choice]

    def request_hashes_of(self, row: int) -> tuple[str, ...]:
        """Every request key behind the decision of ``row``, in request order (``()`` for none)."""
        group = int(self.request_group[row])
        return () if group < 0 else self.request_groups[group]


def _check_spec(spec: SkillSpec, dreams: DreamSet) -> None:
    for label, have, want in (
        ("skill", dreams.skill, spec.name),
        ("situation_names", dreams.situation_names, spec.situation_names),
        ("action_names", dreams.action_names, spec.action_names),
        ("outcome_names", dreams.outcome_names, spec.outcome_names),
        ("primary_outcome", dreams.primary_outcome, spec.primary_outcome),
    ):
        if have != want:
            raise ValueError(f"DreamSet.{label} {have!r} does not match the spec's {want!r}")


def dreams_from_decisions(
    spec: SkillSpec, decisions: Sequence[Decision], *, spread_levels: tuple[float, float]
) -> DreamSet:
    """Stack ``decisions`` (one per situation) into a :class:`DreamSet`.

    All decisions must belong to ``spec``, share one method label and have the same number
    of candidates C (ValueError otherwise). ``spread_levels`` must be the levels the
    decisions were made with (they are recorded, not recomputed). Every request key of every
    decision (:func:`~learned_judgment.engine.rows.decision_request_group`) is recorded.
    """
    if not decisions:
        raise ValueError("decisions is empty; a DreamSet needs at least one decision")
    methods = {decision.method for decision in decisions}
    if len(methods) != 1:
        raise ValueError(f"decisions mix methods {sorted(methods)}")
    counts = sorted({int(decision.candidates.shape[0]) for decision in decisions})
    if len(counts) != 1:
        raise ValueError(f"candidate counts differ between situations: {counts}")
    primary = spec.outcome_index(spec.primary_outcome)
    groups: dict[tuple[str, ...], int] = {}
    request_group = []
    for position, decision in enumerate(decisions):
        if decision.skill != spec.name:
            raise ValueError(f"decisions[{position}] belongs to skill {decision.skill!r}, not {spec.name!r}")
        if decision.predicted.names != spec.outcome_names:
            raise ValueError(f"decisions[{position}] outcomes {decision.predicted.names} differ from the spec")
        try:
            group = decision_request_group(decision)
        except ValueError as error:
            raise ValueError(f"decisions[{position}]: {error}") from error
        if not group:
            request_group.append(-1)
            continue
        request_group.append(groups.setdefault(group, len(groups)))
    ordered_groups = tuple(groups)

    def chosen_rows(attribute: str) -> np.ndarray:
        return np.stack([getattr(d.predicted, attribute)[d.chosen_index] for d in decisions])

    outcome_spread = chosen_rows("spread")
    return DreamSet(
        skill=spec.name,
        method=methods.pop(),
        situations=np.stack([decision.situation for decision in decisions]),
        candidates=np.stack([decision.candidates for decision in decisions]),
        scores=np.stack([decision.scores for decision in decisions]),
        chosen=np.array([decision.chosen_index for decision in decisions], dtype=np.int64),
        physics_choice=np.array([decision.physics_index for decision in decisions], dtype=np.int64),
        chosen_spread=outcome_spread[:, primary],
        request_hashes=tuple(dict.fromkeys(item for group in ordered_groups for item in group)),
        request_groups=ordered_groups,
        request_group=np.array(request_group, dtype=np.int64),
        situation_names=spec.situation_names,
        action_names=spec.action_names,
        outcome_names=spec.outcome_names,
        primary_outcome=spec.primary_outcome,
        spread_levels=spread_levels,
        chosen_physics=chosen_rows("physics"),
        chosen_mean=chosen_rows("mean"),
        chosen_median=chosen_rows("median"),
        chosen_outcome_spread=outcome_spread,
    )


def dream(
    spec: SkillSpec,
    model: ResidualModel | None,
    layout: FeatureLayout,
    *,
    n_dreams: int,
    rng: np.random.Generator,
    points_per_dim: Sequence[int],
    quantile_levels: Sequence[float],
    spread_levels: tuple[float, float],
    method: str,
    extra: np.ndarray | None = None,
) -> DreamSet:
    """Sample ``n_dreams`` situations (n, S) with ``spec.sample_situations(rng, n_dreams)`` and
    decide all of them in one :func:`decide_batch` call (one predict call of ``model``; every
    provider request it made is recorded in the DreamSet). Situation i does not depend on
    ``n_dreams`` when the spec samples row by row, as the default ``sample_situations`` does.

    Args:
        model:   the fitted residual model (the teacher), or ``None`` with ``method ==
                 PHYSICS_ONLY`` for physics-only decisions.
        rng:     the dream-situation stream (e.g. ``stream_rng(seed, "<skill>/dreams/situations")``).
        extra:   (n_dreams, E) observed context when ``layout`` has an extra block.

    Raises ValueError when situations do not all get the same number of candidates C.
    """
    if isinstance(n_dreams, bool) or not isinstance(n_dreams, (int, np.integer)) or n_dreams < 1:
        raise ValueError(f"n_dreams must be an int >= 1; got {n_dreams!r}")
    situations = np.asarray(spec.sample_situations(rng, int(n_dreams)), dtype=np.float64)
    if situations.shape != (int(n_dreams), spec.n_situation):
        raise ValueError(
            f"sample_situations must return ({n_dreams}, {spec.n_situation}); got {situations.shape}"
        )
    decisions = decide_batch(
        spec,
        situations,
        model=model,
        layout=layout,
        points_per_dim=points_per_dim,
        quantile_levels=quantile_levels,
        spread_levels=spread_levels,
        method=method,
        extra=extra,
    )
    return dreams_from_decisions(spec, decisions, spread_levels=spread_levels)


def keep_confident(dreams: DreamSet, max_chosen_spread: float) -> DreamSet:
    """Rows whose ``chosen_spread <= max_chosen_spread`` (exact comparison).

    The threshold is chosen on dev data by the caller and passed in; nothing is defaulted.
    The result may have zero rows.
    """
    if isinstance(max_chosen_spread, bool) or not isinstance(max_chosen_spread, (int, float, np.floating)):
        raise ValueError(f"max_chosen_spread must be a number; got {max_chosen_spread!r}")
    threshold = float(max_chosen_spread)
    if math.isnan(threshold) or threshold < 0.0:
        raise ValueError(f"max_chosen_spread must be >= 0 (a spread is non-negative); got {threshold}")
    keep = dreams.chosen_spread <= threshold
    per_row = {name: getattr(dreams, name)[keep] for name in (*_FLOAT_ARRAYS, *_INT_ARRAYS)}
    return dataclasses.replace(dreams, **per_row)


def dream_rows(
    spec: SkillSpec,
    dreams: DreamSet,
    *,
    run_id: str,
    seed: int,
    world_preset: str,
    timestamp_utc: str,
) -> list[AttemptRow]:
    """One shared-schema row with ``source == "dream"`` per dream (see engine.rows); each row's
    teacher summary lists every request key behind its decision."""
    _check_spec(spec, dreams)
    rows = []
    for i in range(dreams.n_rows):
        rows.append(
            _dream_row(
                spec,
                run_id=run_id,
                seed=seed,
                world_preset=world_preset,
                timestamp_utc=timestamp_utc,
                method=dreams.method,
                request_hashes=dreams.request_hashes_of(i),
                situation=dreams.situations[i],
                candidates=dreams.candidates[i],
                scores=dreams.scores[i],
                chosen_index=int(dreams.chosen[i]),
                physics_index=int(dreams.physics_choice[i]),
                chosen_physics=dreams.chosen_physics[i],
                chosen_mean=dreams.chosen_mean[i],
                chosen_median=dreams.chosen_median[i],
                chosen_spread=dreams.chosen_outcome_spread[i],
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def _canonical_json(value: Any) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _paths(path: Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"a DreamSet path must end in .npz; got {path}")
    return path, path.with_suffix(".json")


def save_dreams(path: Path, dreams: DreamSet, meta: dict) -> None:
    """Write ``dreams`` to ``path`` (``.npz``) and its metadata to the sibling ``.json``.

    ``meta`` is the caller's provenance (config digest, seeds, threshold, ...) and must be
    JSON-serializable with finite numbers. The arrays file is written first; the JSON names
    its sha256, so a crash between the two writes is detected by :func:`load_dreams`.
    """
    arrays_path, meta_path = _paths(path)
    if not isinstance(meta, dict):
        raise TypeError(f"meta must be a dict; got {type(meta).__name__}")
    buffer = io.BytesIO()
    np.savez(buffer, **{name: getattr(dreams, name) for name in (*_FLOAT_ARRAYS, *_INT_ARRAYS)})
    arrays = buffer.getvalue()
    header = {
        "format": DREAM_FORMAT,
        "skill": dreams.skill,
        "method": dreams.method,
        "situation_names": list(dreams.situation_names),
        "action_names": list(dreams.action_names),
        "outcome_names": list(dreams.outcome_names),
        "primary_outcome": dreams.primary_outcome,
        "spread_levels": list(dreams.spread_levels),
        "request_hashes": list(dreams.request_hashes),
        "request_groups": [list(group) for group in dreams.request_groups],
        "n_rows": dreams.n_rows,
        "n_candidates": dreams.n_candidates,
        "arrays_file": arrays_path.name,
        "arrays_sha256": hashlib.sha256(arrays).hexdigest(),
        "meta": meta,
    }
    try:
        encoded_header = _canonical_json(header)
    except (TypeError, ValueError) as error:
        raise ValueError(f"meta is not JSON-serializable with finite numbers: {error}") from error
    atomic_write_bytes(arrays_path, arrays)
    atomic_write_bytes(meta_path, encoded_header)


def load_dreams(path: Path) -> tuple[DreamSet, dict]:
    """Inverse of :func:`save_dreams`: returns ``(dreams, meta)``.

    Raises ValueError when the format tag, the arrays digest or the stored array names do
    not match, and names the reason when the file is of a retired format (``v1``). Arrays are
    read with ``allow_pickle=False``.
    """
    arrays_path, meta_path = _paths(path)
    header = json.loads(meta_path.read_text(encoding="utf-8"))
    found = header.get("format") if isinstance(header, dict) else None
    if isinstance(found, str) and found in _RETIRED_FORMATS:
        raise ValueError(f"{meta_path} is an old {found} file, which is no longer read: {_RETIRED_FORMATS[found]}")
    if found != DREAM_FORMAT:
        raise ValueError(f"{meta_path} is not a {DREAM_FORMAT} file (format {found!r})")
    if set(header) != _HEADER_KEYS:
        raise ValueError(
            f"{meta_path} header keys do not match; missing={sorted(_HEADER_KEYS - set(header))} "
            f"extra={sorted(set(header) - _HEADER_KEYS)}"
        )
    groups = header["request_groups"]
    if not isinstance(groups, list) or not all(isinstance(group, list) for group in groups):
        raise ValueError(f"{meta_path}: request_groups must be a list of lists of request hashes")
    data = arrays_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != header["arrays_sha256"]:
        raise ValueError(f"{arrays_path} does not match the sha256 recorded in {meta_path}")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        names = set(archive.files)
        expected = set(_FLOAT_ARRAYS) | set(_INT_ARRAYS)
        if names != expected:
            raise ValueError(
                f"{arrays_path} arrays do not match; missing={sorted(expected - names)} extra={sorted(names - expected)}"
            )
        arrays = {name: archive[name] for name in expected}
    dreams = DreamSet(
        skill=header["skill"],
        method=header["method"],
        situation_names=tuple(header["situation_names"]),
        action_names=tuple(header["action_names"]),
        outcome_names=tuple(header["outcome_names"]),
        primary_outcome=header["primary_outcome"],
        spread_levels=tuple(header["spread_levels"]),
        request_hashes=tuple(header["request_hashes"]),
        request_groups=tuple(tuple(group) for group in header["request_groups"]),
        **arrays,
    )
    if dreams.n_rows != header["n_rows"] or dreams.n_candidates != header["n_candidates"]:
        raise ValueError(f"{arrays_path} shapes disagree with the n_rows / n_candidates in {meta_path}")
    return dreams, header["meta"]
