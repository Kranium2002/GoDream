"""The learning run of a skill whose inputs are observed inside the episode (image windows).

Lane contract (Person B, observed-input skills such as dodge_vision and ramp)
-----------------------------------------------------------------------------
Inputs are the files of the shared array contract (C candidates in the spec's action order, K
outcomes in the spec's outcome order, D = frames x N x N window features in [0, 1], oldest
frame first as learned_judgment.vision.flatten_window; a skill whose features are not image
windows, such as ramp, writes "frames": null and any D features in [0, 1], which train uses as
stored and save_student records as "input": {"features": D, "encoding"}):

    rows file   <dir>/rows.npz   scenario_index (n,) int64, decision_time_s (n,) float64,
                                 action (n,) int64, features (n, D) float32,
                                 physics (n, K) float64, measured (n, K) float64
                <dir>/rows.json  skill, split, action_names, outcome_names, outcome_kinds,
                                 outcome_bounds, frames, n_situations, seeds, world_seed,
                                 world_preset, config_digest, kind counts, scenarios
                                 (list of dicts, index = scenario_index; "kind" read by
                                 one_shot_report)
    dream file  <dir>/dream.npz  scenario_index (m,), decision_time_s (m,), features (m, D)
                                 float32, physics (m, C, K) float64; <dir>/dream.json (same meta)

The run, with nothing skill-specific in this module (the spec object carries the skill):

    teacher_matrix   [window (D) | action features (A) | physics guess (K)] per row, in the
                     blocks named by skills.<name>.teacher_features
    make_teacher     the production TabPFN residual teacher fitted on measured - physics of
                     the rows file (fit stores the context; no network)
    quote_dream      the provider's free quote of every request a prediction would send
    predict_outcomes physics + predicted residual, clipped to the spec's outcome bounds; the
                     ONLY billable call here, and only with allow_billing=True
    dream_scores     spec.objective of every dream candidate, (m, C), one batched prediction
    dream_targets    soft action targets (students.targets.soft_targets) and threat targets
                     [1 - p, p]: p = logistic((best active score - NO_OP score) / scale), or by
                     skills.<name>.threat_target (threat_target_settings): the dream's predicted
                     NO_OP collision (collision_course) or its predicted NO_OP separation below a
                     set distance (near_approach)
    train_student    a spiking trunk with two readout heads (threat 2 | action C) trained on
                     the dream targets; exported as ONE numpy SpikingNetwork whose outputs are
                     [threat (2) | action (C)], parity-checked against the torch model
    save_student     network.npz + network.json (snn_core.save_network) and student.json

Student input encoding, image windows: the stored RAW window of F frames (newest F of the
dream's, vision.take_frames) -> :func:`encode_window_features` with config.yaml vision.encoding
and vision.diff_floor ("abs_diff": the F - 1 adjacent-frame changes |frame k - frame k-1|,
divided per window by max(largest change, diff_floor); "raw": the F frames) -> x -> (x - 0.5) *
sqrt(12) per feature (the existing InputEncoder's map for a float feature declared on [0, 1]);
all stated in student.json "input" ("frames" = F camera frames, "frame_size", "window_encoding",
"diff_floor", "encoding"). learned_judgment.runtime applies the same function. The teacher's
matrix stays on the raw windows. Readout: softmax over each head's logits; threat confidence =
softmax(threat logits)[1].

Billing: only :func:`predict_outcomes` (and :func:`dream_scores`, which calls it) sends a
billable request, and both raise RuntimeError before any network use unless the caller passes
``allow_billing=True``. :func:`quote_dream` calls only free provider endpoints:
TabPFNProvider.connect (prior_labs.connect -> tabpfn_client.init), TabPFNProvider.usage
(ServiceClient.get_api_usage) and TabPFNProvider.quote (tabpfn_client.estimate_cost on zero
arrays of the request's shape). The teacher's own gate still decides every request.

Command line (from the repository root), writing under <observed.runs_dir>/<RUN_ID>/
(events.jsonl, result.json, scores.npz, student/)::

    python -m learned_judgment.observed quote --run RUN --skill NAME --context DIR --dream DIR
    python -m learned_judgment.observed teach --run RUN --skill NAME --context DIR --dream DIR \
        [--dev DIR] [--confirm-billing]
    python -m learned_judgment.observed train --run RUN --skill NAME --dream DIR --scores FILE \
        [--frames F] [--hidden H ...] [--steps N]

Exit codes: 0 done, 1 failed (traceback), 3 refused (no --confirm-billing, or the teacher's
gate refused).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import io
import json
import math
import pkgutil
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.config import CONFIG_PATH, ConfigError, config_digest, load_config, optional, require
from learned_judgment.engine.features import FeatureLayout
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.fileio import atomic_write_bytes, atomic_write_text
from learned_judgment.students.snn_core import SpikingNetwork, save_network
from learned_judgment.students.targets import score_spread, soft_targets
from learned_judgment.teacher import (
    BudgetError,
    CacheLockedError,
    LedgerMissingError,
    TabPFNProvider,
    TabPFNTeacher,
    TeacherSettings,
    UnresolvedDispatchError,
    budget_refusal_reasons,
    constant_outcomes,
    remaining_tokens,
)
from learned_judgment.vision import VisionSettings, flatten_window, take_frames, unflatten_window

__all__ = [
    "EVENT_STATES",
    "EVENT_STEPS",
    "FEATURE_BLOCKS",
    "INPUT_ENCODING",
    "STUDENT_FORMAT",
    "EventLog",
    "available_observed_skills",
    "dream_request_plan",
    "dream_scores",
    "WINDOW_ENCODINGS",
    "dream_targets",
    "encode_inputs",
    "encode_window_features",
    "encoded_width",
    "load_dream",
    "load_rows",
    "load_spec",
    "main",
    "make_teacher",
    "one_shot_report",
    "predict_outcomes",
    "quote_dream",
    "save_student",
    "student_readout",
    "teacher_matrix",
    "threat_target_settings",
    "train_student",
    "window_encoding_from_config",
]

EVENT_STEPS = ("baseline", "attempts", "teacher_fit", "dream", "train", "evaluate")
EVENT_STATES = ("running", "done", "failed", "refused", "stopped")
FEATURE_BLOCKS = ("window", "action", "physics")  # the only order a FeatureLayout can describe
OUTCOME_KINDS = ("regression", "binary")
STUDENT_FORMAT = "learned-judgment/observed-student/v1"
INPUT_ENCODING = "(x - 0.5) * sqrt(12)"
WINDOW_ENCODINGS = ("raw", "abs_diff", "signed_diff", "signed_diff_first", "events")  # values of config.yaml vision.encoding and student.json input.window_encoding
THREAT, ACTION = "threat", "action"
THREAT_MODES = ("collision_course", "near_approach")  # values of skills.<name>.threat_target (threat_target_settings)
EXIT_REFUSED = 3

_SKILLS_PACKAGE = "learned_judgment.skills"
_BUILDER = "build_observed_spec"
_SPEC_ATTRIBUTES = (
    "name",
    "action_names",
    "action_features",
    "action_feature_names",
    "no_op_index",
    "outcome_names",
    "outcome_kinds",
    "outcome_bounds",
    "objective",
)
_ROWS_ARRAYS = ("scenario_index", "decision_time_s", "action", "features", "physics", "measured")
_DREAM_ARRAYS = ("scenario_index", "decision_time_s", "features", "physics")
_REFUSALS = (BudgetError, UnresolvedDispatchError, LedgerMissingError, CacheLockedError)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_QUOTE_CALLS = (
    "TabPFNProvider.connect (prior_labs.connect -> tabpfn_client.init; once per provider)",
    "TabPFNProvider.usage (ServiceClient.get_api_usage)",
    "TabPFNProvider.quote (tabpfn_client.estimate_cost on zero arrays; once per distinct request shape)",
    "CallLog.tokens_quoted_for_billed_calls (local read of tabpfn.call_log)",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Spec object
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Spec:
    """The checked attributes of a spec object (arrays normalized to float64)."""

    name: str
    action_names: tuple[str, ...]
    action_features: np.ndarray  # (C, A)
    action_feature_names: tuple[str, ...]
    no_op_index: int | None
    outcome_names: tuple[str, ...]
    outcome_kinds: tuple[str, ...]
    low: np.ndarray  # (K,)
    high: np.ndarray  # (K,)

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def n_outcomes(self) -> int:
        return len(self.outcome_names)


def _names(label: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"spec.{label} must be a sequence of names; got {value!r}")
    names = tuple(value)
    if not names or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"spec.{label} must hold >= 1 distinct non-empty names; got {list(names)}")
    return names


def _check_spec(spec: Any) -> _Spec:
    """Validate the spec object's attributes (see the module docstring); raise on any problem."""
    missing = [name for name in _SPEC_ATTRIBUTES if not hasattr(spec, name)]
    if missing:
        raise TypeError(f"the spec object lacks the attributes {missing}")
    if not isinstance(spec.name, str) or not spec.name:
        raise ValueError(f"spec.name must be a non-empty string; got {spec.name!r}")
    actions = _names("action_names", spec.action_names)
    features = np.asarray(spec.action_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(actions) or features.shape[1] < 1:
        raise ValueError(f"spec.action_features must have shape ({len(actions)}, A >= 1); got {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("spec.action_features contains non-finite values")
    feature_names = _names("action_feature_names", spec.action_feature_names)
    if len(feature_names) != features.shape[1]:
        raise ValueError(f"spec.action_feature_names has {len(feature_names)} names for {features.shape[1]} columns")
    no_op = spec.no_op_index
    if no_op is not None:
        if isinstance(no_op, bool) or not isinstance(no_op, (int, np.integer)) or not 0 <= int(no_op) < len(actions):
            raise ValueError(f"spec.no_op_index must be None or an index in [0, {len(actions)}); got {no_op!r}")
        if len(actions) < 2:
            raise ValueError("a spec with a no_op_index needs at least one other action")
        no_op = int(no_op)
    outcomes = _names("outcome_names", spec.outcome_names)
    kinds = tuple(spec.outcome_kinds)
    if len(kinds) != len(outcomes) or any(kind not in OUTCOME_KINDS for kind in kinds):
        raise ValueError(f"spec.outcome_kinds must be {len(outcomes)} values out of {OUTCOME_KINDS}; got {list(kinds)}")
    bounds = np.asarray(spec.outcome_bounds, dtype=np.float64)
    if bounds.shape != (len(outcomes), 2) or np.any(np.isnan(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError(f"spec.outcome_bounds must be ({len(outcomes)}, 2) rows [low <= high]; got {bounds.tolist()}")
    if not callable(spec.objective):
        raise TypeError("spec.objective must be callable: objective(OutcomeDistribution) -> (n,)")
    return _Spec(
        name=spec.name,
        action_names=actions,
        action_features=features,
        action_feature_names=feature_names,
        no_op_index=no_op,
        outcome_names=outcomes,
        outcome_kinds=kinds,
        low=np.ascontiguousarray(bounds[:, 0]),
        high=np.ascontiguousarray(bounds[:, 1]),
    )


def _heads(checked: _Spec) -> dict[str, list[int]]:
    """Output slices [start, stop) of the exported network: [threat (2) | action (C)]."""
    c = checked.n_actions
    if checked.no_op_index is None:
        return {ACTION: [0, c]}
    return {THREAT: [0, 2], ACTION: [2, 2 + c]}


def _objective(spec: Any, distribution: OutcomeDistribution) -> np.ndarray:
    scores = np.asarray(spec.objective(distribution), dtype=np.float64)
    if scores.shape != (distribution.n_rows,) or not np.all(np.isfinite(scores)):
        raise ValueError(f"spec.objective must return finite scores of shape ({distribution.n_rows},); got {scores.shape}")
    return scores


def _skill_module(name: Any) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"a skill name must be a Python identifier; got {name!r}")
    return f"{_SKILLS_PACKAGE}.{name}"


def load_spec(name: str, config: Mapping[str, Any]) -> Any:
    """The spec object of skill ``name``: ``learned_judgment.skills.<name>.build_observed_spec(config)``.

    Raises:
        ValueError naming the module when it does not exist or defines no build_observed_spec,
        or when the returned spec's name is not ``name``; whatever the builder raises; the
        errors of the spec check (TypeError / ValueError) for an invalid spec object.
    """
    module_name = _skill_module(name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ValueError(f"no module {module_name}: skill {name!r} has no observed spec") from None
    builder = getattr(module, _BUILDER, None)
    if not callable(builder):
        raise ValueError(f"{module_name} defines no function {_BUILDER}(config)")
    spec = builder(config)
    if _check_spec(spec).name != name:
        raise ValueError(f"{module_name}.{_BUILDER} built a spec named {spec.name!r}, not {name!r}")
    return spec


def available_observed_skills() -> tuple[str, ...]:
    """Sorted names of the modules under learned_judgment/skills that define build_observed_spec.

    A module that fails to import is skipped (it cannot be run either).
    """
    package = importlib.import_module(_SKILLS_PACKAGE)
    found: list[str] = []
    for info in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"{_SKILLS_PACKAGE}.{info.name}")
        except Exception:  # noqa: BLE001 - a broken module is not an available skill
            continue
        if callable(getattr(module, _BUILDER, None)):
            found.append(info.name)
    return tuple(sorted(found))


# --------------------------------------------------------------------------------------
# Progress events
# --------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """numpy scalars and arrays, tuples and Paths as plain JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(_jsonable(value), indent=indent, allow_nan=False, ensure_ascii=False)


def _count(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be None or a non-negative integer; got {value!r}")
    return int(value)


class EventLog:
    """Progress of one run: one JSON line per event appended to ``path`` (events.jsonl).

    Line: {"run_id", "skill", "step", "state", "done", "total", "time_utc", "data"} with step in
    EVENT_STEPS, state in EVENT_STATES, done/total None or non-negative ints, data None or a
    JSON-serializable mapping, time_utc ISO 8601 with a zero UTC offset from ``clock``.
    """

    def __init__(self, path: str | Path, run_id: str, skill: str, clock: Callable[[], datetime] | None = None) -> None:
        if not isinstance(run_id, str) or not run_id or not isinstance(skill, str) or not skill:
            raise ValueError(f"run_id and skill must be non-empty strings; got {run_id!r}, {skill!r}")
        self.path = Path(path)
        self.run_id = run_id
        self.skill = skill
        self._clock = clock if clock is not None else _utc_now

    def emit(
        self,
        step: str,
        state: str,
        done: int | None = None,
        total: int | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one event and return it. ValueError for an unknown step or state, a bad count,
        a clock that is not UTC, or data that is not a JSON mapping."""
        if step not in EVENT_STEPS:
            raise ValueError(f"step must be one of {EVENT_STEPS}; got {step!r}")
        if state not in EVENT_STATES:
            raise ValueError(f"state must be one of {EVENT_STATES}; got {state!r}")
        if data is not None and not isinstance(data, Mapping):
            raise ValueError(f"data must be None or a mapping; got {type(data).__name__}")
        now = self._clock()
        if not isinstance(now, datetime) or now.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError(f"the clock must return a timezone-aware UTC datetime; got {now!r}")
        event = {
            "run_id": self.run_id,
            "skill": self.skill,
            "step": step,
            "state": state,
            "done": _count("done", done),
            "total": _count("total", total),
            "time_utc": now.isoformat(),
            "data": None if data is None else _jsonable(dict(data)),
        }
        line = json.dumps(event, allow_nan=False, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return event


# --------------------------------------------------------------------------------------
# Files of the shared array contract
# --------------------------------------------------------------------------------------


def _load(directory: str | Path, stem: str, names: Sequence[str]) -> dict[str, Any]:
    folder = Path(directory)
    with np.load(folder / f"{stem}.npz", allow_pickle=False) as archive:
        missing = [name for name in names if name not in archive.files]
        if missing:
            raise ValueError(f"{folder / stem}.npz lacks the arrays {missing}")
        data: dict[str, Any] = {name: archive[name] for name in names}
    data["meta"] = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
    data["path"] = folder
    return data


def _check_common(data: Mapping[str, Any], n: int, label: str) -> None:
    for name in ("scenario_index", "decision_time_s"):
        if data[name].shape != (n,):
            raise ValueError(f"{label} {name} must have shape ({n},); got {data[name].shape}")
    if data["scenario_index"].dtype.kind not in "iu":
        raise ValueError(f"{label} scenario_index must be integers; got {data['scenario_index'].dtype}")
    features = data["features"]
    if features.ndim != 2 or features.shape[0] != n or not np.all(np.isfinite(features)):
        raise ValueError(f"{label} features must be finite with shape ({n}, D); got {features.shape}")


def load_rows(directory: str | Path) -> dict[str, Any]:
    """A rows file: {"scenario_index", "decision_time_s", "action", "features", "physics",
    "measured", "meta" (rows.json), "path"}; shapes checked against each other."""
    rows = _load(directory, "rows", _ROWS_ARRAYS)
    n = int(rows["action"].shape[0])
    _check_common(rows, n, "rows")
    if rows["action"].shape != (n,) or rows["action"].dtype.kind not in "iu":
        raise ValueError(f"rows action must be integers of shape ({n},); got {rows['action'].dtype} {rows['action'].shape}")
    for name in ("physics", "measured"):
        if rows[name].ndim != 2 or rows[name].shape[0] != n or not np.all(np.isfinite(rows[name])):
            raise ValueError(f"rows {name} must be finite with shape ({n}, K); got {rows[name].shape}")
    if rows["physics"].shape != rows["measured"].shape:
        raise ValueError(f"rows physics {rows['physics'].shape} and measured {rows['measured'].shape} differ")
    return rows


def load_dream(directory: str | Path) -> dict[str, Any]:
    """A dream file: {"scenario_index", "decision_time_s", "features", "physics" (m, C, K),
    "meta" (dream.json), "path"}; shapes checked against each other."""
    dream = _load(directory, "dream", _DREAM_ARRAYS)
    m = int(dream["scenario_index"].shape[0])
    _check_common(dream, m, "dream")
    physics = dream["physics"]
    if physics.ndim != 3 or physics.shape[0] != m or not np.all(np.isfinite(physics)):
        raise ValueError(f"dream physics must be finite with shape ({m}, C, K); got {physics.shape}")
    return dream


def _check_meta(checked: _Spec, meta: Mapping[str, Any], label: str) -> None:
    for key, expected in (
        ("skill", checked.name),
        ("action_names", list(checked.action_names)),
        ("outcome_names", list(checked.outcome_names)),
    ):
        if key not in meta:
            raise ValueError(f"{label} meta has no {key!r}")
        if meta[key] != expected:
            raise ValueError(f"{label} meta {key} is {meta[key]!r}; the spec has {expected!r}")
    if "frames" not in meta:
        raise ValueError(f"{label} meta has no 'frames'")


# --------------------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------------------


def _matrix(name: str, value: Any, *, rows: int | None = None, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or (rows is not None and array.shape[0] != rows) or (width is not None and array.shape[1] != width):
        expected = (rows if rows is not None else "n", width if width is not None else "columns")
        raise ValueError(f"{name} must have shape ({expected[0]}, {expected[1]}); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _actions(value: Any, n: int, c: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (n,) or (n and array.dtype.kind not in "iu"):
        raise ValueError(f"action must be integers of shape ({n},); got {array.dtype} {array.shape}")
    array = array.astype(np.int64)
    if np.any(array < 0) or np.any(array >= c):
        raise ValueError(f"action holds an index outside [0, {c})")
    return array


def _feature_set(feature_set: Any) -> tuple[str, ...]:
    if isinstance(feature_set, (str, bytes)) or not isinstance(feature_set, Sequence):
        raise ValueError(f"feature_set must be a list of blocks out of {FEATURE_BLOCKS}; got {feature_set!r}")
    blocks = tuple(feature_set)
    if any(block not in FEATURE_BLOCKS for block in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError(f"feature_set must name distinct blocks out of {FEATURE_BLOCKS}; got {list(blocks)}")
    if "action" not in blocks or "physics" not in blocks:
        raise ValueError(f"feature_set must contain 'action' and 'physics'; got {list(blocks)}")
    if blocks != tuple(block for block in FEATURE_BLOCKS if block in blocks):
        raise ValueError(
            f"feature_set must keep the FeatureLayout block order {FEATURE_BLOCKS} "
            f"(situation | action | physics); got {list(blocks)}"
        )
    return blocks


def teacher_matrix(
    spec: Any, features: np.ndarray, action: np.ndarray, physics: np.ndarray, feature_set: Sequence[str]
) -> tuple[np.ndarray, FeatureLayout]:
    """The residual teacher's feature matrix and its layout.

    Args:
        spec:        the spec object.
        features:    (n, D) window features in [0, 1] (used only with the "window" block).
        action:      (n,) int candidate indices into spec.action_names.
        physics:     (n, K) physics guess in the spec's outcome order.
        feature_set: blocks in this order, "action" and "physics" required:
                     ["window", "action", "physics"] or ["action", "physics"].

    Returns:
        X float64 (n, F): "window" -> D columns window_<i>; "action" -> spec.action_features[action]
        (A columns named spec.action_feature_names); "physics" -> K columns physics_<outcome>.
        FeatureLayout: skill spec.name, no categorical column, n_situation D or 0, n_action A,
        n_physics K, n_extra 0, outcome names and kinds of the spec.
    """
    checked = _check_spec(spec)
    blocks = _feature_set(feature_set)
    physics = _matrix("physics", physics, width=checked.n_outcomes)
    n = physics.shape[0]
    action = _actions(action, n, checked.n_actions)
    window = _matrix("features", features, rows=n)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for block in blocks:
        if block == "window":
            columns.append(window)
            names.extend(f"window_{index}" for index in range(window.shape[1]))
        elif block == "action":
            columns.append(checked.action_features[action])
            names.extend(checked.action_feature_names)
        else:
            columns.append(physics)
            names.extend(f"physics_{name}" for name in checked.outcome_names)
    if len(set(names)) != len(names):
        raise ValueError(f"teacher column names are not unique: {sorted({x for x in names if names.count(x) > 1})}")
    layout = FeatureLayout(
        skill=checked.name,
        names=tuple(names),
        categorical=(),
        n_situation=window.shape[1] if "window" in blocks else 0,
        n_action=checked.action_features.shape[1],
        n_physics=checked.n_outcomes,
        n_extra=0,
        outcome_names=checked.outcome_names,
        outcome_kinds=checked.outcome_kinds,
    )
    return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float64), layout


def make_teacher(spec: Any, rows: Mapping[str, Any], config: Mapping[str, Any], feature_set: Sequence[str]) -> TabPFNTeacher:
    """The production TabPFN residual teacher fitted on a rows file (a :func:`load_rows` dict).

    Context: teacher_matrix of the rows; targets: residuals measured - physics (n, K). Settings
    from TeacherSettings.from_config (tabpfn.*, engine.*). fit stores the context: no network.
    """
    x, layout = teacher_matrix(spec, rows["features"], rows["action"], rows["physics"], feature_set)
    measured = _matrix("measured", rows["measured"], rows=x.shape[0], width=layout.n_outcomes)
    residuals = measured - np.asarray(rows["physics"], dtype=np.float64)
    settings = TeacherSettings.from_config(config)
    teacher = TabPFNTeacher.from_settings(layout, settings, provider=TabPFNProvider(settings), clock=_utc_now)
    return teacher.fit(x, residuals)


def _fitted_context(teacher: TabPFNTeacher) -> tuple[np.ndarray, np.ndarray]:
    # TabPFNTeacher has no public accessor of its fitted context (reported to the lead).
    features, residuals = getattr(teacher, "_features", None), getattr(teacher, "_residuals", None)
    if features is None or residuals is None:
        raise RuntimeError("the teacher is not fitted: call make_teacher first")
    return features, residuals


def _connected(provider: TabPFNProvider) -> TabPFNProvider:
    # TabPFNProvider has no public "connected" flag (reported to the lead); connect() is free.
    if not getattr(provider, "_connected", False):
        provider.connect()
    return provider


def dream_request_plan(teacher: TabPFNTeacher, n_query_rows: int) -> dict[str, Any]:
    """Every provider request a prediction of ``n_query_rows`` rows would send, by the teacher's
    own rules (no network, cache not consulted): "stacked" -> one group of A*n training rows
    (n when A == 1) and A*n_query_rows query rows, width F + 1 when A >= 2; "per_outcome" -> one
    group per active outcome; each group in chunks of tabpfn.max_query_rows_per_request rows,
    a chunk of exactly Q rows padded with one row."""
    if isinstance(n_query_rows, bool) or not isinstance(n_query_rows, (int, np.integer)) or n_query_rows < 1:
        raise ValueError(f"n_query_rows must be an integer >= 1; got {n_query_rows!r}")
    features, residuals = _fitted_context(teacher)
    settings = teacher.settings
    names = teacher.layout.outcome_names
    active = tuple(int(index) for index in np.flatnonzero(~constant_outcomes(residuals)))
    n_train, width = (int(size) for size in features.shape)
    cap, n_levels = settings.max_query_rows_per_request, len(settings.quantile_levels)
    # (outcomes, training rows, columns, query rows) of each request group, as the teacher sends them
    if not active:
        groups: list[tuple[list[str], int, int, int]] = []
    elif settings.multi_outcome_mode == "stacked":
        several = len(active) > 1
        rows_per = len(active) if several else 1
        groups = [([names[i] for i in active], n_train * rows_per, width + int(several), len(active) * int(n_query_rows))]
    else:
        groups = [([names[i]], n_train, width, int(n_query_rows)) for i in active]
    requests: list[dict[str, Any]] = []
    for outcomes, train_rows, columns, total in groups:
        chunks = list(range(0, total, cap))
        for chunk, start in enumerate(chunks):
            rows = min(start + cap, total) - start
            padded = int(rows == n_levels)
            requests.append({
                "outcomes": outcomes,
                "chunk": chunk,
                "n_chunks": len(chunks),
                "n_train": train_rows,
                "n_query": rows + padded,
                "n_features": columns,
                "padded_rows": padded,
            })
    return {
        "mode": settings.multi_outcome_mode,
        "n_query_rows": int(n_query_rows),
        "n_train_rows": n_train,
        "n_features": width,
        "active_outcomes": [names[i] for i in active],
        "constant_outcomes": [name for i, name in enumerate(names) if i not in active],
        "max_query_rows_per_request": cap,
        "requests": requests,
    }


def quote_dream(teacher: TabPFNTeacher, n_query_rows: int) -> dict[str, Any]:
    """The provider's free quote of every request a prediction of ``n_query_rows`` rows would
    send, the account usage now, and an advisory replay of the teacher's budget gate.

    No prediction call: only the free endpoints listed under "calls". The cache is not
    consulted, so a request already cached (and not billed again) is counted too. The teacher
    re-reads usage and a quote before every uncached request and its own gate decides.
    """
    plan = dream_request_plan(teacher, n_query_rows)
    settings = teacher.settings
    provider = _connected(teacher.provider)
    usage = provider.usage()
    quotes: dict[tuple[int, int, int], dict[str, Any]] = {}
    for request in plan["requests"]:
        shape = (request["n_train"], request["n_query"], request["n_features"])
        if shape not in quotes:
            quotes[shape] = provider.quote(*shape)
        request["quote_tokens"] = quotes[shape]["estimated_cost"]
        request["pricing_version"] = quotes[shape]["pricing_version"]
    remaining = remaining_tokens(usage)
    already = teacher.call_log.tokens_quoted_for_billed_calls()
    reasons: list[str] = []
    spent, left = already, remaining
    for request in plan["requests"]:
        reasons = budget_refusal_reasons(
            quote_tokens=request["quote_tokens"],
            max_tokens_per_request=settings.max_tokens_per_request,
            project_token_budget=settings.project_token_budget,
            tokens_already_quoted=spent,
            remaining_account_tokens=left,
        )
        if reasons:
            break
        spent += request["quote_tokens"]
        left = None if left is None else left - request["quote_tokens"]
    return {
        **plan,
        "total_quoted_tokens": int(sum(request["quote_tokens"] for request in plan["requests"])),
        "usage": usage,
        "remaining_account_tokens": remaining,
        "tokens_quoted_for_billed_calls": already,
        "project_token_budget": settings.project_token_budget,
        "max_tokens_per_request": settings.max_tokens_per_request,
        "gate_preview": {"would_refuse": bool(reasons), "reasons": reasons},
        "calls": list(_QUOTE_CALLS),
        "note": "free quotes only; the cache was not consulted (a cached request is not billed again)",
    }


def _require_billing(allow_billing: Any) -> None:
    if allow_billing is not True:
        raise RuntimeError(
            "this sends billable TabPFN requests: pass allow_billing=True only after the quote was reviewed "
            "(command line: --confirm-billing)"
        )


def predict_outcomes(
    teacher: TabPFNTeacher, spec: Any, x_query: np.ndarray, physics: np.ndarray, *, allow_billing: bool = False
) -> OutcomeDistribution:
    """Outcome distribution of q query rows: clip(physics + predicted residual, spec bounds),
    statistic by statistic (OutcomeDistribution.from_physics_and_residual, as engine.decide).

    BILLABLE: raises RuntimeError before anything else unless ``allow_billing is True``. Then
    the teacher serves each request from its cache or sends it through its own gate.

    Args:
        x_query: (q, F) rows in the teacher's layout (teacher_matrix with its feature_set).
        physics: (q, K) the physics guess of the same rows.
    """
    _require_billing(allow_billing)
    checked = _check_spec(spec)
    physics = _matrix("physics", physics, width=checked.n_outcomes)
    x_query = _matrix("x_query", x_query, rows=physics.shape[0], width=teacher.layout.n_features)
    if tuple(teacher.layout.outcome_names) != checked.outcome_names:
        raise ValueError(f"the teacher predicts {teacher.layout.outcome_names}; the spec has {checked.outcome_names}")
    _connected(teacher.provider)
    residual = teacher.predict_residual_distribution(x_query)
    if residual.mean.shape != physics.shape:
        raise ValueError(f"the teacher returned residuals {residual.mean.shape}; expected {physics.shape}")
    if not np.array_equal(residual.quantile_levels, np.asarray(teacher.settings.quantile_levels)):
        raise ValueError("the teacher returned other quantile levels than engine.quantile_levels")
    return OutcomeDistribution.from_physics_and_residual(checked.outcome_names, physics, residual, checked.low, checked.high)


def _candidate_rows(
    spec: Any, features: np.ndarray, physics: np.ndarray, feature_set: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, FeatureLayout]:
    """Every candidate of every dream situation, situation-major (row i * C + c)."""
    checked = _check_spec(spec)
    m, c = physics.shape[0], checked.n_actions
    if physics.shape[1:] != (c, checked.n_outcomes):
        raise ValueError(f"dream physics must have shape (m, {c}, {checked.n_outcomes}); got {physics.shape}")
    flat_physics = physics.reshape(m * c, checked.n_outcomes)
    x, layout = teacher_matrix(spec, np.repeat(features, c, axis=0), np.tile(np.arange(c), m), flat_physics, feature_set)
    return x, flat_physics, layout


def dream_scores(
    teacher: TabPFNTeacher, spec: Any, dream: Mapping[str, Any], feature_set: Sequence[str], allow_billing: bool
) -> np.ndarray:
    """spec.objective of every candidate of every dream situation: float64 (m, C), higher is better.

    ``dream`` holds "features" (m, D) and "physics" (m, C, K) (a :func:`load_dream` dict). One
    :func:`predict_outcomes` call over the m * C rows (the teacher chunks it); BILLABLE, so
    RuntimeError first unless ``allow_billing is True``. The request keys are then in
    ``teacher.last_request_hashes``.
    """
    _require_billing(allow_billing)
    physics = np.asarray(dream["physics"], dtype=np.float64)
    if physics.ndim != 3:
        raise ValueError(f"dream physics must have shape (m, C, K); got {physics.shape}")
    x, flat_physics, layout = _candidate_rows(spec, np.asarray(dream["features"]), physics, feature_set)
    if layout.names != teacher.layout.names:
        raise ValueError("the teacher was fitted on other columns than this feature_set builds")
    predicted = predict_outcomes(teacher, spec, x, flat_physics, allow_billing=allow_billing)
    # The dream world's predicted outcome means (m, C, K), kept for the threat target "collision_course".
    dream_scores.last_predicted_mean = np.asarray(predicted.mean, dtype=np.float64).reshape(
        physics.shape[0], physics.shape[1], physics.shape[2])
    return _objective(spec, predicted).reshape(physics.shape[0], physics.shape[1])


# --------------------------------------------------------------------------------------
# Student
# --------------------------------------------------------------------------------------


def threat_target_settings(config: Mapping[str, Any], skill: str) -> dict[str, Any] | None:
    """skills.<skill>.threat_target, checked: what the student's threat head is taught and what the scorer counts
    as a true threat.

      absent                                     None: the score-margin target (skills.<skill>.threat_target_scale)
      collision_course, or {mode: collision_course}
                                                 {"mode": "collision_course", "outcome": skills.<skill>.threat_outcome}:
                                                 doing nothing ends in that binary outcome (a collision)
      {mode: near_approach, separation_m: d, outcome: name}
                                                 {"mode": "near_approach", "separation_m": d, "outcome": name}: under
                                                 doing nothing, the outcome ``name`` (the closest object-to-body
                                                 separation, m) is below d (d > 0)

    Raises ConfigError for any other value.
    """
    key = f"skills.{skill}.threat_target"
    value = optional(config, key)
    if value is None:
        return None
    block = {"mode": value} if isinstance(value, str) else value
    if not isinstance(block, Mapping) or block.get("mode") not in THREAT_MODES:
        raise ConfigError(f"config.yaml key {key!r} must be one of {THREAT_MODES} or a mapping with such a 'mode'; got {value!r}")
    if block["mode"] == "collision_course":
        if set(block) - {"mode"}:
            raise ConfigError(f"config.yaml key {key!r} (collision_course) takes no other keys; got {sorted(block)}")
        return {"mode": "collision_course", "outcome": str(require(config, f"skills.{skill}.threat_outcome"))}
    if set(block) != {"mode", "separation_m", "outcome"}:
        raise ConfigError(f"config.yaml key {key!r} (near_approach) needs exactly mode, separation_m, outcome; got {sorted(block)}")
    distance, outcome = block["separation_m"], block["outcome"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not (math.isfinite(distance) and distance > 0):
        raise ConfigError(f"config.yaml key '{key}.separation_m' must be a finite number > 0; got {distance!r}")
    if not isinstance(outcome, str) or not outcome:
        raise ConfigError(f"config.yaml key '{key}.outcome' must be an outcome name; got {outcome!r}")
    return {"mode": "near_approach", "separation_m": float(distance), "outcome": outcome}


def dream_targets(scores: np.ndarray, spec: Any, config: Mapping[str, Any], predicted_mean: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Training targets of the student from the dream scores (m, C).

    "action": students.targets.soft_targets with students.snn.target_temperature and
    students.snn.target_spread (spread = score_spread of these rows for "global_median").
    "threat" (only when spec.no_op_index is not None): (m, 2) = [1 - p, p] with, by threat_target_settings:
    collision_course: p = the dream's predicted mean of the threat outcome under NO_OP, clipped to [0, 1];
    near_approach: p = 1 when the dream's predicted mean of the separation outcome under NO_OP is below
    separation_m, else 0; absent: p = 1 / (1 + exp(-(max over active candidates of score - score[no_op]) /
    skills.<name>.threat_target_scale)). predicted_mean (m, C, K) is needed by the first two.
    """
    checked = _check_spec(spec)
    scores = _matrix("scores", scores, width=checked.n_actions)
    temperature = require(config, "students.snn.target_temperature")
    mode = require(config, "students.snn.target_spread")
    spread = score_spread(scores) if mode == "global_median" else None
    targets = {ACTION: soft_targets(scores, temperature=temperature, target_spread=mode, spread=spread)}
    decision = optional(config, f"skills.{checked.name}.action_target")
    if decision is not None and decision.get("mode") == "decision":
        # The dream's DECISION as the target: the first action, in readout order, whose dream score is within
        # tie_tolerance of the best one, with label smoothing. Two actions that the dream scores alike (stepping
        # left or right from a punch at the centre) then give one consistent answer instead of a 50/50 target,
        # which a command gate asking for one clear action can never pass; NO_OP comes first in the order, so a
        # tie between acting and not acting stays NO_OP.
        tolerance, smoothing = float(decision["tie_tolerance"]), float(decision["smoothing"])
        if not (tolerance >= 0.0 and 0.0 <= smoothing < 1.0):
            raise ConfigError(f"skills.{checked.name}.action_target needs tie_tolerance >= 0 and 0 <= smoothing < 1")
        tied = scores >= scores.max(axis=1, keepdims=True) - tolerance
        choice = tied.argmax(axis=1)
        hard = np.full(scores.shape, smoothing / (scores.shape[1] - 1), dtype=np.float64)
        hard[np.arange(scores.shape[0]), choice] = 1.0 - smoothing
        targets[ACTION] = hard
    target = threat_target_settings(config, checked.name)
    if checked.no_op_index is not None and target is not None:
        # collision_course: the dream world (physics guess corrected by the teacher) says doing nothing ends in the
        # outcome named by skills.<name>.threat_outcome (a binary outcome: its predicted mean under NO_OP, clipped to
        # [0, 1]); talkhere.md: "threat means an object approaching on a collision course". near_approach: the dream
        # world says doing nothing lets the object come closer than separation_m (a hit or a near miss at the head).
        # Only the simulation scorer labels it.
        if predicted_mean is None:
            raise ValueError(f"threat_target {target['mode']} needs predicted_mean (scores.npz of a teach run that saved it)")
        if target["outcome"] not in checked.outcome_names:
            raise ConfigError(f"threat_target outcome {target['outcome']!r} is not an outcome of {checked.outcome_names}")
        mean = np.asarray(predicted_mean, dtype=np.float64)
        if mean.shape != (scores.shape[0], checked.n_actions, checked.n_outcomes):
            raise ValueError(f"predicted_mean must have shape {(scores.shape[0], checked.n_actions, checked.n_outcomes)}; got {mean.shape}")
        predicted = mean[:, checked.no_op_index, list(checked.outcome_names).index(target["outcome"])]
        if target["mode"] == "near_approach":
            threat = (predicted < target["separation_m"]).astype(np.float64)
        else:
            threat = np.clip(predicted, 0.0, 1.0)
        targets[THREAT] = np.column_stack([1.0 - threat, threat])
    elif checked.no_op_index is not None:
        key = f"skills.{checked.name}.threat_target_scale"
        scale = require(config, key)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not (math.isfinite(scale) and scale > 0):
            raise ConfigError(f"config.yaml key {key!r} must be a finite number > 0; got {scale!r}")
        active = [index for index in range(checked.n_actions) if index != checked.no_op_index]
        z = (scores[:, active].max(axis=1) - scores[:, checked.no_op_index]) / float(scale)
        threat = np.exp(-np.logaddexp(0.0, -z))  # logistic(z) without overflow
        targets[THREAT] = np.column_stack([np.exp(-np.logaddexp(0.0, z)), threat])
    return targets


def _predicted_mean(scores_path: Any) -> np.ndarray | None:
    """predicted_mean (m, C, K) of a scores.npz, or None for a file written before it was saved."""
    with np.load(scores_path) as data:
        return np.array(data["predicted_mean"]) if "predicted_mean" in data.files else None


def encode_inputs(features: np.ndarray) -> np.ndarray:
    """The student's input encoding: float64 (n, D) = (x - 0.5) * sqrt(12) for window features
    (n, D) in [0, 1] (the InputEncoder's map for a float feature declared on [0, 1])."""
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[1] < 1 or array.dtype.kind not in "fiu":
        raise ValueError(f"features must be a numeric (n, D >= 1) array; got {array.dtype} {array.shape}")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("features must be finite and inside [0, 1]")
    return (array - 0.5) * math.sqrt(12.0)


def _window_encoding(encoding: Any) -> str:
    if encoding not in WINDOW_ENCODINGS:
        raise ValueError(f"the window encoding must be one of {WINDOW_ENCODINGS}; got {encoding!r}")
    return encoding


def _diff_floor(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"diff_floor must be a finite number > 0; got {value!r}")
    if not (math.isfinite(float(value)) and float(value) > 0.0):
        raise ValueError(f"diff_floor must be a finite number > 0; got {value!r}")
    return float(value)


def encoded_width(frames: int, frame_size: int, encoding: str) -> int:
    """Student input width for windows of ``frames`` frames of N x N: (F - 1) * N * N for "abs_diff",
    F * N * N for "raw". ValueError for an unknown encoding, or F < 2 with "abs_diff"."""
    if _window_encoding(encoding) != "raw" and frames < 2:
        raise ValueError(f"the {encoding} encoding needs windows of >= 2 frames; got {frames}")
    return (frames - 1 if encoding != "raw" else frames) * frame_size * frame_size


def window_encoding_from_config(config: Mapping[str, Any]) -> tuple[str, float]:
    """(vision.encoding, vision.diff_floor) of config.yaml; ConfigError for a missing or invalid value."""
    encoding = require(config, "vision.encoding")
    # "events" takes its event threshold in the place of the divisor floor (one parameter per encoding).
    floor = require(config, "vision.event_threshold" if encoding == "events" else "vision.diff_floor")
    if encoding not in WINDOW_ENCODINGS:
        raise ConfigError(f"config.yaml key 'vision.encoding' must be one of {WINDOW_ENCODINGS}; got {encoding!r}")
    try:
        return encoding, _diff_floor(floor)
    except ValueError:
        raise ConfigError(f"config.yaml key 'vision.diff_floor' must be a finite number > 0; got {floor!r}") from None


def encode_window_features(windows: np.ndarray, encoding: str, diff_floor: float | None) -> np.ndarray:
    """The student's features of image windows: float32 (n, F' * N * N) in [0, 1].

    Args:
        windows:    float (n, F, N, N) grey levels in [0, 1], oldest frame first (float32 by contract).
        encoding:   "raw": the windows unchanged, flattened (F' = F; per window what vision.flatten_window gives).
                    "abs_diff": d = |w[:, 1:] - w[:, :-1]| (the F - 1 adjacent pairs, oldest pair first,
                    F' = F - 1) divided per window by max(largest d of that window, diff_floor). A still scene
                    gives 0; an object darker or brighter than its backdrop by the same amount gives the same
                    features.
        diff_floor: the smallest divisor of "abs_diff", a finite number > 0 (unused by "raw"; may be None there).

    Deterministic and vectorised (float32 arithmetic). The observed run's training, :func:`student_readout` and
    learned_judgment.runtime all use this one function.

    Raises:
        ValueError for an unknown encoding, an invalid diff_floor for "abs_diff", windows that are not a finite
        float (n, F, N, N) array, or F < 2 for "abs_diff".
    """
    encoding = _window_encoding(encoding)
    array = np.asarray(windows)
    if array.ndim != 4 or array.shape[2] != array.shape[3] or array.shape[2] < 1 or array.dtype.kind != "f":
        raise ValueError(f"windows must be a float (n, F, N, N) array; got {array.dtype} {array.shape}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("windows contain non-finite values")
    n = array.shape[0]
    if encoding == "raw":
        return np.array(array.reshape(n, -1), dtype=np.float32, copy=True)
    floor = np.float32(_diff_floor(diff_floor))
    encoded_width(array.shape[1], array.shape[2], encoding)  # F >= 2
    if encoding == "events":
        # Event-camera style: each pixel of each adjacent frame pair gives an ON event (1.0) when it got brighter by
        # more than the threshold, an OFF event (0.0) when it got darker by more than it, and 0.5 (no event, zero
        # input current) otherwise. For this encoding the ``diff_floor`` argument IS the event threshold
        # (config vision.event_threshold; stored in student.json input.diff_floor).
        change = array[:, 1:] - array[:, :-1]
        events = np.full(change.shape, 0.5, dtype=np.float32)
        events[change > floor] = 1.0
        events[change < -floor] = 0.0
        return events.reshape(n, -1)
    if encoding == "signed_diff_first":
        # Change of every later frame against the OLDEST frame of the window: a slow approach adds up over the
        # window (7 frames = 233 ms) instead of showing only one frame step; a still scene is still 0.5 everywhere.
        change = (array[:, 1:] - array[:, :1]).reshape(n, -1)
    else:
        change = (array[:, 1:] - array[:, :-1]).reshape(n, -1)
    scale = np.maximum(np.abs(change).max(axis=1), floor)
    if encoding == "abs_diff":
        return np.abs(change) / scale[:, None]
    # "signed_diff": 0.5 is "no change"; brighter than the frame before is above it, darker below. A still scene
    # gives 0.5 everywhere, which the student's input map (x - 0.5) * sqrt(12) turns into zero current.
    return np.float32(0.5) + np.float32(0.5) * change / scale[:, None]


def _student_features(features: np.ndarray, student_input: Mapping[str, Any] | None) -> np.ndarray:
    """Raw flattened windows (n, F * N * N) -> the features the student was trained on, by student.json "input"
    (None or no "window_encoding": used as given, i.e. raw windows or features that are not image windows)."""
    if student_input is None or "window_encoding" not in student_input:
        return features
    frames, size = int(student_input["frames"]), int(student_input["frame_size"])
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[1] != frames * size * size:
        raise ValueError(f"features must be raw flattened windows (n, {frames * size * size}); got {array.shape}")
    windows = array.reshape(array.shape[0], frames, size, size)
    return encode_window_features(windows, student_input["window_encoding"], student_input.get("diff_floor"))


def _combine(networks: Mapping[str, SpikingNetwork]) -> SpikingNetwork:
    """One network with outputs [threat | action] from two heads exported from one trunk."""
    if THREAT not in networks:
        return networks[ACTION]
    threat, action = networks[THREAT], networks[ACTION]
    same = (
        all(np.array_equal(a, b) for a, b in zip(threat.hidden_weights, action.hidden_weights))
        and all(np.array_equal(a, b) for a, b in zip(threat.hidden_biases, action.hidden_biases))
        and threat.betas == action.betas
        and threat.thresholds == action.thresholds
        and threat.readout_beta == action.readout_beta
        and threat.time_steps == action.time_steps
    )
    if not same:
        raise ValueError("the two heads do not share one trunk and readout leak; they cannot be concatenated")
    return SpikingNetwork(
        hidden_weights=action.hidden_weights,
        hidden_biases=action.hidden_biases,
        betas=action.betas,
        thresholds=action.thresholds,
        readout_weight=np.vstack([threat.readout_weight, action.readout_weight]),
        readout_bias=np.concatenate([threat.readout_bias, action.readout_bias]),
        readout_beta=action.readout_beta,
        time_steps=action.time_steps,
    )


def _combined_parity(
    combined: SpikingNetwork,
    networks: Mapping[str, SpikingNetwork],
    heads: Mapping[str, list[int]],
    inputs: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    """The combined network's logits against each exported head's logits on the same inputs."""
    from learned_judgment.students.snn_train import ParityError

    logits = combined.forward(inputs)
    differences, action_differing = {}, 0
    for head, (start, stop) in heads.items():
        own = networks[head].forward(inputs)
        differences[head] = float(np.max(np.abs(logits[:, start:stop] - own)))
        if head == ACTION:
            action_differing = int(np.sum(np.argmax(logits[:, start:stop], axis=1) != np.argmax(own, axis=1)))
    report = {
        "n_inputs_checked": int(inputs.shape[0]),
        "logit_tolerance": tolerance,
        "max_abs_logit_difference": differences,
        "action_decisions_differing": action_differing,
    }
    if not all(value <= tolerance for value in differences.values()) or action_differing:
        raise ParityError(f"the combined network does not reproduce its heads: {report}", report)
    return report


def train_student(
    spec: Any,
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    hidden_sizes: Sequence[int],
    max_steps: int,
    seed: int,
    groups: np.ndarray | None = None,
    on_evaluation: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train the two-head spiking student on the dream and export it.

    Args:
        features:      (m, D) window features in [0, 1]; encoded with :func:`encode_inputs`.
        targets:       :func:`dream_targets` output: "action" (m, C) and "threat" (m, 2) exactly
                       when spec.no_op_index is not None.
        hidden_sizes, max_steps: override students.snn.hidden_sizes / max_steps; every other
                       setting from students.snn (SNNSettings.from_config).
        seed:          run seed; the validation split, the initialisation and the batch order use
                       the named streams "<skill>/observed/validation_split", ".../init", ".../train".
        groups:        (m,) ints (the dream's scenario_index): rows of one group stay on one side
                       of the validation split (students.snn.validation_fraction of the groups);
                       None splits by row.
        on_evaluation: passed to snn_train.fit (progress observer).

    Heads are trained on the same rows with equal batch shares. Deterministic for a seed.

    Returns:
        {"model": torch model, "network": the combined SpikingNetwork [threat (2) | action (C)],
        "heads": {head: [start, stop)}, "parity": {head: check_parity report, "combined": report},
        "training": TrainingReport.to_json(), "split": {...}, "settings": SNNSettings.describe(),
        "seeds": {...}, "wall_s": float}.

    Raises:
        snn_train.TrainingError when fit refuses the network, snn_train.ParityError when an
        exported network does not reproduce the torch model, ValueError for invalid inputs.
    """
    from learned_judgment.engine.streams import stream_rng, stream_seed
    from learned_judgment.students import snn_train
    from learned_judgment.students.snn import split_rows_by_situation

    started = time.perf_counter()
    checked = _check_spec(spec)
    inputs = encode_inputs(features)
    m = inputs.shape[0]
    heads = _heads(checked)
    if set(targets) != set(heads):
        raise ValueError(f"targets must hold exactly the heads {list(heads)}; got {list(targets)}")
    sizes = {THREAT: 2, ACTION: checked.n_actions}
    checked_targets = {head: _matrix(f"targets[{head!r}]", targets[head], rows=m, width=sizes[head]) for head in heads}
    settings = dataclasses.replace(
        snn_train.SNNSettings.from_config(config), hidden_sizes=tuple(hidden_sizes), max_steps=max_steps
    )
    if groups is None:
        keys, split_by = np.arange(m, dtype=np.float64), "row"
    else:
        group_array = np.asarray(groups)
        if group_array.shape != (m,) or (m and group_array.dtype.kind not in "iu"):
            raise ValueError(f"groups must be integers of shape ({m},); got {group_array.dtype} {group_array.shape}")
        keys, split_by = group_array.astype(np.float64), "group"
    names = {part: f"{checked.name}/observed/{part}" for part in ("validation_split", "init", "train")}
    train_rows, validation_rows = split_rows_by_situation(
        keys[:, None], fraction=settings.validation_fraction, rng=stream_rng(seed, names["validation_split"])
    )
    init_seed, train_seed = stream_seed(seed, names["init"]), stream_seed(seed, names["train"])
    model = snn_train.build_model(inputs.shape[1], {head: sizes[head] for head in heads}, settings, seed=init_seed)
    report = snn_train.fit(
        model,
        {head: (inputs[train_rows], checked_targets[head][train_rows]) for head in heads},
        validation={head: (inputs[validation_rows], checked_targets[head][validation_rows]) for head in heads},
        sampling={head: 1.0 / len(heads) for head in heads},
        settings=settings,
        seed=train_seed,
        on_evaluation=on_evaluation,
    )
    tolerance = settings.parity_logit_tolerance
    networks = {head: snn_train.export_network(model, head) for head in heads}
    parity: dict[str, Any] = {
        head: snn_train.check_parity(model, head, networks[head], inputs, logit_tolerance=tolerance) for head in heads
    }
    combined = _combine(networks)
    parity["combined"] = _combined_parity(combined, networks, heads, inputs, tolerance)
    return {
        "model": model,
        "network": combined,
        "heads": heads,
        "parity": parity,
        "training": report.to_json(),
        "split": {
            "by": split_by,
            "validation_fraction": settings.validation_fraction,
            "n_train_rows": int(train_rows.size),
            "n_validation_rows": int(validation_rows.size),
            "n_groups": int(np.unique(keys).size),
        },
        "settings": settings.describe(),
        "seeds": {"seed": int(seed), "streams": names, "init": init_seed, "train": train_seed},
        "wall_s": float(time.perf_counter() - started),
    }


def save_student(
    directory: str | Path,
    network: SpikingNetwork,
    spec: Any,
    *,
    frames: int | None,
    config: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any],
) -> str:
    """Write ``directory``/network.npz + network.json (snn_core.save_network) and student.json;
    return the network digest.

    student.json: {"format": STUDENT_FORMAT, "skill", "action_names", "heads": {"threat": [0, 2],
    "action": [2, 2 + C]} (only "action": [0, C] without a NO_OP), "no_op_index", "input":
    {"frames" (F: camera frames the runtime is given), "frame_size", "window_encoding",
    "diff_floor" (config.yaml vision.encoding / vision.diff_floor, see encode_window_features),
    "encoding"}, "vision": the VisionSettings for ``frames``, "thresholds": thresholds or null,
    "provenance", "network": {"files", "sha256"}}. The network takes encoded_width(F, N,
    window_encoding) inputs. frames None (features that are not image windows): "input":
    {"features": D (the network's input width), "encoding"} and "vision": null.

    Raises ValueError when the network's input or output width does not fit the spec, frames and
    the window encoding.
    """
    checked = _check_spec(spec)
    vision = None if frames is None else VisionSettings.from_config(config, frames=frames)
    window_encoding, diff_floor = (None, None) if vision is None else window_encoding_from_config(config)
    heads = _heads(checked)
    outputs = max(stop for _, stop in heads.values())
    if vision is None:
        inputs = getattr(network, "n_inputs", None)
    else:
        inputs = encoded_width(vision.frames, vision.frame_size, window_encoding)
    if not isinstance(network, SpikingNetwork) or network.n_inputs != inputs or network.n_outputs != outputs:
        raise ValueError(
            f"the network must take {inputs} inputs and give {outputs} outputs; "
            f"got {getattr(network, 'layer_sizes', type(network).__name__)}"
        )
    if thresholds is not None and not isinstance(thresholds, Mapping):
        raise TypeError(f"thresholds must be None or a mapping; got {type(thresholds).__name__}")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"provenance must be a mapping; got {type(provenance).__name__}")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    if vision is None:
        input_block = {"features": network.n_inputs, "encoding": INPUT_ENCODING}
    else:
        input_block = {
            "frames": vision.frames,
            "frame_size": vision.frame_size,
            "window_encoding": window_encoding,
            "diff_floor": diff_floor,
            "encoding": INPUT_ENCODING,
        }
    digest = save_network(
        folder / "network",
        network,
        {"skill": checked.name, "action_names": list(checked.action_names), "heads": heads, "input": input_block},
    )
    record = {
        "format": STUDENT_FORMAT,
        "skill": checked.name,
        "action_names": list(checked.action_names),
        "heads": heads,
        "no_op_index": checked.no_op_index,
        "input": input_block,
        "vision": None if vision is None else dataclasses.asdict(vision),
        "thresholds": None if thresholds is None else dict(thresholds),
        "provenance": dict(provenance),
        "network": {"files": ["network.npz", "network.json"], "sha256": digest},
    }
    atomic_write_text(folder / "student.json", _dumps(record, indent=1) + "\n")
    return digest


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def student_readout(
    network: SpikingNetwork,
    features: np.ndarray,
    heads: Mapping[str, Sequence[int]] | None = None,
    student_input: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(threat_confidence (n,), action_scores (n, C)) for features (n, D) in [0, 1].

    ``student_input`` is student.json "input". With a "window_encoding" key, ``features`` are RAW
    flattened windows (n, frames * frame_size**2) and :func:`encode_window_features` turns them into
    the student's features first; None or no such key (a student saved before the key, or features
    that are not image windows) uses ``features`` as given. Then encode_inputs and one forward.

    Softmax over each head's logits; threat_confidence = softmax(threat logits)[:, 1].
    ``heads`` is student.json "heads"; None means the layout [threat (2) | action (rest)].
    Without a "threat" head, threat_confidence is 0.0 for every row.
    """
    logits = network.forward(encode_inputs(_student_features(features, student_input)))
    if heads is None:
        if network.n_outputs < 3:
            raise ValueError(f"a [threat (2) | action] network needs >= 3 outputs; got {network.n_outputs}")
        heads = {THREAT: [0, 2], ACTION: [2, network.n_outputs]}
    if ACTION not in heads or set(heads) - {THREAT, ACTION}:
        raise ValueError(f"heads must name 'action' and optionally 'threat'; got {list(heads)}")
    start, stop = (int(value) for value in heads[ACTION])
    action = _softmax(logits[:, start:stop])
    if THREAT not in heads:
        return np.zeros(logits.shape[0]), action
    low, high = (int(value) for value in heads[THREAT])
    if high - low != 2:
        raise ValueError(f"the threat head must have 2 outputs; got [{low}, {high})")
    return _softmax(logits[:, low:high])[:, 1], action


# --------------------------------------------------------------------------------------
# One-shot decision rules on measured rows
# --------------------------------------------------------------------------------------


def _situation_table(rows: Mapping[str, Any], n_actions: int) -> tuple[np.ndarray, np.ndarray]:
    """(scenarios (S,) sorted distinct scenario_index, table (S, C) row index of each action)."""
    scenario = np.asarray(rows["scenario_index"]).astype(np.int64)
    action = _actions(rows["action"], scenario.shape[0], n_actions)
    scenarios = np.unique(scenario)
    table = np.full((scenarios.size, n_actions), -1, dtype=np.int64)
    position = np.searchsorted(scenarios, scenario)
    for row, (s, a) in enumerate(zip(position, action)):
        if table[s, a] != -1:
            raise ValueError(f"scenario {scenarios[s]} has more than one row for action {a}")
        table[s, a] = row
    if np.any(table < 0):
        s, a = np.argwhere(table < 0)[0]
        raise ValueError(f"scenario {scenarios[s]} has no row for action {a}: every situation needs all {n_actions} actions")
    return scenarios, table


def _rule_metrics(
    checked: _Spec, picked: np.ndarray, objective: np.ndarray, regret: np.ndarray, chosen: np.ndarray
) -> dict[str, Any]:
    """Metrics of the chosen rows: picked (S, K) measured outcomes, objective / regret / chosen (S,)."""
    outcomes: dict[str, Any] = {}
    for column, (name, kind) in enumerate(zip(checked.outcome_names, checked.outcome_kinds)):
        values = picked[:, column]
        if kind == "binary":
            outcomes[name] = {"count": int(np.sum(values)), "rate": float(np.mean(values))}
        else:
            outcomes[name] = {"mean": float(np.mean(values)), "min": float(np.min(values))}
    return {
        "n": int(picked.shape[0]),
        "outcomes": outcomes,
        "mean_measured_objective": float(np.mean(objective)),
        "mean_regret": float(np.mean(regret)),
        "max_regret": float(np.max(regret)),
        "action_counts": {name: int(np.sum(chosen == index)) for index, name in enumerate(checked.action_names)},
    }


def one_shot_report(
    spec: Any, rows: Mapping[str, Any], chosen: np.ndarray, *, quantile_levels: Sequence[float]
) -> dict[str, Any]:
    """Measured result of a one-shot decision rule, by lookup in a rows file (every situation
    holds a measured row of every action).

    Args:
        rows:            a :func:`load_rows` dict.
        chosen:          (S,) int action per situation, situations in ascending scenario_index.
        quantile_levels: engine.quantile_levels, for the point-mass OutcomeDistribution of the
                         measured outcome that spec.objective scores.

    Returns:
        {"n", "outcomes": {name: {"count", "rate"} for binary, {"mean", "min"} for regression},
        "mean_measured_objective", "mean_regret", "max_regret" (best measured objective of the
        situation minus that of the chosen action), "action_counts", "by_kind": the same per
        scenario kind when rows meta "scenarios" carry "kind" (else absent)}.
    """
    checked = _check_spec(spec)
    scenarios, table = _situation_table(rows, checked.n_actions)
    choice = _actions(chosen, scenarios.size, checked.n_actions)
    measured = _matrix("measured", rows["measured"], width=checked.n_outcomes)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    objective = _objective(spec, OutcomeDistribution.point(checked.outcome_names, measured, levels))
    picked_rows = table[np.arange(scenarios.size), choice]
    regret = objective[table].max(axis=1) - objective[picked_rows]
    report = _rule_metrics(checked, measured[picked_rows], objective[picked_rows], regret, choice)
    listed = rows.get("meta", {}).get("scenarios")
    if not isinstance(listed, list) or not all(isinstance(item, Mapping) and "kind" in item for item in listed):
        return report
    if scenarios.size and (int(scenarios.min()) < 0 or int(scenarios.max()) >= len(listed)):
        raise ValueError(f"scenario_index runs past the {len(listed)} scenarios listed in the rows meta")
    kinds = np.array([str(listed[int(index)]["kind"]) for index in scenarios])
    report["by_kind"] = {}
    for kind in sorted(set(kinds.tolist())):
        mask = kinds == kind
        picked = picked_rows[mask]
        report["by_kind"][kind] = _rule_metrics(checked, measured[picked], objective[picked], regret[mask], choice[mask])
    return report


def _choose(spec: Any, rows: Mapping[str, Any], row_scores: np.ndarray) -> np.ndarray:
    """(S,) argmax over each situation's actions of per-row scores (n,)."""
    checked = _check_spec(spec)
    _, table = _situation_table(rows, checked.n_actions)
    return np.argmax(np.asarray(row_scores, dtype=np.float64)[table], axis=1)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


class _Refused(Exception):
    """A step was refused (no --confirm-billing, or the teacher's gate)."""


def _run_dir(config: Mapping[str, Any], run_id: str) -> Path:
    if not _RUN_ID.match(run_id):
        raise ValueError(f"--run must match {_RUN_ID.pattern}; got {run_id!r}")
    root = Path(require(config, "observed.runs_dir"))
    return (root if root.is_absolute() else CONFIG_PATH.parent / root) / run_id


class _Result:
    """result.json of one run: one section per command, merged on every write; other sections
    are never dropped. ``start`` replaces the command's section with a fresh one."""

    def __init__(self, run_dir: Path, run_id: str, skill: str, section: str) -> None:
        self.path = run_dir / "result.json"
        self.run_id, self.skill, self.section = run_id, skill, section

    def _write(self, record: Mapping[str, Any], *, fresh: bool) -> None:
        if self.path.exists():
            result = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            result = {"run_id": self.run_id, "skill": self.skill}
        if result.get("run_id") != self.run_id or result.get("skill") != self.skill:
            raise ValueError(f"{self.path} belongs to run {result.get('run_id')!r} of skill {result.get('skill')!r}")
        result[self.section] = _jsonable(dict(record) if fresh else {**result.get(self.section, {}), **record})
        result["updated_utc"] = _utc_now().isoformat()
        atomic_write_text(self.path, _dumps(result, indent=1) + "\n")

    def start(self, record: Mapping[str, Any]) -> None:
        self._write({"state": "running", "started_utc": _utc_now().isoformat(), **record}, fresh=True)

    def merge(self, record: Mapping[str, Any]) -> None:
        self._write(record, fresh=False)


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _step(events: EventLog, step: str, body: Callable[[], tuple[Any, Mapping[str, Any]]]) -> Any:
    """Emit running, run ``body`` -> (value, data), emit done(data) / refused / failed."""
    events.emit(step, "running")
    try:
        value, data = body()
    except _REFUSALS as error:
        events.emit(step, "refused", data=_error(error))
        raise _Refused(str(error)) from error
    except Exception as error:
        events.emit(step, "failed", data=_error(error))
        raise
    events.emit(step, "done", data=data)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print(label: str, value: Any) -> None:
    print(f"{label}: {_dumps(value, indent=1)}", flush=True)


def _teacher_setup(args: argparse.Namespace, config: Mapping[str, Any], spec: Any) -> tuple[dict, dict, list[str]]:
    checked = _check_spec(spec)
    context, dream = load_rows(args.context), load_dream(args.dream)
    for label, data in (("context", context), ("dream", dream)):
        _check_meta(checked, data["meta"], label)
    if context["features"].shape[1] != dream["features"].shape[1]:
        raise ValueError(f"context D {context['features'].shape[1]} and dream D {dream['features'].shape[1]} differ")
    feature_set = list(require(config, f"skills.{checked.name}.teacher_features"))
    return context, dream, feature_set


def _fit_step(
    events: EventLog, spec: Any, context: dict, config: Mapping[str, Any], feature_set: list[str]
) -> TabPFNTeacher:
    def body() -> tuple[TabPFNTeacher, dict[str, Any]]:
        teacher = make_teacher(spec, context, config, feature_set)
        return teacher, {"n_context_rows": int(context["action"].shape[0]), "n_features": teacher.layout.n_features}

    return _step(events, "teacher_fit", body)


def _quotes(events: EventLog, teacher: TabPFNTeacher, counts: Mapping[str, int]) -> dict[str, Any]:
    """Free quotes for each named query-row count; the "dream" step is running meanwhile."""
    events.emit("dream", "running", data={"phase": "quote", "query_rows": dict(counts)})
    try:
        quotes = {label: quote_dream(teacher, count) for label, count in counts.items()}
    except Exception as error:
        events.emit("dream", "failed", data=_error(error))
        raise
    _print("quotes (free; nothing billed)", quotes)
    return quotes


def _quote_command(args: argparse.Namespace, config: Mapping[str, Any], spec: Any, events: EventLog, result: _Result) -> int:
    context, dream, feature_set = _teacher_setup(args, config, spec)
    result.merge({"context": context["path"], "dream": dream["path"], "feature_set": feature_set})
    teacher = _fit_step(events, spec, context, config, feature_set)
    quotes = _quotes(events, teacher, {"dream": int(dream["physics"].shape[0] * dream["physics"].shape[1])})
    total = quotes["dream"]["total_quoted_tokens"]
    events.emit("dream", "stopped", data={"reason": "quote only", "total_quoted_tokens": total})
    result.merge({"state": "done", "quote": quotes})
    return 0


def _dev_rules(spec: Any, dev: dict, levels: Sequence[float]) -> dict[str, Any]:
    """One-shot reports of the rules that need no teacher: no_op and physics."""
    checked = _check_spec(spec)
    n_situations = _situation_table(dev, checked.n_actions)[0].size
    physics_values = np.asarray(dev["physics"], dtype=np.float64)
    physics = OutcomeDistribution.point(checked.outcome_names, physics_values, np.asarray(levels))
    reports = {"physics": one_shot_report(spec, dev, _choose(spec, dev, _objective(spec, physics)), quantile_levels=levels)}
    if checked.no_op_index is not None:
        chosen = np.full(n_situations, checked.no_op_index, dtype=np.int64)
        reports["no_op"] = one_shot_report(spec, dev, chosen, quantile_levels=levels)
    return reports


def _dev_teacher(
    teacher: TabPFNTeacher, spec: Any, dev: dict, feature_set: list[str], levels: Sequence[float]
) -> dict[str, Any]:
    """Residual MAE per outcome (physics only vs teacher) and the teacher rule's one-shot report;
    one billed prediction over the dev rows."""
    checked = _check_spec(spec)
    x, _ = teacher_matrix(spec, dev["features"], dev["action"], dev["physics"], feature_set)
    predicted = predict_outcomes(teacher, spec, x, dev["physics"], allow_billing=True)
    keys = list(teacher.last_request_hashes)
    measured = np.asarray(dev["measured"], dtype=np.float64)
    physics = np.asarray(dev["physics"], dtype=np.float64)
    binary = np.array([kind == "binary" for kind in checked.outcome_kinds])
    point = np.where(binary[None, :], predicted.mean, predicted.median)
    mae = {
        name: {"physics_only": float(np.mean(np.abs(measured[:, k] - physics[:, k]))),
               "teacher": float(np.mean(np.abs(measured[:, k] - point[:, k])))}
        for k, name in enumerate(checked.outcome_names)
    }
    chosen = _choose(spec, dev, _objective(spec, predicted))
    return {
        "residual_mae": mae,
        "residual_mae_note": (
            "mean |measured - point| per outcome; physics only: the physics guess; teacher: the predicted mean for "
            "binary outcomes and the predicted median for regression outcomes, clipped to the outcome bounds"
        ),
        "teacher": one_shot_report(spec, dev, chosen, quantile_levels=levels),
        "request_keys": keys,
    }


def _teach_command(args: argparse.Namespace, config: Mapping[str, Any], spec: Any, events: EventLog, result: _Result) -> int:
    context, dream, feature_set = _teacher_setup(args, config, spec)
    dev = load_rows(args.dev) if args.dev else None
    if dev is not None:
        _check_meta(_check_spec(spec), dev["meta"], "dev")
    levels = tuple(float(level) for level in require(config, "engine.quantile_levels"))
    result.merge({"context": context["path"], "dream": dream["path"], "dev": None if dev is None else dev["path"],
                  "feature_set": feature_set})
    teacher = _fit_step(events, spec, context, config, feature_set)
    result.merge({"teacher": {"n_context_rows": int(context["action"].shape[0]),
                              "n_features": teacher.layout.n_features, "call_log": teacher.settings.call_log}})

    if dev is not None:
        def rules() -> tuple[dict[str, Any], dict[str, Any]]:
            reports = _dev_rules(spec, dev, levels)
            return reports, {"rules": sorted(reports)}

        result.merge({"dev_rules": _step(events, "baseline", rules)})

    counts = {"dream": int(dream["physics"].shape[0] * dream["physics"].shape[1])}
    if dev is not None:
        counts["dev"] = int(dev["action"].shape[0])
    result.merge({"quote": _quotes(events, teacher, counts)})
    if not args.confirm_billing:
        reason = "--confirm-billing was not given: nothing was sent"
        events.emit("dream", "refused", data={"reason": reason})
        raise _Refused(reason)

    def score() -> tuple[dict[str, Any], dict[str, Any]]:
        scores = dream_scores(teacher, spec, dream, feature_set, allow_billing=True)
        keys = list(teacher.last_request_hashes)
        buffer = io.BytesIO()
        np.savez(buffer, scores=scores, scenario_index=dream["scenario_index"], decision_time_s=dream["decision_time_s"],
                 request_keys=np.array(keys, dtype=str), predicted_mean=dream_scores.last_predicted_mean)
        path = result.path.parent / "scores.npz"
        atomic_write_bytes(path, buffer.getvalue())
        record = {"path": path, "sha256": _sha256(path), "shape": list(scores.shape), "request_keys": keys}
        return record, {"scores": path, "n_requests": len(keys)}

    result.merge({"scores": _step(events, "dream", score)})
    if dev is not None:
        def evaluate() -> tuple[dict[str, Any], dict[str, Any]]:
            report = _dev_teacher(teacher, spec, dev, feature_set, levels)
            return report, {"request_keys": report["request_keys"], "residual_mae": report["residual_mae"]}

        result.merge({"dev_teacher": _step(events, "evaluate", evaluate)})
    result.merge({"state": "done"})
    return 0


def _newest_frames(features: np.ndarray, config: Mapping[str, Any], stored: int, frames: int) -> np.ndarray:
    """The newest ``frames`` frames of each stored window (vision.take_frames), flattened."""
    if not 1 <= frames <= stored:
        raise ValueError(f"--frames must be in [1, {stored}] (the stored frames); got {frames}")
    stored_settings = VisionSettings.from_config(config, frames=stored)
    VisionSettings.from_config(config, frames=frames)  # frames must be vision.frames or a variant
    if features.shape[1] != stored_settings.n_features:
        raise ValueError(
            f"dream features have {features.shape[1]} columns; {stored} frames need {stored_settings.n_features}"
        )
    if frames == stored:
        return features
    return np.stack([flatten_window(take_frames(unflatten_window(row, stored_settings), frames)) for row in features])


def _train_command(args: argparse.Namespace, config: Mapping[str, Any], spec: Any, events: EventLog, result: _Result) -> int:
    checked = _check_spec(spec)
    dream = load_dream(args.dream)
    _check_meta(checked, dream["meta"], "dream")
    scores_path = Path(args.scores)
    with np.load(scores_path, allow_pickle=False) as archive:
        scores = archive["scores"]
        for name in ("scenario_index", "decision_time_s"):
            if not np.array_equal(archive[name], dream[name]):
                raise ValueError(f"{scores_path} {name} differs from the dream's: the scores belong to another dream")
        request_keys = [str(key) for key in archive["request_keys"]]
    if dream["meta"]["frames"] is None:  # features that are not image windows (e.g. ramp): used as stored
        if args.frames is not None:
            raise ValueError("--frames applies to image windows only; this dream's meta has frames null")
        frames, features = None, np.asarray(dream["features"])
        window_encoding, diff_floor = None, None
    else:
        stored = int(dream["meta"]["frames"])
        frames = stored if args.frames is None else int(args.frames)
        raw = _newest_frames(dream["features"], config, stored, frames)
        window_encoding, diff_floor = window_encoding_from_config(config)
        size = VisionSettings.from_config(config, frames=frames).frame_size
        features = encode_window_features(raw.reshape(raw.shape[0], frames, size, size), window_encoding, diff_floor)
    hidden = list(args.hidden) if args.hidden else list(require(config, f"observed.{checked.name}.snn_hidden_sizes"))
    steps = int(args.steps) if args.steps is not None else int(require(config, f"observed.{checked.name}.snn_max_steps"))
    seed = int(require(config, "runtime.seed"))
    targets = dream_targets(scores, spec, config, predicted_mean=_predicted_mean(args.scores))

    def progress(info: dict[str, Any]) -> None:
        events.emit("train", "running", done=int(info["step"]), total=int(info["max_steps"]), data={
            key: info[key] for key in ("train_loss", "validation_loss", "best_validation_loss", "best_step")
        })

    settings = {"dream": dream["path"], "scores": scores_path, "frames": frames, "window_encoding": window_encoding,
                "diff_floor": diff_floor, "hidden_sizes": hidden, "max_steps": steps,
                "threat_target": threat_target_settings(config, checked.name)}
    result.merge(settings)
    events.emit("train", "running", done=0, total=steps, data={**settings, "rows": int(features.shape[0])})
    try:
        trained = train_student(spec, features, targets, config, hidden_sizes=hidden, max_steps=steps, seed=seed,
                                groups=dream["scenario_index"], on_evaluation=progress)
    except Exception as error:
        events.emit("train", "failed", data={**_error(error), "report": _error_report(error)})
        raise
    provenance = {
        **settings, "run_id": args.run, "dream_config_digest": dream["meta"].get("config_digest"),
        "scores_sha256": _sha256(scores_path), "teacher_request_keys": request_keys, "config_digest": config_digest(config),
        "seeds": trained["seeds"], "training_settings": trained["settings"],
    }
    student_dir = result.path.parent / "student"
    digest = save_student(
        student_dir, trained["network"], spec, frames=frames, config=config, thresholds=None, provenance=provenance
    )
    training = trained["training"]
    events.emit("train", "done", done=training["steps_run"], total=steps, data={
        "student": student_dir, "network_sha256": digest, "best_step": training["best_step"],
        "best_validation_loss_per_head": training["best_validation_loss_per_head"], "wall_s": trained["wall_s"],
    })
    result.merge({
        "state": "done", "student": student_dir, "network_sha256": digest, "split": trained["split"],
        "parity": trained["parity"], "training": training, "wall_s": trained["wall_s"],
    })
    _print("train", {"student": student_dir, "network_sha256": digest, "split": trained["split"],
                     "wall_s": trained["wall_s"]})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.observed", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--run", required=True, help="run id: the directory under observed.runs_dir")
        sub.add_argument("--skill", required=True, help="module under learned_judgment/skills with build_observed_spec")

    quote = commands.add_parser("quote", help="free quote of the dream prediction and current usage; nothing billed")
    common(quote)
    quote.add_argument("--context", required=True, help="rows file directory of the teacher's context")
    quote.add_argument("--dream", required=True, help="dream file directory")
    teach = commands.add_parser("teach", help="fit the teacher and score the dream (billed with --confirm-billing)")
    common(teach)
    teach.add_argument("--context", required=True)
    teach.add_argument("--dream", required=True)
    teach.add_argument("--dev", help="rows file directory of the dev split (one extra billed prediction)")
    teach.add_argument("--confirm-billing", action="store_true", help="send the quoted billable requests")
    train = commands.add_parser("train", help="train and export the spiking student on scored dream rows")
    common(train)
    train.add_argument("--dream", required=True)
    train.add_argument("--scores", required=True, help="scores.npz written by teach")
    train.add_argument("--frames", type=int, help="use the newest F frames of each window (vision.frame_variants)")
    train.add_argument("--hidden", type=int, nargs="+", help="hidden sizes (default observed.<skill>.snn_hidden_sizes)")
    train.add_argument("--steps", type=int, help="max optimizer steps (default observed.<skill>.snn_max_steps)")
    return parser


def _error_report(error: BaseException) -> Any:
    """The report a TrainingError (TrainingReport) or ParityError (dict) carries, else None."""
    report = getattr(error, "report", None)
    return report.to_json() if hasattr(report, "to_json") else report


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point; returns the exit code (0 done, 3 refused; a failure is recorded
    in result.json and raised)."""
    args = _parser().parse_args(argv)
    config = load_config()
    spec = load_spec(args.skill, config)
    run_dir = _run_dir(config, args.run)
    events = EventLog(run_dir / "events.jsonl", args.run, spec.name)
    result = _Result(run_dir, args.run, spec.name, args.command)
    result.start({"argv": list(sys.argv[1:] if argv is None else argv), "config_digest": config_digest(config)})
    command = {"quote": _quote_command, "teach": _teach_command, "train": _train_command}[args.command]
    try:
        code = command(args, config, spec, events, result)
    except _Refused as refusal:
        result.merge({"state": "refused", "reason": str(refusal)})
        print(f"refused: {refusal}", flush=True)
        return EXIT_REFUSED
    except Exception as error:
        result.merge({"state": "failed", "error": _error(error), "report": _error_report(error)})
        raise
    print(f"{args.command}: done; {result.path}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
