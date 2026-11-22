"""End-to-end run of one skill in one world: collect, residual model, dream, train, evaluate.

Lane contract (Person B, pipeline)
----------------------------------
Every step below already exists as a tested function; this module composes them into one
run that a person (command line) or a web interface can drive, and saves the state of every
step so that a run can be stopped, reopened and continued.

    step       composed functions
    collect    eval.splits.split_situations (split "context"), engine.collect.exploration_actions,
               eval.splits.noise_seed (split "context", repeat 0), engine.collect.collect
    teacher    engine.learn.fit_residual_model with teacher.TabPFNTeacher or a baseline of
               eval.baselines (fitting sends nothing to Prior Labs; the request is made when the
               model predicts)
    dream      engine.dream.dream, engine.dream.keep_confident, engine.dream.save_dreams
    train      engine.train_in_dream.train_in_dream, students.snn.SNNStudent.save
    evaluate   engine.decide.decide_batch, students.snn.SNNStudent.decide,
               eval.harness.compare_methods, eval.metrics, eval.report.write_run_result

Nothing here knows a skill: the skill is a SkillSpec, the world comes from a factory, and every
tunable is read from config.yaml with :func:`learned_judgment.config.require`.

One run
-------
A run is (run_id, skill, world preset, base seed, config digest, world). Its hidden world is
fixed for the whole run:

    world_seed = engine.streams.stream_seed(base_seed, "pipeline/world/<world_preset>")

The same world seed builds the world of every collect and every evaluate call of the run
(``world_factory(world_seed, world_preset)``); each episode has its own attempt seed. The world
seed depends on the base seed and the preset only, so every skill run with the same base seed
and preset meets the same hidden world.

Named random streams (engine.streams) under the base seed:

    context situations   "<skill>/context/<preset>/situations"   (eval.splits, split "context")
    context attempt seed "<skill>/context/<preset>/noise/0", index = attempt index
    context actions      "<skill>/context/<preset>/exploration"   (engine.collect.exploration_actions)
    dream situations     "<skill>/dreams/<preset>/situations"
    baseline seed        "<skill>/residual/<method>"               (stream_seed)
    student              "<skill>/snn/..." (engine.train_in_dream)

Saved state (paths from config.yaml, relative paths taken from the repository root; computed by
:class:`RunPaths`)::

    engine.attempts_dir/<run_id>.jsonl                      attempt rows (engine.rows)
    engine.dreams_dir/<run_id>.npz + .json                  the run's dream set (engine.dream)
    pipeline.students_dir/<run_id>.npz + .json              the dream-trained spiking student
    pipeline.evaluation_rows_dir/<run_id>/<evaluation_id>/  attempt files + harness manifests
    pipeline.run_results_dir/<run_id>/<evaluation_id>/<method>.json   run-result files
    pipeline.manifests_dir/<run_id>.json                    the run manifest (below)

The run manifest is one JSON file per run, rewritten atomically (learned_judgment.fileio) at
every state change. It holds the format tag, the run identity, the world seed, the config
digest, a description of the world factory, every path, every stream name, and per step
(``steps``) the state of its latest call plus ``history``, the record of every finished call:

    state         not_run | running | done | stopped | refused | failed
    started_utc / finished_utc   from the run's clock (ISO 8601, zero UTC offset)
    wall_s        seconds of wall time (time.perf_counter)
    settings      the settings the step ran with
    counts        rows, dreams, episodes, ...
    outputs       paths written
    reason        why a step stopped or was refused
    error         {"type", "message"} of a failure

Run identity and a changed config (the rule)
    A manifest that already exists is reopened only with the same skill, world preset, base
    seed, world seed, world description and config digest. A different value of any of them
    raises :class:`RunMismatchError` and nothing is written. The config digest
    (``learned_judgment.config.config_digest`` of the whole parsed config.yaml) must be equal:
    ANY change to config.yaml after a run started (a tunable, a path, Person A's world or
    command settings) refuses the run, because its attempts, dreams, student and evaluations
    would otherwise come from different settings. Start a new run_id under the new config.

Single writer: one process at a time drives a run. A step left ``running`` by a process that
ended stays ``running`` in the manifest until the step is called again.

Never overwritten: attempts are appended and never rerun (an attempt index already in the
attempt file is skipped); a run has one dream set and one student (a second dream or train
call raises :class:`RunStateError`); every evaluate call gets a new evaluation id.

Never adopted: a run_id whose manifest does not exist while any of its attempt, dream, student
or evaluation files does is refused at opening (:class:`RunStateError`), because those files'
identity is unknown. Every collect call compares the stored attempt rows with the plan
(situation, action, attempt seed, world preset), also when no attempt is missing.

Readers: :meth:`SkillRun.decide_with`, :meth:`SkillRun.cost_preview` and
:meth:`SkillRun.dream_cost_preview` never write the manifest (their residual-model fits stay in
memory), so they may be called on a run that another process drives.

Billed steps
------------
Methods ``teacher`` and ``direct_tabpfn`` predict through the hosted TabPFN service. Before a
billed prediction (dream, evaluate, decide_with) the run reads the current account usage and a
cost quote of every request the prediction would send that is not already in the teacher's
cache, with the teacher's own request rules (stacking per ``engine.multi_outcome_mode``,
chunks of ``tabpfn.max_query_rows_per_request`` rows, one padding row for a chunk of exactly
Q rows). dream and evaluate write this preview into the manifest before predicting;
decide_with returns it in its answer and writes nothing to the manifest. The teacher's own gate
then decides each request: before every uncached request it reads usage and a quote itself and
appends them to its call log (``tabpfn.call_log``) before anything is dispatched.
A refusal by the teacher (BudgetError, an unresolved earlier dispatch, a missing call log, a
held cache lock) ends the step in state ``refused``; nothing is retried. An unreachable
provider refuses the step before anything is sent.

Stopping
--------
``should_stop()`` is checked before every step and, through :class:`ObservedWorld`, before
every episode of collect and evaluate: the observer raises :class:`Stopped` from ``reset``.
Rows measured before the stop stay on disk; the step ends ``stopped`` and the run stays usable.

Progress
--------
``progress(event)`` receives dicts with the keys ``run_id``, ``step``, ``event`` ("start" |
"progress" | "finish" | "stopped" | "refused" | "failed"), ``done``, ``total`` (int or None)
and ``message``.

Command line (from the repository root)::

    python -m learned_judgment.pipeline run --skill NAME --preset PRESET --run-id ID \\
        --steps collect,teacher,dream,train,evaluate --method METHOD [--attempts N] [--dreams N]
    python -m learned_judgment.pipeline status [--tabpfn]
    python -m learned_judgment.pipeline cost --skill NAME --preset PRESET --run-id ID --method METHOD
    python -m learned_judgment.pipeline decide --skill NAME --preset PRESET --run-id ID --brain BRAIN \\
        --situation name=value [name=value ...]

Exit codes: 0 every step done, 1 a step failed (or the run could not be opened), 2 invalid
arguments, 3 a step was refused, 4 a step was stopped.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import re
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.config import CONFIG_PATH, config_digest, load_config, optional, require
from learned_judgment.engine.collect import collect, exploration_actions
from learned_judgment.engine.decide import PHYSICS_ONLY, decide, decide_batch
from learned_judgment.engine.dream import DreamSet, keep_confident, load_dreams, save_dreams
from learned_judgment.engine.dream import dream as engine_dream
from learned_judgment.engine.features import FeatureLayout, build_features, build_layout
from learned_judgment.engine.learn import fit_residual_model
from learned_judgment.engine.rows import AttemptRow, read_rows, rows_to_arrays
from learned_judgment.engine.streams import stream_rng, stream_seed
from learned_judgment.engine.train_in_dream import train_in_dream
from learned_judgment.engine.types import ResidualDistribution, ResidualModel
from learned_judgment.eval import baselines
from learned_judgment.eval.harness import MethodEvaluation, compare_methods
from learned_judgment.eval.metrics import decision_agreement, rate_with_count, tolerance_rate
from learned_judgment.eval.report import write_run_result
from learned_judgment.eval.splits import (
    SealedSplitError,
    noise_seed,
    noise_seeds,
    split_situations,
    test_situations,
)
from learned_judgment.fileio import atomic_write_text
from learned_judgment.skills import available_skills, has_for_world, load_skill, load_skill_for_world
from learned_judgment.skills.base import CommandSchema, SkillSpec, World, check_command_binding
from learned_judgment.students.snn import SNNStudent
from learned_judgment.teacher import (
    BudgetError,
    LedgerMissingError,
    TabPFNProvider,
    TabPFNTeacher,
    TeacherSettings,
    UnresolvedDispatchError,
    budget_refusal_reasons,
    constant_outcomes,
    remaining_tokens,
    request_key,
    stack_query,
    stack_training,
)
from learned_judgment.teacher_cache import CacheCorruptError, CacheLock, CacheLockedError, CallLog, RequestCache

__all__ = [
    "BILLED_METHODS",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_STOPPED",
    "EXIT_USAGE",
    "MANIFEST_FORMAT",
    "SNN_PER_SKILL",
    "STATES",
    "STEPS",
    "MethodNotBuiltError",
    "ObservedWorld",
    "RunMismatchError",
    "RunPaths",
    "RunStateError",
    "SkillRun",
    "StepRefused",
    "Stopped",
    "describe_world_factory",
    "find_manifests",
    "go2_world_factory",
    "main",
    "open_run",
    "read_run_manifest",
    "skill_states",
    "tabpfn_status",
    "utc_now",
    "world_seed_for",
]

MANIFEST_FORMAT = "learned_judgment.pipeline_run/v1"
STEPS = ("collect", "teacher", "dream", "train", "evaluate")
STATES = ("not_run", "running", "done", "stopped", "refused", "failed")
EVENTS = ("start", "progress", "finish", "stopped", "refused", "failed")
SNN_PER_SKILL = "snn_per_skill"
SNN_SHARED = "snn_shared"
ORACLE = "oracle"
TEACHER = "teacher"
DIRECT_TEACHER = "direct_tabpfn"
BILLED_METHODS = (TEACHER, DIRECT_TEACHER)
OPEN_EVALUATION_SPLITS = ("dev", "shift")
FINAL_SPLIT = "test"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_STOPPED = 4

WORLD_SEED_STREAM = "pipeline/world/{preset}"
_CONTEXT_SPLIT = "context"
_CONTEXT_REPEAT = 0  # the context attempts use noise repeat 0 of the context split
_ACTION_STREAM = "{skill}/context/{preset}/exploration"
_DREAM_STREAM = "{skill}/dreams/{preset}/situations"
_RESIDUAL_STREAM = "{skill}/residual/{method}"

_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # used in file names; no "." so file suffixes stay unambiguous
_LOCAL_RESIDUAL_CLASSES = (
    baselines.ConstantBiasResidual,
    baselines.GaussianProcessResidual,
    baselines.GradientBoostingResidual,
    baselines.MLPAttemptsResidual,
)
_LOCAL_RESIDUAL = {cls.method: cls for cls in _LOCAL_RESIDUAL_CLASSES}
# Teacher exceptions that are refusals: raised before a billable call, or instead of re-sending one.
_TEACHER_REFUSALS = (BudgetError, UnresolvedDispatchError, LedgerMissingError, CacheLockedError)
_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "format",
        "run_id",
        "skill",
        "world_preset",
        "base_seed",
        "world_seed",
        "world_seed_stream",
        "config_digest",
        "world",
        "created_utc",
        "paths",
        "streams",
        "steps",
        "history",
        "evaluations",
    }
)
_SNN_SCORE_LABEL = (
    "snn_readout: the spiking network's readout per candidate (mean readout membrane over its time steps); "
    "the student chooses the largest. It is neither an objective score nor a predicted outcome."
)
_OBJECTIVE_SCORE_LABEL = (
    "objective: spec.objective of the predicted outcome distribution (physics guess plus predicted "
    "residual) of each candidate; higher is better"
)


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class Stopped(RuntimeError):
    """``should_stop()`` returned true; the step ended at an episode or step boundary."""


class StepRefused(RuntimeError):
    """A step was refused before anything billable was sent (teacher gate, unreachable provider).

    Attributes:
        step:   the step name.
        reason: the refusal reason, also written to the manifest.
    """

    def __init__(self, step: str, reason: str) -> None:
        super().__init__(f"step {step!r} refused: {reason}")
        self.step = step
        self.reason = reason


class RunMismatchError(ValueError):
    """The run's manifest exists with another skill, preset, base seed, world or config digest."""


class RunStateError(ValueError):
    """A step's precondition is not met (no attempts yet, dreams or student already saved, ...).

    Raised before the step starts; the manifest is not changed.
    """


class MethodNotBuiltError(NotImplementedError):
    """The method is listed in evaluation.methods but this pipeline does not build it."""


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def utc_now() -> str:
    """Current time as an ISO 8601 string with a zero UTC offset (the engine's clock convention)."""
    return datetime.now(timezone.utc).isoformat()


def _check_utc(label: str, value: Any) -> str:
    try:
        moment = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an ISO 8601 string with a zero UTC offset; got {value!r}")
    return value


def _check_count(label: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _check_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise ValueError(f"run_id must match {_RUN_ID.pattern} (it names files); got {run_id!r}")
    return run_id


def _jsonable(value: Any) -> Any:
    """``value`` as strict JSON data: numpy scalars and arrays become Python values, tuples become
    lists, paths become strings, and a non-finite float becomes {"non_finite": repr}."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else {"non_finite": repr(number)}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(item) for item in items]
    return str(value)


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _resolve_dir(config: Mapping[str, Any], key: str) -> Path:
    value = require(config, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config.yaml key {key!r} must be a directory path string; got {value!r}")
    path = Path(value)
    return path if path.is_absolute() else CONFIG_PATH.parent / path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world_seed_for(base_seed: int, world_preset: str) -> int:
    """The hidden-world seed of every run with this base seed and preset:
    ``stream_seed(base_seed, "pipeline/world/<world_preset>")`` (a 32-bit integer)."""
    base_seed = _check_count("base_seed", base_seed, 0)
    if not isinstance(world_preset, str) or not world_preset:
        raise ValueError(f"world_preset must be a non-empty string; got {world_preset!r}")
    return stream_seed(base_seed, WORLD_SEED_STREAM.format(preset=world_preset))


def describe_world_factory(world_factory: Callable[..., Any]) -> str:
    """The factory's ``description`` attribute, or its module and qualified name.

    Recorded in the manifest and in every run result, so a result says which world produced it
    (the Go2 simulator or a plumbing fixture)."""
    description = getattr(world_factory, "description", None)
    if isinstance(description, str) and description:
        return description
    module = getattr(world_factory, "__module__", None) or type(world_factory).__module__
    name = getattr(world_factory, "__qualname__", None) or type(world_factory).__qualname__
    return f"{module}.{name}"


def go2_world_factory(config: Mapping[str, Any]) -> Callable[[int, str], World]:
    """``factory(world_seed, world_preset) -> Go2World(config, world_seed, preset=world_preset)``.

    ``learned_judgment.env`` (MuJoCo, the walking policy) is imported on the first call only.
    """

    def factory(world_seed: int, world_preset: str) -> World:
        from learned_judgment.env import Go2World

        return Go2World(dict(config), int(world_seed), preset=world_preset)

    factory.description = (  # type: ignore[attr-defined]
        "learned_judgment.env.Go2World: MuJoCo Go2 with the borrowed, frozen walking policy (Person A)"
    )
    return factory


# --------------------------------------------------------------------------------------
# Paths and manifests
# --------------------------------------------------------------------------------------


class RunPaths:
    """Every file and directory of one run, computed from config.yaml (relative paths are taken
    from the repository root, the directory of config.yaml).

    Attributes:
        run_id:          the run id.
        manifest:        pipeline.manifests_dir/<run_id>.json
        attempts:        engine.attempts_dir/<run_id>.jsonl
        dreams:          engine.dreams_dir/<run_id>.npz (with dreams_meta, the sibling .json)
        student_stem:    pipeline.students_dir/<run_id> (student_npz and student_json beside it)
        evaluation_rows_root: pipeline.evaluation_rows_dir/<run_id>
        run_results_root:     pipeline.run_results_dir/<run_id>
    """

    def __init__(self, config: Mapping[str, Any], run_id: str) -> None:
        self.run_id = _check_run_id(run_id)
        self.manifest = _resolve_dir(config, "pipeline.manifests_dir") / f"{run_id}.json"
        self.attempts = _resolve_dir(config, "engine.attempts_dir") / f"{run_id}.jsonl"
        self.dreams = _resolve_dir(config, "engine.dreams_dir") / f"{run_id}.npz"
        self.dreams_meta = self.dreams.with_suffix(".json")
        self.student_stem = _resolve_dir(config, "pipeline.students_dir") / run_id
        self.student_npz = self.student_stem.with_suffix(".npz")
        self.student_json = self.student_stem.with_suffix(".json")
        self.evaluation_rows_root = _resolve_dir(config, "pipeline.evaluation_rows_dir") / run_id
        self.run_results_root = _resolve_dir(config, "pipeline.run_results_dir") / run_id

    def evaluation_rows(self, evaluation_id: str) -> Path:
        """Directory of the attempt files and harness manifests of one evaluation."""
        return self.evaluation_rows_root / evaluation_id

    def run_results(self, evaluation_id: str) -> Path:
        """Directory of the run-result files (one per method) of one evaluation."""
        return self.run_results_root / evaluation_id

    def as_dict(self) -> dict[str, str]:
        return {
            "manifest": self.manifest.as_posix(),
            "attempts": self.attempts.as_posix(),
            "dreams": self.dreams.as_posix(),
            "dreams_meta": self.dreams_meta.as_posix(),
            "student_npz": self.student_npz.as_posix(),
            "student_json": self.student_json.as_posix(),
            "evaluation_rows_root": self.evaluation_rows_root.as_posix(),
            "run_results_root": self.run_results_root.as_posix(),
        }


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate key {key!r}")
        record[key] = value
    return record


def read_run_manifest(path: str | Path) -> dict[str, Any]:
    """Read a run manifest written by this module; returns it as a dict.

    Raises ValueError when the file is not strict JSON, has another format tag, lacks a
    required key, or names a step or state that does not exist. Opens no other file.
    """
    path = Path(path)
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not a pipeline run manifest: {error}") from error
    if not isinstance(record, dict) or record.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not a pipeline run manifest (format tag {MANIFEST_FORMAT!r} missing)")
    missing = sorted(_REQUIRED_MANIFEST_KEYS - set(record))
    if missing:
        raise ValueError(f"{path}: the manifest lacks the keys {missing}")
    steps = record["steps"]
    if not isinstance(steps, dict) or set(steps) != set(STEPS):
        raise ValueError(f"{path}: steps must name exactly {list(STEPS)}")
    for step, entry in steps.items():
        if not isinstance(entry, dict) or entry.get("state") not in STATES:
            raise ValueError(f"{path}: step {step!r} has no valid state")
    if not isinstance(record["history"], list) or not isinstance(record["evaluations"], list):
        raise ValueError(f"{path}: history and evaluations must be lists")
    return record


def find_manifests(config: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any] | None, str | None]]:
    """Every ``*.json`` under pipeline.manifests_dir, sorted by name, as (path, manifest or None,
    read error or None)."""
    directory = _resolve_dir(config, "pipeline.manifests_dir")
    found: list[tuple[Path, dict[str, Any] | None, str | None]] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        try:
            found.append((path, read_run_manifest(path), None))
        except (OSError, ValueError) as error:
            found.append((path, None, f"{type(error).__name__}: {error}"))
    return found


def _manifest_summary(path: Path, manifest: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "world_preset": manifest["world_preset"],
        "base_seed": manifest["base_seed"],
        "world": manifest["world"],
        "path": path.as_posix(),
        "config_digest_matches_current": manifest["config_digest"] == digest,
        "steps": {step: manifest["steps"][step]["state"] for step in STEPS},
        "evaluations": [entry.get("evaluation_id") for entry in manifest["evaluations"]],
    }


# --------------------------------------------------------------------------------------
# Observer
# --------------------------------------------------------------------------------------


class ObservedWorld:
    """A transparent wrapper of a world that counts finished episodes and stops between them.

    ``reset``, ``step`` and ``finish_episode`` are forwarded to the wrapped world with the same
    arguments and their return values are returned unchanged; every other attribute is read
    from the wrapped world. Before forwarding ``reset`` it calls ``should_stop()`` (when given)
    and raises :class:`Stopped` when that is true, so no new episode starts. After
    ``finish_episode`` returns it increments ``finished`` and calls ``on_episode(finished)``.
    """

    def __init__(
        self,
        world: Any,
        *,
        should_stop: Callable[[], bool] | None = None,
        on_episode: Callable[[int], None] | None = None,
    ) -> None:
        self._world = world
        self._should_stop = should_stop
        self._on_episode = on_episode
        self.finished = 0

    @property
    def wrapped(self) -> Any:
        """The world this observer forwards to."""
        return self._world

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self._should_stop is not None and self._should_stop():
            raise Stopped(f"should_stop() was true before episode {self.finished + 1} of this world started")
        return self._world.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self._world.step(*args, **kwargs)

    def finish_episode(self, *args: Any, **kwargs: Any) -> Any:
        outcome = self._world.finish_episode(*args, **kwargs)
        self.finished += 1
        if self._on_episode is not None:
            self._on_episode(self.finished)
        return outcome

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)


class _Captured(Exception):
    """Carries the query features a decide_batch call would send to its residual model."""

    def __init__(self, features: np.ndarray) -> None:
        super().__init__("query features captured")
        self.features = features


class _QueryCapture:
    """Stands in for a residual model inside :func:`decide_batch` only to read the exact query
    feature matrix that call would build; it never answers (predicting raises _Captured)."""

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "_QueryCapture":
        raise RuntimeError("_QueryCapture is never fitted")

    def predict_residuals(self, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise _Captured(np.array(query_features, dtype=np.float64, copy=True))

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        raise _Captured(np.array(query_features, dtype=np.float64, copy=True))


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


class SkillRun:
    """One run of one skill in one world, with saved state (see the module docstring).

    Args:
        config:        the parsed config.yaml.
        spec:          a built SkillSpec.
        world_factory: ``world_factory(world_seed: int, world_preset: str) -> World``; for the
                       simulator, :func:`go2_world_factory`.
        run_id:        names the run's files; letters, digits, "_" and "-".
        world_preset:  a key of config.yaml ``world.presets``.
        base_seed:     integer >= 0; the root of every named stream of the run.
        clock:         ``clock() -> str``, the current time as ISO 8601 with a zero UTC offset.
        progress:      optional ``progress(event: dict)`` (keys in the module docstring).
        should_stop:   optional ``should_stop() -> bool``, polled before every step and episode.

    Opening creates the manifest, or reopens an existing one with the same identity (skill,
    world preset, base seed, world seed, world description, config digest); anything else raises
    RunMismatchError and writes nothing. Creating a manifest while files of this run_id already
    exist raises RunStateError and writes nothing (files without a manifest are never adopted). Changed config digest: any change to config.yaml after
    the run started refuses the run; continue under the new config with a new run_id.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        spec: SkillSpec,
        *,
        world_factory: Callable[[int, str], World],
        run_id: str,
        world_preset: str,
        base_seed: int,
        clock: Callable[[], str],
        progress: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if not callable(world_factory):
            raise TypeError("world_factory must be callable as world_factory(world_seed, world_preset)")
        presets = require(config, "world.presets")
        if not isinstance(presets, Mapping) or world_preset not in presets:
            raise ValueError(f"world_preset {world_preset!r} is not a key of config.yaml world.presets")
        self.config = config
        self.spec = spec
        self.run_id = _check_run_id(run_id)
        self.world_preset = str(world_preset)
        self.base_seed = _check_count("base_seed", base_seed, 0)
        self.world_factory = world_factory
        self.clock = clock
        self._progress = progress
        self._should_stop = should_stop
        _check_utc("clock()", clock())

        self.command_schema = CommandSchema.from_config(config)
        check_command_binding(spec, self.command_schema)
        points = require(config, f"skills.{spec.name}.candidate_points_per_dim")
        self.points_per_dim = tuple(_check_count("candidate_points_per_dim entry", p, 1) for p in points)
        self.quantile_levels = tuple(float(level) for level in require(config, "engine.quantile_levels"))
        spread = require(config, "engine.spread_levels")
        if not isinstance(spread, (list, tuple)) or len(spread) != 2:
            raise ValueError(f"config.yaml engine.spread_levels must hold two levels; got {spread!r}")
        self.spread_levels = (float(spread[0]), float(spread[1]))
        self.exploration_fraction = require(config, "engine.attempt_exploration_fraction")
        self.physics_mismatch_policy = require(config, "pipeline.on_physics_mismatch")
        self.methods = tuple(require(config, "evaluation.methods"))
        self.layout = build_layout(spec)
        self._direct_layout = build_layout(spec, include_physics=False)

        self.paths = RunPaths(config, self.run_id)
        self.world_seed = world_seed_for(self.base_seed, self.world_preset)
        self.config_digest = config_digest(config)
        self.world_description = describe_world_factory(world_factory)
        self.streams = {
            "world_seed": WORLD_SEED_STREAM.format(preset=self.world_preset),
            "context_situations": f"{spec.name}/{_CONTEXT_SPLIT}/{self.world_preset}/situations (eval.splits)",
            "context_attempt_seeds": f"{spec.name}/{_CONTEXT_SPLIT}/{self.world_preset}/noise/{_CONTEXT_REPEAT} (eval.splits)",
            "context_actions": _ACTION_STREAM.format(skill=spec.name, preset=self.world_preset),
            "dream_situations": _DREAM_STREAM.format(skill=spec.name, preset=self.world_preset),
            "residual_seed": _RESIDUAL_STREAM.format(skill=spec.name, method="<method>"),
        }

        self._models: dict[tuple[str, int], ResidualModel] = {}  # fits recorded as manifest step "teacher"
        self._unrecorded_models: dict[tuple[str, int], ResidualModel] = {}  # fits of decide_with and cost previews
        self._teacher_settings: TeacherSettings | None = None
        self._provider: TabPFNProvider | None = None
        self._provider_connected = False
        self._student: SNNStudent | None = None
        self._manifest = self._open_manifest()

    # ----- manifest ------------------------------------------------------------------------

    def _identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "skill": self.spec.name,
            "world_preset": self.world_preset,
            "base_seed": self.base_seed,
            "world_seed": self.world_seed,
            "world": self.world_description,
            "config_digest": self.config_digest,
        }

    def _open_manifest(self) -> dict[str, Any]:
        identity = self._identity()
        if self.paths.manifest.exists():
            manifest = read_run_manifest(self.paths.manifest)
            differing = [
                f"{key}: manifest has {manifest.get(key)!r}, this call has {value!r}"
                for key, value in identity.items()
                if manifest.get(key) != value
            ]
            if differing:
                raise RunMismatchError(
                    f"run {self.run_id!r} already exists ({self.paths.manifest}) with another identity; "
                    + "; ".join(differing)
                    + ". A run is never continued under another skill, preset, base seed, world or config "
                    "(any change to config.yaml changes its digest): start a new run_id"
                )
            return manifest
        orphans = self._existing_run_files()
        if orphans:
            raise RunStateError(
                f"run {self.run_id!r} has files without a manifest ({self.paths.manifest} does not exist): {orphans}. "
                "Their identity (skill, preset, base seed, world, config digest) is unknown, so a run never adopts "
                "them: start a new run_id, or move these files away"
            )
        manifest = {
            "format": MANIFEST_FORMAT,
            **identity,
            "world_seed_stream": self.streams["world_seed"],
            "created_utc": self.clock(),
            "paths": self.paths.as_dict(),
            "streams": dict(self.streams),
            "steps": {step: {"state": "not_run"} for step in STEPS},
            "history": [],
            "evaluations": [],
        }
        self._write(manifest)
        return manifest

    def _existing_run_files(self) -> list[str]:
        """Paths of this run_id's attempt, dream, student and evaluation files that exist now."""
        candidates = (
            self.paths.attempts,
            self.paths.dreams,
            self.paths.dreams_meta,
            self.paths.student_npz,
            self.paths.student_json,
            self.paths.evaluation_rows_root,
            self.paths.run_results_root,
        )
        return [path.as_posix() for path in candidates if path.exists()]

    def _write(self, manifest: Mapping[str, Any]) -> None:
        text = json.dumps(_jsonable(manifest), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
        atomic_write_text(self.paths.manifest, text + "\n")

    def _save(self) -> None:
        self._write(self._manifest)

    @property
    def manifest(self) -> dict[str, Any]:
        """A deep copy of the run manifest as last written."""
        return copy.deepcopy(self._manifest)

    def step_state(self, step: str) -> str:
        """State of the latest call of ``step`` (one of STATES)."""
        if step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}; got {step!r}")
        return str(self._manifest["steps"][step]["state"])

    # ----- events and the step state machine ------------------------------------------------

    def _emit(self, step: str, event: str, done: int | None, total: int | None, message: str) -> None:
        if self._progress is not None:
            self._progress(
                {"run_id": self.run_id, "step": step, "event": event, "done": done, "total": total, "message": message}
            )

    def _stop_requested(self) -> bool:
        return self._should_stop is not None and bool(self._should_stop())

    def _new_record(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "state": "running",
            "started_utc": _check_utc("clock()", self.clock()),
            "finished_utc": None,
            "wall_s": None,
            "settings": _jsonable(settings),
            "counts": {},
            "outputs": {},
            "reason": None,
            "error": None,
        }

    def _finish(self, step: str, record: dict[str, Any], started: float, state: str, *, reason: str | None = None,
                error: BaseException | None = None) -> None:
        record.update(
            state=state,
            finished_utc=_check_utc("clock()", self.clock()),
            wall_s=time.perf_counter() - started,
            reason=reason,
            error=None if error is None else _error_record(error),
        )
        self._manifest["history"].append({"step": step, **_jsonable(record)})
        self._save()

    def _run_step(
        self,
        step: str,
        settings: Mapping[str, Any],
        body: Callable[[dict[str, Any]], Any],
        *,
        total: int | None = None,
    ) -> Any:
        """Run ``body(record)`` as step ``step``: state running, then done / stopped / refused /
        failed, with times, settings, counts, outputs and the reason or error in the manifest."""
        record = self._new_record(settings)
        self._manifest["steps"][step] = record
        started = time.perf_counter()
        if self._stop_requested():
            reason = "should_stop() was true before the step started; nothing was run"
            self._finish(step, record, started, "stopped", reason=reason)
            self._emit(step, "stopped", 0, total, reason)
            raise Stopped(reason)
        self._save()
        self._emit(step, "start", 0, total, f"{step} started")
        try:
            result = body(record)
        except Stopped as stop:
            self._finish(step, record, started, "stopped", reason=str(stop))
            self._emit(step, "stopped", record["counts"].get("done"), total, str(stop))
            raise
        except KeyboardInterrupt as interrupt:
            reason = "interrupted (KeyboardInterrupt) during the step"
            self._finish(step, record, started, "stopped", reason=reason, error=interrupt)
            self._emit(step, "stopped", record["counts"].get("done"), total, reason)
            raise
        except StepRefused as refusal:
            self._finish(step, record, started, "refused", reason=refusal.reason)
            self._emit(step, "refused", record["counts"].get("done"), total, refusal.reason)
            raise
        except _TEACHER_REFUSALS as refusal:
            reason = f"{type(refusal).__name__}: {refusal}"
            self._finish(step, record, started, "refused", reason=reason)
            self._emit(step, "refused", record["counts"].get("done"), total, reason)
            raise StepRefused(step, reason) from refusal
        except Exception as failure:
            self._finish(step, record, started, "failed", error=failure)
            self._emit(step, "failed", record["counts"].get("done"), total, f"{type(failure).__name__}: {failure}")
            raise
        self._finish(step, record, started, "done")
        self._emit(step, "finish", record["counts"].get("done", total), total, f"{step} done")
        return result

    # ----- attempts --------------------------------------------------------------------------

    def attempts(self) -> list[AttemptRow]:
        """Every attempt row of this run in file order (attempt index order); [] before collecting."""
        if not self.paths.attempts.exists():
            return []
        rows = read_rows(self.paths.attempts, spec=self.spec)
        foreign = [index for index, row in enumerate(rows) if row.run_id != self.run_id or row.source != "real"]
        if foreign:
            raise RunStateError(
                f"{self.paths.attempts} holds rows {foreign[:5]} that are not real attempts of run {self.run_id!r}"
            )
        return rows

    def _context_count(self, rows: Sequence[AttemptRow], n_context: int | None) -> int:
        if not rows:
            raise RunStateError(f"run {self.run_id!r} has no attempts yet: run collect_attempts first")
        if n_context is None:
            return len(rows)
        n = _check_count("n_context", n_context, 1)
        if n > len(rows):
            raise RunStateError(f"n_context {n} exceeds the {len(rows)} attempts collected so far")
        return n

    def _plan_context(self, n_attempts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Situations (n, S), actions (n, A) and attempt seeds int64 (n,) of context attempts
        0..n-1; attempt i does not depend on n."""
        spec = self.spec
        situations = split_situations(
            spec, base_seed=self.base_seed, split=_CONTEXT_SPLIT, n=n_attempts, world_preset=self.world_preset
        )
        actions = exploration_actions(
            spec,
            situations,
            rng=stream_rng(self.base_seed, self.streams["context_actions"]),
            points_per_dim=self.points_per_dim,
            exploration_fraction=self.exploration_fraction,
            quantile_levels=self.quantile_levels,
            spread_levels=self.spread_levels,
            layout=self.layout,
        )
        seeds = np.array(
            [
                noise_seed(
                    spec,
                    base_seed=self.base_seed,
                    split=_CONTEXT_SPLIT,
                    world_preset=self.world_preset,
                    situation_index=index,
                    repeat=_CONTEXT_REPEAT,
                )
                for index in range(n_attempts)
            ],
            dtype=np.int64,
        )
        return situations, actions, seeds

    def collect_attempts(self, n_attempts: int) -> list[AttemptRow]:
        """Make sure attempts 0..n_attempts-1 exist; run only the missing ones.

        Attempt i has situation i of the context split, the action of row i of
        ``exploration_actions`` (stream "<skill>/context/<preset>/exploration") and attempt seed
        ``noise_seed(split="context", situation_index=i, repeat=0)``; all depend on i only, so a
        call with 10 after 5 rows exist runs attempts 5..9 and the 10 rows equal those of one call
        with 10. The stored rows are compared with the plan first (situation, action, attempt
        seed); a difference raises RunStateError. Episodes run through
        :func:`engine.collect.collect` in ``world_factory(world_seed, world_preset)`` wrapped in
        :class:`ObservedWorld`. Returns every attempt row of the run (existing and new).
        """
        n_attempts = _check_count("n_attempts", n_attempts, 1)
        settings = {
            "n_attempts": n_attempts,
            "split": _CONTEXT_SPLIT,
            "exploration_fraction": self.exploration_fraction,
            "points_per_dim": list(self.points_per_dim),
            "world_seed": self.world_seed,
            "streams": {
                key: self.streams[key] for key in ("context_situations", "context_attempt_seeds", "context_actions")
            },
        }

        def body(record: dict[str, Any]) -> list[AttemptRow]:
            existing = self.attempts()
            have = len(existing)
            record["counts"].update(rows_before=have, done=min(have, n_attempts))
            record["outputs"]["attempts"] = self.paths.attempts.as_posix()
            # The stored rows are compared with the plan on every call, also when none is missing,
            # so a call never returns rows that this run's streams did not plan.
            situations, actions, seeds = self._plan_context(max(have, n_attempts))
            if have:
                stored = rows_to_arrays(self.spec, existing, on_physics_mismatch=self.physics_mismatch_policy)
                same = (
                    np.array_equal(stored.situations, situations[:have])
                    and np.array_equal(stored.actions, actions[:have])
                    and [row.seed for row in existing] == [int(seed) for seed in seeds[:have]]
                    and all(preset == self.world_preset for preset in stored.world_presets)
                )
                if not same:
                    raise RunStateError(
                        f"the {have} stored attempts of run {self.run_id!r} differ from the planned attempts 0..{have - 1} "
                        "(situation, action, attempt seed or world preset): the plan changed; start a new run_id"
                    )
            if have >= n_attempts:
                record["counts"].update(new_rows=0, rows_after=have)
                return existing

            def on_episode(finished: int) -> None:
                record["counts"]["done"] = have + finished
                self._emit("collect", "progress", have + finished, n_attempts, f"attempt {have + finished} of {n_attempts}")

            world = ObservedWorld(
                self.world_factory(self.world_seed, self.world_preset),
                should_stop=self._should_stop,
                on_episode=on_episode,
            )
            try:
                collect(
                    self.spec,
                    world,
                    situations[have:],
                    actions[have:],
                    attempt_seeds=seeds[have:],
                    command_schema=self.command_schema,
                    run_id=self.run_id,
                    world_preset=self.world_preset,
                    out_path=self.paths.attempts,
                    clock=self.clock,
                )
            finally:
                after = len(self.attempts()) if self.paths.attempts.exists() else 0
                record["counts"].update(new_rows=after - have, rows_after=after, done=min(after, n_attempts))
            return self.attempts()

        return self._run_step("collect", settings, body, total=n_attempts)

    # ----- residual models ---------------------------------------------------------------------

    def _method_kind(self, method: str) -> str:
        """"physics" | "local" | "billed" | "student" for a method of evaluation.methods."""
        if not isinstance(method, str) or method not in self.methods:
            raise ValueError(f"method {method!r} is not listed in config.yaml evaluation.methods {list(self.methods)}")
        if method == PHYSICS_ONLY:
            return "physics"
        if method in _LOCAL_RESIDUAL:
            return "local"
        if method in BILLED_METHODS:
            return "billed"
        if method == SNN_PER_SKILL:
            return "student"
        if method == ORACLE:
            raise MethodNotBuiltError(
                "the oracle method is not built in this pipeline: it needs the evaluation-only oracle search over "
                "supported actions with repeated noise draws, which does not exist yet"
            )
        if method == SNN_SHARED:
            raise MethodNotBuiltError(
                "snn_shared is a multi-skill student (students.shared_snn) and is not part of a single-skill run"
            )
        raise MethodNotBuiltError(f"method {method!r} has no builder in learned_judgment.pipeline")

    def _layout_for(self, method: str) -> FeatureLayout:
        return self._direct_layout if method == DIRECT_TEACHER else self.layout

    def _teacher_settings_from_config(self) -> TeacherSettings:
        if self._teacher_settings is None:
            self._teacher_settings = TeacherSettings.from_config(self.config)
        return self._teacher_settings

    def _datetime_clock(self) -> datetime:
        return datetime.fromisoformat(_check_utc("clock()", self.clock()))

    def _provider_object(self) -> TabPFNProvider:
        if self._provider is None:
            self._provider = TabPFNProvider(self._teacher_settings_from_config())
        return self._provider

    def _connected_provider(self) -> TabPFNProvider:
        provider = self._provider_object()
        if not self._provider_connected:
            provider.connect()
            self._provider_connected = True
        return provider

    def _new_model(self, method: str) -> ResidualModel:
        kind = self._method_kind(method)
        if kind == "local":
            seed = stream_seed(self.base_seed, _RESIDUAL_STREAM.format(skill=self.spec.name, method=method))
            return _LOCAL_RESIDUAL[method].from_config(self.config, self.layout, seed=seed)
        if kind != "billed":
            raise ValueError(f"method {method!r} is not a residual method")
        layout = self._layout_for(method)
        teacher = TabPFNTeacher.from_settings(
            layout, self._teacher_settings_from_config(), provider=self._provider_object(), clock=self._datetime_clock
        )
        if method == DIRECT_TEACHER:
            return baselines.DirectTargetAdapter(teacher, spec=self.spec, layout=layout)
        return teacher

    def residual_model(self, method: str, n_context: int | None = None) -> ResidualModel | None:
        """The residual model ``method`` fitted on the first ``n_context`` attempts (None = all).

        ``"physics_only"`` returns None (zero residual by definition). ``"teacher"`` builds
        :class:`TabPFNTeacher` (``TabPFNTeacher.from_settings`` with a :class:`TabPFNProvider`);
        ``"direct_tabpfn"`` wraps a teacher on the layout without physics columns in
        ``eval.baselines.DirectTargetAdapter``; any other residual method is the baseline of
        ``eval.baselines`` whose ``method`` attribute has that name, built with ``from_config``
        and the seed ``stream_seed(base_seed, "<skill>/residual/<method>")``. Fitting goes
        through ``engine.learn.fit_residual_model`` and sends nothing to Prior Labs. A fit is
        recorded as manifest step "teacher" (method, n_context) and kept in memory for this
        object, so asking again for the same method and number of attempts does not refit.
        :meth:`decide_with` and the cost previews never call this: they use the recorded fit when
        this object has one and otherwise fit in memory without writing the manifest.
        """
        kind = self._method_kind(method)
        if kind == "physics":
            return None
        if kind not in ("local", "billed"):
            raise ValueError(f"method {method!r} is not a residual method")
        rows = self.attempts()
        n = self._context_count(rows, n_context)
        key = (method, n)
        if key in self._models:
            return self._models[key]
        settings = {
            "method": method,
            "n_context_requested": n_context,
            "n_context": n,
            "on_physics_mismatch": self.physics_mismatch_policy,
            "layout": self._layout_for(method).describe(),
            "billed_at_prediction": kind == "billed",
        }

        def body(record: dict[str, Any]) -> ResidualModel:
            fitted = self._fit(method, rows[:n])
            record["counts"].update(n_context=n, n_attempts_available=len(rows))
            describe = getattr(fitted, "describe", None)
            record["model"] = _jsonable(describe()) if callable(describe) else {"class": type(fitted).__name__}
            return fitted

        fitted = self._run_step("teacher", settings, body)
        self._models[key] = fitted
        return fitted

    def _fit(self, method: str, rows: Sequence[AttemptRow]) -> ResidualModel:
        """A new residual model of ``method`` fitted on ``rows`` (engine.learn.fit_residual_model)."""
        return fit_residual_model(
            self.spec, self._new_model(method), rows, self._layout_for(method), on_physics_mismatch=self.physics_mismatch_policy
        )

    def _unrecorded_model(self, method: str, n_context: int | None) -> ResidualModel | None:
        """The residual model for calls that only read the run (decide_with, cost previews).

        Returns this object's recorded fit of (method, n_context) when it has one; otherwise fits
        one and keeps it in memory for this object only. Writes nothing to the manifest, so such a
        call never rewrites the manifest of a run another process may be driving. The fit is the
        same as the recorded one: same attempts, same seed (stream "<skill>/residual/<method>").
        """
        kind = self._method_kind(method)
        if kind == "physics":
            return None
        if kind not in ("local", "billed"):
            raise ValueError(f"method {method!r} is not a residual method")
        rows = self.attempts()
        n = self._context_count(rows, n_context)
        key = (method, n)
        if key in self._models:
            return self._models[key]
        if key not in self._unrecorded_models:
            self._unrecorded_models[key] = self._fit(method, rows[:n])
        return self._unrecorded_models[key]

    # ----- cost preview ---------------------------------------------------------------------

    def _training_arrays(self, method: str, n: int) -> tuple[FeatureLayout, np.ndarray, np.ndarray]:
        """(layout, features (n, F), targets (n, K)) exactly as the billed model's teacher receives
        them in ``fit``: residuals for "teacher", outcomes rebuilt by the adapter for "direct_tabpfn"."""
        rows = self.attempts()[:n]
        arrays = rows_to_arrays(self.spec, rows, on_physics_mismatch=self.physics_mismatch_policy)
        layout = self._layout_for(method)
        features = build_features(
            layout, arrays.situations, arrays.actions, physics=arrays.physics if layout.n_physics else None
        )
        targets = arrays.residuals
        if method == DIRECT_TEACHER:
            adapter = self._unrecorded_model(method, n)
            targets = targets + adapter.physics(features)  # type: ignore[union-attr]
        return layout, features, targets

    def _request_plan(
        self,
        layout: FeatureLayout,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        query: np.ndarray | None,
        n_query_rows: int,
    ) -> dict[str, Any]:
        """Every provider request the teacher would make for ``n_query_rows`` query rows, by the
        teacher's own rules; with ``query`` (n_query_rows, F) each request's key and whether the
        teacher's cache already holds it."""
        settings = self._teacher_settings_from_config()
        n_levels = len(settings.quantile_levels)
        cap = settings.max_query_rows_per_request
        active = tuple(int(index) for index in np.flatnonzero(~constant_outcomes(targets)))
        plan: dict[str, Any] = {
            "mode": settings.multi_outcome_mode,
            "n_query_rows": int(n_query_rows),
            "n_train_rows": int(features.shape[0]),
            "active_outcomes": [layout.outcome_names[index] for index in active],
            "constant_outcomes": [name for index, name in enumerate(layout.outcome_names) if index not in active],
            "max_query_rows_per_request": cap,
            "cache_checked": query is not None,
            "requests": [],
        }
        if not active:
            return plan
        width = features.shape[1]
        groups: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, int, list[str], tuple[int, ...]]] = []
        if settings.multi_outcome_mode == "stacked":
            x_train, y_train, _, _ = stack_training(features, targets, active)
            several = len(active) > 1
            categorical = layout.categorical_indices + ((width,) if several else ())
            stacked = stack_query(query, active) if query is not None else None
            groups.append((x_train, y_train, stacked, len(active) * int(n_query_rows), list(plan["active_outcomes"]), categorical))
        else:
            for index in active:
                groups.append(
                    (
                        features,
                        np.ascontiguousarray(targets[:, index]),
                        query,
                        int(n_query_rows),
                        [layout.outcome_names[index]],
                        layout.categorical_indices,
                    )
                )
        base_url = self._provider_object().base_url() if query is not None else None
        cache = RequestCache(settings.cache_dir) if query is not None else None
        for x_train, y_train, rows, total, outcomes, categorical in groups:
            n_chunks = max(1, math.ceil(total / cap))
            for chunk, start in enumerate(range(0, total, cap)):
                stop = min(start + cap, total)
                padded = 1 if stop - start == n_levels else 0
                request: dict[str, Any] = {
                    "outcomes": outcomes,
                    "chunk": chunk,
                    "n_chunks": n_chunks,
                    "n_train": int(x_train.shape[0]),
                    "n_query": int(stop - start + padded),
                    "n_features": int(x_train.shape[1]),
                    "padded_rows": padded,
                }
                if rows is not None and cache is not None and base_url is not None:
                    sent = rows[start:stop]
                    if padded:
                        sent = np.concatenate([sent, sent[-1:]], axis=0)
                    key = request_key(
                        settings=settings,
                        x_train=x_train,
                        y_train=y_train,
                        x_query=np.ascontiguousarray(sent),
                        categorical_indices=categorical,
                        base_url=base_url,
                    )
                    request["key"] = key
                    try:
                        request["cached"] = cache.get(key) is not None
                    except CacheCorruptError as error:
                        request["cached"] = False
                        request["cache_error"] = f"{type(error).__name__}: {error}"
                plan["requests"].append(request)
        return plan

    def _quote(self, method: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Usage and quotes (free provider calls) for the requests of ``plan`` not in the cache,
        and an advisory replay of the teacher's budget gate. Never guesses a number: an
        unreachable provider gives ``reachable: False`` and the error."""
        settings = self._teacher_settings_from_config()
        requests = plan["requests"]
        to_send = [request for request in requests if request.get("cached") is not True]
        preview: dict[str, Any] = {
            "method": method,
            "billed": True,
            **{key: value for key, value in plan.items() if key != "requests"},
            "n_requests": len(requests),
            "n_cached": len(requests) - len(to_send),
            "n_to_send": len(to_send),
            "requests": requests,
            "project_token_budget": settings.project_token_budget,
            "max_tokens_per_request": settings.max_tokens_per_request,
            "note": (
                "the cost of every request not in the cache, quoted by the provider's free estimate_cost; "
                "a cached request is served without a charge"
                if plan["cache_checked"]
                else "the cost of every request the teacher would make, quoted by the provider's free estimate_cost; "
                "the cache was not consulted (no query rows given), so requests already cached are counted too"
            ),
        }
        try:
            provider = self._connected_provider()
            usage = provider.usage()
            quotes: dict[tuple[int, int, int], dict[str, Any]] = {}
            for request in to_send:
                shape = (request["n_train"], request["n_query"], request["n_features"])
                if shape not in quotes:
                    quotes[shape] = provider.quote(*shape)
                request["quote_tokens"] = quotes[shape]["estimated_cost"]
                request["pricing_version"] = quotes[shape]["pricing_version"]
        except Exception as error:  # noqa: BLE001 - any failure to reach the provider is reported, never guessed
            preview.update(reachable=False, error=f"{type(error).__name__}: {error}")
            return preview
        total = int(sum(request["quote_tokens"] for request in to_send))
        preview.update(reachable=True, usage=usage, total_quoted_tokens=total)
        try:
            preview["remaining_account_tokens"] = remaining_tokens(usage)
        except Exception as error:  # noqa: BLE001
            preview["remaining_account_tokens"] = None
            preview["remaining_account_tokens_error"] = f"{type(error).__name__}: {error}"
        try:
            quoted_before = self._call_log().tokens_quoted_for_billed_calls()
            preview["tokens_quoted_for_billed_calls"] = quoted_before
        except Exception as error:  # noqa: BLE001
            preview["tokens_quoted_for_billed_calls"] = None
            preview["ledger_error"] = f"{type(error).__name__}: {error}"
            quoted_before = None
        reasons: list[str] = []
        if quoted_before is not None:
            already = quoted_before
            remaining = preview["remaining_account_tokens"]
            for request in to_send:
                tokens = request["quote_tokens"]
                reasons = budget_refusal_reasons(
                    quote_tokens=tokens,
                    max_tokens_per_request=settings.max_tokens_per_request,
                    project_token_budget=settings.project_token_budget,
                    tokens_already_quoted=already,
                    remaining_account_tokens=remaining,
                )
                if reasons:
                    break
                already += tokens
                remaining = None if remaining is None else remaining - tokens
        preview["gate_preview"] = {
            "would_refuse": bool(reasons),
            "reasons": reasons,
            "note": "advisory replay of teacher.budget_refusal_reasons; the teacher re-reads usage and a quote "
            "before each uncached request and its own gate decides",
        }
        return preview

    def _call_log(self) -> CallLog:
        settings = self._teacher_settings_from_config()
        cache = RequestCache(settings.cache_dir)
        lock = CacheLock(cache.lock_path, timeout_s=settings.lock_timeout_s, poll_s=settings.lock_poll_s)
        return CallLog(settings.call_log, clock=self._datetime_clock, lock=lock)

    def _capture_query(self, method: str, situations: np.ndarray) -> np.ndarray:
        """The exact query feature matrix ``decide_batch`` builds for ``situations`` under
        ``method``'s layout (every situation x candidate, physics columns included when the
        layout has them)."""
        try:
            decide_batch(
                self.spec,
                situations,
                model=_QueryCapture(),  # type: ignore[arg-type]
                layout=self._layout_for(method),
                points_per_dim=self.points_per_dim,
                quantile_levels=self.quantile_levels,
                spread_levels=self.spread_levels,
                method=method,
            )
        except _Captured as captured:
            return captured.features
        raise RuntimeError("decide_batch did not query its residual model")

    def _billed_preview(self, method: str, n_context: int | None, situations: np.ndarray, *, record: bool) -> dict[str, Any]:
        """Cost preview of the requests a prediction of ``situations`` would send. ``record``: fit
        the model as a recorded step (inside dream and evaluate) or in memory only (decide_with,
        dream_cost_preview), which writes nothing to the manifest."""
        n = self._context_count(self.attempts(), n_context)
        if record:
            self.residual_model(method, n)
        else:
            self._unrecorded_model(method, n)
        layout, features, targets = self._training_arrays(method, n)
        query = self._capture_query(method, situations)
        plan = self._request_plan(layout, features, targets, query=query, n_query_rows=int(query.shape[0]))
        preview = self._quote(method, plan)
        preview["n_context"] = n
        preview["n_situations"] = int(situations.shape[0])
        return preview

    @staticmethod
    def _require_reachable(step: str, preview: Mapping[str, Any]) -> None:
        if preview["n_to_send"] and not preview.get("reachable"):
            raise StepRefused(
                step,
                f"the provider could not be reached for the usage read and the cost quote ({preview.get('error')}); "
                "nothing was sent",
            )

    def cost_preview(self, method: str, n_query_rows: int, *, n_context: int | None = None) -> dict[str, Any]:
        """What a prediction of ``n_query_rows`` query rows (situations x candidates) by ``method``
        would cost, without sending it.

        Billed methods ("teacher", "direct_tabpfn"): the residual model is fitted on the first
        ``n_context`` attempts (free), the requests are planned with the teacher's rules, and the
        provider's free usage read and ``estimate_cost`` quote give ``usage``, per-request
        ``quote_tokens`` and ``total_quoted_tokens``; ``gate_preview`` replays the teacher's
        budget rules. The cache is not consulted here (no query rows are given; see
        :meth:`dream_cost_preview`). An unreachable provider gives ``reachable: False`` and
        ``error``. Local methods return ``{"method", "billed": False, "n_query_rows", "message"}``.
        Sends no prediction and writes nothing to the manifest.
        """
        kind = self._method_kind(method)
        n_query_rows = _check_count("n_query_rows", n_query_rows, 1)
        if kind != "billed":
            return {
                "method": method,
                "billed": False,
                "n_query_rows": n_query_rows,
                "message": "local method: it runs on this machine; nothing is sent to Prior Labs and nothing is billed",
            }
        n = self._context_count(self.attempts(), n_context)
        self._unrecorded_model(method, n)
        layout, features, targets = self._training_arrays(method, n)
        plan = self._request_plan(layout, features, targets, query=None, n_query_rows=n_query_rows)
        preview = self._quote(method, plan)
        preview["n_context"] = n
        return preview

    def _dream_situations(self, n_dreams: int) -> np.ndarray:
        rng = stream_rng(self.base_seed, self.streams["dream_situations"])
        return np.asarray(self.spec.sample_situations(rng, n_dreams), dtype=np.float64)

    def dream_cost_preview(self, method: str, n_dreams: int, n_context: int | None = None) -> dict[str, Any]:
        """The cost preview of exactly the prediction :meth:`dream` would send for ``n_dreams``
        dreams: the same dream situations and candidates, each request's key, and only the
        requests not already in the teacher's cache are quoted. Local methods: nothing billed.
        Sends no prediction and writes nothing to the manifest."""
        kind = self._method_kind(method)
        n_dreams = _check_count("n_dreams", n_dreams, 1)
        if kind != "billed":
            return self.cost_preview(method, 1) | {"n_dreams": n_dreams, "n_query_rows": None}
        preview = self._billed_preview(method, n_context, self._dream_situations(n_dreams), record=False)
        preview["n_dreams"] = n_dreams
        return preview

    # ----- dream -------------------------------------------------------------------------------

    def _confidence_threshold(self) -> tuple[float | None, str]:
        thresholds = optional(self.config, "engine.confidence_thresholds_by_skill")
        if thresholds is None:
            return None, (
                "config.yaml engine.confidence_thresholds_by_skill is null: no threshold has been selected on dev "
                "data yet, so every dream is kept"
            )
        if not isinstance(thresholds, Mapping):
            raise ValueError("config.yaml engine.confidence_thresholds_by_skill must be null or a mapping")
        if self.spec.name not in thresholds or thresholds[self.spec.name] is None:
            return None, (
                f"config.yaml engine.confidence_thresholds_by_skill has no threshold for skill {self.spec.name!r}, "
                "so every dream is kept"
            )
        value = thresholds[self.spec.name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"the confidence threshold of {self.spec.name!r} must be a number; got {value!r}")
        return float(value), "config.yaml engine.confidence_thresholds_by_skill"

    def dream(self, method: str, n_dreams: int, n_context: int | None = None) -> DreamSet:
        """Dream ``n_dreams`` situations with ``method`` and save the kept dreams.

        ``engine.dream.dream`` samples the situations from the stream
        "<skill>/dreams/<preset>/situations" and decides them in one batched call of the residual
        model fitted on the first ``n_context`` attempts (``physics_only``: no model). Rows
        whose chosen spread exceeds ``engine.confidence_thresholds_by_skill[<skill>]`` are
        dropped (``keep_confident``); when no threshold is set every dream is kept and the
        manifest records ``confidence_threshold`` null and the reason. The kept set is saved with
        ``save_dreams`` to engine.dreams_dir/<run_id>.npz.

        For a billed method the step first writes the cost preview (usage and quotes of the
        uncached requests) into the manifest; an unreachable provider or a teacher refusal ends
        the step ``refused`` (StepRefused) with nothing retried. A run has one dream set: when it
        is already saved this raises RunStateError.
        """
        kind = self._method_kind(method)
        if kind not in ("physics", "local", "billed"):
            raise ValueError(f"method {method!r} cannot dream; use physics_only or a residual method")
        n_dreams = _check_count("n_dreams", n_dreams, 1)
        if self.paths.dreams.exists() or self.paths.dreams_meta.exists():
            raise RunStateError(
                f"run {self.run_id!r} already has a dream set ({self.paths.dreams}); a run has one dream set and it is "
                "never overwritten: start a new run_id for other dreams"
            )
        threshold, threshold_reason = self._confidence_threshold()
        settings = {
            "method": method,
            "n_dreams": n_dreams,
            "n_context_requested": n_context,
            "situation_stream": self.streams["dream_situations"],
            "points_per_dim": list(self.points_per_dim),
            "quantile_levels": list(self.quantile_levels),
            "spread_levels": list(self.spread_levels),
            "confidence_threshold": threshold,
            "confidence_threshold_reason": threshold_reason,
        }

        def body(record: dict[str, Any]) -> DreamSet:
            model = self.residual_model(method, n_context)
            n_used = None if model is None else self._context_count(self.attempts(), n_context)
            record["counts"]["n_context"] = n_used
            if kind == "billed":
                preview = self._billed_preview(method, n_used, self._dream_situations(n_dreams), record=True)
                record["cost_preview"] = _jsonable(preview)
                self._save()
                self._emit(
                    "dream",
                    "progress",
                    0,
                    n_dreams,
                    f"cost preview: {preview['n_to_send']} request(s) to send, {preview['n_cached']} cached, "
                    f"quoted {preview.get('total_quoted_tokens')} tokens",
                )
                self._require_reachable("dream", preview)
            dreams = engine_dream(
                self.spec,
                model,
                self._layout_for(method),
                n_dreams=n_dreams,
                rng=stream_rng(self.base_seed, self.streams["dream_situations"]),
                points_per_dim=self.points_per_dim,
                quantile_levels=self.quantile_levels,
                spread_levels=self.spread_levels,
                method=method,
            )
            kept = dreams if threshold is None else keep_confident(dreams, threshold)
            meta = {
                "run_id": self.run_id,
                "skill": self.spec.name,
                "world_preset": self.world_preset,
                "base_seed": self.base_seed,
                "config_digest": self.config_digest,
                "method": method,
                "n_context": n_used,
                "situation_stream": self.streams["dream_situations"],
                "n_dreamed": dreams.n_rows,
                "n_kept": kept.n_rows,
                "confidence_threshold": threshold,
                "confidence_threshold_reason": threshold_reason,
                "attempts_file": self.paths.attempts.as_posix() if model is not None else None,
                "attempts_sha256": _sha256_file(self.paths.attempts) if model is not None else None,
            }
            save_dreams(self.paths.dreams, kept, meta)
            record["counts"].update(
                n_dreamed=dreams.n_rows,
                n_kept=kept.n_rows,
                n_candidates=dreams.n_candidates,
                n_requests=len(dreams.request_hashes),
                done=n_dreams,
            )
            record["outputs"].update(dreams=self.paths.dreams.as_posix(), dreams_meta=self.paths.dreams_meta.as_posix())
            record["request_hashes"] = list(dreams.request_hashes)
            return kept

        return self._run_step("dream", settings, body, total=n_dreams)

    # ----- train ----------------------------------------------------------------------------

    def student(self) -> SNNStudent:
        """The run's saved dream-trained spiking student (loaded once with SNNStudent.load)."""
        if self._student is None:
            if not (self.paths.student_npz.exists() and self.paths.student_json.exists()):
                raise RunStateError(f"run {self.run_id!r} has no student yet: run train_student first")
            self._student = SNNStudent.load(self.paths.student_stem)
        return self._student

    def train_student(self) -> SNNStudent:
        """Train the per-skill spiking student inside the saved dreams and save it.

        ``engine.train_in_dream.train_in_dream`` on the dream set of this run (settings from
        config.yaml ``students.snn``), then ``SNNStudent.save`` to pipeline.students_dir/<run_id>.
        The manifest records the training report, the in-dream numbers of the early-stopping
        rows, the parity report, the deployed network's sha256 and the sha256 of the dream file.
        Emits start and finish events only. RunStateError when there are no dreams yet or the
        student already exists (never overwritten).
        """
        if not (self.paths.dreams.exists() and self.paths.dreams_meta.exists()):
            raise RunStateError(f"run {self.run_id!r} has no saved dreams yet: run dream first")
        if self.paths.student_npz.exists() or self.paths.student_json.exists():
            raise RunStateError(
                f"run {self.run_id!r} already has a student ({self.paths.student_stem}); it is never overwritten"
            )
        settings = {
            "dreams": self.paths.dreams.as_posix(),
            "students": _jsonable(require(self.config, "students")),
            "points_per_dim": list(self.points_per_dim),
            "base_seed": self.base_seed,
        }

        def body(record: dict[str, Any]) -> SNNStudent:
            dreams, _ = load_dreams(self.paths.dreams)
            student = train_in_dream(self.spec, dreams, self.config, base_seed=self.base_seed)
            digest = student.save(self.paths.student_stem)
            record["counts"].update(n_dream_rows=dreams.n_rows, n_candidates=dreams.n_candidates)
            record["outputs"].update(student_npz=self.paths.student_npz.as_posix(), student_json=self.paths.student_json.as_posix())
            record["network_sha256"] = digest
            record["dreams_sha256"] = _sha256_file(self.paths.dreams)
            record["training_report"] = _jsonable(student.meta.get("training"))
            record["early_stopping_rows_in_dream"] = _jsonable(student.meta.get("early_stopping_rows_in_dream"))
            record["parity"] = _jsonable(student.meta.get("parity"))
            self._student = student
            return student

        return self._run_step("train", settings, body)

    # ----- evaluate -----------------------------------------------------------------------------

    def _next_evaluation_id(self, split: str) -> str:
        index = 1 + sum(1 for entry in self._manifest["history"] if entry.get("step") == "evaluate")
        while True:
            evaluation_id = f"eval{index:03d}-{split}"
            if not (self.paths.evaluation_rows(evaluation_id).exists() or self.paths.run_results(evaluation_id).exists()):
                return evaluation_id
            index += 1

    def _policy(self, method: str, n_context: int | None) -> Callable[[np.ndarray], np.ndarray]:
        kind = self._method_kind(method)
        if kind == "student":
            student = self.student()
            return lambda situations: student.decide(self.spec, situations)
        model = self.residual_model(method, n_context) if kind != "physics" else None

        def policy(situations: np.ndarray) -> np.ndarray:
            decisions = decide_batch(
                self.spec,
                situations,
                model=model,
                layout=self._layout_for(method),
                points_per_dim=self.points_per_dim,
                quantile_levels=self.quantile_levels,
                spread_levels=self.spread_levels,
                method=method,
            )
            return np.stack([decision.action for decision in decisions])

        return policy

    def _metrics(self, evaluation: MethodEvaluation, physics_actions: np.ndarray | None) -> dict[str, float | int]:
        spec = self.spec
        n, repeats, k = evaluation.measured.shape
        measured = evaluation.measured.reshape(n * repeats, k)
        metrics: dict[str, float | int] = {
            "n_situations": n,
            "n_repeats": repeats,
            "n_episodes": n * repeats,
            "loss_mean": float(np.mean(evaluation.losses)),
        }
        tolerance = tolerance_rate(spec, measured)
        metrics.update(tolerance_rate=tolerance["rate"], tolerance_count=tolerance["count"])
        for column, outcome in enumerate(spec.outcomes):
            if outcome.kind == "binary":
                rate = rate_with_count(measured[:, column] == 1.0)
                metrics[f"{outcome.name}_rate"] = rate["rate"]
                metrics[f"{outcome.name}_count"] = rate["count"]
        if physics_actions is not None and evaluation.method != PHYSICS_ONLY:
            agreement = decision_agreement(evaluation.actions, physics_actions)
            metrics["agreement_with_physics_only"] = agreement["rate"]
        return metrics

    def evaluate(
        self,
        methods: Sequence[str],
        *,
        split: str,
        n_situations: int,
        n_repeats: int,
        method_context: Mapping[str, int | None] | None = None,
        final_evaluation: bool = False,
    ) -> dict[str, Any]:
        """Evaluate ``methods`` on the same situations, hidden world and attempt seeds.

        Args:
            methods:          names from evaluation.methods: "physics_only", residual methods
                              (decide_batch with the fitted model), "snn_per_skill" (the saved
                              student). "oracle" and "snn_shared" raise MethodNotBuiltError.
            split:            "dev" or "shift" (eval.splits.split_situations / noise_seeds);
                              "test" only together with ``final_evaluation=True``.
            n_situations:     situations of the split (indices 0..n-1).
            n_repeats:        noise repeats per situation.
            method_context:   {residual method: n_context}; a method not listed uses every attempt.
            final_evaluation: True reads the sealed test split through eval.splits.test_situations
                              (logged to evaluation.test_access_log); only with split "test".

        Runs ``eval.harness.compare_methods`` with world_factory(world_seed, preset) wrapped in
        :class:`ObservedWorld` (one fresh world per method, same world seed), writes one
        run-result file per method with ``eval.report.write_run_result`` and returns::

            {"evaluation_id", "split", "final_evaluation", "n_situations", "n_repeats", "world_seed",
             "methods": {method: {"run_result", "harness_manifest", "attempts", "metrics"}}}

        Metrics per method: n_situations, n_repeats, n_episodes, loss_mean (spec.loss averaged over
        every episode), tolerance_rate / tolerance_count (eval.metrics.tolerance_rate),
        <outcome>_rate / <outcome>_count for each binary outcome (eval.metrics.rate_with_count),
        and agreement_with_physics_only (eval.metrics.decision_agreement) when physics_only is
        evaluated in the same call. Every call gets a new evaluation id; nothing is overwritten.
        """
        methods = list(methods)
        if not methods or len(set(methods)) != len(methods):
            raise ValueError(f"methods must be a non-empty list without repeats; got {methods}")
        kinds = {method: self._method_kind(method) for method in methods}
        n_situations = _check_count("n_situations", n_situations, 1)
        n_repeats = _check_count("n_repeats", n_repeats, 1)
        if final_evaluation is True:
            if split != FINAL_SPLIT:
                raise ValueError(f"final_evaluation=True reads the sealed test split; pass split={FINAL_SPLIT!r}")
        else:
            if final_evaluation is not False:
                raise ValueError("final_evaluation must be True or False")
            if isinstance(split, str) and split.strip().casefold() == FINAL_SPLIT:
                raise SealedSplitError(
                    "the test split is sealed: only evaluate(..., split='test', final_evaluation=True) reads it"
                )
            if split not in OPEN_EVALUATION_SPLITS:
                raise ValueError(f"split must be one of {OPEN_EVALUATION_SPLITS}; got {split!r}")
        contexts = dict(method_context or {})
        for method in contexts:
            if method not in methods or kinds[method] not in ("local", "billed"):
                raise ValueError(f"method_context names {method!r}, which is not a residual method of this evaluation")
        if "student" in kinds.values():
            self.student()
        evaluation_id = self._next_evaluation_id(split)
        compare_run_id = f"{self.run_id}__{evaluation_id}"
        rows_dir = self.paths.evaluation_rows(evaluation_id)
        results_dir = self.paths.run_results(evaluation_id)
        total = len(methods) * n_situations * n_repeats
        settings = {
            "evaluation_id": evaluation_id,
            "methods": methods,
            "split": split,
            "final_evaluation": final_evaluation,
            "n_situations": n_situations,
            "n_repeats": n_repeats,
            "method_context": contexts,
            "world_seed": self.world_seed,
            "compare_run_id": compare_run_id,
        }
        entry: dict[str, Any] = {"evaluation_id": evaluation_id, "split": split, "methods": methods, "state": "running"}

        def body(record: dict[str, Any]) -> dict[str, Any]:
            record["outputs"].update(evaluation_rows=rows_dir.as_posix(), run_results=results_dir.as_posix())
            if final_evaluation is True:
                sealed = test_situations(
                    self.spec,
                    self.config,
                    base_seed=self.base_seed,
                    n=n_situations,
                    n_repeats=n_repeats,
                    world_preset=self.world_preset,
                    final_evaluation=True,
                    clock=self.clock,
                )
                situations, seeds = sealed.situations, sealed.noise_seeds
            else:
                situations = split_situations(
                    self.spec, base_seed=self.base_seed, split=split, n=n_situations, world_preset=self.world_preset
                )
                seeds = noise_seeds(
                    self.spec,
                    base_seed=self.base_seed,
                    split=split,
                    world_preset=self.world_preset,
                    n_situations=n_situations,
                    n_repeats=n_repeats,
                )
            details: dict[str, Any] = {}
            for method in methods:
                if kinds[method] == "billed":
                    preview = self._billed_preview(method, contexts.get(method), situations, record=True)
                    record.setdefault("cost_previews", {})[method] = _jsonable(preview)
                    self._save()
                    self._require_reachable("evaluate", preview)
            policies = {method: self._policy(method, contexts.get(method)) for method in methods}
            for method in methods:
                if kinds[method] in ("local", "billed"):
                    n_used = self._context_count(self.attempts(), contexts.get(method))
                    model = self.residual_model(method, n_used)
                    describe = getattr(model, "describe", None)
                    details[method] = {
                        "n_context": n_used,
                        "model": _jsonable(describe()) if callable(describe) else {"class": type(model).__name__},
                    }
                elif kinds[method] == "student":
                    student = self.student()
                    details[method] = {"network_sha256": student.digest, "student": self.paths.student_stem.as_posix()}
                else:
                    details[method] = {"model": None}
            finished = {"episodes": 0}

            def on_episode(_: int) -> None:
                finished["episodes"] += 1
                record["counts"]["done"] = finished["episodes"]
                self._emit("evaluate", "progress", finished["episodes"], total, f"episode {finished['episodes']} of {total}")

            def observed_factory(world_seed: int) -> ObservedWorld:
                return ObservedWorld(
                    self.world_factory(world_seed, self.world_preset), should_stop=self._should_stop, on_episode=on_episode
                )

            results = compare_methods(
                self.spec,
                policies,
                world_factory=observed_factory,
                world_seed=self.world_seed,
                situations=situations,
                attempt_seeds=seeds,
                command_schema=self.command_schema,
                run_id=compare_run_id,
                world_preset=self.world_preset,
                split=split,
                base_seed=self.base_seed,
                config_digest=self.config_digest,
                out_dir=rows_dir,
                clock=self.clock,
            )
            physics_actions = results[PHYSICS_ONLY].actions if PHYSICS_ONLY in results else None
            summary: dict[str, Any] = {}
            for method, evaluation in results.items():
                metrics = self._metrics(evaluation, physics_actions)
                path = results_dir / f"{method}.json"
                provenance = {
                    "manifests": [evaluation.manifest_path],
                    "pipeline_manifest": self.paths.manifest.as_posix(),
                    "pipeline_run_id": self.run_id,
                    "evaluation_id": evaluation_id,
                    "config_digest": self.config_digest,
                    "base_seed": self.base_seed,
                    "world_seed": self.world_seed,
                    "world": self.world_description,
                    "simulated": "every number comes from simulated episodes in the world named under 'world'",
                    "decisions_computed_on": (
                        f"this machine ({platform.platform()}, CPU)"
                        + ("; predictions by the hosted Prior Labs TabPFN service" if kinds[method] == "billed" else "")
                    ),
                    "method_detail": details[method],
                    "metric_functions": [
                        "eval.harness.compare_methods losses (spec.loss)",
                        "eval.metrics.tolerance_rate",
                        "eval.metrics.rate_with_count",
                        "eval.metrics.decision_agreement",
                    ],
                }
                write_run_result(
                    path,
                    skill=self.spec.name,
                    method=method,
                    run_id=compare_run_id,
                    world_preset=self.world_preset,
                    split=split,
                    metrics=metrics,
                    provenance=_jsonable(provenance),
                )
                summary[method] = {
                    "run_result": path.as_posix(),
                    "harness_manifest": Path(evaluation.manifest_path).as_posix(),
                    "attempts": Path(evaluation.rows_path).as_posix(),
                    "metrics": metrics,
                }
            record["counts"].update(done=total, n_episodes=total)
            record["outputs"]["run_result_files"] = {method: item["run_result"] for method, item in summary.items()}
            entry["results"] = {method: item["run_result"] for method, item in summary.items()}
            return {
                "evaluation_id": evaluation_id,
                "split": split,
                "final_evaluation": final_evaluation,
                "n_situations": n_situations,
                "n_repeats": n_repeats,
                "world_seed": self.world_seed,
                "methods": summary,
            }

        try:
            result = self._run_step("evaluate", settings, body, total=total)
        except BaseException:
            entry["state"] = self._manifest["steps"]["evaluate"]["state"]
            self._manifest["evaluations"].append(entry)
            self._save()
            raise
        entry["state"] = "done"
        self._manifest["evaluations"].append(entry)
        self._save()
        return result

    # ----- one decision ---------------------------------------------------------------------------

    def _situation_row(self, situation: Mapping[str, Any] | Sequence[float] | np.ndarray) -> np.ndarray:
        if isinstance(situation, Mapping):
            return self.spec.encode_situation(dict(situation))
        row = np.asarray(situation, dtype=np.float64)
        if row.shape != (self.spec.n_situation,):
            raise ValueError(f"an encoded situation must have shape ({self.spec.n_situation},); got {row.shape}")
        self.spec.encode_situation(self.spec.decode_situation(row))
        return row

    def _outcome_map(self, values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.spec.outcome_names, values)}

    def decide_with(
        self,
        brain: str,
        situation: Mapping[str, Any] | Sequence[float] | np.ndarray,
        *,
        n_context: int | None = None,
    ) -> dict[str, Any]:
        """One decision of ``brain`` for one situation; returns JSON-serialisable data.

        Args:
            brain:     "physics_only", a residual method of evaluation.methods (fitted on the first
                       ``n_context`` attempts, None = all), or "snn_per_skill" (the saved student).
            situation: {situation name: value} (categorical values as level strings) or an
                       encoded row (S,) in spec order.

        Returns::

            {"run_id", "skill", "brain", "situation": {name: value}, "situation_clipped": bool,
             "action_names": [A], "action": {name: value}, "chosen_index": int,
             "candidates": [[A floats] x C], "scores": [C floats], "score_kind": "objective" | "snn_readout",
             "score_label": str, "physics_index": int | None,
             "physics_guess": {outcome: float} of the chosen action,
             "predicted_outcome": {"mean", "median", "spread": {outcome: float}} | None (None for the student),
             "request_hashes": [str], "cost_preview": dict | None, "n_context": int | None}

        plus "network_sha256" for the student. ``situation`` is the situation actually decided on
        (for the student under students.snn.out_of_range "clip", the clipped one; then
        ``situation_clipped`` is true). For a billed brain the cost preview of the one request
        group is taken first and returned under "cost_preview" (StepRefused when the provider
        cannot be reached and something would be sent); the teacher's gate then decides, and it
        appends the usage and quote of the request to its call log (tabpfn.call_log) before any
        dispatch; a teacher refusal raises StepRefused. Nothing is written to the run manifest: a
        residual model this object has not fitted as a recorded step is fitted in memory only.
        """
        kind = self._method_kind(brain)
        row = self._situation_row(situation)
        if kind == "student":
            return self._decide_with_student(row)
        if kind not in ("physics", "local", "billed"):
            raise ValueError(f"brain {brain!r} cannot decide")
        preview = None
        n_used = None
        if kind != "physics":
            n_used = self._context_count(self.attempts(), n_context)
        if kind == "billed":
            preview = self._billed_preview(brain, n_used, row[None, :], record=False)
            self._require_reachable("decide", preview)
        model = self._unrecorded_model(brain, n_used) if kind != "physics" else None
        try:
            decision = decide(
                self.spec,
                row,
                model=model,
                layout=self._layout_for(brain),
                points_per_dim=self.points_per_dim,
                quantile_levels=self.quantile_levels,
                spread_levels=self.spread_levels,
                method=brain,
            )
        except _TEACHER_REFUSALS as refusal:
            raise StepRefused("decide", f"{type(refusal).__name__}: {refusal}") from refusal
        chosen = decision.chosen_index
        predicted = decision.predicted
        return _jsonable(
            {
                "run_id": self.run_id,
                "skill": self.spec.name,
                "brain": brain,
                "situation": self.spec.decode_situation(row),
                "situation_clipped": False,
                "action_names": list(self.spec.action_names),
                "action": self.spec.decode_action(decision.action),
                "chosen_index": chosen,
                "candidates": decision.candidates,
                "scores": decision.scores,
                "score_kind": "objective",
                "score_label": _OBJECTIVE_SCORE_LABEL,
                "physics_index": decision.physics_index,
                "physics_guess": self._outcome_map(predicted.physics[chosen]),
                "predicted_outcome": {
                    "mean": self._outcome_map(predicted.mean[chosen]),
                    "median": self._outcome_map(predicted.median[chosen]),
                    "spread": self._outcome_map(predicted.spread[chosen]),
                },
                "request_hashes": list(decision.request_hashes),
                "cost_preview": preview,
                "n_context": n_used,
            }
        )

    def _decide_with_student(self, row: np.ndarray) -> dict[str, Any]:
        student = self.student()
        deployed = student.deployed()
        student.check_spec(self.spec)
        decided_on, clipped = deployed.check_situations(row[None, :])
        chosen = int(student.decide_index_one(row))
        readout = deployed.network.forward(deployed.encode(decided_on))[0]
        if int(np.argmax(readout)) != chosen:
            raise RuntimeError(
                "the student's per-decision path chose another candidate than the argmax of its readout on the same "
                "input; report this (it would mean the two numpy paths disagree)"
            )
        situation = decided_on[0]
        grid = np.asarray(self.spec.candidate_actions(situation, student.points_per_dim), dtype=np.float64)
        action = grid[chosen]
        physics = np.asarray(self.spec.physics_guess(situation[None, :], action[None, :]), dtype=np.float64)[0]
        return _jsonable(
            {
                "run_id": self.run_id,
                "skill": self.spec.name,
                "brain": SNN_PER_SKILL,
                "situation": self.spec.decode_situation(situation),
                "situation_clipped": bool(np.any(clipped)),
                "action_names": list(self.spec.action_names),
                "action": self.spec.decode_action(action),
                "chosen_index": chosen,
                "candidates": grid,
                "scores": readout,
                "score_kind": "snn_readout",
                "score_label": _SNN_SCORE_LABEL,
                "physics_index": None,
                "physics_guess": self._outcome_map(physics),
                "predicted_outcome": None,
                "request_hashes": [],
                "cost_preview": None,
                "n_context": None,
                "network_sha256": student.digest,
            }
        )


# --------------------------------------------------------------------------------------
# Module-level entry points
# --------------------------------------------------------------------------------------


def open_run(
    config: Mapping[str, Any],
    skill_name: str,
    *,
    run_id: str,
    world_preset: str,
    base_seed: int,
    clock: Callable[[], str],
    progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SkillRun:
    """A :class:`SkillRun` of the registered skill ``skill_name`` on the Go2 simulator.

    Builds one Go2World (go2_world_factory, the run's world seed) to read what the skill needs
    from the world, builds the spec with ``skills.load_skill_for_world`` and opens the run with
    the same factory. The probe world is not used for any episode.
    """
    factory = go2_world_factory(config)
    world = factory(world_seed_for(base_seed, world_preset), world_preset)
    spec = load_skill_for_world(skill_name, config, world)
    return SkillRun(
        config,
        spec,
        world_factory=factory,
        run_id=run_id,
        world_preset=world_preset,
        base_seed=base_seed,
        clock=clock,
        progress=progress,
        should_stop=should_stop,
    )


def skill_states(config: Mapping[str, Any]) -> dict[str, Any]:
    """Every registered skill, whether its spec can be built now, and the run manifests found.

    The spec is built with ``skills.load_skill(name, config)``, without a world; a skill whose
    module defines ``for_world`` gets its world arguments only in :func:`open_run`, so its
    ``buildable_without_world`` may be false while it builds with a world. Returns::

        {"config_digest": str,
         "skills": [{"skill", "buildable_without_world": bool, "reason": str | None,
                     "has_for_world": bool, "manifests": [summary]}],
         "other_manifests": [summary of manifests whose skill is not registered],
         "unreadable_manifests": [{"path", "error"}]}

    A manifest summary is {"run_id", "world_preset", "base_seed", "world", "path",
    "config_digest_matches_current", "steps": {step: state}, "evaluations": [evaluation ids]}.
    """
    digest = config_digest(config)
    found = find_manifests(config)
    names = available_skills()
    entries = []
    for name in names:
        entry: dict[str, Any] = {"skill": name}
        try:
            load_skill(name, config)
            entry.update(buildable_without_world=True, reason=None)
        except Exception as error:  # noqa: BLE001 - the reason is the answer
            entry.update(buildable_without_world=False, reason=f"{type(error).__name__}: {error}")
        entry["has_for_world"] = has_for_world(name)
        entry["manifests"] = [
            _manifest_summary(path, manifest, digest)
            for path, manifest, _ in found
            if manifest is not None and manifest["skill"] == name
        ]
        entries.append(entry)
    return {
        "config_digest": digest,
        "skills": entries,
        "other_manifests": [
            _manifest_summary(path, manifest, digest)
            for path, manifest, _ in found
            if manifest is not None and manifest["skill"] not in names
        ],
        "unreadable_manifests": [{"path": path.as_posix(), "error": error} for path, manifest, error in found if manifest is None],
    }


def tabpfn_status(config: Mapping[str, Any]) -> dict[str, Any]:
    """Account usage read live (free), the project budget and what remains of it.

    Returns {"reachable": bool, "usage": {...} (allow-listed counters and limits, no
    identifiers), "remaining_account_tokens", "project_token_budget",
    "tokens_quoted_for_billed_calls" (sum of quotes of dispatched requests in the call log),
    "project_tokens_remaining", "max_tokens_per_request", "call_log", "error" when the provider
    could not be reached, "ledger_error" when the call log could not be read}. Sends no
    prediction.
    """
    settings = TeacherSettings.from_config(config)
    status: dict[str, Any] = {
        "project_token_budget": settings.project_token_budget,
        "max_tokens_per_request": settings.max_tokens_per_request,
        "call_log": settings.call_log.as_posix(),
    }
    try:
        cache = RequestCache(settings.cache_dir)
        lock = CacheLock(cache.lock_path, timeout_s=settings.lock_timeout_s, poll_s=settings.lock_poll_s)
        log = CallLog(settings.call_log, clock=lambda: datetime.now(timezone.utc), lock=lock)
        quoted = log.tokens_quoted_for_billed_calls()
        status["tokens_quoted_for_billed_calls"] = quoted
        status["project_tokens_remaining"] = (
            None if settings.project_token_budget is None else settings.project_token_budget - quoted
        )
    except Exception as error:  # noqa: BLE001
        status["ledger_error"] = f"{type(error).__name__}: {error}"
    try:
        provider = TabPFNProvider(settings)
        provider.connect()
        usage = provider.usage()
        status.update(reachable=True, usage=usage)
        try:
            status["remaining_account_tokens"] = remaining_tokens(usage)
        except Exception as error:  # noqa: BLE001
            status["remaining_account_tokens"] = None
            status["remaining_account_tokens_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001
        status.update(reachable=False, error=f"{type(error).__name__}: {error}")
    return status


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def _print_event(event: Mapping[str, Any]) -> None:
    done = "-" if event.get("done") is None else event["done"]
    total = "-" if event.get("total") is None else event["total"]
    print(f"[{event['run_id']}] {event['step']:<8} {event['event']:<8} {done}/{total} {event['message']}", flush=True)


def _comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_situation(pairs: Sequence[str]) -> dict[str, Any]:
    situation: dict[str, Any] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name:
            raise ValueError(f"--situation takes name=value pairs; got {pair!r}")
        try:
            situation[name] = float(value)
        except ValueError:
            situation[name] = value
    return situation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.pipeline", description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="config.yaml to use (default: the repository's config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)

    def run_identity(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--skill", required=True, help="registered skill name")
        sub.add_argument("--preset", required=True, help="world preset (config.yaml world.presets)")
        sub.add_argument("--run-id", required=True, help="run id (letters, digits, '_' and '-')")
        sub.add_argument("--base-seed", type=int, default=None, help="base seed (default: config.yaml runtime.seed)")

    run = commands.add_parser("run", help="run steps of one skill run")
    run_identity(run)
    run.add_argument("--steps", required=True, help=f"comma list from {','.join(STEPS)}")
    run.add_argument("--method", default=None, help="residual method of the teacher and dream steps")
    run.add_argument("--attempts", type=int, default=None, help="context attempts (default: pipeline.default_attempts)")
    run.add_argument("--context", type=int, default=None, help="attempts the residual model uses (default: all)")
    run.add_argument("--dreams", type=int, default=None, help="dreams (default: engine.dreams_per_skill_initial)")
    run.add_argument("--split", default="dev", choices=OPEN_EVALUATION_SPLITS, help="evaluation split")
    run.add_argument("--eval-situations", type=int, default=None,
                     help="evaluation situations (default: evaluation.dev_situations_per_skill or "
                     "evaluation.shift_situations_per_preset)")
    run.add_argument("--eval-repeats", type=int, default=None, help="noise repeats (default: pipeline.evaluation_repeats)")
    run.add_argument("--eval-methods", default=None,
                     help="comma list of methods to evaluate (default: physics_only, --method, and snn_per_skill when "
                     "the run has a student)")

    status = commands.add_parser("status", help="registered skills and run manifests")
    status.add_argument("--tabpfn", action="store_true", help="also read the TabPFN account usage (free call)")

    cost = commands.add_parser("cost", help="cost preview of the dream step (no prediction is sent)")
    run_identity(cost)
    cost.add_argument("--method", required=True)
    cost.add_argument("--dreams", type=int, default=None, help="dreams (default: engine.dreams_per_skill_initial)")
    cost.add_argument("--context", type=int, default=None)

    decide_parser = commands.add_parser("decide", help="one decision for one situation")
    run_identity(decide_parser)
    decide_parser.add_argument("--brain", required=True)
    decide_parser.add_argument("--situation", nargs="+", required=True, help="name=value pairs")
    decide_parser.add_argument("--context", type=int, default=None)
    return parser


def _open_from_args(opener: Callable[..., SkillRun], config: Mapping[str, Any], args: argparse.Namespace,
                    should_stop: Callable[[], bool] | None) -> SkillRun:
    base_seed = args.base_seed if args.base_seed is not None else require(config, "runtime.seed")
    return opener(
        config,
        args.skill,
        run_id=args.run_id,
        world_preset=args.preset,
        base_seed=base_seed,
        clock=utc_now,
        progress=_print_event,
        should_stop=should_stop,
    )


def _summary(run: SkillRun) -> None:
    manifest = run.manifest
    print("summary:")
    print(f"  manifest  {manifest['paths']['manifest']}")
    for step in STEPS:
        entry = manifest["steps"][step]
        print(f"  {step:<9} {entry['state']}" + (f"  ({entry.get('reason')})" if entry.get("reason") else ""))
    for label, key in (("attempts", "attempts"), ("dreams", "dreams"), ("student", "student_npz")):
        path = Path(manifest["paths"][key])
        if path.exists():
            print(f"  {label:<9} {path.as_posix()}")
    if manifest["evaluations"]:
        last = manifest["evaluations"][-1]
        for method, path in (last.get("results") or {}).items():
            print(f"  result    {last['evaluation_id']} {method}: {path}")


def _run_command(run: SkillRun, config: Mapping[str, Any], args: argparse.Namespace, steps: list[str]) -> int:
    for step in STEPS:
        if step not in steps:
            continue
        try:
            if step == "collect":
                attempts = args.attempts if args.attempts is not None else require(config, "pipeline.default_attempts")
                run.collect_attempts(attempts)
            elif step == "teacher":
                run.residual_model(args.method, args.context)
            elif step == "dream":
                dreams = args.dreams if args.dreams is not None else require(config, "engine.dreams_per_skill_initial")
                run.dream(args.method, dreams, args.context)
            elif step == "train":
                run.train_student()
            elif step == "evaluate":
                if args.eval_methods:
                    methods = _comma_list(args.eval_methods)
                else:
                    methods = [PHYSICS_ONLY]
                    if args.method and args.method != PHYSICS_ONLY:
                        methods.append(args.method)
                    if run.paths.student_npz.exists():
                        methods.append(SNN_PER_SKILL)
                key = "evaluation.dev_situations_per_skill" if args.split == "dev" else "evaluation.shift_situations_per_preset"
                situations = args.eval_situations if args.eval_situations is not None else require(config, key)
                repeats = args.eval_repeats if args.eval_repeats is not None else require(config, "pipeline.evaluation_repeats")
                contexts = {method: args.context for method in methods if method in _LOCAL_RESIDUAL or method in BILLED_METHODS}
                run.evaluate(methods, split=args.split, n_situations=situations, n_repeats=repeats,
                             method_context=contexts if args.context is not None else None)
        except Stopped as stop:
            print(f"STOPPED {step}: {stop}", flush=True)
            _summary(run)
            return EXIT_STOPPED
        except StepRefused as refusal:
            print(f"REFUSED {step}: {refusal.reason}", flush=True)
            _summary(run)
            return EXIT_REFUSED
        except Exception as error:  # noqa: BLE001 - reported, exit code 1
            print(f"FAILED {step}: {type(error).__name__}: {error}", flush=True)
            _summary(run)
            return EXIT_FAILED
    _summary(run)
    return EXIT_OK


def main(argv: Sequence[str] | None = None, *, opener: Callable[..., SkillRun] | None = None) -> int:
    """Command-line entry point; returns the exit code (module docstring).

    ``opener`` builds the SkillRun from (config, skill_name, run_id=, world_preset=, base_seed=,
    clock=, progress=, should_stop=); the default is :func:`open_run` (the Go2 simulator).
    """
    parser = _parser()
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else load_config()
    opener = opener or open_run

    if args.command == "status":
        report: dict[str, Any] = {"skill_states": skill_states(config)}
        if args.tabpfn:
            report["tabpfn_status"] = tabpfn_status(config)
        print(json.dumps(_jsonable(report), indent=1, sort_keys=True))
        return EXIT_OK

    stop = {"requested": False}

    def request_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
        stop["requested"] = True
        print("stop requested: the current step ends before its next episode", flush=True)

    previous = None
    try:
        previous = signal.signal(signal.SIGINT, request_stop)
    except ValueError:  # not the main thread: Ctrl+C keeps its default behaviour
        previous = None
    try:
        if args.command == "run":
            steps = _comma_list(args.steps)
            unknown = [step for step in steps if step not in STEPS]
            if not steps or unknown:
                parser.error(f"--steps takes a comma list from {','.join(STEPS)}; unknown: {unknown}")
            if ("teacher" in steps or "dream" in steps) and not args.method:
                parser.error("--method is required for the teacher and dream steps")
            try:
                run = _open_from_args(opener, config, args, lambda: stop["requested"])
            except Exception as error:  # noqa: BLE001
                print(f"FAILED open: {type(error).__name__}: {error}", flush=True)
                return EXIT_FAILED
            return _run_command(run, config, args, steps)

        try:
            run = _open_from_args(opener, config, args, lambda: stop["requested"])
        except Exception as error:  # noqa: BLE001
            print(f"FAILED open: {type(error).__name__}: {error}", flush=True)
            return EXIT_FAILED
        if args.command == "cost":
            dreams = args.dreams if args.dreams is not None else require(config, "engine.dreams_per_skill_initial")
            try:
                preview = run.dream_cost_preview(args.method, dreams, args.context)
            except Exception as error:  # noqa: BLE001
                print(f"FAILED cost: {type(error).__name__}: {error}", flush=True)
                return EXIT_FAILED
            print(json.dumps(_jsonable(preview), indent=1, sort_keys=True))
            return EXIT_OK
        if args.command == "decide":
            try:
                answer = run.decide_with(args.brain, _parse_situation(args.situation), n_context=args.context)
            except StepRefused as refusal:
                print(f"REFUSED decide: {refusal.reason}", flush=True)
                return EXIT_REFUSED
            except Exception as error:  # noqa: BLE001
                print(f"FAILED decide: {type(error).__name__}: {error}", flush=True)
                return EXIT_FAILED
            print(json.dumps(answer, indent=1, sort_keys=True))
            return EXIT_OK
        parser.error(f"unknown command {args.command!r}")
        return EXIT_USAGE
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    raise SystemExit(main())
