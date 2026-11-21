"""Common real-run evaluation harness (written now; NOT RUN until Person A's Go2World exists).

Lane contract (Person B, evaluation)
------------------------------------
:func:`evaluate_actions` runs ONE ACTUAL episode per (situation, action) through
``spec.episode(world, situation, action)`` on a fresh world from ``world_factory(seed)``,
where ``seed`` is the per-episode noise seed shared by every method
(:func:`learned_judgment.eval.splits.noise_seed`). Each measured episode is appended at once
to a JSON Lines attempt file as a ``source="real"``
:class:`~learned_judgment.engine.rows.AttemptRow` (canonical line from
``AttemptRow.to_json_line``; the row's ``seed`` is the seed the world was built with), so a
crash loses nothing already measured. The measured outcome comes only from
``spec.episode``; nothing here derives an outcome from the intended action.

:func:`compare_methods` asks every method for its actions on the SAME situations, then runs
every method on the SAME noise seeds, and returns per-method per-episode losses
(``spec.loss``) shaped for :mod:`learned_judgment.eval.bootstrap`.

Run files (review finding: a rerun appended to the old attempt file)
    A run never touches an existing file: it refuses to start (FileExistsError) when its
    attempt file or its manifest already exists. Before the first episode it writes a
    manifest next to the attempt file (:func:`manifest_path`, ``<stem>.manifest.json``)::

        format          "learned_judgment.run_manifest/v1"
        run_id          the run id given to the call; row_run_id: the run_id stamped on rows
        method, skill, split, world_preset, base_seed, config_digest (sha256 hex of config.yaml)
        n_situations, n_repeats
        episodes        ordered [situation_index, repeat, seed] -- row k of the attempt file
                        is episode k (rows themselves carry no split, repeat or index)
        attempt_file    file name of the attempt file, in the manifest's directory
        started_utc     clock() before the first episode
        completed       false until the last episode is written and read back, then true
        finished_utc    null, then clock() after the last episode

    After the last episode the attempt file is read back and checked against the planned
    episodes (count, order of seeds, row run id), and only then is the manifest rewritten
    with ``completed: true``. A crashed run keeps ``completed: false``; run-result files
    (:mod:`learned_judgment.eval.report`) may only be written from completed manifests.

:func:`measure_latency` times a decision function with ``time.perf_counter_ns``.

Never call these with a stand-in world: the only valid ``world_factory`` builds Person A's
Go2World. Episodes run sequentially in one process (single writer per attempt file);
parallel episodes are out of scope until seed parity is verified on the real simulator.

Arrays: situations (n, S), actions (n, A), measured (n, K) / (n, R, K), losses (n, R),
noise seeds int (n,) / (n, R); float64 unless stated.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.engine.rows import AttemptRow, append_rows, build_real_row, read_rows, rows_to_arrays
from learned_judgment.eval.splits import SPLITS
from learned_judgment.fileio import atomic_write_text
from learned_judgment.skills.base import SkillSpec, World

__all__ = [
    "MANIFEST_FORMAT",
    "MethodEvaluation",
    "compare_methods",
    "evaluate_actions",
    "manifest_path",
    "measure_latency",
    "read_manifest",
]

MANIFEST_FORMAT = "learned_judgment.run_manifest/v1"
_ATTEMPT_SUFFIX = ".jsonl"
_MANIFEST_SUFFIX = ".manifest.json"
_MANIFEST_KEYS = frozenset(
    {
        "format",
        "run_id",
        "row_run_id",
        "method",
        "skill",
        "split",
        "world_preset",
        "base_seed",
        "config_digest",
        "n_situations",
        "n_repeats",
        "episodes",
        "attempt_file",
        "started_utc",
        "completed",
        "finished_utc",
    }
)
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# Definition of "p95" (03_METRICS.md) and its estimator (numpy type 7).
_P95 = 0.95
_QUANTILE_METHOD = "linear"


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def _check_name(label: str, value: Any) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ValueError(f"{label} must match {_NAME.pattern} (it is used in file names); got {value!r}")
    return value


def _check_text(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string; got {value!r}")
    return value


def _check_int(label: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _check_split(value: Any) -> str:
    if not isinstance(value, str) or value not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {value!r}")
    return value


def _check_digest(value: Any) -> str:
    if not isinstance(value, str) or not _DIGEST.match(value):
        raise ValueError(f"config_digest must be a lowercase sha256 hex digest (learned_judgment.config.config_digest); got {value!r}")
    return value


def _check_utc(label: str, value: Any) -> str:
    try:
        moment = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an ISO 8601 string with a zero UTC offset; got {value!r}")
    return value


def _check_situations(spec: SkillSpec, situations: Any) -> np.ndarray:
    array = np.asarray(situations, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != spec.n_situation or array.shape[0] < 1:
        raise ValueError(f"situations must have shape (n >= 1, S={spec.n_situation}); got {array.shape}")
    for row, situation in enumerate(array):
        try:
            spec.encode_situation(spec.decode_situation(situation))
        except ValueError as error:
            raise ValueError(f"situation at row {row} is invalid: {error}") from error
    return array


def _check_actions(spec: SkillSpec, actions: Any, n: int, label: str) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.shape != (n, spec.n_action):
        raise ValueError(f"{label} must have shape ({n}, {spec.n_action}); got {array.shape}")
    outside = np.flatnonzero(~spec.action_in_bounds(array))
    if outside.size:
        raise ValueError(f"{label}: rows {outside.tolist()} are outside the bounds of skill {spec.name!r}")
    return array


def _check_seeds(seeds: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(seeds)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"noise_seeds must be an integer array of shape {shape}; got {array.dtype} {array.shape}")
    if np.any(array < 0):
        raise ValueError("noise_seeds must be non-negative")
    return array.astype(np.int64)


def _check_attempt_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    if path.suffix != _ATTEMPT_SUFFIX or path.name.endswith(_MANIFEST_SUFFIX):
        raise ValueError(f"the attempt file must be a {_ATTEMPT_SUFFIX} file; got {path.name!r}")
    return path


def _refuse_existing(path: Path) -> None:
    for target in (path, manifest_path(path)):
        if target.exists():
            raise FileExistsError(
                f"{target} already exists: a run never appends to or replaces an earlier run's files; "
                "use a new run_id or out_dir (inspect the old manifest's 'completed' flag first)"
            )


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def manifest_path(attempt_path: str | Path) -> Path:
    """Manifest next to an attempt file: ``<dir>/<stem>.manifest.json`` for ``<dir>/<stem>.jsonl``."""
    path = _check_attempt_path(attempt_path)
    return path.with_name(path.stem + _MANIFEST_SUFFIX)


def _write_manifest(path: Path, record: Mapping[str, Any]) -> None:
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    atomic_write_text(path, text + "\n")


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate key {key!r}")
        record[key] = value
    return record


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a run manifest written by this module; returns its fields as a dict.

    Raises ValueError for anything that is not a well-formed manifest (format tag, exact key
    set, types, episode list of n_situations x n_repeats entries, completed/finished_utc
    consistency). It does not open the attempt file.
    """
    path = Path(path)
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not a run manifest: {error}") from error
    if not isinstance(record, dict) or record.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not a run manifest (format tag {MANIFEST_FORMAT!r} missing)")
    if set(record) != _MANIFEST_KEYS:
        raise ValueError(
            f"{path}: manifest keys do not match; missing={sorted(_MANIFEST_KEYS - set(record))} "
            f"extra={sorted(set(record) - _MANIFEST_KEYS)}"
        )
    where = str(path)
    for key in ("run_id", "row_run_id", "skill", "world_preset", "attempt_file"):
        _check_text(f"{where}: {key}", record[key])
    _check_name(f"{where}: method", record["method"])
    _check_split(record["split"])
    _check_int(f"{where}: base_seed", record["base_seed"], 0)
    _check_digest(record["config_digest"])
    n_situations = _check_int(f"{where}: n_situations", record["n_situations"], 1)
    n_repeats = _check_int(f"{where}: n_repeats", record["n_repeats"], 1)
    episodes = record["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{where}: episodes must be a non-empty list")
    for position, episode in enumerate(episodes):
        if not isinstance(episode, list) or len(episode) != 3:
            raise ValueError(f"{where}: episodes[{position}] must be [situation_index, repeat, seed]")
        for label, value in zip(("situation_index", "repeat", "seed"), episode):
            _check_int(f"{where}: episodes[{position}].{label}", value, 0)
        if episode[0] >= n_situations:
            raise ValueError(f"{where}: episodes[{position}] situation index {episode[0]} >= n_situations")
    if len(episodes) != n_situations * n_repeats or len({(e[0], e[1]) for e in episodes}) != len(episodes):
        raise ValueError(f"{where}: episodes must list each (situation, repeat) once: {n_situations} x {n_repeats}")
    if Path(record["attempt_file"]).name != record["attempt_file"]:
        raise ValueError(f"{where}: attempt_file must be a bare file name in the manifest's directory")
    _check_utc(f"{where}: started_utc", record["started_utc"])
    if not isinstance(record["completed"], bool):
        raise ValueError(f"{where}: completed must be true or false")
    if record["completed"]:
        _check_utc(f"{where}: finished_utc", record["finished_utc"])
    elif record["finished_utc"] is not None:
        raise ValueError(f"{where}: an incomplete run has finished_utc null")
    return record


# --------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------


def _run_plan(
    spec: SkillSpec,
    world_factory: Callable[[int], World],
    situations: np.ndarray,
    actions: np.ndarray,
    plan: Sequence[tuple[int, int, int]],
    *,
    path: Path,
    manifest: dict[str, Any],
    clock: Callable[[], str],
) -> np.ndarray:
    """Write the manifest, run every planned episode in order, verify, complete the manifest.

    plan: ordered (situation_index, repeat, seed); returns measured (len(plan), K) read back
    from the attempt file in plan order.
    """
    _refuse_existing(path)
    record = dict(manifest)
    record.update(
        format=MANIFEST_FORMAT,
        episodes=[[int(index), int(repeat), int(seed)] for index, repeat, seed in plan],
        attempt_file=path.name,
        started_utc=_check_utc("clock()", clock()),
        completed=False,
        finished_utc=None,
    )
    manifest_file = manifest_path(path)
    _write_manifest(manifest_file, record)
    row_run_id = record["row_run_id"]
    for index, _, seed in plan:
        world = world_factory(int(seed))
        measured_map = spec.episode(
            world, spec.decode_situation(situations[index]), spec.decode_action(actions[index])
        )
        row = build_real_row(
            spec,
            run_id=row_run_id,
            seed=int(seed),
            world_preset=record["world_preset"],
            situation=situations[index],
            action=actions[index],
            measured_outcome=measured_map,
            timestamp_utc=clock(),
        )
        append_rows(path, [row])
    rows: list[AttemptRow] = read_rows(path, spec=spec)
    expected_seeds = [int(seed) for _, _, seed in plan]
    if (
        len(rows) != len(plan)
        or [row.seed for row in rows] != expected_seeds
        or any(row.run_id != row_run_id for row in rows)
    ):
        raise RuntimeError(f"{path} does not hold exactly the planned episodes; the manifest stays incomplete")
    record.update(completed=True, finished_utc=_check_utc("clock()", clock()))
    _write_manifest(manifest_file, record)
    return rows_to_arrays(spec, rows).measured


def evaluate_actions(
    spec: SkillSpec,
    world_factory: Callable[[int], World],
    situations: np.ndarray,
    actions: np.ndarray,
    *,
    noise_seeds: np.ndarray,
    run_id: str,
    method: str,
    world_preset: str,
    split: str,
    base_seed: int,
    repeat: int,
    config_digest: str,
    out_path: str | Path,
    clock: Callable[[], str],
) -> np.ndarray:
    """Run one actual episode per row and return the measured outcomes (n, K).

    Args:
        world_factory: ``seed -> Go2World`` for ``world_preset``; called once per episode with
                       that episode's noise seed, so every episode starts from a fresh world.
        situations:    (n, S) encoded situations; row i is situation index i of ``split`` (the
                       index its noise seed was drawn for).
        actions:       (n, A) encoded actions, all inside the spec's declared bounds.
        noise_seeds:   int (n,) per-episode seeds (identical for every method being compared).
        run_id:        run id stamped on every row and recorded in the manifest.
        method:        method label recorded in the manifest.
        split:         one of ``splits.SPLITS``; base_seed: the split's base seed; repeat: the
                       noise repeat these seeds belong to (all recorded in the manifest).
        config_digest: ``learned_judgment.config.config_digest`` of the config the run used.
        out_path:      new JSON Lines attempt file (``.jsonl``); refused if it or its manifest
                       already exists.
        clock:         current time as an ISO 8601 string with a zero UTC offset; called for
                       the manifest start, after each episode, and for the manifest finish.

    Every input is validated before the manifest is written and before the first world is
    built. The returned array is read back from the attempt file through ``rows_to_arrays``.
    """
    situations = _check_situations(spec, situations)
    n = situations.shape[0]
    actions = _check_actions(spec, actions, n, "actions")
    seeds = _check_seeds(noise_seeds, (n,))
    repeat = _check_int("repeat", repeat, 0)
    manifest = {
        "run_id": _check_text("run_id", run_id),
        "row_run_id": run_id,
        "method": _check_name("method", method),
        "skill": spec.name,
        "split": _check_split(split),
        "world_preset": _check_text("world_preset", world_preset),
        "base_seed": _check_int("base_seed", base_seed, 0),
        "config_digest": _check_digest(config_digest),
        "n_situations": n,
        "n_repeats": 1,
    }
    path = _check_attempt_path(out_path)
    _refuse_existing(path)
    plan = [(index, repeat, int(seeds[index])) for index in range(n)]
    return _run_plan(spec, world_factory, situations, actions, plan, path=path, manifest=manifest, clock=clock)


@dataclass(frozen=True, eq=False)
class MethodEvaluation:
    """Actual-episode results of one method on the shared situations and noise seeds.

    Attributes:
        method:        method label (a key of ``policies``).
        actions:       (n, A) actions the method chose, one per situation.
        measured:      (n, R, K) measured outcomes, repeat r used noise_seeds[:, r].
        losses:        (n, R) ``spec.loss`` of each episode (input of the paired bootstrap).
        rows_path:     attempt file holding this method's real rows.
        manifest_path: the run manifest of that attempt file (completed when returned).
    """

    method: str
    actions: np.ndarray
    measured: np.ndarray
    losses: np.ndarray
    rows_path: str
    manifest_path: str


def compare_methods(
    spec: SkillSpec,
    policies: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    *,
    world_factory: Callable[[int], World],
    situations: np.ndarray,
    noise_seeds: np.ndarray,
    run_id: str,
    world_preset: str,
    split: str,
    base_seed: int,
    config_digest: str,
    out_dir: str | Path,
    clock: Callable[[], str],
) -> dict[str, MethodEvaluation]:
    """Evaluate every method on the SAME situations (n, S) and noise seeds int (n, R).

    Rows of method m go to ``out_dir / f"{run_id}__{m}.jsonl"`` with row run_id
    ``f"{run_id}/{m}"``, and its manifest to ``out_dir / f"{run_id}__{m}.manifest.json"``.
    Before any policy is called, every target attempt file and manifest is checked: if one
    exists the call raises FileExistsError and nothing runs. Each policy then maps all
    situations (n, S) to actions (n, A) in one call (so a batched teacher makes one request);
    every method's actions are validated before the first episode of any method. Episodes of
    a method run repeat by repeat (all situations of repeat 0, then repeat 1, ...), which is
    the row order of its attempt file and the episode order of its manifest.
    Returns {method: MethodEvaluation} in ``policies`` order.
    """
    run_id = _check_name("run_id", run_id)
    if not policies:
        raise ValueError("policies must name at least one method")
    for method in policies:
        _check_name("method", method)
    situations = _check_situations(spec, situations)
    n = situations.shape[0]
    seeds = np.asarray(noise_seeds)
    if seeds.ndim != 2 or seeds.shape[0] != n or seeds.shape[1] < 1:
        raise ValueError(f"noise_seeds must have shape ({n}, R >= 1); got {seeds.shape}")
    seeds = _check_seeds(seeds, seeds.shape)
    n_repeats = seeds.shape[1]
    common = {
        "skill": spec.name,
        "split": _check_split(split),
        "world_preset": _check_text("world_preset", world_preset),
        "base_seed": _check_int("base_seed", base_seed, 0),
        "config_digest": _check_digest(config_digest),
        "n_situations": n,
        "n_repeats": n_repeats,
    }
    directory = Path(out_dir)
    paths = {method: directory / f"{run_id}__{method}{_ATTEMPT_SUFFIX}" for method in policies}
    for path in paths.values():
        _refuse_existing(path)

    chosen: dict[str, np.ndarray] = {}
    for method, policy in policies.items():
        chosen[method] = _check_actions(spec, policy(situations.copy()), n, f"actions of method {method!r}")

    plan = [(index, repeat, int(seeds[index, repeat])) for repeat in range(n_repeats) for index in range(n)]
    results: dict[str, MethodEvaluation] = {}
    for method, actions in chosen.items():
        path = paths[method]
        manifest = {**common, "run_id": run_id, "row_run_id": f"{run_id}/{method}", "method": method}
        flat = _run_plan(spec, world_factory, situations, actions, plan, path=path, manifest=manifest, clock=clock)
        measured = np.ascontiguousarray(flat.reshape(n_repeats, n, spec.n_outcome).transpose(1, 0, 2))
        losses = np.asarray(spec.loss(measured.reshape(n * n_repeats, spec.n_outcome)), dtype=np.float64)
        if losses.shape != (n * n_repeats,) or not np.all(np.isfinite(losses)):
            raise ValueError(f"{spec.name}.loss must return finite ({n * n_repeats},); got {losses.shape}")
        results[method] = MethodEvaluation(
            method=method,
            actions=actions,
            measured=measured,
            losses=losses.reshape(n, n_repeats),
            rows_path=str(path),
            manifest_path=str(manifest_path(path)),
        )
    return results


def measure_latency(decide_fn: Callable[[Any], Any], inputs: Sequence[Any], n_decisions: int) -> dict[str, Any]:
    """Wall-clock latency of ``decide_fn`` over ``n_decisions`` calls.

    Call i uses ``inputs[i % len(inputs)]``; each call is timed alone with
    ``time.perf_counter_ns``. Returns {"median_s", "p95_s", "n_decisions", "n_inputs",
    "timer", "platform", "processor", "cpu_count"}: timings of THIS machine only. The caller
    labels what ``decide_fn`` includes (network, cache hits, model forward pass).
    """
    if isinstance(n_decisions, bool) or not isinstance(n_decisions, (int, np.integer)) or n_decisions < 1:
        raise ValueError(f"n_decisions must be a positive integer; got {n_decisions!r}")
    if len(inputs) < 1:
        raise ValueError("inputs must hold at least one decision input")
    elapsed = np.empty(int(n_decisions), dtype=np.int64)
    for call in range(int(n_decisions)):
        argument = inputs[call % len(inputs)]
        start = time.perf_counter_ns()
        decide_fn(argument)
        elapsed[call] = time.perf_counter_ns() - start
    seconds = elapsed.astype(np.float64) / 1e9
    return {
        "median_s": float(np.median(seconds)),
        "p95_s": float(np.quantile(seconds, _P95, method=_QUANTILE_METHOD)),
        "n_decisions": int(n_decisions),
        "n_inputs": len(inputs),
        "timer": "time.perf_counter_ns",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
