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

:func:`measure_latency` times a decision function with ``time.perf_counter_ns``.

Never call these with a stand-in world: the only valid ``world_factory`` builds Person A's
Go2World. Episodes run sequentially in one process (single writer per attempt file);
parallel episodes are out of scope until seed parity is verified on the real simulator.

Arrays: situations (n, S), actions (n, A), measured (n, K) / (n, R, K), losses (n, R),
noise seeds int (n,) / (n, R); float64 unless stated.
"""

from __future__ import annotations

import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.engine.rows import AttemptRow, append_rows, build_real_row, rows_to_arrays
from learned_judgment.skills.base import SkillSpec, World

__all__ = ["MethodEvaluation", "compare_methods", "evaluate_actions", "measure_latency"]

_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
# Definition of "p95" (03_METRICS.md) and its estimator (numpy type 7).
_P95 = 0.95
_QUANTILE_METHOD = "linear"


def _check_name(label: str, value: Any) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ValueError(f"{label} must match {_NAME.pattern} (it is used in file names); got {value!r}")
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


def evaluate_actions(
    spec: SkillSpec,
    world_factory: Callable[[int], World],
    situations: np.ndarray,
    actions: np.ndarray,
    *,
    noise_seeds: np.ndarray,
    run_id: str,
    world_preset: str,
    out_path: str | Path,
    clock: Callable[[], str],
) -> np.ndarray:
    """Run one actual episode per row and return the measured outcomes (n, K).

    Args:
        world_factory: ``seed -> Go2World`` for ``world_preset``; called once per episode with
                       that episode's noise seed, so every episode starts from a fresh world.
        situations:    (n, S) encoded situations.
        actions:       (n, A) encoded actions, all inside the spec's declared bounds.
        noise_seeds:   int (n,) per-episode seeds (identical for every method being compared).
        out_path:      JSON Lines attempt file (append-only, single writer).
        clock:         current time as an ISO 8601 string with a zero UTC offset; called once
                       per episode after it finished.

    Every input is validated before the first world is built. The returned array is read
    back through ``rows_to_arrays`` from the rows that were written.
    """
    situations = _check_situations(spec, situations)
    n = situations.shape[0]
    actions = _check_actions(spec, actions, n, "actions")
    seeds = _check_seeds(noise_seeds, (n,))
    if not isinstance(run_id, str) or not run_id or not isinstance(world_preset, str) or not world_preset:
        raise ValueError("run_id and world_preset must be non-empty strings")
    path = Path(out_path)

    rows: list[AttemptRow] = []
    for situation, action, seed in zip(situations, actions, seeds):
        world = world_factory(int(seed))
        measured_map = spec.episode(world, spec.decode_situation(situation), spec.decode_action(action))
        row = build_real_row(
            spec,
            run_id=run_id,
            seed=int(seed),
            world_preset=world_preset,
            situation=situation,
            action=action,
            measured_outcome=measured_map,
            timestamp_utc=clock(),
        )
        append_rows(path, [row])
        rows.append(row)
    return rows_to_arrays(spec, rows).measured


@dataclass(frozen=True, eq=False)
class MethodEvaluation:
    """Actual-episode results of one method on the shared situations and noise seeds.

    Attributes:
        method:    method label (a key of ``policies``).
        actions:   (n, A) actions the method chose, one per situation.
        measured:  (n, R, K) measured outcomes, repeat r used noise_seeds[:, r].
        losses:    (n, R) ``spec.loss`` of each episode (input of the paired bootstrap).
        rows_path: attempt file holding this method's real rows.
    """

    method: str
    actions: np.ndarray
    measured: np.ndarray
    losses: np.ndarray
    rows_path: str


def compare_methods(
    spec: SkillSpec,
    policies: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    *,
    world_factory: Callable[[int], World],
    situations: np.ndarray,
    noise_seeds: np.ndarray,
    run_id: str,
    world_preset: str,
    out_dir: str | Path,
    clock: Callable[[], str],
) -> dict[str, MethodEvaluation]:
    """Evaluate every method on the SAME situations (n, S) and noise seeds int (n, R).

    Each policy maps all situations (n, S) to actions (n, A) in one call (so a batched
    teacher makes one request). Every method's actions are validated before the first
    episode of any method. Rows of method m go to ``out_dir / f"{run_id}__{m}.jsonl"`` with
    row run_id ``f"{run_id}/{m}"``. Returns {method: MethodEvaluation} in ``policies`` order.
    """
    run_id = _check_name("run_id", run_id)
    if not policies:
        raise ValueError("policies must name at least one method")
    situations = _check_situations(spec, situations)
    n = situations.shape[0]
    seeds = np.asarray(noise_seeds)
    if seeds.ndim != 2 or seeds.shape[0] != n or seeds.shape[1] < 1:
        raise ValueError(f"noise_seeds must have shape ({n}, R >= 1); got {seeds.shape}")
    seeds = _check_seeds(seeds, seeds.shape)
    n_repeats = seeds.shape[1]

    chosen: dict[str, np.ndarray] = {}
    for method, policy in policies.items():
        _check_name("method", method)
        chosen[method] = _check_actions(spec, policy(situations.copy()), n, f"actions of method {method!r}")

    directory = Path(out_dir)
    results: dict[str, MethodEvaluation] = {}
    for method, actions in chosen.items():
        path = directory / f"{run_id}__{method}.jsonl"
        measured = np.empty((n, n_repeats, spec.n_outcome))
        for repeat in range(n_repeats):
            measured[:, repeat, :] = evaluate_actions(
                spec,
                world_factory,
                situations,
                actions,
                noise_seeds=seeds[:, repeat],
                run_id=f"{run_id}/{method}",
                world_preset=world_preset,
                out_path=path,
                clock=clock,
            )
        losses = np.asarray(spec.loss(measured.reshape(n * n_repeats, spec.n_outcome)), dtype=np.float64)
        if losses.shape != (n * n_repeats,) or not np.all(np.isfinite(losses)):
            raise ValueError(f"{spec.name}.loss must return finite ({n * n_repeats},); got {losses.shape}")
        results[method] = MethodEvaluation(
            method=method,
            actions=actions,
            measured=measured,
            losses=losses.reshape(n, n_repeats),
            rows_path=str(path),
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
