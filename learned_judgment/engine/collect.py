"""Collecting actual attempts (engine lane, Person B).

Lane contract
-------------
:func:`exploration_actions` chooses the action of each context attempt: with probability
``exploration_fraction`` a uniformly random candidate of ``spec.candidate_actions``,
otherwise the physics-only decision. All randomness comes from ONE draw
``rng.random((n, 2))`` on the ``rng`` passed in: column 0 decides explore / physics and
column 1 picks the candidate, so row i's action depends only on row i's two numbers (the
first rows of a larger set equal a smaller set drawn from the same stream), and the number of
draws does not depend on which branch a row takes.

:func:`collect` runs one actual episode per (situation, action) on ONE world, the one Person
A's ``Go2World(config, seed, preset)`` built for the whole run (the world seed fixes the
hidden world). Each attempt calls ``spec.episode(world, situation, action, attempt_seed)``
with its own attempt seed (it seeds only that episode's noise), and the attempt seed is the
row's ``seed``. Before anything is stepped, the spec's command dimensions are checked against
the verified command schema and every situation, action, attempt seed and run field is
validated. Each real row is appended to a JSON Lines file as soon as it is measured, so a
crash loses nothing already measured. It is NOT RUN until Person A's Go2World exists; it must
never be called with a stand-in world.

Arrays: situations (n, S), actions (n, A), float64, in spec order; attempt seeds int (n,).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch
from learned_judgment.engine.features import FeatureLayout
from learned_judgment.engine.rows import AttemptRow, append_rows, build_real_row, check_new_attempts
from learned_judgment.skills.base import CommandSchema, SkillSpec, World, check_command_binding

__all__ = ["collect", "exploration_actions"]

_EXPLORE, _PICK = 0, 1  # columns of the (n, 2) uniform draw of exploration_actions


def exploration_actions(
    spec: SkillSpec,
    situations: np.ndarray,
    *,
    rng: np.random.Generator,
    points_per_dim: Sequence[int],
    exploration_fraction: float,
    quantile_levels: Sequence[float],
    spread_levels: tuple[float, float],
    layout: FeatureLayout,
) -> np.ndarray:
    """Actions (n, A) for the context attempts in ``situations`` (n, S).

    One draw ``u = rng.random((n, 2))``. Row i explores when ``u[i, 0] < exploration_fraction``
    and then takes candidate ``floor(u[i, 1] * C_i)`` of its grid of C_i candidates; otherwise
    it takes the physics-only decision (:func:`decide_batch` with ``model=None``). Row i
    therefore does not depend on n, and the stream position after the call depends only on n.
    ``exploration_fraction`` is a probability in [0, 1] read from config.yaml; 0.0 gives
    exactly the physics-only actions.
    """
    if isinstance(exploration_fraction, bool) or not isinstance(exploration_fraction, (int, float, np.floating)):
        raise ValueError(f"exploration_fraction must be a number; got {exploration_fraction!r}")
    fraction = float(exploration_fraction)
    if math.isnan(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"exploration_fraction is a probability and must lie in [0, 1]; got {fraction}")
    decisions = decide_batch(
        spec,
        situations,
        model=None,
        layout=layout,
        points_per_dim=points_per_dim,
        quantile_levels=quantile_levels,
        spread_levels=spread_levels,
        method=PHYSICS_ONLY,
    )
    n = len(decisions)
    actions = np.empty((n, spec.n_action), dtype=np.float64)
    if n == 0:
        return actions
    counts = np.array([decision.candidates.shape[0] for decision in decisions], dtype=np.int64)
    uniform = rng.random((n, 2))
    explore = uniform[:, _EXPLORE] < fraction
    # floor(u * C) < C for u in [0, 1); the minimum only guards the last bit of rounding.
    picks = np.minimum(np.floor(uniform[:, _PICK] * counts).astype(np.int64), counts - 1)
    for row, decision in enumerate(decisions):
        actions[row] = decision.candidates[picks[row]] if explore[row] else decision.physics_action
    return actions


def _attempt_seeds(attempt_seeds: Any, n: int) -> list[int]:
    """Attempt seeds as Python ints: an integer array of shape (n,), each >= 0, all distinct."""
    array = np.asarray(attempt_seeds)
    if array.shape != (n,) or (array.size and not np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"attempt_seeds must be an integer array of shape ({n},); got {array.dtype} {array.shape}")
    return [int(seed) for seed in array]


def collect(
    spec: SkillSpec,
    world: World,
    situations: np.ndarray,
    actions: np.ndarray,
    *,
    attempt_seeds: np.ndarray,
    command_schema: CommandSchema,
    run_id: str,
    world_preset: str,
    out_path: Path,
    clock: Callable[[], str],
) -> list[AttemptRow]:
    """Run one actual episode per row in ``world`` and record it.

    Args:
        world:          Person A's Go2World built by the caller for ``world_preset`` from the
                        run's world seed; every attempt of the run uses this one world.
        situations:     (n, S) encoded situations.
        actions:        (n, A) encoded actions; every row must be inside the spec bounds.
        attempt_seeds:  int (n,) distinct seeds >= 0; attempt i calls
                        ``spec.episode(world, situation_i, action_i, attempt_seeds[i])`` and its
                        row's ``seed`` is ``attempt_seeds[i]``.
        command_schema: the verified command schema (``CommandSchema.from_config``); the spec's
                        command dimensions must lie inside it (:func:`check_command_binding`).
        run_id:         run identifier stamped on every row; ``(run_id, attempt seed)`` must not
                        already be in ``out_path``.
        out_path:       JSON Lines attempt file; each row is appended right after its episode.
        clock:          returns the current time as an ISO 8601 UTC string; called once before
                        the first episode (to validate its format) and once per row after the
                        episode has finished.

    Everything is checked before the first episode -- command binding, every situation and
    action, the attempt seeds, run_id, world_preset, the clock's format and the existing
    attempt file -- so an invalid input stops the run before the world is stepped.
    Returns the rows in input order.
    """
    check_command_binding(spec, command_schema)
    situations = np.asarray(situations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if situations.ndim != 2 or situations.shape[1] != spec.n_situation:
        raise ValueError(f"situations must have shape (n, {spec.n_situation}); got {situations.shape}")
    n = situations.shape[0]
    if actions.shape != (n, spec.n_action):
        raise ValueError(f"actions must have shape ({n}, {spec.n_action}); got {actions.shape}")
    outside = np.flatnonzero(~spec.action_in_bounds(actions)) if n else np.empty(0, dtype=np.int64)
    if outside.size:
        raise ValueError(f"actions at rows {outside.tolist()} are outside the bounds of skill {spec.name!r}")
    for row, situation in enumerate(situations):
        try:
            spec.encode_situation(spec.decode_situation(situation))
        except ValueError as error:
            raise ValueError(f"situation at row {row} is invalid: {error}") from error
    seeds = _attempt_seeds(attempt_seeds, n)
    out_path = Path(out_path)
    check_new_attempts(out_path, run_id=run_id, world_preset=world_preset, seeds=seeds, timestamp_utc=clock())

    rows: list[AttemptRow] = []
    for situation, action, seed in zip(situations, actions, seeds):
        measured = spec.episode(world, spec.decode_situation(situation), spec.decode_action(action), seed)
        row = build_real_row(
            spec,
            run_id=run_id,
            seed=seed,
            world_preset=world_preset,
            situation=situation,
            action=action,
            measured_outcome=measured,
            timestamp_utc=clock(),
        )
        append_rows(out_path, [row], spec=spec)
        rows.append(row)
    return rows
