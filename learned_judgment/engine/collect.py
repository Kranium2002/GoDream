"""Collecting actual attempts (engine lane, Person B).

Lane contract
-------------
:func:`exploration_actions` chooses the action of each context attempt: with probability
``exploration_fraction`` a uniformly random candidate of ``spec.candidate_actions``,
otherwise the physics-only decision. All randomness comes from the ``rng`` passed in, and
the number of draws does not depend on which branch a row takes.

:func:`collect` runs one actual episode per (situation, action) through
``spec.episode(world, ...)`` on Person A's Go2World and appends each real row to a JSON
Lines file as soon as it is measured, so a crash loses nothing already measured. It is
NOT RUN until Person A's Go2World exists; it must never be called with a stand-in world.

Arrays: situations (n, S), actions (n, A), float64, in spec order.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch
from learned_judgment.engine.features import FeatureLayout
from learned_judgment.engine.rows import AttemptRow, append_rows, build_real_row
from learned_judgment.skills.base import SkillSpec, World

__all__ = ["collect", "exploration_actions"]


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

    Row i explores when ``rng.random() < exploration_fraction`` and then takes candidate
    ``rng.integers(0, C_i)`` of its grid; otherwise it takes the physics-only decision
    (:func:`decide_batch` with ``model=None``). Both draws are made for every row, so the
    stream position after the call depends only on n. ``exploration_fraction`` is a
    probability in [0, 1] read from config.yaml; 0.0 gives exactly the physics-only actions.
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
    explore = rng.random(n) < fraction
    picks = rng.integers(0, counts)
    for row, decision in enumerate(decisions):
        actions[row] = decision.candidates[picks[row]] if explore[row] else decision.physics_action
    return actions


def collect(
    spec: SkillSpec,
    world: World,
    situations: np.ndarray,
    actions: np.ndarray,
    *,
    run_id: str,
    seed: int,
    world_preset: str,
    out_path: Path,
    clock: Callable[[], str],
) -> list[AttemptRow]:
    """Run one actual episode per row and record it.

    Args:
        world:        Person A's Go2World built for ``world_preset`` and ``seed``.
        situations:   (n, S) encoded situations.
        actions:      (n, A) encoded actions; every row must be inside the spec bounds.
        out_path:     JSON Lines attempt file; each row is appended right after its episode.
        clock:        returns the current time as an ISO 8601 UTC string; called once per
                      row after the episode has finished.

    Every situation and action is checked before the first episode, so an invalid row
    stops the run before the world is stepped. Returns the rows in input order.
    """
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

    out_path = Path(out_path)
    rows: list[AttemptRow] = []
    for situation, action in zip(situations, actions):
        measured = spec.episode(world, spec.decode_situation(situation), spec.decode_action(action))
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
        append_rows(out_path, [row])
        rows.append(row)
    return rows
