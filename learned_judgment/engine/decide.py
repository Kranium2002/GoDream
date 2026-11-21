"""Generic decision step: score bounded candidate commands and choose the best.

    outcome distribution = physics guess + predicted residual distribution
    score                = spec.objective(outcome distribution)
    decision             = argmax score over the candidate grid

Nothing here knows a skill. All situations of a batch and all their candidates go to the
residual model in ONE call, so a teacher backed by a billed provider makes one request per
batch (one decision, a whole dev set, or thousands of dreams).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from learned_judgment.engine.features import FeatureLayout, build_features
from learned_judgment.engine.types import Decision, OutcomeDistribution, ResidualDistribution, ResidualModel
from learned_judgment.skills.base import SkillSpec

__all__ = ["PHYSICS_ONLY", "decide", "decide_batch"]

PHYSICS_ONLY = "physics_only"


def decide_batch(
    spec: SkillSpec,
    situations: np.ndarray,
    *,
    model: ResidualModel | None,
    layout: FeatureLayout,
    points_per_dim: Sequence[int],
    quantile_levels: Sequence[float],
    spread_levels: tuple[float, float],
    method: str,
    extra: np.ndarray | None = None,
) -> list[Decision]:
    """Decide for every row of ``situations`` (n, S) with one residual-model call.

    Args:
        model:           a fitted :class:`ResidualModel`, or ``None`` for physics only (zero
                         residual by definition of that method; ``method`` must then be
                         ``PHYSICS_ONLY``).
        layout:          feature layout the model was fitted with.
        points_per_dim:  candidate grid resolution per action dimension (config.yaml).
        quantile_levels: predictive quantile levels (config.yaml).
        spread_levels:   (low, high) members of ``quantile_levels`` defining the spread.
        method:          label stored on each decision.
        extra:           (n, E) observed context per situation when the layout has an extra block.

    Returns:
        One :class:`Decision` per situation. Each lists every scored candidate, the physics
        action among them, and the predicted outcome distribution per candidate.
    """
    situations = np.asarray(situations, dtype=np.float64)
    if situations.ndim != 2 or situations.shape[1] != spec.n_situation:
        raise ValueError(f"situations must have shape (n, {spec.n_situation}); got {situations.shape}")
    if (model is None) != (method == PHYSICS_ONLY):
        raise ValueError(f"method {method!r} and model {'None' if model is None else 'given'} disagree")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    n = situations.shape[0]
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float64)
        if extra.ndim != 2 or extra.shape[0] != n:
            raise ValueError(f"extra must have shape ({n}, E); got {extra.shape}")

    grids, situation_rows, extra_rows = [], [], []
    for row, situation in enumerate(situations):
        grid = np.asarray(spec.candidate_actions(situation, points_per_dim), dtype=np.float64)
        if grid.ndim != 2 or grid.shape[1] != spec.n_action or grid.shape[0] < 1:
            raise ValueError(f"candidate_actions must return (C, {spec.n_action}); got {grid.shape}")
        if not np.all(spec.action_in_bounds(grid)):
            raise ValueError(f"skill {spec.name!r} proposed a candidate outside its verified action bounds")
        grids.append(grid)
        situation_rows.append(np.repeat(situation[None, :], grid.shape[0], axis=0))
        if extra is not None:
            extra_rows.append(np.repeat(extra[row][None, :], grid.shape[0], axis=0))
    all_situations = np.concatenate(situation_rows) if n else np.empty((0, spec.n_situation))
    all_actions = np.concatenate(grids) if n else np.empty((0, spec.n_action))
    total = all_actions.shape[0]

    physics = np.asarray(spec.physics_guess(all_situations, all_actions), dtype=np.float64)
    if physics.shape != (total, spec.n_outcome):
        raise ValueError(f"physics_guess must return ({total}, {spec.n_outcome}); got {physics.shape}")
    physics_scores = np.asarray(
        spec.objective(OutcomeDistribution.point(spec.outcome_names, physics, levels)), dtype=np.float64
    )
    if physics_scores.shape != (total,) or not np.all(np.isfinite(physics_scores)):
        raise ValueError(f"objective must return finite physics-only scores of shape ({total},)")

    request_hash = None
    request_hashes: tuple[str, ...] = ()
    if model is None:
        residual = ResidualDistribution.zeros(total, spec.n_outcome, levels, spread_levels)
    else:
        features = build_features(
            layout,
            all_situations,
            all_actions,
            physics=physics if layout.n_physics else None,
            extra=np.concatenate(extra_rows) if extra_rows else None,
        )
        residual = model.predict_residual_distribution(features)
        if residual.mean.shape != (total, spec.n_outcome):
            raise ValueError(
                f"the residual model returned {residual.mean.shape}; expected ({total}, {spec.n_outcome})"
            )
        if not np.array_equal(residual.quantile_levels, levels):
            raise ValueError("the residual model returned different quantile levels than were requested")
        if tuple(residual.spread_levels) != (float(spread_levels[0]), float(spread_levels[1])):
            raise ValueError(
                f"the residual model computed its spread from levels {tuple(residual.spread_levels)}, "
                f"not the requested {tuple(spread_levels)}"
            )
        request_hash = getattr(model, "last_request_hash", None)
        request_hashes = tuple(getattr(model, "last_request_hashes", None) or ())
        if not request_hashes and request_hash is not None:
            request_hashes = (request_hash,)

    predicted = OutcomeDistribution.from_physics_and_residual(
        spec.outcome_names, physics, residual, spec.outcome_low, spec.outcome_high
    )
    scores = physics_scores if model is None else np.asarray(spec.objective(predicted), dtype=np.float64)
    if scores.shape != (total,) or not np.all(np.isfinite(scores)):
        raise ValueError(f"objective must return finite scores of shape ({total},)")

    decisions: list[Decision] = []
    start = 0
    for situation, grid in zip(situations, grids):
        stop = start + grid.shape[0]
        rows = slice(start, stop)
        decisions.append(
            Decision(
                skill=spec.name,
                method=method,
                situation=situation.copy(),
                candidates=grid,
                scores=scores[rows].copy(),
                chosen_index=int(np.argmax(scores[rows])),
                physics_index=int(np.argmax(physics_scores[rows])),
                predicted=OutcomeDistribution(
                    names=predicted.names,
                    physics=predicted.physics[rows],
                    mean=predicted.mean[rows],
                    median=predicted.median[rows],
                    quantile_levels=predicted.quantile_levels,
                    quantiles=predicted.quantiles[rows],
                    spread=predicted.spread[rows],
                ),
                request_hash=request_hash,
                request_hashes=request_hashes,
            )
        )
        start = stop
    return decisions


def decide(
    spec: SkillSpec,
    situation: np.ndarray,
    *,
    model: ResidualModel | None,
    layout: FeatureLayout,
    points_per_dim: Sequence[int],
    quantile_levels: Sequence[float],
    spread_levels: tuple[float, float],
    method: str,
    extra: np.ndarray | None = None,
) -> Decision:
    """Decide for one situation (S,). See :func:`decide_batch`."""
    situation = np.asarray(situation, dtype=np.float64)
    if situation.shape != (spec.n_situation,):
        raise ValueError(f"situation must have shape ({spec.n_situation},); got {situation.shape}")
    return decide_batch(
        spec,
        situation[None, :],
        model=model,
        layout=layout,
        points_per_dim=points_per_dim,
        quantile_levels=quantile_levels,
        spread_levels=spread_levels,
        method=method,
        extra=None if extra is None else np.asarray(extra, dtype=np.float64)[None, :],
    )[0]
