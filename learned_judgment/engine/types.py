"""Shared array types for the generic decision engine.

This module is deliberately free of skill, simulator and provider imports: every other
Person B module (skills, teacher, baselines, students, evaluation) depends on it.

Array conventions used across the engine
----------------------------------------
    n   rows in a batch (training attempts, or candidate actions being scored)
    S   situation features, in ``spec.situation`` order
    A   action dimensions, in ``spec.actions`` order (1 <= A <= 3)
    K   outcomes, in ``spec.outcomes`` order -- the fixed per-skill outcome order
    Q   predictive quantile levels, strictly increasing in (0, 1)

    residual = measured_outcome - physics_guess          (n, K), column by column

Categorical values are stored as the float index of the level. Binary outcomes are stored
as 0.0 / 1.0, so the *mean* of a predicted binary outcome is a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Decision",
    "OutcomeDistribution",
    "ResidualDistribution",
    "ResidualModel",
    "cdf_from_quantiles",
]


def _as_2d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, K); got {array.shape}")
    return array


def _check_levels(levels: np.ndarray) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 1:
        raise ValueError(f"quantile_levels must have shape (Q,) with Q >= 1; got {levels.shape}")
    if np.any(levels <= 0.0) or np.any(levels >= 1.0) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile_levels must be strictly increasing and inside (0, 1)")
    return levels


def _check_quantiles(quantiles: np.ndarray, n: int, k: int, q: int) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.shape != (n, k, q):
        raise ValueError(f"quantiles must have shape (n, K, Q)={(n, k, q)}; got {quantiles.shape}")
    return quantiles


@dataclass(frozen=True)
class ResidualDistribution:
    """Predicted residual distribution for ``n`` query rows and ``K`` outcomes.

    Attributes:
        mean:            (n, K) predictive mean of the residual.
        median:          (n, K) predictive median of the residual.
        quantile_levels: (Q,)   strictly increasing levels in (0, 1).
        quantiles:       (n, K, Q) residual quantiles, non-decreasing along the last axis.
        spread:          (n, K) half-width of the central predictive interval between
                         ``spread_levels`` -- the "predictive spread" of the teacher API.
        spread_levels:   the (low, high) quantile levels ``spread`` was computed from.
    """

    mean: np.ndarray
    median: np.ndarray
    quantile_levels: np.ndarray
    quantiles: np.ndarray
    spread: np.ndarray
    spread_levels: tuple[float, float]

    def __post_init__(self) -> None:
        mean = _as_2d("mean", self.mean)
        median = _as_2d("median", self.median)
        spread = _as_2d("spread", self.spread)
        levels = _check_levels(self.quantile_levels)
        n, k = mean.shape
        if median.shape != (n, k) or spread.shape != (n, k):
            raise ValueError("mean, median and spread must share the shape (n, K)")
        quantiles = _check_quantiles(self.quantiles, n, k, levels.size)
        if np.any(spread < 0.0):
            raise ValueError("spread must be non-negative")
        for name, value in (
            ("mean", mean),
            ("median", median),
            ("spread", spread),
            ("quantile_levels", levels),
            ("quantiles", quantiles),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_rows(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_outcomes(self) -> int:
        return int(self.mean.shape[1])

    @classmethod
    def from_quantiles(
        cls,
        mean: np.ndarray,
        median: np.ndarray,
        quantile_levels: np.ndarray,
        quantiles: np.ndarray,
        spread_levels: tuple[float, float],
    ) -> "ResidualDistribution":
        """Build a distribution and derive ``spread`` from two of its quantile levels.

        Quantiles are sorted along the last axis first, so a provider that returns crossing
        quantiles cannot produce a negative spread. ``spread_levels`` must be exact members of
        ``quantile_levels`` (both come from config.yaml); nothing is interpolated or extrapolated.
        """
        levels = _check_levels(quantile_levels)
        quantiles = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
        low, high = spread_levels
        if not low < high:
            raise ValueError(f"spread_levels must be (low, high) with low < high; got {spread_levels}")
        positions = []
        for level in (low, high):
            matches = np.flatnonzero(levels == level)
            if matches.size != 1:
                raise ValueError(f"spread level {level} is not one of quantile_levels {levels.tolist()}")
            positions.append(int(matches[0]))
        spread = 0.5 * (quantiles[..., positions[1]] - quantiles[..., positions[0]])
        return cls(
            mean=mean,
            median=median,
            quantile_levels=levels,
            quantiles=quantiles,
            spread=spread,
            spread_levels=(float(low), float(high)),
        )

    @classmethod
    def zeros(
        cls, n_rows: int, n_outcomes: int, quantile_levels: np.ndarray, spread_levels: tuple[float, float]
    ) -> "ResidualDistribution":
        """The "no correction" distribution: zero residual and zero spread.

        Used only where zero residual is the definition of the method (physics only, or a
        learner with zero context attempts). It is never a stand-in for a provider response.
        """
        levels = _check_levels(quantile_levels)
        shape = (int(n_rows), int(n_outcomes))
        return cls(
            mean=np.zeros(shape),
            median=np.zeros(shape),
            quantile_levels=levels,
            quantiles=np.zeros(shape + (levels.size,)),
            spread=np.zeros(shape),
            spread_levels=(float(spread_levels[0]), float(spread_levels[1])),
        )


def cdf_from_quantiles(quantiles: np.ndarray, levels: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    """P(X <= threshold) per row from a finite set of predictive quantiles.

    Args:
        quantiles: (n, Q) non-decreasing along axis 1.
        levels:    (Q,) strictly increasing in (0, 1).
        threshold: scalar or (n,).

    Returns:
        (n,) probabilities. Between quantiles the CDF is interpolated linearly. Below the
        lowest quantile the lowest level is returned and above the highest quantile the
        highest level is returned, i.e. the tails are never extrapolated to 0 or 1. A row
        whose quantiles are all equal is a point mass and returns exactly 0.0 or 1.0.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    levels = _check_levels(levels)
    if quantiles.ndim != 2 or quantiles.shape[1] != levels.size:
        raise ValueError(f"quantiles must have shape (n, Q={levels.size}); got {quantiles.shape}")
    n, q = quantiles.shape
    t = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n,))
    count = (quantiles <= t[:, None]).sum(axis=1)
    lower = np.clip(count - 1, 0, q - 1)
    upper = np.clip(count, 0, q - 1)
    rows = np.arange(n)
    q_lower, q_upper = quantiles[rows, lower], quantiles[rows, upper]
    gap = q_upper - q_lower
    weight = np.divide(t - q_lower, gap, out=np.zeros(n), where=gap > 0.0)
    prob = levels[lower] + weight * (levels[upper] - levels[lower])
    prob = np.where(count == 0, levels[0], prob)
    prob = np.where(count == q, levels[-1], prob)
    point_mass = quantiles[:, 0] == quantiles[:, -1]
    return np.where(point_mass, (t >= quantiles[:, 0]).astype(np.float64), prob)


@dataclass(frozen=True)
class OutcomeDistribution:
    """Predicted outcome distribution: physics guess plus predicted residual, per row.

    Attributes:
        names:           K outcome names in the skill's fixed outcome order.
        physics:         (n, K) the physics guess the prediction was built on.
        mean:            (n, K) for a binary outcome this is its predicted probability.
        median:          (n, K)
        quantile_levels: (Q,)
        quantiles:       (n, K, Q)
        spread:          (n, K) predictive spread of the residual model (0 for physics only).
    """

    names: tuple[str, ...]
    physics: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    quantile_levels: np.ndarray
    quantiles: np.ndarray
    spread: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(self.names)
        physics = _as_2d("physics", self.physics)
        mean = _as_2d("mean", self.mean)
        median = _as_2d("median", self.median)
        spread = _as_2d("spread", self.spread)
        levels = _check_levels(self.quantile_levels)
        n, k = mean.shape
        if len(names) != k or len(set(names)) != k:
            raise ValueError(f"names must be {k} unique outcome names; got {names}")
        if physics.shape != (n, k) or median.shape != (n, k) or spread.shape != (n, k):
            raise ValueError("physics, mean, median and spread must share the shape (n, K)")
        quantiles = _check_quantiles(self.quantiles, n, k, levels.size)
        for name, value in (
            ("names", names),
            ("physics", physics),
            ("mean", mean),
            ("median", median),
            ("spread", spread),
            ("quantile_levels", levels),
            ("quantiles", quantiles),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_rows(self) -> int:
        return int(self.mean.shape[0])

    def index(self, outcome: str) -> int:
        try:
            return self.names.index(outcome)
        except ValueError:
            raise KeyError(f"unknown outcome {outcome!r}; known outcomes: {self.names}") from None

    def mean_of(self, outcome: str) -> np.ndarray:
        return self.mean[:, self.index(outcome)]

    def median_of(self, outcome: str) -> np.ndarray:
        return self.median[:, self.index(outcome)]

    def spread_of(self, outcome: str) -> np.ndarray:
        return self.spread[:, self.index(outcome)]

    def quantile_of(self, outcome: str, level: float) -> np.ndarray:
        """(n,) quantile at ``level``, which must be exactly one of ``quantile_levels``
        (no tolerance, no interpolation, no extrapolation)."""
        matches = np.flatnonzero(self.quantile_levels == level)
        if matches.size != 1:
            raise KeyError(f"level {level} is not one of {self.quantile_levels.tolist()}")
        return self.quantiles[:, self.index(outcome), int(matches[0])]

    def prob_below(self, outcome: str, threshold: np.ndarray | float) -> np.ndarray:
        """(n,) P(outcome <= threshold); see :func:`cdf_from_quantiles` for tail handling."""
        return cdf_from_quantiles(self.quantiles[:, self.index(outcome), :], self.quantile_levels, threshold)

    def prob_above(self, outcome: str, threshold: np.ndarray | float) -> np.ndarray:
        """(n,) P(outcome > threshold) = 1 - prob_below."""
        return 1.0 - self.prob_below(outcome, threshold)

    @classmethod
    def from_physics_and_residual(
        cls,
        names: tuple[str, ...],
        physics: np.ndarray,
        residual: ResidualDistribution,
        low: np.ndarray,
        high: np.ndarray,
    ) -> "OutcomeDistribution":
        """outcome = clip(physics_guess + predicted residual, low, high), statistic by statistic.

        ``low`` / ``high`` are (K,) physical bounds of the outcomes (``-inf`` / ``+inf`` when an
        outcome is unbounded). Clipping the mean is an approximation of the mean of the clipped
        distribution; the quantiles and median are exact under clipping.
        """
        physics = _as_2d("physics", physics)
        if physics.shape != residual.mean.shape:
            raise ValueError(
                f"physics {physics.shape} and residual {residual.mean.shape} must share the shape (n, K)"
            )
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        if low.shape != (physics.shape[1],) or high.shape != (physics.shape[1],):
            raise ValueError("low and high must have shape (K,)")
        return cls(
            names=names,
            physics=physics,
            mean=np.clip(physics + residual.mean, low, high),
            median=np.clip(physics + residual.median, low, high),
            quantile_levels=residual.quantile_levels,
            quantiles=np.clip(physics[:, :, None] + residual.quantiles, low[None, :, None], high[None, :, None]),
            spread=residual.spread,
        )

    @classmethod
    def point(
        cls, names: tuple[str, ...], values: np.ndarray, quantile_levels: np.ndarray
    ) -> "OutcomeDistribution":
        """A point mass at ``values`` (n, K): used for physics-only scoring and for scoring a
        *measured* outcome with the same objective that scores predictions."""
        values = _as_2d("values", values)
        levels = _check_levels(quantile_levels)
        return cls(
            names=names,
            physics=values,
            mean=values,
            median=values,
            quantile_levels=levels,
            quantiles=np.repeat(values[:, :, None], levels.size, axis=2),
            spread=np.zeros_like(values),
        )


@runtime_checkable
class ResidualModel(Protocol):
    """Anything that learns residuals from attempts: the TabPFN teacher and every residual
    baseline (constant bias, Gaussian process, gradient boosting, MLP on attempts).

    ``features`` is float64 (n, F) in the column order of the model's ``FeatureLayout``;
    ``residuals`` is float64 (n, K) in the skill's fixed outcome order.
    """

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "ResidualModel":
        ...

    def predict_residuals(self, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(median_residual, predictive_spread)``, each float64 (n_query, K)."""
        ...

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """Return the full predictive summary for the same single batched request."""
        ...


@dataclass(frozen=True)
class Decision:
    """One decision of the engine for one situation.

    Attributes:
        skill:         SkillSpec name.
        method:        label of what produced the scores (e.g. "physics_only", "teacher").
        situation:     (S,) encoded situation.
        candidates:    (C, A) every candidate action that was scored, physics action included.
        scores:        (C,) objective value per candidate; higher is better.
        chosen_index:  argmax of ``scores``.
        physics_index: index of the action physics alone would choose.
        predicted:     OutcomeDistribution with C rows, aligned with ``candidates``.
        request_hash:  cache key of the last provider request behind the decision, when one was made.
        request_hashes: cache keys of every provider request behind the decision (several when the
                       model sends one request per outcome or splits a large batch).
    """

    skill: str
    method: str
    situation: np.ndarray
    candidates: np.ndarray
    scores: np.ndarray
    chosen_index: int
    physics_index: int
    predicted: OutcomeDistribution
    request_hash: str | None = None
    request_hashes: tuple[str, ...] = ()

    @property
    def action(self) -> np.ndarray:
        """(A,) the chosen action."""
        return self.candidates[self.chosen_index]

    @property
    def physics_action(self) -> np.ndarray:
        """(A,) the action the physics guess alone would choose."""
        return self.candidates[self.physics_index]
