"""Calibration of predictive quantiles, as verified in proof p07.

Lane contract (Person B, evaluation)
------------------------------------
Inputs are predictive quantiles for ONE outcome dimension (the caller loops over outcomes
for "PIT by outcome dimension") and the measured values:

    quantiles   float64 (n, Q)  at ``levels`` (Q,), strictly increasing in (0, 1);
                                sorted per row here before use (crossing quantiles)
    y           float64 (n,)    measured outcome (or residual) of the same rows

Everything is read at the quantile knots; nothing is interpolated between them:

    interval_coverage       share of y inside [q(low), q(high)]; low/high must be exact members
    pit_bin_counts          counts of y per bin between knots, bin j = #{knots < y}
    pit_chisquare           chi-square test of those counts against the level masses
    randomized_pit          PIT drawn uniformly inside the level interval of y's bin (seeded)
    randomized_pit_ks       KS test of the randomized PIT against U(0, 1)
    spread_error_spearman   Spearman rank correlation of predicted spread with actual error;
                            "undefined" (not 0) when either input is constant

A KS test on linearly interpolated PIT values is deliberately absent: p07 showed it rejects
uniformity for a perfectly calibrated model (KS p = 1e-37 at n = 20000).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats

from learned_judgment.eval.metrics import UNDEFINED

__all__ = [
    "interval_coverage",
    "pit_bin_counts",
    "pit_chisquare",
    "randomized_pit",
    "randomized_pit_ks",
    "spread_error_spearman",
]


def _inputs(quantiles: Any, levels: Any, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_array = np.asarray(levels, dtype=np.float64)
    if level_array.ndim != 1 or level_array.size < 1:
        raise ValueError(f"levels must have shape (Q,) with Q >= 1; got {level_array.shape}")
    if np.any(level_array <= 0.0) or np.any(level_array >= 1.0) or np.any(np.diff(level_array) <= 0.0):
        raise ValueError(f"levels must be strictly increasing inside (0, 1); got {level_array.tolist()}")
    q = np.asarray(quantiles, dtype=np.float64)
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"y must have shape (n,) with n >= 1; got {values.shape}")
    if q.shape != (values.size, level_array.size):
        raise ValueError(f"quantiles must have shape (n={values.size}, Q={level_array.size}); got {q.shape}")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(values))):
        raise ValueError("quantiles and y must be finite")
    return np.sort(q, axis=1), level_array, values


def _position(levels: np.ndarray, level: float) -> int:
    matches = np.flatnonzero(levels == float(level))
    if matches.size != 1:
        raise ValueError(f"level {level} is not exactly one of {levels.tolist()} (no interpolation)")
    return int(matches[0])


def interval_coverage(quantiles: Any, levels: Any, y: Any, low_level: float, high_level: float) -> dict[str, Any]:
    """Share of rows with q(low_level) <= y <= q(high_level), read at the knots.

    quantiles (n, Q), levels (Q,), y (n,). Returns {"coverage", "count", "n", "nominal",
    "low_level", "high_level"} where nominal = high_level - low_level.
    """
    q, level_array, values = _inputs(quantiles, levels, y)
    low, high = _position(level_array, low_level), _position(level_array, high_level)
    if not low < high:
        raise ValueError(f"low_level {low_level} must be below high_level {high_level}")
    inside = (values >= q[:, low]) & (values <= q[:, high])
    count = int(inside.sum())
    return {
        "coverage": count / values.size,
        "count": count,
        "n": int(values.size),
        "nominal": float(level_array[high] - level_array[low]),
        "low_level": float(level_array[low]),
        "high_level": float(level_array[high]),
    }


def _bins(q: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(n,) bin index in 0..Q: bin j means q_(j-1) < y <= q_j (knots compared exactly)."""
    return (values[:, None] > q).sum(axis=1)


def pit_bin_counts(quantiles: Any, levels: Any, y: Any) -> np.ndarray:
    """int64 (Q + 1,) number of rows whose y falls in each bin between sorted knots."""
    q, level_array, values = _inputs(quantiles, levels, y)
    return np.bincount(_bins(q, values), minlength=level_array.size + 1).astype(np.int64)


def pit_chisquare(quantiles: Any, levels: Any, y: Any) -> dict[str, Any]:
    """Chi-square test of the bin counts against expected masses diff([0, *levels, 1]) * n.

    Returns {"counts", "expected", "statistic", "pvalue", "n"} (lists of Python numbers).
    """
    q, level_array, values = _inputs(quantiles, levels, y)
    counts = np.bincount(_bins(q, values), minlength=level_array.size + 1)
    expected = np.diff(np.concatenate(([0.0], level_array, [1.0]))) * values.size
    test = stats.chisquare(counts, expected)
    return {
        "counts": [int(value) for value in counts],
        "expected": [float(value) for value in expected],
        "statistic": float(test.statistic),
        "pvalue": float(test.pvalue),
        "n": int(values.size),
    }


def randomized_pit(quantiles: Any, levels: Any, y: Any, *, seed: int) -> np.ndarray:
    """(n,) randomized PIT: U(edge_j, edge_(j+1)) for y in bin j, edges = [0, *levels, 1].

    Exact under calibration without assuming any shape between knots. ``seed`` is a
    non-negative int for ``np.random.default_rng``.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
    q, level_array, values = _inputs(quantiles, levels, y)
    edges = np.concatenate(([0.0], level_array, [1.0]))
    bins = _bins(q, values)
    return np.random.default_rng(int(seed)).uniform(edges[bins], edges[bins + 1])


def randomized_pit_ks(quantiles: Any, levels: Any, y: Any, *, seed: int) -> dict[str, Any]:
    """KS test of :func:`randomized_pit` against U(0, 1): {"statistic", "pvalue", "n", "seed"}."""
    pit = randomized_pit(quantiles, levels, y, seed=seed)
    test = stats.kstest(pit, "uniform")
    return {"statistic": float(test.statistic), "pvalue": float(test.pvalue), "n": int(pit.size), "seed": int(seed)}


def spread_error_spearman(spread: Any, abs_error: Any) -> dict[str, Any]:
    """Spearman rank correlation of predicted spread (n,) with actual absolute error (n,).

    Returns {"rho", "pvalue", "n"}; ``rho`` / ``pvalue`` are ``"undefined"`` (with a
    ``"reason"``) when either input is constant or scipy returns NaN -- never 0.
    """
    a = np.asarray(spread, dtype=np.float64)
    b = np.asarray(abs_error, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError(f"spread and abs_error must share the shape (n >= 2,); got {a.shape}, {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("spread and abs_error must be finite")
    result: dict[str, Any] = {"n": int(a.size)}
    constant = [name for name, values in (("spread", a), ("abs_error", b)) if np.all(values == values[0])]
    if constant:
        result.update(rho=UNDEFINED, pvalue=UNDEFINED, reason=f"constant input: {', '.join(constant)}")
        return result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test = stats.spearmanr(a, b)
    rho, pvalue = float(test.statistic), float(test.pvalue)
    notes = [f"{item.category.__name__}: {item.message}" for item in caught]
    if not np.isfinite(rho):
        result.update(rho=UNDEFINED, pvalue=UNDEFINED, reason="scipy returned NaN; " + "; ".join(notes))
        return result
    result["rho"] = rho
    result["pvalue"] = pvalue if np.isfinite(pvalue) else UNDEFINED
    if notes:
        result["warnings"] = notes
    return result
