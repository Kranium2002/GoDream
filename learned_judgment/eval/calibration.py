"""Calibration of predictive distributions, per outcome dimension and per outcome kind.

Lane contract (Person B, evaluation)
------------------------------------
The caller loops over outcome dimensions ("PIT by outcome dimension") and passes, for each,
the outcome kind and bounds from the SkillSpec (``spec.outcomes[k].kind`` /
``spec.outcomes[k].bounds``, or ``FeatureLayout.outcome_kinds`` with ``spec.outcome_low`` /
``spec.outcome_high``):

    kind "regression"  quantile functions below; ``bounds = (low, high)`` may be infinite
    kind "binary"      probability functions below; every quantile function raises

Quantile functions (regression outcomes only)
    quantiles   float64 (n, Q)  at ``levels`` (Q,), strictly increasing in (0, 1); sorted per row
                                here (crossing quantiles); every knot must lie inside ``bounds``
    y           float64 (n,)    measured outcome of the same rows, inside ``bounds``

    interval_coverage       coverage of [q(low), q(high)], read at the knots, reported
                            separately for rows whose interval does not touch an outcome bound
                            (the nominal level high - low applies to them) and for rows whose
                            interval touches a bound (their nominal coverage is ABOVE high - low,
                            because the closed interval contains the probability mass at the bound)
    randomized_pit          tie- and bound-aware randomized PIT (seeded), see below
    pit_bin_counts          counts of the randomized PIT per level interval [0, l1, ..., lQ, 1]
    pit_chisquare           chi-square test of those counts against the level masses
    randomized_pit_ks       KS test of the randomized PIT against U(0, 1)

Probability functions (binary outcomes; the predicted probability is the predictive mean)
    brier_score             mean squared difference of predicted probability and 0/1 outcome
    reliability_table       equal-width probability bins: count, mean predicted probability,
                            observed rate, and a binned chi-square of observed vs expected events

    spread_error_spearman   Spearman rank correlation of predicted spread with actual error;
                            "undefined" (not 0) when either input has exactly zero range

Randomized PIT with atoms (ties)
--------------------------------
A regression outcome clipped at a physical bound has a probability atom at that bound, and the
engine's clipped quantiles (``OutcomeDistribution.from_physics_and_residual``) then tie at the
bound. A knot is a *separator* when its value is not shared with another knot of the row and
does not lie on an outcome bound; a tied knot and a knot on a bound are where an atom can sit.
Level intervals are merged across every non-separator knot, so the bins of a row are the
intervals between consecutive separators, with edges ``[0, l_s1, ..., l_sm, 1]``. The PIT of a
row is drawn uniformly over the merged bin that contains ``y`` (``y`` equal to a separator
belongs to the bin below it). For a tied row (``y`` equal to tied knots) this is the whole tied
range ``[edges[#{q < y}], edges[#{q <= y} + 1]]`` of the knot edges ``[0, *levels, 1]``; the
untied rows that share that merged bin get the same range. The rule is exact under
calibration whenever every atom sits on tied knots or on a bound: the probability of a merged
bin is ``l_upper - l_lower`` because its separators are continuity points. Drawing the tied
range for the tied rows alone (without merging the neighbouring bins) is not exact: it
rejects a calibrated outcome clipped at 0 (experiments/proofs/B_eval_offline/
c8_calibration_atoms.py prints that p-value next to the merged-bin result).
Without ties and bounds the rule reduces to the plain per-bin draw (proof p07), with the same
draws for the same seed. A KS test on linearly interpolated PIT values is deliberately absent:
p07 showed it rejects uniformity for a perfectly calibrated model (KS p = 1e-37 at n = 20000).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import numpy as np
from scipy import stats

from learned_judgment.eval.metrics import UNDEFINED

__all__ = [
    "KINDS",
    "brier_score",
    "interval_coverage",
    "pit_bin_counts",
    "pit_chisquare",
    "randomized_pit",
    "randomized_pit_ks",
    "reliability_table",
    "spread_error_spearman",
]

KINDS = ("regression", "binary")
_AT_BOUND_NOTE = (
    "these intervals touch an outcome bound; the closed interval contains the probability mass at "
    "that bound, so their nominal coverage is above high_level - low_level"
)


# --------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------


def _regression_bounds(kind: Any, bounds: Any, function: str) -> tuple[float, float]:
    """Refuse binary outcomes for quantile functions; return finite-or-infinite (low, high)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}")
    if kind == "binary":
        raise ValueError(
            f"{function} is a quantile calibration function and is not defined for a binary outcome "
            "(its quantiles only take the values 0 and 1); use brier_score and reliability_table on "
            "the predicted probability instead"
        )
    values = tuple(float(value) for value in bounds)
    if len(values) != 2 or any(math.isnan(value) for value in values) or not values[0] < values[1]:
        raise ValueError(f"bounds must be (low, high) with low < high (either may be infinite); got {bounds!r}")
    return values


def _inputs(
    quantiles: Any, levels: Any, y: Any, bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    low, high = bounds
    if np.any(values < low) or np.any(values > high):
        raise ValueError(f"y must lie inside the outcome bounds [{low}, {high}]")
    if np.any(q < low) or np.any(q > high):
        raise ValueError(
            f"quantiles must lie inside the outcome bounds [{low}, {high}] (clip them as "
            "OutcomeDistribution.from_physics_and_residual does)"
        )
    return np.sort(q, axis=1), level_array, values


def _position(levels: np.ndarray, level: float) -> int:
    matches = np.flatnonzero(levels == float(level))
    if matches.size != 1:
        raise ValueError(f"level {level} is not exactly one of {levels.tolist()} (no interpolation)")
    return int(matches[0])


def _check_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
    return int(seed)


# --------------------------------------------------------------------------------------
# Interval coverage (regression)
# --------------------------------------------------------------------------------------


def interval_coverage(
    quantiles: Any,
    levels: Any,
    y: Any,
    low_level: float,
    high_level: float,
    *,
    kind: str,
    bounds: Sequence[float],
) -> dict[str, Any]:
    """Coverage of [q(low_level), q(high_level)] (both ends included), read at the knots.

    quantiles (n, Q), levels (Q,), y (n,); ``kind`` must be "regression" (a binary outcome
    raises ValueError); ``bounds`` = (low, high) of the outcome.

    A row's interval *touches a bound* when q(low_level) <= low or q(high_level) >= high.
    Returns a flat dict of Python scalars:
        low_level, high_level, nominal (= high_level - low_level), n
        interior_n, interior_count, interior_coverage  rows not touching a bound; the nominal
                                                       level applies to them
        at_bound_n, at_bound_share, at_bound_count, at_bound_coverage, at_bound_note
                                                       rows touching a bound; nominal is above
                                                       high_level - low_level for them
    A coverage over zero rows is ``"undefined"``.
    """
    outcome_bounds = _regression_bounds(kind, bounds, "interval_coverage")
    q, level_array, values = _inputs(quantiles, levels, y, outcome_bounds)
    low, high = _position(level_array, low_level), _position(level_array, high_level)
    if not low < high:
        raise ValueError(f"low_level {low_level} must be below high_level {high_level}")
    inside = (values >= q[:, low]) & (values <= q[:, high])
    touches = (q[:, low] <= outcome_bounds[0]) | (q[:, high] >= outcome_bounds[1])
    n = int(values.size)
    interior_n = int(np.count_nonzero(~touches))
    at_bound_n = n - interior_n
    interior_count = int(np.count_nonzero(inside & ~touches))
    at_bound_count = int(np.count_nonzero(inside & touches))
    return {
        "low_level": float(level_array[low]),
        "high_level": float(level_array[high]),
        "nominal": float(level_array[high] - level_array[low]),
        "n": n,
        "interior_n": interior_n,
        "interior_count": interior_count,
        "interior_coverage": interior_count / interior_n if interior_n else UNDEFINED,
        "at_bound_n": at_bound_n,
        "at_bound_share": at_bound_n / n,
        "at_bound_count": at_bound_count,
        "at_bound_coverage": at_bound_count / at_bound_n if at_bound_n else UNDEFINED,
        "at_bound_note": _AT_BOUND_NOTE,
    }


# --------------------------------------------------------------------------------------
# Randomized PIT (regression)
# --------------------------------------------------------------------------------------


def _pit_ranges(
    q: np.ndarray, levels: np.ndarray, values: np.ndarray, bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merged-bin PIT range per row: (lower (n,), upper (n,), tied bool (n,), merged bool (n,)).

    q (n, Q) sorted, levels (Q,), values (n,). See the module docstring for the rule.
    """
    n, n_levels = q.shape
    shared = np.zeros(q.shape, dtype=bool)
    shared[:, 1:] |= q[:, 1:] == q[:, :-1]
    shared[:, :-1] |= q[:, :-1] == q[:, 1:]
    separator = ~shared & (q > bounds[0]) & (q < bounds[1])
    index = np.arange(n_levels)
    below = separator & (q < values[:, None])
    above = separator & (q >= values[:, None])
    lower_knot = np.where(below, index, -1).max(axis=1)  # -1: no separator below
    upper_knot = np.where(above, index, n_levels).min(axis=1)  # Q: no separator above
    edges = np.concatenate(([0.0], levels, [1.0]))
    lower, upper = edges[lower_knot + 1], edges[upper_knot + 1]
    tied = np.any(q == values[:, None], axis=1)
    merged = (upper_knot - lower_knot) > 1
    return lower, upper, tied, merged


def randomized_pit(
    quantiles: Any, levels: Any, y: Any, *, kind: str, bounds: Sequence[float], seed: int
) -> np.ndarray:
    """(n,) randomized PIT: uniform over the merged level bin that contains y (module docstring).

    ``kind`` must be "regression"; ``bounds`` = (low, high); ``seed`` is a non-negative int for
    ``np.random.default_rng``. Exact under calibration without assuming any shape between
    knots, provided every atom of the outcome sits on tied knots or on an outcome bound.
    """
    seed = _check_seed(seed)
    outcome_bounds = _regression_bounds(kind, bounds, "randomized_pit")
    q, level_array, values = _inputs(quantiles, levels, y, outcome_bounds)
    lower, upper, _, _ = _pit_ranges(q, level_array, values, outcome_bounds)
    return np.random.default_rng(seed).uniform(lower, upper)


def _pit_with_ties(
    quantiles: Any, levels: Any, y: Any, kind: str, bounds: Sequence[float], seed: int, function: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = _check_seed(seed)
    outcome_bounds = _regression_bounds(kind, bounds, function)
    q, level_array, values = _inputs(quantiles, levels, y, outcome_bounds)
    lower, upper, tied, merged = _pit_ranges(q, level_array, values, outcome_bounds)
    pit = np.random.default_rng(seed).uniform(lower, upper)
    n = int(values.size)
    ties = {
        "tied_rows": int(np.count_nonzero(tied)),
        "tied_share": float(np.count_nonzero(tied) / n),
        "merged_rows": int(np.count_nonzero(merged)),
        "merged_share": float(np.count_nonzero(merged) / n),
    }
    return pit, level_array, ties


def _bin_counts(pit: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """int64 (Q + 1,): PIT values per level interval [edge_j, edge_(j+1)), edges [0, *levels, 1]."""
    return np.bincount(np.searchsorted(levels, pit, side="right"), minlength=levels.size + 1).astype(np.int64)


def pit_bin_counts(
    quantiles: Any, levels: Any, y: Any, *, kind: str, bounds: Sequence[float], seed: int
) -> np.ndarray:
    """int64 (Q + 1,) number of rows whose randomized PIT (same ``seed``) falls in each level
    interval of ``[0, *levels, 1]``. Regression outcomes only."""
    pit, level_array, _ = _pit_with_ties(quantiles, levels, y, kind, bounds, seed, "pit_bin_counts")
    return _bin_counts(pit, level_array)


def pit_chisquare(
    quantiles: Any, levels: Any, y: Any, *, kind: str, bounds: Sequence[float], seed: int
) -> dict[str, Any]:
    """Chi-square test of the randomized-PIT bin counts against diff([0, *levels, 1]) * n.

    Regression outcomes only. Returns {"counts", "expected", "statistic", "pvalue", "n", "seed",
    "tied_rows", "tied_share", "merged_rows", "merged_share"} (Python numbers): a tied row has
    y equal to one of its knots; a merged row drew its PIT over more than one level interval.
    """
    pit, level_array, ties = _pit_with_ties(quantiles, levels, y, kind, bounds, seed, "pit_chisquare")
    counts = _bin_counts(pit, level_array)
    expected = np.diff(np.concatenate(([0.0], level_array, [1.0]))) * pit.size
    test = stats.chisquare(counts, expected)
    return {
        "counts": [int(value) for value in counts],
        "expected": [float(value) for value in expected],
        "statistic": float(test.statistic),
        "pvalue": float(test.pvalue),
        "n": int(pit.size),
        "seed": int(seed),
        **ties,
    }


def randomized_pit_ks(
    quantiles: Any, levels: Any, y: Any, *, kind: str, bounds: Sequence[float], seed: int
) -> dict[str, Any]:
    """KS test of :func:`randomized_pit` against U(0, 1). Regression outcomes only.

    Returns {"statistic", "pvalue", "n", "seed", "tied_rows", "tied_share", "merged_rows",
    "merged_share"}.
    """
    pit, _, ties = _pit_with_ties(quantiles, levels, y, kind, bounds, seed, "randomized_pit_ks")
    test = stats.kstest(pit, "uniform")
    return {"statistic": float(test.statistic), "pvalue": float(test.pvalue), "n": int(pit.size), "seed": int(seed), **ties}


# --------------------------------------------------------------------------------------
# Binary outcomes
# --------------------------------------------------------------------------------------


def _binary_inputs(predicted_probability: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predicted_probability, dtype=np.float64)
    values = np.asarray(y, dtype=np.float64)
    if p.ndim != 1 or p.size < 1 or values.shape != p.shape:
        raise ValueError(f"predicted_probability and y must share the shape (n >= 1,); got {p.shape}, {values.shape}")
    if not (np.all(np.isfinite(p)) and np.all((p >= 0.0) & (p <= 1.0))):
        raise ValueError("predicted_probability must be finite and inside [0, 1]")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("y of a binary outcome must be 0.0 or 1.0")
    return p, values


def brier_score(predicted_probability: Any, y: Any) -> dict[str, Any]:
    """Brier score mean((p - y)^2) of predicted probabilities (n,) for 0/1 outcomes y (n,).

    Returns {"brier", "n"}. Lower is better; it mixes calibration and sharpness, so judge
    calibration with :func:`reliability_table`.
    """
    p, values = _binary_inputs(predicted_probability, y)
    return {"brier": float(np.mean((p - values) ** 2)), "n": int(p.size)}


def reliability_table(predicted_probability: Any, y: Any, n_bins: int) -> dict[str, Any]:
    """Reliability table of predicted probabilities (n,) against 0/1 outcomes y (n,).

    Bins are equal-width on [0, 1]: ``edges = numpy.linspace(0, 1, n_bins + 1)``; bin b holds
    edges[b] <= p < edges[b + 1], and the last bin also holds p = 1. ``n_bins`` comes from
    config.yaml ``evaluation.reliability_bins`` at the call site.

    Returns {"n", "n_bins", "edges", "bins", "binned_chisquare"} where each of ``bins`` is
    {"low", "high", "count", "mean_predicted", "observed_rate", "expected_events",
    "observed_events", "variance"} ("undefined" means for an empty bin), and
    ``binned_chisquare`` is {"statistic", "df", "pvalue", "zero_variance_mismatch", "note"}:
    the sum over bins with variance sum p(1 - p) > 0 of (observed - expected events)^2 /
    variance, compared with chi-square(df = number of such bins). Under calibration with
    independent rows it is approximately chi-square (poor for bins with few events).
    ``zero_variance_mismatch`` counts events in bins whose predictions are all exactly 0 or 1
    that differ from the prediction (any such event refutes those predictions outright).
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer; got {n_bins!r}")
    p, values = _binary_inputs(predicted_probability, y)
    n_bins = int(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    bins: list[dict[str, Any]] = []
    statistic, df, mismatch = 0.0, 0, 0.0
    for b in range(n_bins):
        members = index == b
        count = int(np.count_nonzero(members))
        expected = float(p[members].sum())
        observed = float(values[members].sum())
        variance = float(np.sum(p[members] * (1.0 - p[members])))
        bins.append(
            {
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "count": count,
                "mean_predicted": expected / count if count else UNDEFINED,
                "observed_rate": observed / count if count else UNDEFINED,
                "expected_events": expected,
                "observed_events": observed,
                "variance": variance,
            }
        )
        if variance > 0.0:
            statistic += (observed - expected) ** 2 / variance
            df += 1
        else:
            mismatch += abs(observed - expected)
    test: dict[str, Any] = {
        "statistic": float(statistic) if df else UNDEFINED,
        "df": df,
        "pvalue": float(stats.chi2.sf(statistic, df)) if df else UNDEFINED,
        "zero_variance_mismatch": float(mismatch),
        "note": "sum over bins with sum p(1-p) > 0 of (observed - expected events)^2 / sum p(1-p); "
        "chi-square with df = number of such bins under calibration (approximate for small bins)",
    }
    return {
        "n": int(p.size),
        "n_bins": n_bins,
        "edges": [float(value) for value in edges],
        "bins": bins,
        "binned_chisquare": test,
    }


# --------------------------------------------------------------------------------------
# Spread versus error
# --------------------------------------------------------------------------------------


def spread_error_spearman(spread: Any, abs_error: Any) -> dict[str, Any]:
    """Spearman rank correlation of predicted spread (n,) with actual absolute error (n,).

    Returns {"rho", "pvalue", "n"}; ``rho`` / ``pvalue`` are ``"undefined"`` (with a
    ``"reason"``) when either input has exactly zero range (``numpy.ptp == 0``, e.g. the
    homoscedastic baselines, whose spread is one value broadcast to every row) or scipy
    returns NaN -- never 0.
    """
    a = np.asarray(spread, dtype=np.float64)
    b = np.asarray(abs_error, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError(f"spread and abs_error must share the shape (n >= 2,); got {a.shape}, {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("spread and abs_error must be finite")
    result: dict[str, Any] = {"n": int(a.size)}
    constant = [name for name, values in (("spread", a), ("abs_error", b)) if float(np.ptp(values)) == 0.0]
    if constant:
        result.update(rho=UNDEFINED, pvalue=UNDEFINED, reason=f"zero range: {', '.join(constant)}")
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
