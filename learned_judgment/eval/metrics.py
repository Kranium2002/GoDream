"""Pure metric functions over measured outcomes and per-episode losses.

Lane contract (Person B, evaluation)
------------------------------------
Every function here is a pure function of the arrays it is given; none runs an episode,
reads a file or fixes a threshold. Tolerances, quantile levels and attempt counts are
arguments (from config.yaml / PREREGISTRATION.md). Comparisons against a given tolerance
are exact. A ratio refuses negative errors / losses (ValueError), and a ratio with a zero
denominator is reported as the literal ``"undefined"``, never as a number.

Shapes
    abs_errors, losses           float64 (n,)
    flags                        bool (n,)
    measured outcomes            float64 (n, K) in the skill's outcome order
    learning-curve counts        int (P,) strictly increasing, errors float64 (P,)
    actions                      float64 (n, A)

Results are plain dicts of Python scalars so they can be saved as JSON run artifacts.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.skills.base import SkillSpec

__all__ = [
    "UNDEFINED",
    "attempts_to_tolerance",
    "decision_agreement",
    "learning_curve_area",
    "rate_with_count",
    "regression_summary",
    "regret",
    "student_teacher_ratio",
    "tolerance_rate",
]

UNDEFINED: Literal["undefined"] = "undefined"
# Definition of the "90th-percentile" error (03_METRICS.md) and its estimator (numpy type 7).
_P90 = 0.9
_QUANTILE_METHOD = "linear"


def _vector(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must have shape (n,) with n >= 1; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _outcomes(spec: SkillSpec, name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != spec.n_outcome or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n >= 1, K={spec.n_outcome}); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def regression_summary(abs_errors: Any) -> dict[str, float | int]:
    """abs_errors (n,) non-negative -> {"mean", "median", "p90", "n"} (p90: numpy linear quantile)."""
    errors = _vector("abs_errors", abs_errors)
    if np.any(errors < 0.0):
        raise ValueError("abs_errors must be non-negative")
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, _P90, method=_QUANTILE_METHOD)),
        "n": int(errors.size),
    }


def rate_with_count(flags: Any) -> dict[str, float | int]:
    """flags bool (n,) -> {"rate", "count", "n"} with rate = count / n."""
    array = np.asarray(flags)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"flags must have shape (n,) with n >= 1; got {array.shape}")
    if array.dtype != np.bool_:
        raise ValueError(f"flags must be a boolean array; got dtype {array.dtype}")
    count = int(array.sum())
    return {"rate": count / array.size, "count": count, "n": int(array.size)}


def tolerance_rate(spec: SkillSpec, measured: Any) -> dict[str, float | int]:
    """Skill-specific hit / tolerance rate of measured outcomes (n, K) via ``spec.success``."""
    outcomes = _outcomes(spec, "measured", measured)
    flags = np.asarray(spec.success(outcomes))
    if flags.shape != (outcomes.shape[0],) or flags.dtype != np.bool_:
        raise ValueError(f"{spec.name}.success must return bool ({outcomes.shape[0]},); got {flags.dtype} {flags.shape}")
    return rate_with_count(flags)


def regret(spec: SkillSpec, measured_method: Any, measured_oracle: Any, quantile_levels: Sequence[float]) -> np.ndarray:
    """Per-episode regret (n,) = utility(oracle) - utility(method).

    Both inputs are measured outcomes (n, K) of the SAME situations and noise seeds (row i of
    each belongs to the same episode pairing). Utility is ``spec.objective`` applied to a
    point mass at the measured outcome, the same objective that scored the predictions.
    A negative value means the method's episode scored higher than the oracle's.
    """
    method = _outcomes(spec, "measured_method", measured_method)
    oracle = _outcomes(spec, "measured_oracle", measured_oracle)
    if method.shape != oracle.shape:
        raise ValueError(f"measured_method {method.shape} and measured_oracle {oracle.shape} must be paired rows")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    utilities = []
    for label, values in (("oracle", oracle), ("method", method)):
        utility = np.asarray(
            spec.objective(OutcomeDistribution.point(spec.outcome_names, values, levels)), dtype=np.float64
        )
        if utility.shape != (values.shape[0],) or not np.all(np.isfinite(utility)):
            raise ValueError(f"{spec.name}.objective must return finite ({values.shape[0]},) for the {label} rows")
        utilities.append(utility)
    return utilities[0] - utilities[1]


def _curve(counts: Any, errors: Any) -> tuple[np.ndarray, np.ndarray]:
    count_array = np.asarray(counts)
    if count_array.ndim != 1 or count_array.size < 1 or not np.issubdtype(count_array.dtype, np.integer):
        raise ValueError(f"counts must be an integer array of shape (P,) with P >= 1; got {count_array.dtype} {count_array.shape}")
    if np.any(count_array < 0) or np.any(np.diff(count_array) <= 0):
        raise ValueError(f"counts must be non-negative and strictly increasing; got {count_array.tolist()}")
    error_array = _vector("errors", errors)
    if error_array.shape != count_array.shape:
        raise ValueError(f"errors {error_array.shape} must match counts {count_array.shape}")
    return count_array, error_array


def learning_curve_area(counts: Any, errors: Any) -> float:
    """Trapezoid area under errors (P,) over attempt counts (P,), in error x attempts.

    Not normalized; needs P >= 2 points (a single point has no area).
    """
    count_array, error_array = _curve(counts, errors)
    if count_array.size < 2:
        raise ValueError("a learning-curve area needs at least two attempt counts")
    return float(np.trapezoid(error_array, x=count_array.astype(np.float64)))


def attempts_to_tolerance(counts: Any, errors: Any, tolerance: float) -> int | None:
    """Smallest evaluated attempt count whose error is <= ``tolerance`` (exact comparison),
    or ``None`` when no evaluated count reaches it ("not reached")."""
    count_array, error_array = _curve(counts, errors)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance):
        raise ValueError(f"tolerance must be finite; got {tolerance}")
    reached = np.flatnonzero(error_array <= tolerance)
    return int(count_array[reached[0]]) if reached.size else None


def student_teacher_ratio(student_errors: Any, teacher_errors: Any) -> dict[str, Any]:
    """mean(student_errors) / mean(teacher_errors) over paired episodes, each (n,).

    Both inputs are errors / losses and must be non-negative: a ratio of means of negative or
    sign-changing losses does not read as "relative error" (a student better everywhere on a
    loss of -utility would show a ratio above 1), so a negative value raises ValueError.
    Returns {"ratio", "student_mean", "teacher_mean", "n"}; ``ratio`` is ``"undefined"`` (with
    a ``"reason"``) when the teacher mean is exactly 0.
    """
    student = _vector("student_errors", student_errors)
    teacher = _vector("teacher_errors", teacher_errors)
    if student.shape != teacher.shape:
        raise ValueError(f"student_errors {student.shape} and teacher_errors {teacher.shape} must be paired")
    for label, values in (("student_errors", student), ("teacher_errors", teacher)):
        if np.any(values < 0.0):
            raise ValueError(f"{label} must be non-negative for a ratio; {int(np.sum(values < 0.0))} values are negative")
    result: dict[str, Any] = {
        "student_mean": float(student.mean()),
        "teacher_mean": float(teacher.mean()),
        "n": int(student.size),
    }
    if result["teacher_mean"] == 0.0:
        result.update(ratio=UNDEFINED, reason="teacher mean error is 0")
    else:
        result["ratio"] = result["student_mean"] / result["teacher_mean"]
    return result


def decision_agreement(actions_a: Any, actions_b: Any) -> dict[str, float | int]:
    """Share of situations where two methods chose exactly the same action.

    actions_a, actions_b: (n, A) chosen actions for the same n situations, both taken from the
    deterministic candidate grid (a continuous output must be mapped to its grid candidate
    by the caller first). Equality is exact over all A dimensions.
    """
    a = np.asarray(actions_a, dtype=np.float64)
    b = np.asarray(actions_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 1:
        raise ValueError(f"actions must share the shape (n >= 1, A); got {a.shape} and {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("actions must be finite")
    return rate_with_count(np.all(a == b, axis=1))
