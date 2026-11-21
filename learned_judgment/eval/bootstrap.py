"""Paired cluster bootstrap confidence intervals, as verified in proof p06.

Lane contract (Person B, evaluation)
------------------------------------
Inputs are per-episode losses of two methods on IDENTICAL situations and noise seeds:

    loss_a, loss_b   float64 (n_situations, n_repeats)   row = situation, column = noise repeat

Each situation is a cluster: the losses are averaged over its repeats first
(``a_s = loss_a.mean(1)``, ``b_s = loss_b.mean(1)``) and situations are resampled with
replacement, keeping a and b paired. For mean statistics with an equal number of repeats per
situation this equals resampling whole clusters (p06). The interval comes from
``scipy.stats.bootstrap(..., paired=True, vectorized=True, rng=np.random.default_rng(seed))``
-- always ``rng=``, never the legacy ``random_state=`` (p06: an int there seeds a different
generator without a warning).

When the requested method is BCa and its interval is not finite (degenerate data, e.g.
identical methods; scipy emits DegenerateDataWarning), the interval is recomputed with
``method="percentile"`` from the same seed and the fallback is recorded in the result.

Resample count, confidence level, method and seed are arguments (config.yaml
``evaluation.bootstrap_*``); nothing is defaulted here. A ratio whose denominator mean is 0
(on the data or in any resample) is reported as the literal ``"undefined"``.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy import stats

from learned_judgment.eval.metrics import UNDEFINED

__all__ = ["METHODS", "paired_difference_ci", "paired_ratio_ci"]

METHODS = ("BCa", "percentile", "basic")
_FALLBACK_METHOD = "percentile"


def _cluster_means(loss_a: Any, loss_b: Any) -> tuple[np.ndarray, np.ndarray, int, int]:
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"loss_a and loss_b must share the shape (n_situations, n_repeats); got {a.shape}, {b.shape}")
    n_situations, n_repeats = a.shape
    if n_situations < 2 or n_repeats < 1:
        raise ValueError(f"need >= 2 situations and >= 1 repeat; got {a.shape} (scipy raises below 2)")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("losses must be finite")
    return a.mean(axis=1), b.mean(axis=1), n_situations, n_repeats


def _check_settings(n_resamples: Any, confidence_level: Any, method: Any, seed: Any) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive integer; got {n_resamples!r}")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError(f"confidence_level must be inside (0, 1); got {confidence_level!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")


def _bootstrap(
    data: tuple[np.ndarray, np.ndarray],
    statistic: Callable[..., np.ndarray],
    *,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stats.bootstrap(
            data,
            statistic,
            paired=True,
            vectorized=True,
            n_resamples=int(n_resamples),
            confidence_level=float(confidence_level),
            method=method,
            rng=np.random.default_rng(int(seed)),
        )
    return result, [f"{item.category.__name__}: {item.message}" for item in caught]


def _interval(
    data: tuple[np.ndarray, np.ndarray],
    statistic: Callable[..., np.ndarray],
    *,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    result, caught = _bootstrap(
        data, statistic, n_resamples=n_resamples, confidence_level=confidence_level, method=method, seed=seed
    )
    record: dict[str, Any] = {"method_used": method, "fallback_reason": None, "warnings": caught}
    low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    if method == "BCa" and not (np.isfinite(low) and np.isfinite(high)):
        record["fallback_reason"] = (
            "BCa interval was not finite (degenerate bootstrap distribution); recomputed with "
            f"method={_FALLBACK_METHOD!r} from the same seed"
        )
        result, fallback_caught = _bootstrap(
            data,
            statistic,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=_FALLBACK_METHOD,
            seed=seed,
        )
        record["method_used"] = _FALLBACK_METHOD
        record["warnings"] = caught + fallback_caught
        low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    record.update(low=low, high=high, bootstrap_distribution=np.asarray(result.bootstrap_distribution))
    return record


def _common(
    statistic_name: str,
    n_situations: int,
    n_repeats: int,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "statistic": statistic_name,
        "n_situations": n_situations,
        "n_repeats": n_repeats,
        "n_resamples": int(n_resamples),
        "confidence_level": float(confidence_level),
        "method_requested": method,
        "seed": int(seed),
    }


def paired_difference_ci(
    loss_a: Any,
    loss_b: Any,
    *,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Paired cluster-bootstrap CI of mean(loss_b) - mean(loss_a).

    Args:
        loss_a, loss_b: (n_situations, n_repeats) per-episode losses on identical situations
            and noise seeds. Positive estimates mean method b has the higher loss.

    Returns a dict with: statistic, estimate, low, high, method_requested, method_used,
    fallback_reason (None or text), warnings (list of text), n_situations, n_repeats,
    n_resamples, confidence_level, seed, per_repeat_estimates (n_repeats point estimates,
    one per noise repeat, to show seed spread; they are not bootstrapped).
    """
    _check_settings(n_resamples, confidence_level, method, seed)
    a_s, b_s, n_situations, n_repeats = _cluster_means(loss_a, loss_b)

    def mean_difference(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.mean(b - a, axis=axis)

    interval = _interval(
        (a_s, b_s),
        mean_difference,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        seed=seed,
    )
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    result = _common("mean(b) - mean(a)", n_situations, n_repeats, n_resamples, confidence_level, method, seed)
    result.update(
        estimate=float(np.mean(b_s - a_s)),
        low=interval["low"],
        high=interval["high"],
        method_used=interval["method_used"],
        fallback_reason=interval["fallback_reason"],
        warnings=interval["warnings"],
        per_repeat_estimates=[float(value) for value in (b - a).mean(axis=0)],
    )
    return result


def paired_ratio_ci(
    loss_a: Any,
    loss_b: Any,
    *,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Paired cluster-bootstrap CI of mean(loss_b) / mean(loss_a) (ratio of means), e.g.
    student error / teacher error with a = teacher. Intended for non-negative losses.

    Same arguments and result keys as :func:`paired_difference_ci`. ``estimate``, ``low`` and
    ``high`` are the literal ``"undefined"`` (with ``undefined_reason``) when mean(loss_a) is 0
    on the data or in any bootstrap resample; ``per_repeat_estimates`` holds ``"undefined"``
    for a repeat whose mean(loss_a) is 0.
    """
    _check_settings(n_resamples, confidence_level, method, seed)
    a_s, b_s, n_situations, n_repeats = _cluster_means(loss_a, loss_b)
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    result = _common("mean(b) / mean(a)", n_situations, n_repeats, n_resamples, confidence_level, method, seed)
    per_repeat: list[float | str] = []
    for repeat in range(n_repeats):
        denominator = float(a[:, repeat].mean())
        per_repeat.append(UNDEFINED if denominator == 0.0 else float(b[:, repeat].mean()) / denominator)
    result["per_repeat_estimates"] = per_repeat

    def undefined(reason: str) -> dict[str, Any]:
        result.update(
            estimate=UNDEFINED,
            low=UNDEFINED,
            high=UNDEFINED,
            method_used=None,
            fallback_reason=None,
            warnings=[],
            undefined_reason=reason,
        )
        return result

    denominator = float(a_s.mean())
    if denominator == 0.0:
        return undefined("mean(loss_a) is 0 on the data")

    def ratio_of_means(a_sample: np.ndarray, b_sample: np.ndarray, axis: int = -1) -> np.ndarray:
        numerator = np.asarray(np.mean(b_sample, axis=axis))
        denominators = np.asarray(np.mean(a_sample, axis=axis))
        out = np.full(np.shape(numerator), np.nan)
        return np.divide(numerator, denominators, out=out, where=denominators != 0.0)

    interval = _interval(
        (a_s, b_s),
        ratio_of_means,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        seed=seed,
    )
    zero_resamples = int(np.count_nonzero(~np.isfinite(interval["bootstrap_distribution"])))
    if zero_resamples:
        return undefined(f"mean(loss_a) is 0 in {zero_resamples} of {int(n_resamples)} bootstrap resamples")
    result.update(
        estimate=float(b_s.mean()) / denominator,
        low=interval["low"],
        high=interval["high"],
        method_used=interval["method_used"],
        fallback_reason=interval["fallback_reason"],
        warnings=interval["warnings"],
        undefined_reason=None,
    )
    return result
