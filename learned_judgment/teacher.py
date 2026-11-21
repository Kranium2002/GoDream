"""TabPFN residual teacher: settings, request keys, the hosted provider and the teacher itself.

Lane contract (teacher lane, Person B)
--------------------------------------
* :class:`TabPFNTeacher` implements :class:`learned_judgment.engine.types.ResidualModel`::

      fit(features (n, F), residuals (n, K))          stores the context; no network
      predict_residual_distribution(query (m, F))     -> ResidualDistribution (m, K[, Q])
      predict_residuals(query (m, F))                 -> (median (m, K), spread (m, K))

  F and the column order come from the teacher's FeatureLayout, K from its outcome order,
  Q from ``engine.quantile_levels``; the spread is half the width between
  ``engine.spread_levels``.
* The hosted regressor accepts only 1-D targets (E7 p1), so outcomes are combined per
  ``engine.multi_outcome_mode``:

      stacked      all non-constant ("active") outcomes in long format with outcome-major
                   blocks: block j repeats every training row with a final categorical column
                   holding the outcome index k = active[j], and its target is
                   (r[:, k] - mean_k) / std_k (training mean, std with ddof 0). Queries are
                   expanded the same way; mean, median and every quantile are de-standardized
                   (value * std_k + mean_k). With one active outcome no column is added.
      per_outcome  one request per active outcome on the original features and raw residuals.

  An outcome whose training residuals are all identical is constant: its predicted residual
  is that constant with zero spread, and it is never sent.
* Query rows are sent in consecutive chunks of at most ``tabpfn.max_query_rows_per_request``
  rows (after stacking). Every chunk is its own request with the same training context, its own
  key, gate, cache entry and call-log records; the answers are concatenated in order. A chunk
  whose row count equals the number of quantile levels Q is sent with its last row duplicated
  once (the extra answer row is dropped), so the provider's (Q, n) quantile layout can always be
  told from the transposed one by shape; the key covers the rows exactly as sent.
* Every request has a :func:`request_key`. A stored key is served from the
  :class:`RequestCache` without any network call. An uncached request runs entirely under the
  cache's inter-process :class:`CacheLock` (<cache_dir>/.lock):

      re-check the cache -> the call log must exist when the cache holds entries
      (LedgerMissingError) -> the key must have no unresolved dispatch (UnresolvedDispatchError)
      -> offline request checks -> the cache, unvalidated and call-log directories must be
      writable with the configured persistence settings -> free usage read and cost quote
      (E7 p2/p3) -> gate -> dispatch record -> ONE billable call -> raw response written to
      unvalidated/<key>.npz -> validation -> validated entry <key>.npz -> unvalidated file
      deleted -> completed record.

  The gate refuses (:class:`BudgetError`) when the quote exceeds
  ``tabpfn.max_tokens_per_request``, when ``tabpfn.project_token_budget`` is unset, when tokens
  already quoted for dispatched requests plus this quote exceed that budget, when the account's
  remaining tokens are below the quote, or when a configured
  ``tabpfn.verified_request_limit_per_minute`` / ``_per_hour`` would be exceeded by this
  dispatch. A request is never re-sent automatically: a dispatch without a later completed or
  abandoned record blocks its key until an operator calls :meth:`TabPFNTeacher.resolve_unresolved`.
* :class:`TabPFNProvider` is the only code that imports ``tabpfn_client``, lazily. It follows
  the pinned client 0.10.1 contract executed or read in experiments/proofs/E7_tabpfn_contract:
  categoricals are declared explicitly (no auto-detection below 100 rows), every
  output-affecting setting is pinned, and ``predict(output_type="main")`` returns mean, median
  and mode as (n,) and quantiles in the server's (Q, n) layout.

Shapes: n training rows, m query rows, F features, K outcomes, A active outcomes, Q levels.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.config import CONFIG_PATH, ConfigError, optional, require
from learned_judgment.engine.features import FeatureLayout
from learned_judgment.engine.types import ResidualDistribution
from learned_judgment.teacher_cache import (
    DISPATCH_FLAG,
    QUOTE_FIELD,
    CacheCorruptError,
    CachedEntry,
    CacheLock,
    CacheLockedError,
    CallLog,
    RequestCache,
    canonical_json,
    dispatch_times,
    tokens_quoted,
    unresolved_dispatches,
    update_digest,
    utc_timestamp,
)

__all__ = [
    "MAIN_OUTPUT_KEYS",
    "MIN_TRAIN_ROWS",
    "MULTI_OUTCOME_MODES",
    "STACKED_OUTCOME_COLUMN",
    "BudgetError",
    "CacheCorruptError",
    "CacheLock",
    "CacheLockedError",
    "CachedEntry",
    "CallLog",
    "LedgerMissingError",
    "ProviderResponseError",
    "RequestCache",
    "TabPFNProvider",
    "TabPFNTeacher",
    "TeacherSettings",
    "UnresolvedDispatchError",
    "budget_refusal_reasons",
    "constant_outcomes",
    "destandardize",
    "fill_outcomes",
    "median_quantile_gap",
    "parse_main_output",
    "rate_limit_refusal_reasons",
    "remaining_tokens",
    "request_key",
    "stack_query",
    "stack_training",
    "unstack_rows",
]

MULTI_OUTCOME_MODES = ("stacked", "per_outcome")
MIN_TRAIN_ROWS = 2  # the hosted client rejects fewer training rows (E7 p1, MIN_TRAIN_ROWS)
TASK = "regression"
OUTPUT_TYPE = "main"
MAIN_OUTPUT_KEYS = ("mean", "median", "mode", "quantiles")
STACKED_OUTCOME_COLUMN = "stacked_outcome_index"
RESOLVE_ACTIONS = ("adopt", "abandon")

_KEY_SCHEMA = "learned-judgment/teacher-request-key/v2"
_PROJECT_ROOT = CONFIG_PATH.parent
_AUTO_MODEL_PATHS = ("auto", "default")  # aliases that let the server choose the model
# Constructor flags that change what the client sends besides tabpfn_config: tabpfn_systems
# (paper_version), thinking_config (thinking_mode) and text_handling (E7 p1 "what is SENT").
_PINNED_REQUEST_FLAGS = (("paper_version", False), ("text_handling", "advanced"), ("thinking_mode", False))
# api_models.APIUsageData fields: counters, limits and reset times only, no identifiers (E7 p2).
_USAGE_FIELDS = (
    "current_usage",
    "usage_limit",
    "reset_time",
    "daily_tokens_used",
    "daily_token_limit",
    "daily_usage",
    "daily_limit",
    "daily_reset_time",
    "monthly_tokens_used",
    "monthly_token_limit",
    "monthly_usage",
    "monthly_limit",
    "monthly_reset_time",
    "thinking_fit_count",
    "thinking_fit_limit",
)
_LIMIT_PAIRS = (
    ("daily_token_limit", "daily_tokens_used"),
    ("monthly_token_limit", "monthly_tokens_used"),
    ("usage_limit", "current_usage"),
)
_UNLIMITED_USAGE_LIMIT = -1  # documented "unlimited" marker of usage_limit (E7 p2)
_KEYED_SETTINGS = ("client_version", "model_path", "n_estimators", "random_state", "quantile_levels")
_USAGE_COUNTERS = ("current_usage", "daily_tokens_used", "monthly_tokens_used")
_BILLED_PHASES = ("fit", "predict", "received")  # provider phases at or after the network stage
# Behaviours config.yaml names as switches; the teacher always performs them and refuses to be
# configured otherwise (review finding T7).
_ALWAYS_ON_SWITCHES = (
    ("tabpfn.check_usage_before_uncached_request", "reads account usage before every uncached request"),
    ("tabpfn.estimate_cost_before_uncached_request", "reads a cost quote before every uncached request"),
    ("tabpfn.cache_requests_and_responses", "caches every request and response"),
)
_RATE_WINDOWS = (("verified_request_limit_per_minute", 60.0), ("verified_request_limit_per_hour", 3600.0))


class BudgetError(RuntimeError):
    """An uncached request was refused before any billable call; reasons are in the message and the call log."""


class ProviderResponseError(RuntimeError):
    """A provider response (prediction, quote or usage) violated the verified contract."""


class UnresolvedDispatchError(RuntimeError):
    """The call log holds a dispatch of this key without a later completed or abandoned record.

    The request may already have been billed. It is never re-sent automatically; an operator
    decides with :meth:`TabPFNTeacher.resolve_unresolved` (``adopt`` or ``abandon``).
    """


class LedgerMissingError(RuntimeError):
    """The call log (the project's spend ledger) is missing while the cache holds entries."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherSettings:
    """Everything the teacher reads from config.yaml; no field has a default.

    Keyed into every request (they change what is sent): client_version, model_path,
    n_estimators, random_state and quantile_levels (the provider base URL is keyed too; it comes
    from the installed client, not from here). Not keyed: spread_levels (post-processing),
    multi_outcome_mode (visible in the request arrays), cache_dir and call_log (paths),
    max_tokens_per_request, project_token_budget and the verified request limits (the gate),
    max_query_rows_per_request (each chunk's rows are keyed), lock_timeout_s and lock_poll_s.

    lock_poll_s is ``tabpfn.lock_poll_s``: the pause between attempts on a file another
    process holds, used here between attempts on the cache lock.
    """

    client_version: str
    model_path: str
    n_estimators: int
    random_state: int
    quantile_levels: tuple[float, ...]
    spread_levels: tuple[float, float]
    multi_outcome_mode: str
    cache_dir: Path
    call_log: Path
    max_tokens_per_request: int
    project_token_budget: int | None
    max_query_rows_per_request: int
    lock_timeout_s: float
    lock_poll_s: float
    verified_request_limit_per_minute: int | None
    verified_request_limit_per_hour: int | None

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.client_version, str) or not self.client_version:
            problems.append("client_version must be a non-empty string")
        if not isinstance(self.model_path, str) or not self.model_path or self.model_path in _AUTO_MODEL_PATHS:
            problems.append(f"model_path must name a pinned model, not {self.model_path!r}")
        for name in ("n_estimators", "max_tokens_per_request", "max_query_rows_per_request"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                problems.append(f"{name} must be a positive integer; got {value!r}")
        if not _is_int(self.random_state) or self.random_state < 0:
            problems.append(f"random_state must be a non-negative integer; got {self.random_state!r}")
        if self.project_token_budget is not None and (
            not _is_int(self.project_token_budget) or self.project_token_budget < 0
        ):
            problems.append(f"project_token_budget must be null or a non-negative integer; got {self.project_token_budget!r}")
        for name, _ in _RATE_WINDOWS:
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 1):
                problems.append(f"{name} must be null or a positive integer; got {value!r}")
        for name in ("lock_timeout_s", "lock_poll_s"):
            value = getattr(self, name)
            if not _is_real(value) or not math.isfinite(value) or value <= 0:
                problems.append(f"{name} must be a positive finite number of seconds; got {value!r}")
        if self.multi_outcome_mode not in MULTI_OUTCOME_MODES:
            problems.append(f"multi_outcome_mode must be one of {MULTI_OUTCOME_MODES}; got {self.multi_outcome_mode!r}")
        for name in ("cache_dir", "call_log"):
            if not isinstance(getattr(self, name), Path):
                problems.append(f"{name} must be a pathlib.Path")

        levels = tuple(self.quantile_levels) if isinstance(self.quantile_levels, (list, tuple)) else ()
        if not levels or not all(_is_real(level) for level in levels):
            problems.append(f"quantile_levels must be a non-empty sequence of numbers; got {self.quantile_levels!r}")
            levels = ()
        else:
            levels = tuple(float(level) for level in levels)
            if any(not 0.0 < level < 1.0 for level in levels) or any(b <= a for a, b in zip(levels, levels[1:])):
                problems.append(f"quantile_levels must be strictly increasing inside (0, 1); got {list(levels)}")
        if levels and _is_int(self.max_query_rows_per_request) and self.max_query_rows_per_request <= len(levels):
            # Structural: a chunk of exactly Q rows is padded to Q + 1 rows, which must still fit.
            problems.append(
                f"max_query_rows_per_request must exceed the number of quantile levels ({len(levels)}); "
                f"got {self.max_query_rows_per_request}"
            )
        spread = tuple(self.spread_levels) if isinstance(self.spread_levels, (list, tuple)) else ()
        if len(spread) != 2 or not all(_is_real(level) for level in spread):
            problems.append(f"spread_levels must be two numbers (low, high); got {self.spread_levels!r}")
        else:
            spread = (float(spread[0]), float(spread[1]))
            if not spread[0] < spread[1]:
                problems.append(f"spread_levels must satisfy low < high; got {list(spread)}")
            missing = [level for level in spread if level not in levels]
            if missing:
                problems.append(f"spread_levels {missing} are not members of quantile_levels {list(levels)}")
        if problems:
            raise ValueError("invalid TeacherSettings: " + "; ".join(problems))
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(self, "spread_levels", spread)
        object.__setattr__(self, "lock_timeout_s", float(self.lock_timeout_s))
        object.__setattr__(self, "lock_poll_s", float(self.lock_poll_s))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TeacherSettings":
        """Read ``tabpfn.*`` and ``engine.*`` with :func:`require`;
        ``tabpfn.project_token_budget`` and ``tabpfn.verified_request_limit_per_minute`` /
        ``_per_hour`` with :func:`optional` (a null budget refuses every uncached request; a null
        limit is not enforced and every dispatch record says so).

        ``tabpfn.check_usage_before_uncached_request``, ``tabpfn.estimate_cost_before_uncached_request``
        and ``tabpfn.cache_requests_and_responses`` must be true: the teacher always does these
        things, so any other value raises ConfigError instead of being silently ignored.
        Relative paths are resolved against the repository root (the directory of config.yaml).
        """

        def integer(key: str) -> int:
            value = require(config, key)
            if not _is_int(value):
                raise ConfigError(f"config.yaml key {key!r} must be an integer; got {value!r}")
            return value

        def seconds(key: str) -> float:
            value = require(config, key)
            if not _is_real(value):
                raise ConfigError(f"config.yaml key {key!r} must be a number of seconds; got {value!r}")
            return float(value)

        def text(key: str) -> str:
            value = require(config, key)
            if not isinstance(value, str):
                raise ConfigError(f"config.yaml key {key!r} must be a string; got {value!r}")
            return value

        def numbers(key: str) -> tuple[float, ...]:
            value = require(config, key)
            if not isinstance(value, list) or not all(_is_real(item) for item in value):
                raise ConfigError(f"config.yaml key {key!r} must be a list of numbers; got {value!r}")
            return tuple(float(item) for item in value)

        def path(key: str) -> Path:
            value = Path(text(key))
            return value if value.is_absolute() else _PROJECT_ROOT / value

        def optional_integer(key: str) -> int | None:
            value = optional(config, key)
            if value is not None and not _is_int(value):
                raise ConfigError(f"config.yaml key {key!r} must be null or an integer; got {value!r}")
            return value

        for key, behaviour in _ALWAYS_ON_SWITCHES:
            value = require(config, key)
            if value is not True:
                raise ConfigError(
                    f"config.yaml key {key!r} must be true; got {value!r}. The teacher always {behaviour}; "
                    f"this cannot be switched off"
                )
        spread = numbers("engine.spread_levels")
        if len(spread) != 2:
            raise ConfigError(f"config.yaml key 'engine.spread_levels' must hold exactly two levels; got {list(spread)}")
        try:
            return cls(
                client_version=text("tabpfn.client_version"),
                model_path=text("tabpfn.model_path"),
                n_estimators=integer("tabpfn.n_estimators"),
                random_state=integer("tabpfn.random_state"),
                quantile_levels=numbers("engine.quantile_levels"),
                spread_levels=(spread[0], spread[1]),
                multi_outcome_mode=text("engine.multi_outcome_mode"),
                cache_dir=path("tabpfn.cache_dir"),
                call_log=path("tabpfn.call_log"),
                max_tokens_per_request=integer("tabpfn.max_tokens_per_request"),
                project_token_budget=optional_integer("tabpfn.project_token_budget"),
                max_query_rows_per_request=integer("tabpfn.max_query_rows_per_request"),
                lock_timeout_s=seconds("tabpfn.lock_timeout_s"),
                lock_poll_s=seconds("tabpfn.lock_poll_s"),
                verified_request_limit_per_minute=optional_integer("tabpfn.verified_request_limit_per_minute"),
                verified_request_limit_per_hour=optional_integer("tabpfn.verified_request_limit_per_hour"),
            )
        except ValueError as exc:
            raise ConfigError(f"config.yaml tabpfn/engine settings are invalid: {exc}") from exc


# --------------------------------------------------------------------------------------
# Request description and key
# --------------------------------------------------------------------------------------


def _tabpfn_config(settings: TeacherSettings, categorical_indices: Sequence[int]) -> dict[str, Any]:
    """The tabpfn_config the provider sends (compared with the client's own before any call)."""
    return {
        "categorical_features_indices": [int(index) for index in categorical_indices],
        "ignore_pretraining_limits": False,
        "model_path": settings.model_path,
        "n_estimators": settings.n_estimators,
        "random_state": settings.random_state,
    }


def _request_description(settings: TeacherSettings, categorical_indices: Sequence[int], base_url: str) -> dict[str, Any]:
    """Everything keyed besides the arrays. Column names are NOT keyed: the provider uploads
    numpy arrays, so the server never sees them (they are kept in the cache metadata)."""
    return {
        "schema": _KEY_SCHEMA,
        "base_url": base_url,
        "client_version": settings.client_version,
        "task": TASK,
        "tabpfn_config": _tabpfn_config(settings, categorical_indices),
        "request_flags": dict(_PINNED_REQUEST_FLAGS),
        "predict_params": {"output_type": OUTPUT_TYPE, "quantiles": list(settings.quantile_levels)},
    }


def _float64(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.dtype.itemsize != 8:
        raise TypeError(f"{name} must be a float64 array; got dtype {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D; got shape {array.shape}")
    return array


def _check_categorical(n_features: int, categorical_indices: Sequence[int]) -> tuple[int, ...]:
    indices = tuple(categorical_indices)
    if not all(_is_int(index) for index in indices):
        raise ValueError(f"categorical_indices must be integers; got {list(indices)}")
    if any(b <= a for a, b in zip(indices, indices[1:])) or any(not 0 <= i < n_features for i in indices):
        raise ValueError(f"categorical_indices must be strictly increasing inside [0, {n_features}); got {list(indices)}")
    return indices


def _check_base_url(base_url: Any) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f"base_url must be a non-empty string; got {base_url!r}")
    return base_url


def _check_request_arrays(x_train: Any, y_train: Any, x_query: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = _float64("x_train", x_train, 2)
    y_train = _float64("y_train", y_train, 1)
    x_query = _float64("x_query", x_query, 2)
    if y_train.shape != (x_train.shape[0],):
        raise ValueError(f"y_train must have shape ({x_train.shape[0]},); got {y_train.shape}")
    if x_query.shape[1] != x_train.shape[1]:
        raise ValueError(f"x_query must have {x_train.shape[1]} columns like x_train; got {x_query.shape}")
    return x_train, y_train, x_query


def request_key(
    *,
    settings: TeacherSettings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    categorical_indices: Sequence[int],
    base_url: str,
) -> str:
    """sha256 hex key of one provider request (recipe of proof B_students_eval_contract/p09_hash).

    Args:
        x_train:             float64 (n, F), the training features exactly as sent.
        y_train:             float64 (n,), the training target exactly as sent.
        x_query:             float64 (m, F), the query rows exactly as sent (including a padding row).
        categorical_indices: strictly increasing column indices declared categorical.
        base_url:            the provider URL the request goes to (TabPFNProvider.base_url()).

    Keyed: key schema version (v2), base URL, client version, task, the tabpfn_config sent
    (model_path, n_estimators, random_state, categorical indices, ignore_pretraining_limits),
    the pinned request flags, predict params (output_type "main" and the exact quantile list, in
    order) and canonical digests of x_train, y_train and x_query (dtype, shape, C-order
    little-endian bytes, -0.0 folded to +0.0). Not keyed: column names (never sent), paths,
    budgets, rate limits, spread levels and the multi-outcome mode (visible in the arrays).
    """
    x_train, y_train, x_query = _check_request_arrays(x_train, y_train, x_query)
    indices = _check_categorical(x_train.shape[1], categorical_indices)
    base_url = _check_base_url(base_url)
    digest = hashlib.sha256()
    digest.update(_KEY_SCHEMA.encode("utf-8") + b"\x00")
    for tag, array in (("x_train", x_train), ("y_train", y_train), ("x_query", x_query)):
        update_digest(digest, array, tag)
    digest.update(canonical_json(_request_description(settings, indices, base_url)).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Pure multi-outcome helpers
# --------------------------------------------------------------------------------------


def constant_outcomes(residuals: np.ndarray) -> np.ndarray:
    """bool (K,): True for an outcome whose training residuals (n, K), n >= 1, are all identical (exact)."""
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise ValueError(f"residuals must have shape (n >= 1, K); got {residuals.shape}")
    return np.all(residuals == residuals[0:1, :], axis=0)


def _check_active(active: Sequence[int], n_outcomes: int | None = None) -> tuple[int, ...]:
    active = tuple(active)
    if not active or not all(_is_int(index) for index in active):
        raise ValueError(f"active must be a non-empty sequence of outcome indices; got {list(active)}")
    if any(b <= a for a, b in zip(active, active[1:])) or active[0] < 0:
        raise ValueError(f"active outcome indices must be strictly increasing and >= 0; got {list(active)}")
    if n_outcomes is not None and active[-1] >= n_outcomes:
        raise ValueError(f"active outcome index {active[-1]} is out of range for K={n_outcomes}")
    return active


def stack_query(query: np.ndarray, active: Sequence[int]) -> np.ndarray:
    """Expand rows for the stacked request.

    Args:
        query:  (m, F) float64 feature rows.
        active: A >= 1 strictly increasing outcome indices.

    Returns:
        (A*m, F+1) outcome-major (rows [j*m, (j+1)*m) carry outcome index active[j] in the last
        column) when A >= 2; a contiguous copy of ``query`` (m, F) when A == 1.
    """
    query = np.asarray(query, dtype=np.float64)
    if query.ndim != 2:
        raise ValueError(f"query must have shape (m, F); got {query.shape}")
    active = _check_active(active)
    if len(active) == 1:
        return np.array(query, dtype=np.float64, order="C", copy=True)
    rows = query.shape[0]
    blocks = [np.column_stack([query, np.full(rows, float(index))]) for index in active]
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


def stack_training(
    features: np.ndarray, residuals: np.ndarray, active: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long-format training request for the "stacked" mode.

    Args:
        features:  (n, F) float64.
        residuals: (n, K) float64.
        active:    A >= 1 strictly increasing indices of non-constant outcomes.

    Returns:
        x_long: (A*n, F+1), or (n, F) when A == 1; rows as in :func:`stack_query`.
        y_long: (A*n,) standardized residual (r[:, k] - mean_k) / std_k of each row's outcome.
        mean:   (A,) training mean of each active outcome.
        std:    (A,) training standard deviation (ddof 0) of each active outcome; > 0.
    """
    features = np.asarray(features, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    if features.ndim != 2 or residuals.ndim != 2 or residuals.shape[0] != features.shape[0]:
        raise ValueError(f"features (n, F) and residuals (n, K) must share n; got {features.shape}, {residuals.shape}")
    active = _check_active(active, residuals.shape[1])
    selected = residuals[:, list(active)]
    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    if not np.all(std > 0.0) or not np.all(np.isfinite(std)):
        raise ValueError(f"every active outcome needs a finite positive standard deviation; got {std.tolist()}")
    standardized = (selected - mean) / std
    if not np.all(np.isfinite(standardized)):
        raise ValueError("standardized residuals are not finite")
    y_long = np.ascontiguousarray(standardized.T.reshape(-1))
    return stack_query(features, active), y_long, mean, std


def unstack_rows(values: np.ndarray, n_rows: int, n_active: int) -> np.ndarray:
    """Outcome-major long rows back to per-outcome columns: (A*n, ...) -> (n, A, ...)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != n_rows * n_active:
        raise ValueError(f"values must have {n_rows} x {n_active} rows; got shape {values.shape}")
    return np.ascontiguousarray(values.reshape((n_active, n_rows) + values.shape[1:]).swapaxes(0, 1))


def destandardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """values (n, A, ...) in standardized units -> values * std + mean, per outcome along axis 1."""
    values = np.asarray(values, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.ndim != 1 or std.shape != mean.shape or values.ndim < 2 or values.shape[1] != mean.shape[0]:
        raise ValueError(f"values (n, A, ...) need mean and std of shape (A,); got {values.shape}, {mean.shape}, {std.shape}")
    shape = (1, mean.shape[0]) + (1,) * (values.ndim - 2)
    return values * std.reshape(shape) + mean.reshape(shape)


def fill_outcomes(active_values: np.ndarray, active: Sequence[int], constants: np.ndarray) -> np.ndarray:
    """Place active-outcome predictions into the full outcome order.

    Args:
        active_values: (m, A, ...) predictions of the active outcomes, in ``active`` order.
        active:        A outcome indices (may be empty).
        constants:     (K,) value used for every outcome not in ``active`` (its constant residual).

    Returns:
        (m, K, ...) with column active[j] = active_values[:, j] and every other column = its constant.
    """
    constants = np.asarray(constants, dtype=np.float64)
    active_values = np.asarray(active_values, dtype=np.float64)
    active = tuple(active)
    if constants.ndim != 1 or active_values.ndim < 2 or active_values.shape[1] != len(active):
        raise ValueError(f"active_values (m, {len(active)}, ...) and constants (K,) do not fit; got {active_values.shape}")
    if active:
        _check_active(active, constants.shape[0])
    trailing = active_values.shape[2:]
    out = np.empty((active_values.shape[0], constants.shape[0]) + trailing, dtype=np.float64)
    out[...] = constants.reshape((1, constants.shape[0]) + (1,) * len(trailing))
    if active:
        out[:, list(active)] = active_values
    return out


# --------------------------------------------------------------------------------------
# Provider-response and gate rules (pure)
# --------------------------------------------------------------------------------------


def parse_main_output(raw: Any, *, n_rows: int, quantile_levels: Sequence[float]) -> dict[str, np.ndarray]:
    """Validate the dict returned by ``predict(X, output_type="main", quantiles=levels)``.

    The pinned client returns mean, median and mode as (n,) and quantiles as the server's
    "one list for each quantile" array (Q, n) (estimator.py ``_FULL_OUTPUT_ROW_AXIS["quantiles"]
    == 1``). This function transposes that documented layout and checks exact keys, exact
    shapes, finiteness and non-decreasing quantiles along the level axis. Nothing is sorted or
    repaired: any violation raises ProviderResponseError. ``n_rows == Q`` raises ValueError,
    because (Q, n) and (n, Q) then have the same shape; the teacher never sends such a request
    (it pads it with one duplicated row).

    Returns:
        {"mean": (n,), "median": (n,), "mode": (n,), "quantiles": (n, Q)}, all float64.
    """
    q = len(quantile_levels)
    if n_rows < 1 or q < 1:
        raise ValueError(f"n_rows and the number of quantile levels must be >= 1; got {n_rows}, {q}")
    if n_rows == q:
        raise ValueError(f"n_rows == {q} quantile levels: the quantile layout cannot be told from its shape")
    if not isinstance(raw, Mapping):
        raise ProviderResponseError(f"'main' output must be a dict; got {type(raw).__name__}")
    keys = set(raw)
    if keys != set(MAIN_OUTPUT_KEYS):
        raise ProviderResponseError(
            f"'main' output keys {sorted(map(str, keys))} differ from the verified {list(MAIN_OUTPUT_KEYS)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for name in MAIN_OUTPUT_KEYS:
        try:
            array = np.asarray(raw[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ProviderResponseError(f"'main' output {name!r} is not numeric: {exc}") from None
        expected = (q, n_rows) if name == "quantiles" else (n_rows,)
        if array.shape != expected:
            raise ProviderResponseError(f"'main' output {name!r} has shape {array.shape}; the verified layout is {expected}")
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            raise ProviderResponseError(f"'main' output {name!r} has {bad} non-finite values")
        arrays[name] = array
    quantiles = np.ascontiguousarray(arrays["quantiles"].T)
    crossing = int(np.count_nonzero(np.any(np.diff(quantiles, axis=1) < 0.0, axis=1)))
    if crossing:
        raise ProviderResponseError(f"quantiles decrease along the level axis in {crossing} of {n_rows} rows")
    arrays["quantiles"] = quantiles
    return arrays


def median_quantile_gap(arrays: Mapping[str, np.ndarray], quantile_levels: Sequence[float]) -> float | None:
    """max |median - quantile(0.5)| over the rows of a validated response; None when 0.5 is not a level.

    Recorded, never enforced: whether the server's median equals its 0.5 quantile is unverified
    until the first real request.
    """
    levels = [float(level) for level in quantile_levels]
    if 0.5 not in levels:
        return None
    column = np.asarray(arrays["quantiles"])[:, levels.index(0.5)]
    return float(np.max(np.abs(np.asarray(arrays["median"]) - column)))


def remaining_tokens(usage: Mapping[str, Any]) -> int | None:
    """Tokens the account may still spend: min(limit - used) over the reported limit pairs.

    Pairs: (daily_token_limit, daily_tokens_used), (monthly_token_limit, monthly_tokens_used),
    (usage_limit, current_usage). A pair with both values null/absent is not reported and is
    skipped; usage_limit == -1 is the documented unlimited marker and is skipped. Returns None
    only when every reported pair is unlimited. Raises ProviderResponseError when no pair is
    reported or a reported value is not a non-negative integer.
    """
    remaining: list[int] = []
    reported = 0
    for limit_name, used_name in _LIMIT_PAIRS:
        limit, used = usage.get(limit_name), usage.get(used_name)
        if limit is None and used is None:
            continue
        reported += 1
        if not _is_int(limit) or not _is_int(used):
            raise ProviderResponseError(f"usage {limit_name}={limit!r} / {used_name}={used!r} are not integers")
        if limit_name == "usage_limit" and limit == _UNLIMITED_USAGE_LIMIT:
            continue
        if limit < 0 or used < 0:
            raise ProviderResponseError(f"usage {limit_name}={limit} / {used_name}={used} must be non-negative")
        remaining.append(limit - used)
    if not reported:
        raise ProviderResponseError("the usage response reports no token limit")
    return min(remaining) if remaining else None


def budget_refusal_reasons(
    *,
    quote_tokens: int,
    max_tokens_per_request: int,
    project_token_budget: int | None,
    tokens_already_quoted: int,
    remaining_account_tokens: int | None,
) -> list[str]:
    """Every reason to refuse an uncached request; [] means it may be sent. Integer comparisons are exact.

    Refuse when quote > max_tokens_per_request, when project_token_budget is None, when
    tokens_already_quoted + quote > project_token_budget, or when remaining_account_tokens
    (None = unlimited) < quote.
    """
    reasons: list[str] = []
    if quote_tokens > max_tokens_per_request:
        reasons.append(f"quote {quote_tokens} exceeds tabpfn.max_tokens_per_request {max_tokens_per_request}")
    if project_token_budget is None:
        reasons.append("tabpfn.project_token_budget is not set; the lead must set it before the first billable request")
    elif tokens_already_quoted + quote_tokens > project_token_budget:
        reasons.append(
            f"tokens already quoted for dispatched requests ({tokens_already_quoted}) plus this quote "
            f"({quote_tokens}) exceed tabpfn.project_token_budget {project_token_budget}"
        )
    if remaining_account_tokens is not None and remaining_account_tokens < quote_tokens:
        reasons.append(f"the account has {remaining_account_tokens} tokens left, below the quote {quote_tokens}")
    return reasons


def rate_limit_refusal_reasons(
    *,
    dispatch_times: Sequence[datetime],
    now: datetime,
    per_minute: int | None,
    per_hour: int | None,
) -> tuple[list[str], dict[str, int]]:
    """Reasons to refuse a dispatch at ``now`` given earlier dispatch times, and the window counts.

    A limit of None is not enforced. A dispatch is refused when the dispatches of the last 60 s
    (3600 s) plus this one would exceed tabpfn.verified_request_limit_per_minute (_per_hour).
    Dispatch times later than ``now`` count as recent.

    Returns:
        (reasons, {"dispatches_last_minute": int, "dispatches_last_hour": int}).
    """
    counts: dict[str, int] = {}
    reasons: list[str] = []
    for (name, window_s), limit, label in zip(_RATE_WINDOWS, (per_minute, per_hour), ("minute", "hour")):
        recent = sum(1 for when in dispatch_times if now - when < timedelta(seconds=window_s))
        counts[f"dispatches_last_{label}"] = recent
        if limit is not None and recent + 1 > limit:
            reasons.append(f"{recent} dispatches in the last {label} plus this one exceed tabpfn.{name} {limit}")
    return reasons, counts


def _json_safe(value: Any) -> Any:
    """Provider metadata as strict JSON: non-finite floats become {"non_finite": repr}, unknown types str."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {"non_finite": repr(float(value))}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _guarded(read: Callable[[], Any]) -> Any:
    """``read()``, or {"error": text} when it raises: metadata read after a billed call must never
    cost the response itself."""
    try:
        return read()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# Provider (the only code that imports tabpfn_client)
# --------------------------------------------------------------------------------------


class TabPFNProvider:
    """Hosted TabPFN regressor behind the pinned tabpfn-client.

    ``connect``, ``usage`` and ``quote`` are free; ``base_url``, ``outgoing_config`` and
    ``validate_request`` are offline. ``predict_raw`` (and ``predict_main``, which validates its
    answer) is billable (at least 10,000 tokens per call on v3.5, E7 p3) and is only ever called
    by the teacher after its gate.

    Attributes:
        prediction_calls_started: number of billable calls that reached the network stage.
        last_phase: stage of the latest billable call ("validate_request", "fit", "predict",
            "received", then "validate_response" and "done" in predict_main), so a caller can
            log where a failure happened.
    """

    def __init__(self, settings: TeacherSettings) -> None:
        self.settings = settings
        self.prediction_calls_started = 0
        self.last_phase: str | None = None
        self._connected = False

    def connect(self) -> None:
        """Check the installed client version, then authenticate with ``prior_labs.connect()``."""
        import tabpfn_client

        if tabpfn_client.__version__ != self.settings.client_version:
            raise RuntimeError(
                f"installed tabpfn-client {tabpfn_client.__version__} differs from tabpfn.client_version "
                f"{self.settings.client_version}"
            )
        try:
            import prior_labs
        except ModuleNotFoundError as exc:
            raise RuntimeError("prior_labs.py (repository root) is not importable; run from the repository root") from exc
        prior_labs.connect()
        self._connected = True

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("TabPFNProvider.connect() must succeed before any provider call")

    def base_url(self) -> str:
        """The URL the pinned client sends requests to (offline): ``ServiceClient.base_url``, which
        the client sets from TABPFN_CLIENT_API_URL at import. Keyed into every request."""
        from tabpfn_client.client import ServiceClient

        return _check_base_url(str(ServiceClient.base_url))

    def usage(self) -> dict[str, Any]:
        """Account usage (free read). Only allow-listed counters, limits and reset times:
        {field: int | str | None}, no identifiers."""
        self._require_connected()
        import tabpfn_client
        from tabpfn_client.client import ServiceClient

        raw = ServiceClient.get_api_usage(tabpfn_client.get_access_token())
        if not isinstance(raw, Mapping):
            raise ProviderResponseError(f"usage response must be a dict; got {type(raw).__name__}")
        usage: dict[str, Any] = {}
        for name in _USAGE_FIELDS:
            if name not in raw:
                continue
            value = raw[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, str))):
                raise ProviderResponseError(f"usage field {name} has unexpected type {type(value).__name__}")
            usage[name] = value
        return usage

    def _model_version(self) -> str:
        from tabpfn_client.utils import model_version_from_path

        version = model_version_from_path(self.settings.model_path, ()).value
        if not self.settings.model_path.startswith(f"{version}_"):
            raise ValueError(f"cannot derive the model version of model_path {self.settings.model_path!r} for a quote")
        return version

    def quote(self, n_train: int, n_query: int, n_features: int) -> dict[str, Any]:
        """Free cost estimate of one predict: ``estimate_cost`` on zero arrays (n_train, n_features)
        and (n_query, n_features) with the pinned model version and n_estimators.

        Returns {"estimated_cost": int tokens, "pricing_version": str, "inputs": resolved request}.
        Raises ProviderResponseError when the server resolves other inputs than were sent.
        """
        self._require_connected()
        for name, value, low in (("n_train", n_train, MIN_TRAIN_ROWS), ("n_query", n_query, 1), ("n_features", n_features, 1)):
            if not _is_int(value) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}; got {value!r}")
        from tabpfn_client import estimate_cost

        version = self._model_version()
        response = estimate_cost(
            np.zeros((n_train, n_features)),
            np.zeros((n_query, n_features)),
            model_version=version,
            n_estimators=self.settings.n_estimators,
        )
        cost = response.estimated_cost
        inputs = response.inputs.model_dump(mode="json", exclude_none=True)
        expected = {
            "train_rows": n_train,
            "test_rows": n_query,
            "raw_columns": n_features,
            "model_version": version,
            "n_estimators": self.settings.n_estimators,
        }
        mismatched = {name: inputs.get(name) for name, value in expected.items() if inputs.get(name) != value}
        if not _is_int(cost) or cost < 0 or mismatched:
            raise ProviderResponseError(f"quote {cost!r} resolved {mismatched} instead of {expected}")
        return {"estimated_cost": cost, "pricing_version": str(response.pricing_version), "inputs": inputs}

    def outgoing_config(self, categorical_indices: Sequence[int]) -> dict[str, Any]:
        """The tabpfn_config the pinned client would send for these settings (offline, no network)."""
        _, sent = self._build_regressor(categorical_indices)
        return sent

    def _build_regressor(self, categorical_indices: Sequence[int]) -> tuple[Any, dict[str, Any]]:
        from tabpfn_client import TabPFNRegressor
        from tabpfn_client.models import ApiMode

        expected = _tabpfn_config(self.settings, categorical_indices)
        regressor = TabPFNRegressor(
            model_path=expected["model_path"],
            n_estimators=expected["n_estimators"],
            random_state=expected["random_state"],
            categorical_features_indices=list(expected["categorical_features_indices"]),
            ignore_pretraining_limits=expected["ignore_pretraining_limits"],
            api_mode=ApiMode.SYNC,
            **dict(_PINNED_REQUEST_FLAGS),
        )
        # Private accessor of the pinned 0.10.1 client (E7 p1): the exact config it will send.
        sent = regressor._get_tabpfn_config().model_dump(mode="json", exclude_none=True)
        if sent != expected:
            raise RuntimeError(f"the client would send tabpfn_config {sent}, not the keyed settings {expected}")
        return regressor, sent

    def validate_request(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        categorical_indices: Sequence[int],
        quantile_levels: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[float, ...]]:
        """Every offline check of one request (no network): float64 arrays x_train (n, F) with
        n >= 2, y_train (n,), x_query (m, F) with m >= 1 and m != Q, all finite; valid categorical
        indices; quantile levels equal to the keyed ones; and the client's outgoing tabpfn_config
        equal to the keyed one.

        Returns:
            (x_train, y_train, x_query, categorical indices, quantile levels).
        """
        x_train, y_train, x_query = _check_request_arrays(x_train, y_train, x_query)
        indices = _check_categorical(x_train.shape[1], categorical_indices)
        if x_train.shape[0] < MIN_TRAIN_ROWS or x_query.shape[0] < 1:
            raise ValueError(f"need >= {MIN_TRAIN_ROWS} training rows and >= 1 query row; got {x_train.shape}, {x_query.shape}")
        if not (np.all(np.isfinite(x_train)) and np.all(np.isfinite(y_train)) and np.all(np.isfinite(x_query))):
            raise ValueError("request arrays must be finite")
        levels = tuple(float(level) for level in quantile_levels)
        if levels != self.settings.quantile_levels:
            raise ValueError(f"quantile_levels {list(levels)} differ from the keyed {list(self.settings.quantile_levels)}")
        if x_query.shape[0] == len(levels):
            raise ValueError(
                f"{x_query.shape[0]} query rows equal the number of quantile levels: the response layout would be "
                f"ambiguous; pad the query with one duplicated row"
            )
        self.outgoing_config(indices)
        return x_train, y_train, x_query, indices, levels

    def predict_raw(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        categorical_indices: Sequence[int],
        quantile_levels: Sequence[float],
    ) -> tuple[Any, dict[str, Any]]:
        """BILLABLE: one fit and one ``predict(output_type="main", quantiles=levels)``; the answer
        is returned exactly as the client produced it, NOT validated.

        Args:
            x_train, y_train, x_query, categorical_indices, quantile_levels: see :meth:`validate_request`.

        Returns:
            (raw, metadata): ``raw`` as returned by the client (documented: a dict with mean,
            median, mode (m,) and quantiles (Q, m)); ``metadata`` (strict JSON) holds the fitted
            model id, server timings, response metadata, the tabpfn_config sent, base URL, client
            version and wall-clock seconds. A metadata item that cannot be read is recorded as
            {"error": text} instead of raising, so the response is never lost.
        """
        self._require_connected()
        self.last_phase = "validate_request"
        x_train, y_train, x_query, indices, levels = self.validate_request(
            x_train, y_train, x_query, categorical_indices=categorical_indices, quantile_levels=quantile_levels
        )
        import tabpfn_client
        from tabpfn_client.client import ServiceClient

        regressor, sent = self._build_regressor(indices)
        self.prediction_calls_started += 1
        self.last_phase = "fit"
        start = time.perf_counter()
        regressor.fit(x_train, y_train)
        fit_s = time.perf_counter() - start
        self.last_phase = "predict"
        start = time.perf_counter()
        raw = regressor.predict(x_query, output_type=OUTPUT_TYPE, quantiles=list(levels))
        predict_s = time.perf_counter() - start
        self.last_phase = "received"
        metadata = _json_safe(
            {
                "model_id": _guarded(lambda: str(regressor.model_id_)),
                "timings": _guarded(lambda: _json_safe(regressor.get_timings())),
                # Private attribute of the pinned 0.10.1 client: the server's response metadata (E7).
                "response_metadata": _guarded(lambda: _json_safe(dict(regressor._last_meta))),
                "tabpfn_config_sent": sent,
                "api_mode": "sync",
                "base_url": _guarded(lambda: str(ServiceClient.base_url)),
                "client_version": _guarded(lambda: str(tabpfn_client.__version__)),
                "wall_s": {"fit": fit_s, "predict": predict_s},
            }
        )
        return raw, metadata

    def predict_main(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        categorical_indices: Sequence[int],
        quantile_levels: Sequence[float],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """BILLABLE: :meth:`predict_raw` followed by :func:`parse_main_output`.

        Returns ({"mean": (m,), "median": (m,), "mode": (m,), "quantiles": (m, Q)}, metadata).
        The teacher does not use this method: it stores the raw answer before validating it.
        """
        raw, metadata = self.predict_raw(
            x_train, y_train, x_query, categorical_indices=categorical_indices, quantile_levels=quantile_levels
        )
        self.last_phase = "validate_response"
        arrays = parse_main_output(raw, n_rows=int(np.asarray(x_query).shape[0]), quantile_levels=quantile_levels)
        self.last_phase = "done"
        return arrays, metadata


# --------------------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Request:
    """One provider request as the teacher sends it (one chunk of query rows)."""

    key: str
    x_train: np.ndarray  # (n, F) exactly as sent
    y_train: np.ndarray  # (n,) exactly as sent
    x_sent: np.ndarray  # (m + padded_rows, F) exactly as sent
    kept_rows: int  # m: answer rows returned to the caller
    categorical_indices: tuple[int, ...]
    column_names: tuple[str, ...]
    base_url: str
    context: dict[str, Any]
    chunk: dict[str, int]
    base: dict[str, Any]  # fields of every call-log record of this request

    @property
    def padded_rows(self) -> int:
        return int(self.x_sent.shape[0] - self.kept_rows)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"x_train": self.x_train, "y_train": self.y_train, "x_query": self.x_sent}


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class TabPFNTeacher:
    """Residual model backed by the hosted TabPFN regressor; implements ResidualModel.

    Attributes:
        last_request_hash:   key of the last request behind the latest prediction (None when
                             every outcome was constant or nothing was predicted yet).
        last_request_hashes: every key used by the latest prediction, in request order (one per
                             chunk and per request of the per_outcome mode).
    """

    def __init__(
        self,
        layout: FeatureLayout,
        settings: TeacherSettings,
        *,
        provider: TabPFNProvider,
        cache: RequestCache,
        call_log: CallLog,
        clock: Callable[[], datetime],
    ) -> None:
        if layout.n_outcomes < 1:
            raise ValueError("the layout must declare at least one outcome")
        if STACKED_OUTCOME_COLUMN in layout.names:
            raise ValueError(f"feature name {STACKED_OUTCOME_COLUMN!r} is reserved for the stacked outcome column")
        differing = [name for name in _KEYED_SETTINGS if getattr(provider.settings, name) != getattr(settings, name)]
        if differing:
            raise ValueError(f"provider and teacher settings differ in keyed fields {differing}")
        if not _same_path(cache.directory, settings.cache_dir) or not _same_path(call_log.path, settings.call_log):
            raise ValueError(
                f"cache directory {cache.directory} and call log {call_log.path} must be the configured "
                f"{settings.cache_dir} and {settings.call_log}"
            )
        lock = CacheLock(cache.lock_path, timeout_s=settings.lock_timeout_s, poll_s=settings.lock_poll_s)
        if call_log.lock.identity != lock.identity:
            raise ValueError(f"the call log must lock {cache.lock_path}, the cache's lock file; it locks {call_log.lock.path}")
        self.layout = layout
        self.settings = settings
        self.provider = provider
        self.cache = cache
        self.call_log = call_log
        self._lock = lock
        self._clock = clock
        self._features: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self.last_request_hash: str | None = None
        self.last_request_hashes: tuple[str, ...] = ()

    @classmethod
    def from_settings(
        cls,
        layout: FeatureLayout,
        settings: TeacherSettings,
        *,
        provider: TabPFNProvider,
        clock: Callable[[], datetime],
    ) -> "TabPFNTeacher":
        """A teacher with the configured cache, call log and the cache's lock (settings.cache_dir/.lock)."""
        cache = RequestCache(settings.cache_dir)
        lock = CacheLock(cache.lock_path, timeout_s=settings.lock_timeout_s, poll_s=settings.lock_poll_s)
        call_log = CallLog(settings.call_log, clock=clock, lock=lock)
        return cls(layout, settings, provider=provider, cache=cache, call_log=call_log, clock=clock)

    # ----- ResidualModel -------------------------------------------------------------

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "TabPFNTeacher":
        """Store the context: features (n, F) in layout order, residuals (n, K) in outcome order.

        No network call. Raises ValueError on a shape mismatch, n < 2, non-finite values or a
        categorical column that does not hold valid level indices.
        """
        features = self._check_features("features", features)
        try:
            residuals = np.asarray(residuals, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"residuals must be numeric: {exc}") from None
        n, k = features.shape[0], self.layout.n_outcomes
        if residuals.shape != (n, k):
            raise ValueError(f"residuals must have shape ({n}, {k}) in outcome order {self.layout.outcome_names}; got {residuals.shape}")
        if n < MIN_TRAIN_ROWS:
            raise ValueError(f"the teacher needs at least {MIN_TRAIN_ROWS} context rows; got {n}")
        if not np.all(np.isfinite(residuals)):
            raise ValueError("residuals contain non-finite values")
        self._features = features.copy()
        self._residuals = np.ascontiguousarray(residuals).copy()
        self.last_request_hash = None
        self.last_request_hashes = ()
        return self

    def predict_residuals(self, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(median residual, predictive spread), each float64 (m, K), from the same request(s)
        as :meth:`predict_residual_distribution`."""
        distribution = self.predict_residual_distribution(query_features)
        return distribution.median, distribution.spread

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """Residual distribution for query rows (m, F): mean/median/spread (m, K), quantiles (m, K, Q).

        "stacked" mode makes one request per chunk of at most tabpfn.max_query_rows_per_request
        stacked rows; "per_outcome" mode does the same for each non-constant outcome. Each
        request is served from the cache when stored. Raises RuntimeError before fit(),
        ValueError on invalid rows, BudgetError when the gate refuses, UnresolvedDispatchError /
        LedgerMissingError / CacheLockedError when a request cannot safely be decided,
        ProviderResponseError on a contract violation and CacheCorruptError on a damaged entry.
        """
        if self._features is None or self._residuals is None:
            raise RuntimeError("fit() must be called before predicting")
        self.last_request_hash = None
        self.last_request_hashes = ()
        query = self._check_features("query_features", query_features)
        if query.shape[0] < 1:
            raise ValueError("query_features needs at least one row")
        levels = np.asarray(self.settings.quantile_levels, dtype=np.float64)
        active = tuple(int(index) for index in np.flatnonzero(~constant_outcomes(self._residuals)))
        keys: list[str] = []
        if active:
            mean_a, median_a, quantiles_a = self._predict_active(query, active, keys)
        else:
            mean_a = median_a = np.empty((query.shape[0], 0))
            quantiles_a = np.empty((query.shape[0], 0, levels.size))
        constants = self._residuals[0]
        distribution = ResidualDistribution.from_quantiles(
            fill_outcomes(mean_a, active, constants),
            fill_outcomes(median_a, active, constants),
            levels,
            fill_outcomes(quantiles_a, active, constants),
            self.settings.spread_levels,
        )
        self.last_request_hashes = tuple(keys)
        self.last_request_hash = keys[-1] if keys else None
        return distribution

    # ----- operator decisions ----------------------------------------------------------

    def resolve_unresolved(self, key: str, *, action: str) -> dict[str, Any]:
        """Resolve a dispatch of ``key`` that has no completed or abandoned record. No network call.

        Args:
            key:    the request key named by UnresolvedDispatchError.
            action: "adopt": validate the response stored in unvalidated/<key>.npz now (or take
                    the validated <key>.npz when it already exists); when valid, cache it and log
                    "completed" with adopted true. An invalid or missing response logs
                    "adopt_rejected" and raises; the dispatch stays unresolved.
                    "abandon": log "abandoned" (a new dispatch of the key is allowed afterwards)
                    and move any unvalidated response to unvalidated/abandoned/ as evidence.

        Returns:
            The call-log record written for the decision.

        Raises:
            ValueError for an unknown action or a key without an unresolved dispatch;
            ProviderResponseError / CacheCorruptError / ValueError when adoption is rejected.
        """
        if action not in RESOLVE_ACTIONS:
            raise ValueError(f"action must be one of {RESOLVE_ACTIONS}; got {action!r}")
        self.cache.paths(key)  # validates the key format
        with self._lock:
            self._require_ledger()
            dispatch = unresolved_dispatches(self.call_log.read()).get(key)
            if dispatch is None:
                raise ValueError(f"the call log {self.call_log.path} has no unresolved dispatch of request {key}")
            base = {
                name: dispatch.get(name)
                for name in ("key", "skill", "mode", "outcomes", "n_train", "n_query", "n_features", "chunk", "padded_rows")
            }
            base.update({"operator_action": action, "dispatched_utc": dispatch.get("timestamp_utc")})
            quote_tokens = dispatch.get(QUOTE_FIELD)
            if action == "abandon":
                moved = self.cache.abandon_unvalidated(key, self._now().strftime("%Y%m%dT%H%M%S%fZ"))
                record = {
                    **base,
                    "event": "abandoned",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: quote_tokens,
                    "unvalidated_moved_to": None if moved is None else str(moved),
                }
                self.call_log.append(record)
                return record
            return self._adopt(key, base, quote_tokens)

    # ----- internals -----------------------------------------------------------------

    def _now(self) -> datetime:
        now = self._clock()
        utc_timestamp(lambda: now)  # raises unless timezone-aware UTC
        return now

    def _check_features(self, name: str, values: np.ndarray) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric: {exc}") from None
        width = self.layout.n_features
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must have shape (n, {width}) in layout order {self.layout.names}; got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        for column, levels in self.layout.categorical:
            if not np.all(np.isin(array[:, column], np.arange(len(levels), dtype=np.float64))):
                raise ValueError(
                    f"{name} column {column} ({self.layout.names[column]!r}) must hold level indices 0..{len(levels) - 1}"
                )
        return np.ascontiguousarray(array)

    def _predict_active(
        self, query: np.ndarray, active: tuple[int, ...], keys: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mean (m, A), median (m, A), quantiles (m, A, Q)) of the active outcomes."""
        if self._features is None or self._residuals is None:
            raise RuntimeError("fit() must be called before predicting")
        rows = query.shape[0]
        if self.settings.multi_outcome_mode == "stacked":
            x_train, y_train, mean, std = stack_training(self._features, self._residuals, active)
            extra = len(active) > 1
            arrays = self._request_rows(
                x_train,
                y_train,
                stack_query(query, active),
                categorical_indices=self.layout.categorical_indices + ((self.layout.n_features,) if extra else ()),
                column_names=self.layout.names + ((STACKED_OUTCOME_COLUMN,) if extra else ()),
                context=self._context(active, {"mean": mean.tolist(), "std": std.tolist(), "ddof": 0}),
                keys=keys,
            )
            return tuple(  # type: ignore[return-value]
                destandardize(unstack_rows(arrays[name], rows, len(active)), mean, std)
                for name in ("mean", "median", "quantiles")
            )
        parts = [
            self._request_rows(
                self._features,
                np.ascontiguousarray(self._residuals[:, index]),
                query,
                categorical_indices=self.layout.categorical_indices,
                column_names=self.layout.names,
                context=self._context((index,), None),
                keys=keys,
            )
            for index in active
        ]
        return tuple(np.stack([part[name] for part in parts], axis=1) for name in ("mean", "median", "quantiles"))  # type: ignore[return-value]

    def _context(self, active: tuple[int, ...], standardization: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "skill": self.layout.skill,
            "mode": self.settings.multi_outcome_mode,
            "outcomes": [self.layout.outcome_names[index] for index in active],
            "outcome_indices": list(active),
            "standardization": standardization,
        }

    def _request_rows(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        categorical_indices: tuple[int, ...],
        column_names: tuple[str, ...],
        context: dict[str, Any],
        keys: list[str],
    ) -> dict[str, np.ndarray]:
        """Send x_query (rows, F) in consecutive chunks of at most max_query_rows_per_request rows,
        all with the same training context; returns the answers concatenated in row order:
        {"mean": (rows,), "median": (rows,), "mode": (rows,), "quantiles": (rows, Q)}."""
        cap = self.settings.max_query_rows_per_request
        total = int(x_query.shape[0])
        bounds = [(start, min(start + cap, total)) for start in range(0, total, cap)]
        parts = [
            self._request(
                x_train,
                y_train,
                np.ascontiguousarray(x_query[start:stop]),
                categorical_indices=categorical_indices,
                column_names=column_names,
                context=context,
                chunk={"index": index, "count": len(bounds), "first_row": start, "stop_row": stop, "total_rows": total},
                keys=keys,
            )
            for index, (start, stop) in enumerate(bounds)
        ]
        return {name: np.concatenate([part[name] for part in parts], axis=0) for name in MAIN_OUTPUT_KEYS}

    def _request(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        categorical_indices: tuple[int, ...],
        column_names: tuple[str, ...],
        context: dict[str, Any],
        chunk: dict[str, int],
        keys: list[str],
    ) -> dict[str, np.ndarray]:
        """Serve one request from the cache, or gate, send, store and log it. Returns the answer
        rows of ``x_query`` (m, F): {"mean": (m,), "median": (m,), "mode": (m,), "quantiles": (m, Q)}."""
        rows = int(x_query.shape[0])
        if len(column_names) != x_train.shape[1]:
            raise ValueError(f"{len(column_names)} column names for {x_train.shape[1]} columns")
        # m == Q would make the (Q, m) quantile layout indistinguishable from (m, Q): send one more row.
        if rows == len(self.settings.quantile_levels):
            x_sent = np.ascontiguousarray(np.concatenate([x_query, x_query[-1:]], axis=0))
        else:
            x_sent = x_query
        base_url = self.provider.base_url()
        key = request_key(
            settings=self.settings,
            x_train=x_train,
            y_train=y_train,
            x_query=x_sent,
            categorical_indices=categorical_indices,
            base_url=base_url,
        )
        base = {
            "key": key,
            "skill": context["skill"],
            "mode": context["mode"],
            "outcomes": context["outcomes"],
            "n_train": int(x_train.shape[0]),
            "n_query": int(x_sent.shape[0]),
            "n_features": int(x_train.shape[1]),
            "chunk": chunk,
            "padded_rows": int(x_sent.shape[0] - rows),
        }
        request = _Request(
            key=key,
            x_train=x_train,
            y_train=y_train,
            x_sent=x_sent,
            kept_rows=rows,
            categorical_indices=tuple(categorical_indices),
            column_names=tuple(column_names),
            base_url=base_url,
            context=context,
            chunk=chunk,
            base=base,
        )
        entry = self.cache.get(key)
        arrays = self._serve_cached(request, entry, after_lock_wait=False) if entry is not None else self._uncached(request)
        keys.append(key)
        return {name: np.ascontiguousarray(array[:rows]) for name, array in arrays.items()}

    def _require_ledger(self) -> None:
        """The call log is the project's spend ledger: it must exist once the cache holds anything."""
        if not self.call_log.exists() and self.cache.has_entries():
            raise LedgerMissingError(
                f"the call log {self.call_log.path} does not exist, but the cache {self.cache.directory} holds "
                f"entries: the project budget cannot be counted. Restore the call log (for example from a backup) "
                f"before any further teacher request. Do not create an empty one: that would restart the "
                f"project budget at zero."
            )

    def _append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._require_ledger()
            self.call_log.append(record)

    def _serve_cached(self, request: _Request, entry: CachedEntry, *, after_lock_wait: bool) -> dict[str, np.ndarray]:
        arrays = self._check_cached(entry, request)
        record = {**request.base, "event": "cache_hit", "cache_hit": True, DISPATCH_FLAG: False, QUOTE_FIELD: None}
        if after_lock_wait:
            record["found_after_lock_wait"] = True
        self._append(record)
        return arrays

    def _request_meta(self, request: _Request) -> dict[str, Any]:
        return {
            "description": _request_description(self.settings, request.categorical_indices, request.base_url),
            "categorical_indices": list(request.categorical_indices),
            "base_url": request.base_url,
            "column_names": list(request.column_names),
            "n_train": request.base["n_train"],
            "n_query": request.base["n_query"],
            "n_features": request.base["n_features"],
            "padded_rows": request.padded_rows,
            "chunk": request.chunk,
            "teacher_context": request.context,
        }

    def _uncached(self, request: _Request) -> dict[str, np.ndarray]:
        """The whole uncached path under the cache lock; returns the answer to every sent row."""
        with self._lock:
            entry = self.cache.get(request.key)
            if entry is not None:  # another process stored it while this one waited for the lock
                return self._serve_cached(request, entry, after_lock_wait=True)
            self._require_ledger()
            records = self.call_log.read()
            dispatch = unresolved_dispatches(records).get(request.key)
            if dispatch is not None:
                raise UnresolvedDispatchError(self._unresolved_message(request.key, dispatch))
            self.provider.validate_request(
                request.x_train,
                request.y_train,
                request.x_sent,
                categorical_indices=request.categorical_indices,
                quantile_levels=self.settings.quantile_levels,
            )
            self._require_writable(request)
            usage_before = self.provider.usage()
            quote = self.provider.quote(request.base["n_train"], request.base["n_query"], request.base["n_features"])
            tokens = quote["estimated_cost"]
            spent = tokens_quoted(records, self.call_log.path)
            rate_reasons, rate_counts = rate_limit_refusal_reasons(
                dispatch_times=dispatch_times(records, self.call_log.path),
                now=self._now(),
                per_minute=self.settings.verified_request_limit_per_minute,
                per_hour=self.settings.verified_request_limit_per_hour,
            )
            configured = (
                self.settings.verified_request_limit_per_minute is not None
                or self.settings.verified_request_limit_per_hour is not None
            )
            gate = {
                QUOTE_FIELD: tokens,
                "quote": quote,
                "usage_before": usage_before,
                "tokens_quoted_before": spent,
                "rate_limit_configured": configured,
            }
            if configured:
                gate["rate_limit"] = {
                    "per_minute": self.settings.verified_request_limit_per_minute,
                    "per_hour": self.settings.verified_request_limit_per_hour,
                    **rate_counts,
                }
            reasons = budget_refusal_reasons(
                quote_tokens=tokens,
                max_tokens_per_request=self.settings.max_tokens_per_request,
                project_token_budget=self.settings.project_token_budget,
                tokens_already_quoted=spent,
                remaining_account_tokens=remaining_tokens(usage_before),
            ) + rate_reasons
            if reasons:
                self.call_log.append(
                    {**request.base, **gate, "event": "refused", "cache_hit": False, DISPATCH_FLAG: False, "reasons": reasons}
                )
                raise BudgetError(f"request {request.key} refused before any billable call: " + "; ".join(reasons))
            self.call_log.append({**request.base, **gate, "event": "dispatch", "cache_hit": False, DISPATCH_FLAG: True})
            return self._dispatch(request, usage_before=usage_before, quote=quote)

    def _require_writable(self, request: _Request) -> None:
        """Refuse (BudgetError) unless the persistence settings load and the cache directory, its
        unvalidated folder and the call log's directory can each take a temporary file."""
        problems: list[str] = []
        for label, probe in (
            ("the cache directory or its unvalidated folder", self.cache.probe_writable),
            ("the call-log directory", self.call_log.probe_writable),
        ):
            try:
                probe()
            except Exception as exc:  # OSError, ConfigError (persistence.*), ValueError: all refuse the request
                problems.append(f"{label} cannot be written with the configured persistence settings ({_error_text(exc)})")
        if not problems:
            return
        message = f"request {request.key} refused before any billable call: " + "; ".join(problems)
        try:
            self.call_log.append(
                {**request.base, "event": "refused", "cache_hit": False, DISPATCH_FLAG: False, QUOTE_FIELD: None, "reasons": problems}
            )
        except Exception as exc:
            message += f" (the refusal could not be logged either: {_error_text(exc)})"
        raise BudgetError(message)

    def _unresolved_message(self, key: str, dispatch: Mapping[str, Any]) -> str:
        path = self.cache.unvalidated_path(key)
        stored = f"the response received is stored unvalidated at {path}" if path.exists() else "no response is stored"
        return (
            f"request {key} was dispatched at {dispatch.get('timestamp_utc')} (call log {self.call_log.path}) and has no "
            f"later completed or abandoned record, so it may already have been billed; {stored}. It is not sent again "
            f"automatically. Check the account usage, then call TabPFNTeacher.resolve_unresolved({key!r}, "
            f"action='adopt') to validate and cache the stored response without any network call, or "
            f"action='abandon' to allow a new billable dispatch of this request."
        )

    def _dispatch(self, request: _Request, *, usage_before: dict[str, Any], quote: dict[str, Any]) -> dict[str, np.ndarray]:
        """The billable call, then raw response to disk, validation and the validated entry."""
        tokens = quote["estimated_cost"]
        try:
            raw, metadata = self.provider.predict_raw(
                request.x_train,
                request.y_train,
                request.x_sent,
                categorical_indices=request.categorical_indices,
                quantile_levels=self.settings.quantile_levels,
            )
        except BaseException as exc:
            self.call_log.append(
                {
                    **request.base,
                    "event": "provider_error",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: tokens,
                    "possibly_billed": self.provider.last_phase in _BILLED_PHASES,
                    "phase": self.provider.last_phase,
                    "error": _error_text(exc),
                    "usage_after": self._usage_or_error(),
                }
            )
            raise
        request_meta = self._request_meta(request)
        received = {
            "key": request.key,
            "received_utc": _guarded(lambda: utc_timestamp(self._clock)),
            "quote": quote,
            "usage_before": usage_before,
            "provider": metadata,
        }
        unvalidated_path: Path | None = None
        unvalidated_error: str | None = None
        try:
            unvalidated_path = self.cache.put_unvalidated(
                request.key,
                request_arrays=request.arrays(),
                request_meta=request_meta,
                raw_response=raw,
                response_meta=received,
            )
        except Exception as exc:  # keep going: validation and the validated entry may still succeed
            unvalidated_error = _error_text(exc)
            self.call_log.append(
                {
                    **request.base,
                    "event": "unvalidated_write_failed",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: tokens,
                    "billed": True,
                    "error": unvalidated_error,
                }
            )
        usage_after = self._usage_or_error()
        try:
            arrays = parse_main_output(raw, n_rows=int(request.x_sent.shape[0]), quantile_levels=self.settings.quantile_levels)
        except ProviderResponseError as exc:
            self.call_log.append(
                {
                    **request.base,
                    "event": "response_rejected",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: tokens,
                    "billed": True,
                    "error": _error_text(exc),
                    "unvalidated_path": None if unvalidated_path is None else str(unvalidated_path),
                    "usage_after": usage_after,
                }
            )
            where = f"is kept unvalidated at {unvalidated_path}" if unvalidated_path else f"could not be stored ({unvalidated_error})"
            raise ProviderResponseError(
                f"{exc}. The billed response {where}; the request is not sent again automatically: inspect it, then "
                f"call TabPFNTeacher.resolve_unresolved({request.key!r}, action='adopt' or 'abandon')"
            ) from exc
        response_meta = {
            **received,
            "usage_after": usage_after,
            "usage_delta": _usage_delta(usage_before, usage_after),
        }
        self._store(
            key=request.key,
            base=request.base,
            request_arrays=request.arrays(),
            request_meta=request_meta,
            arrays=arrays,
            response_meta=response_meta,
            quote_tokens=tokens,
            adopted=False,
            extra={
                "usage_after": usage_after,
                "usage_delta": response_meta["usage_delta"],
                "timings": metadata.get("timings"),
                "wall_s": metadata.get("wall_s"),
                "unvalidated_write_error": unvalidated_error,
            },
        )
        return arrays

    def _store(
        self,
        *,
        key: str,
        base: Mapping[str, Any],
        request_arrays: dict[str, np.ndarray],
        request_meta: dict[str, Any],
        arrays: dict[str, np.ndarray],
        response_meta: dict[str, Any],
        quote_tokens: Any,
        adopted: bool,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write the validated entry, then delete the unvalidated file, then log "completed"."""
        gap = median_quantile_gap(arrays, self.settings.quantile_levels)
        response_meta = {**response_meta, "median_vs_q50_max_abs_diff": gap, "adopted": adopted}
        unvalidated = self.cache.unvalidated_path(key)
        try:
            sidecar_error = self.cache.put(
                key,
                request_arrays=request_arrays,
                request_meta=request_meta,
                response_arrays=arrays,
                response_meta=response_meta,
            )
        except BaseException as exc:
            self.call_log.append(
                {
                    **base,
                    "event": "cache_write_failed",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: quote_tokens,
                    "billed": True,
                    "error": _error_text(exc),
                    "unvalidated_path": str(unvalidated) if unvalidated.exists() else None,
                }
            )
            raise
        cleanup_error: str | None = None
        try:
            self.cache.remove_unvalidated(key)
        except OSError as exc:
            cleanup_error = _error_text(exc)
        record = {
            **base,
            **extra,
            "event": "completed",
            "cache_hit": False,
            DISPATCH_FLAG: False,
            QUOTE_FIELD: quote_tokens,
            "adopted": adopted,
            "median_vs_q50_max_abs_diff": gap,
            "sidecar_error": sidecar_error,
            "unvalidated_cleanup_error": cleanup_error,
        }
        self.call_log.append(record)
        return record

    def _adopt(self, key: str, base: dict[str, Any], quote_tokens: Any) -> dict[str, Any]:
        """Adopt the stored response of an unresolved dispatch (caller holds the lock)."""
        existing = self.cache.get(key)
        if existing is not None:  # the validated entry was written before the completed record
            cleanup_error: str | None = None
            try:
                self.cache.remove_unvalidated(key)
            except OSError as exc:
                cleanup_error = _error_text(exc)
            record = {
                **base,
                "event": "completed",
                "cache_hit": False,
                DISPATCH_FLAG: False,
                QUOTE_FIELD: quote_tokens,
                "adopted": True,
                "adopted_from": "validated_cache_entry",
                "median_vs_q50_max_abs_diff": existing.response_meta.get("median_vs_q50_max_abs_diff"),
                "unvalidated_cleanup_error": cleanup_error,
            }
            self.call_log.append(record)
            return record
        try:
            stored = self.cache.get_unvalidated(key)
            if stored is None:
                raise ValueError(
                    f"no response is stored for request {key} at {self.cache.unvalidated_path(key)}: nothing to adopt; "
                    f"use action='abandon'"
                )
            x_train, y_train, x_query = (stored.request_arrays.get(name) for name in ("x_train", "y_train", "x_query"))
            indices = stored.request_meta.get("categorical_indices")
            base_url = stored.request_meta.get("base_url")
            if x_train is None or y_train is None or x_query is None or not isinstance(indices, list):
                raise CacheCorruptError(f"{stored.path} does not hold the request it answers")
            recomputed = request_key(
                settings=self.settings,
                x_train=x_train,
                y_train=y_train,
                x_query=x_query,
                categorical_indices=indices,
                base_url=base_url,
            )
            if recomputed != key:
                raise ValueError(
                    f"the request stored in {stored.path} hashes to {recomputed} under the current settings, not {key}: "
                    f"restore the settings used at dispatch, or abandon"
                )
            arrays = parse_main_output(
                stored.raw_response, n_rows=int(x_query.shape[0]), quantile_levels=self.settings.quantile_levels
            )
        except (ValueError, TypeError, ProviderResponseError, CacheCorruptError) as exc:
            self.call_log.append(
                {
                    **base,
                    "event": "adopt_rejected",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: quote_tokens,
                    "error": _error_text(exc),
                }
            )
            raise
        return self._store(
            key=key,
            base=base,
            request_arrays={"x_train": x_train, "y_train": y_train, "x_query": x_query},
            request_meta=stored.request_meta,
            arrays=arrays,
            response_meta={**stored.response_meta, "usage_after": None, "adopted_utc": utc_timestamp(self._clock)},
            quote_tokens=quote_tokens,
            adopted=True,
            extra={"adopted_from": "unvalidated_response"},
        )

    def _check_cached(self, entry: CachedEntry, request: _Request) -> dict[str, np.ndarray]:
        for name, sent in request.arrays().items():
            stored = entry.request_arrays.get(name)
            if stored is None or stored.dtype != sent.dtype or not np.array_equal(stored, sent):
                raise CacheCorruptError(f"cache entry {entry.key} stores a different {name} than the request it is keyed by")
        rows, q = request.x_sent.shape[0], len(self.settings.quantile_levels)
        arrays: dict[str, np.ndarray] = {}
        for name in MAIN_OUTPUT_KEYS:
            array = entry.response_arrays.get(name)
            expected = (rows, q) if name == "quantiles" else (rows,)
            if array is None or array.dtype != np.float64 or array.shape != expected or not np.all(np.isfinite(array)):
                raise CacheCorruptError(f"cache entry {entry.key} response {name!r} is not finite float64 {expected}")
            arrays[name] = array
        if np.any(np.diff(arrays["quantiles"], axis=1) < 0.0):
            raise CacheCorruptError(f"cache entry {entry.key} holds decreasing quantiles")
        return arrays

    def _usage_or_error(self) -> dict[str, Any]:
        try:
            return self.provider.usage()
        except Exception as exc:  # recorded, never raised: the call outcome matters more
            return {"error": _error_text(exc)}


def _usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    """after - before for the usage counters both reads report as integers."""
    return {
        name: after[name] - before[name]
        for name in _USAGE_COUNTERS
        if _is_int(before.get(name)) and _is_int(after.get(name))
    }
