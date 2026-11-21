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

      stacked      ONE request for all non-constant ("active") outcomes, in long format with
                   outcome-major blocks: block j repeats every training row with a final
                   categorical column holding the outcome index k = active[j], and its target
                   is (r[:, k] - mean_k) / std_k (training mean, std with ddof 0). Queries are
                   expanded the same way; mean, median and every quantile are de-standardized
                   (value * std_k + mean_k). With one active outcome no column is added.
      per_outcome  one request per active outcome on the original features and raw residuals.

  An outcome whose training residuals are all identical is constant: its predicted residual
  is that constant with zero spread, and it is never sent.
* Every request has a :func:`request_key`. A stored key is served from the
  :class:`RequestCache` without any network call. An uncached request first reads account
  usage and a cost quote (both free, E7 p2/p3), then is refused with :class:`BudgetError`
  when the quote exceeds ``tabpfn.max_tokens_per_request``, when
  ``tabpfn.project_token_budget`` is unset, when tokens already quoted for dispatched
  requests plus this quote exceed that budget, or when the account's remaining tokens are
  below the quote. An accepted request is logged as a dispatch BEFORE the billable call, sent
  once (never retried here), cached completely, and logged again with its outcome.
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
import time
from dataclasses import dataclass
from datetime import datetime
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
    CallLog,
    RequestCache,
    canonical_json,
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
    "CachedEntry",
    "CallLog",
    "ProviderResponseError",
    "RequestCache",
    "TabPFNProvider",
    "TabPFNTeacher",
    "TeacherSettings",
    "budget_refusal_reasons",
    "constant_outcomes",
    "destandardize",
    "fill_outcomes",
    "parse_main_output",
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

_KEY_SCHEMA = "learned-judgment/teacher-request-key/v1"
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


class BudgetError(RuntimeError):
    """An uncached request was refused before any billable call; reasons are in the message and the call log."""


class ProviderResponseError(RuntimeError):
    """A provider response (prediction, quote or usage) violated the verified contract."""


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
    n_estimators, random_state and quantile_levels. Not keyed: spread_levels (post-processing),
    multi_outcome_mode (visible in the request arrays), cache_dir and call_log (paths),
    max_tokens_per_request and project_token_budget (the gate).
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

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.client_version, str) or not self.client_version:
            problems.append("client_version must be a non-empty string")
        if not isinstance(self.model_path, str) or not self.model_path or self.model_path in _AUTO_MODEL_PATHS:
            problems.append(f"model_path must name a pinned model, not {self.model_path!r}")
        for name in ("n_estimators", "max_tokens_per_request"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                problems.append(f"{name} must be a positive integer; got {value!r}")
        if not _is_int(self.random_state) or self.random_state < 0:
            problems.append(f"random_state must be a non-negative integer; got {self.random_state!r}")
        if self.project_token_budget is not None and (
            not _is_int(self.project_token_budget) or self.project_token_budget < 0
        ):
            problems.append(f"project_token_budget must be null or a non-negative integer; got {self.project_token_budget!r}")
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

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TeacherSettings":
        """Read ``tabpfn.*`` and ``engine.*`` with :func:`require`; ``tabpfn.project_token_budget``
        with :func:`optional` (null means "no budget set", which refuses every uncached request).

        Relative paths are resolved against the repository root (the directory of config.yaml).
        """

        def integer(key: str) -> int:
            value = require(config, key)
            if not _is_int(value):
                raise ConfigError(f"config.yaml key {key!r} must be an integer; got {value!r}")
            return value

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

        budget = optional(config, "tabpfn.project_token_budget")
        if budget is not None and not _is_int(budget):
            raise ConfigError(f"config.yaml key 'tabpfn.project_token_budget' must be null or an integer; got {budget!r}")
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
                project_token_budget=budget,
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


def _request_description(
    settings: TeacherSettings, categorical_indices: Sequence[int], column_names: Sequence[str]
) -> dict[str, Any]:
    return {
        "schema": _KEY_SCHEMA,
        "client_version": settings.client_version,
        "task": TASK,
        "tabpfn_config": _tabpfn_config(settings, categorical_indices),
        "request_flags": dict(_PINNED_REQUEST_FLAGS),
        "predict_params": {"output_type": OUTPUT_TYPE, "quantiles": list(settings.quantile_levels)},
        "column_names": list(column_names),
    }


def _float64(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.dtype.itemsize != 8:
        raise TypeError(f"{name} must be a float64 array; got dtype {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D; got shape {array.shape}")
    return array


def _check_request_columns(
    n_features: int, categorical_indices: Sequence[int], column_names: Sequence[str]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    names = tuple(column_names)
    if len(names) != n_features or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"column_names must be {n_features} non-empty strings; got {list(names)}")
    if len(set(names)) != len(names):
        raise ValueError(f"column_names must be unique; got {list(names)}")
    indices = tuple(categorical_indices)
    if not all(_is_int(index) for index in indices):
        raise ValueError(f"categorical_indices must be integers; got {list(indices)}")
    if any(b <= a for a, b in zip(indices, indices[1:])) or any(not 0 <= i < n_features for i in indices):
        raise ValueError(f"categorical_indices must be strictly increasing inside [0, {n_features}); got {list(indices)}")
    return indices, names


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
    column_names: Sequence[str],
) -> str:
    """sha256 hex key of one provider request (recipe of proof B_students_eval_contract/p09_hash).

    Args:
        x_train:             float64 (n, F), the training features exactly as sent.
        y_train:             float64 (n,), the training target exactly as sent.
        x_query:             float64 (m, F), the query rows exactly as sent.
        categorical_indices: strictly increasing column indices declared categorical.
        column_names:        F unique names of the columns, in order.

    Keyed: key schema version, client version, task, the tabpfn_config sent (model_path,
    n_estimators, random_state, categorical indices, ignore_pretraining_limits), the pinned
    request flags, predict params (output_type "main" and the exact quantile list, in order),
    column names, and canonical digests of x_train, y_train and x_query (dtype, shape,
    C-order little-endian bytes, -0.0 folded to +0.0). Not keyed: paths, budgets, spread
    levels, the multi-outcome mode (visible in the arrays), and the base URL.
    """
    x_train, y_train, x_query = _check_request_arrays(x_train, y_train, x_query)
    indices, names = _check_request_columns(x_train.shape[1], categorical_indices, column_names)
    digest = hashlib.sha256()
    digest.update(_KEY_SCHEMA.encode("utf-8") + b"\x00")
    for tag, array in (("x_train", x_train), ("y_train", y_train), ("x_query", x_query)):
        update_digest(digest, array, tag)
    digest.update(canonical_json(_request_description(settings, indices, names)).encode("utf-8"))
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
    repaired: any violation raises ProviderResponseError. (When n == Q the orientation cannot be
    told from the shape; the documented layout is assumed.)

    Returns:
        {"mean": (n,), "median": (n,), "mode": (n,), "quantiles": (n, Q)}, all float64.
    """
    q = len(quantile_levels)
    if n_rows < 1 or q < 1:
        raise ValueError(f"n_rows and the number of quantile levels must be >= 1; got {n_rows}, {q}")
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


# --------------------------------------------------------------------------------------
# Provider (the only code that imports tabpfn_client)
# --------------------------------------------------------------------------------------


class TabPFNProvider:
    """Hosted TabPFN regressor behind the pinned tabpfn-client.

    ``connect``, ``usage`` and ``quote`` are free. ``predict_main`` is billable (at least
    10,000 tokens per call on v3.5, E7 p3) and is only ever called by the teacher after its gate.

    Attributes:
        prediction_calls_started: number of predict_main calls that reached the network stage.
        last_phase: stage of the latest predict_main ("fit", "predict", "validate_response",
            "done"), so a caller can log where a failure happened.
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

    def predict_main(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        categorical_indices: Sequence[int],
        quantile_levels: Sequence[float],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """BILLABLE: one fit and one ``predict(output_type="main", quantiles=levels)``.

        Args:
            x_train: float64 (n, F), n >= 2, finite.
            y_train: float64 (n,), finite.
            x_query: float64 (m, F), m >= 1, finite.
            categorical_indices: strictly increasing columns declared categorical.
            quantile_levels: must equal settings.quantile_levels (the keyed list).

        Returns:
            ({"mean": (m,), "median": (m,), "mode": (m,), "quantiles": (m, Q)}, metadata) where
            metadata holds the fitted model id, server timings, response metadata, the
            tabpfn_config sent, base URL, client version and wall-clock seconds.
        """
        self._require_connected()
        self.last_phase = "validate_request"
        x_train, y_train, x_query = _check_request_arrays(x_train, y_train, x_query)
        indices, _ = _check_request_columns(
            x_train.shape[1], categorical_indices, [str(i) for i in range(x_train.shape[1])]
        )
        if x_train.shape[0] < MIN_TRAIN_ROWS or x_query.shape[0] < 1:
            raise ValueError(f"need >= {MIN_TRAIN_ROWS} training rows and >= 1 query row; got {x_train.shape}, {x_query.shape}")
        if not (np.all(np.isfinite(x_train)) and np.all(np.isfinite(y_train)) and np.all(np.isfinite(x_query))):
            raise ValueError("request arrays must be finite")
        levels = tuple(float(level) for level in quantile_levels)
        if levels != self.settings.quantile_levels:
            raise ValueError(f"quantile_levels {list(levels)} differ from the keyed {list(self.settings.quantile_levels)}")
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
        self.last_phase = "validate_response"
        arrays = parse_main_output(raw, n_rows=x_query.shape[0], quantile_levels=levels)
        metadata = {
            "model_id": str(regressor.model_id_),
            "timings": _json_safe(regressor.get_timings()),
            # Private attribute of the pinned 0.10.1 client: the server's response metadata (E7).
            "response_metadata": _json_safe(dict(regressor._last_meta)),
            "tabpfn_config_sent": sent,
            "api_mode": "sync",
            "base_url": str(ServiceClient.base_url),
            "client_version": str(tabpfn_client.__version__),
            "wall_s": {"fit": fit_s, "predict": predict_s},
        }
        self.last_phase = "done"
        return arrays, metadata


# --------------------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------------------


class TabPFNTeacher:
    """Residual model backed by the hosted TabPFN regressor; implements ResidualModel.

    Attributes:
        last_request_hash:   key of the last request behind the latest prediction (None when
                             every outcome was constant or nothing was predicted yet).
        last_request_hashes: every key used by the latest prediction, in request order.
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
        self.layout = layout
        self.settings = settings
        self.provider = provider
        self.cache = cache
        self.call_log = call_log
        self._clock = clock
        self._features: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self.last_request_hash: str | None = None
        self.last_request_hashes: tuple[str, ...] = ()

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

        One request in "stacked" mode, one per non-constant outcome in "per_outcome" mode; each
        is served from the cache when stored. Raises RuntimeError before fit(), ValueError on
        invalid rows, BudgetError when the gate refuses, ProviderResponseError on a contract
        violation and CacheCorruptError on a damaged cache entry.
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

    # ----- internals -----------------------------------------------------------------

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
            arrays = self._request(
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
            self._request(
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

    def _request(
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
        """Serve one request from the cache, or gate, send, cache and log it. Returns the
        response arrays {"mean": (m,), "median": (m,), "mode": (m,), "quantiles": (m, Q)}."""
        key = request_key(
            settings=self.settings,
            x_train=x_train,
            y_train=y_train,
            x_query=x_query,
            categorical_indices=categorical_indices,
            column_names=column_names,
        )
        shape = {"n_train": int(x_train.shape[0]), "n_query": int(x_query.shape[0]), "n_features": int(x_train.shape[1])}
        base = {"key": key, "skill": context["skill"], "mode": context["mode"], "outcomes": context["outcomes"], **shape}

        entry = self.cache.get(key)
        if entry is not None:
            arrays = self._check_cached(entry, x_train, y_train, x_query)
            self.call_log.append({**base, "event": "cache_hit", "cache_hit": True, DISPATCH_FLAG: False, QUOTE_FIELD: None})
            keys.append(key)
            return arrays

        usage_before = self.provider.usage()
        quote = self.provider.quote(shape["n_train"], shape["n_query"], shape["n_features"])
        tokens = quote["estimated_cost"]
        spent = self.call_log.tokens_quoted_for_billed_calls()
        gate = {QUOTE_FIELD: tokens, "quote": quote, "usage_before": usage_before, "tokens_quoted_before": spent}
        reasons = budget_refusal_reasons(
            quote_tokens=tokens,
            max_tokens_per_request=self.settings.max_tokens_per_request,
            project_token_budget=self.settings.project_token_budget,
            tokens_already_quoted=spent,
            remaining_account_tokens=remaining_tokens(usage_before),
        )
        if reasons:
            self.call_log.append({**base, **gate, "event": "refused", "cache_hit": False, DISPATCH_FLAG: False, "reasons": reasons})
            raise BudgetError(f"request {key} refused before any billable call: " + "; ".join(reasons))

        self.call_log.append({**base, **gate, "event": "dispatch", "cache_hit": False, DISPATCH_FLAG: True})
        try:
            arrays, metadata = self.provider.predict_main(
                x_train,
                y_train,
                x_query,
                categorical_indices=categorical_indices,
                quantile_levels=self.settings.quantile_levels,
            )
        except BaseException as exc:
            self.call_log.append(
                {
                    **base,
                    "event": "provider_error",
                    "cache_hit": False,
                    DISPATCH_FLAG: False,
                    QUOTE_FIELD: tokens,
                    "possibly_billed": self.provider.last_phase in ("fit", "predict", "validate_response"),
                    "phase": self.provider.last_phase,
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage_after": self._usage_or_error(),
                }
            )
            raise
        usage_after = self._usage_or_error()
        record = {
            **base,
            "cache_hit": False,
            DISPATCH_FLAG: False,
            QUOTE_FIELD: tokens,
            "usage_after": usage_after,
            "usage_delta": _usage_delta(usage_before, usage_after),
        }
        try:
            self.cache.put(
                key,
                request_arrays={"x_train": x_train, "y_train": y_train, "x_query": x_query},
                request_meta={
                    "description": _request_description(self.settings, categorical_indices, column_names),
                    **shape,
                    "teacher_context": context,
                },
                response_arrays=arrays,
                response_meta={
                    "key": key,
                    "timestamp_utc": utc_timestamp(self._clock),
                    "quote": quote,
                    "usage_before": usage_before,
                    "usage_after": usage_after,
                    "provider": metadata,
                },
            )
        except BaseException as exc:
            self.call_log.append({**record, "event": "cache_write_failed", "billed": True, "error": f"{type(exc).__name__}: {exc}"})
            raise
        self.call_log.append({**record, "event": "completed", "timings": metadata.get("timings"), "wall_s": metadata.get("wall_s")})
        keys.append(key)
        return arrays

    def _check_cached(
        self, entry: CachedEntry, x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray
    ) -> dict[str, np.ndarray]:
        for name, sent in (("x_train", x_train), ("y_train", y_train), ("x_query", x_query)):
            stored = entry.request_arrays.get(name)
            if stored is None or stored.dtype != sent.dtype or not np.array_equal(stored, sent):
                raise CacheCorruptError(f"cache entry {entry.key} stores a different {name} than the request it is keyed by")
        rows, q = x_query.shape[0], len(self.settings.quantile_levels)
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
            return {"error": f"{type(exc).__name__}: {exc}"}


def _usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    """after - before for the usage counters both reads report as integers."""
    return {
        name: after[name] - before[name]
        for name in _USAGE_COUNTERS
        if _is_int(before.get(name)) and _is_int(after.get(name))
    }
