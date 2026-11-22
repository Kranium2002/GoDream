"""Residual baselines B2-B6, all behind the engine's ResidualModel protocol.

Lane contract (Person B, baselines + evaluation)
------------------------------------------------
Every baseline learns the residual ``measured_outcome - physics_guess`` from ACTUAL attempts
and answers with a :class:`~learned_judgment.engine.types.ResidualDistribution`, so
:func:`learned_judgment.engine.decide.decide_batch` scores every method with the same
"physics + predicted residual -> spec.objective" step.

    ConstantBiasResidual      B2  physics + constant residual (works from 1 attempt)
    GaussianProcessResidual   B3  one GaussianProcessRegressor per outcome
    GradientBoostingResidual  B4  HistGradientBoostingRegressor (squared error) per outcome, with
                                  homoscedastic out-of-fold error quantiles
    MLPAttemptsResidual       B5  MLPRegressor (lbfgs) on the actual attempts, with homoscedastic
                                  out-of-fold error quantiles -- a baseline, not a student
    DirectTargetAdapter       B6  wraps any ResidualModel on a layout WITHOUT the physics block and
                                  makes it predict the outcome directly

Predictive quantiles of B4 and B5 (review finding: in-sample errors of a flexible model gave
an interval that collapsed to a point): the point model is fitted on all attempts; the
predictive distribution is that point prediction plus the empirical distribution of
OUT-OF-FOLD errors from a K-fold refit of the same model with the same settings
(K = min(interval_folds, number of attempts); fold assignment from the named stream
``baselines/<method>/interval_folds`` of the baseline's seed). The spread is computed once
per outcome and broadcast, so it is exactly the same value for every query row. Each fitted
B4/B5 also exposes ``oof_interval_coverage``: the cross-validated coverage of its
``coverage_intervals`` on the training attempts (an honesty indicator, not a result).
``describe()`` of every baseline states how its quantiles are produced.

Array conventions (see :mod:`learned_judgment.engine.types`)
    features    float64 (n, F)    columns in the model's FeatureLayout order
    residuals   float64 (n, K)    the skill's fixed outcome order
    quantiles   float64 (n, K, Q) at the configured quantile levels (exact members, no interpolation)

Settings: every tunable is a constructor argument; ``from_config`` reads them with
:func:`learned_judgment.config.require` from ``engine.quantile_levels``, ``engine.spread_levels``,
``evaluation.coverage_intervals`` and ``baselines.<name>.*``. Seeds are passed in by the caller
(engine.streams.stream_seed). Every scikit-learn fit and prediction runs under
``threadpoolctl.threadpool_limits(limits=1, user_api="blas")`` because proof p01b/p01d showed
fitted weights change with the BLAS thread count. Settings not listed in config.yaml are
left at the scikit-learn defaults of the installed version; ``describe()`` records them.

``fit`` is all-or-nothing: everything is fitted into local variables and installed on the
object only after every fit succeeded, so a refit that raises leaves the previous fit intact.

Nothing in this module calls a provider or a simulator, and nothing here produces a metric.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from learned_judgment.config import require
from learned_judgment.engine.features import FeatureLayout
from learned_judgment.engine.streams import stream_rng
from learned_judgment.engine.types import ResidualDistribution, ResidualModel
from learned_judgment.skills.base import SkillSpec

__all__ = [
    "ConstantBiasResidual",
    "DirectTargetAdapter",
    "GaussianProcessResidual",
    "GradientBoostingResidual",
    "MLPAttemptsResidual",
]

# Definition of an "empirical quantile" in this module: numpy's default estimator
# (Hyndman & Fan type 7, linear between order statistics). Structural, not tuned.
_EMPIRICAL_QUANTILE_METHOD = "linear"
# The median is the 0.5 quantile by definition.
_MEDIAN_LEVEL = 0.5
# scikit-learn random_state must fit in an unsigned 32-bit integer.
_MAX_SEED = 2**32 - 1
# Out-of-fold errors need at least two folds (structural).
_MIN_FOLDS = 2


def _blas_single_thread() -> Any:
    return threadpool_limits(limits=1, user_api="blas")


def _check_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer; got {seed!r}")
    if not 0 <= int(seed) <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]; got {seed}")
    return int(seed)


def _check_count(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return int(value)


def _check_positive(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0; got {value!r}")
    return number


def _check_bounds(name: str, bounds: Sequence[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in bounds)
    if len(values) != 2 or not (np.isfinite(values[0]) and np.isfinite(values[1]) and 0.0 < values[0] < values[1]):
        raise ValueError(f"{name} must be (low, high) with 0 < low < high; got {list(bounds)}")
    return values


def _check_levels(quantile_levels: Sequence[float]) -> np.ndarray:
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 1:
        raise ValueError(f"quantile_levels must have shape (Q,) with Q >= 1; got {levels.shape}")
    if np.any(levels <= 0.0) or np.any(levels >= 1.0) or np.any(np.diff(levels) <= 0.0):
        raise ValueError(f"quantile_levels must be strictly increasing inside (0, 1); got {levels.tolist()}")
    return levels


def _check_spread_levels(spread_levels: Sequence[float], levels: np.ndarray) -> tuple[float, float]:
    values = tuple(float(value) for value in spread_levels)
    if len(values) != 2 or not values[0] < values[1]:
        raise ValueError(f"spread_levels must be (low, high) with low < high; got {list(spread_levels)}")
    for value in values:
        if np.flatnonzero(levels == value).size != 1:
            raise ValueError(f"spread level {value} is not exactly one of quantile_levels {levels.tolist()}")
    return values


def _check_intervals(intervals: Sequence[Sequence[float]], levels: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Central intervals [(low, high), ...] whose ends are exact members of ``levels``."""
    pairs: list[tuple[float, float]] = []
    for interval in intervals:
        pair = _check_spread_levels(interval, levels)
        pairs.append(pair)
    if not pairs:
        raise ValueError("coverage_intervals must list at least one (low, high) interval")
    return tuple(pairs)


def _config_levels(config: Mapping[str, Any]) -> tuple[np.ndarray, tuple[float, float]]:
    levels = _check_levels(require(config, "engine.quantile_levels"))
    spread = require(config, "engine.spread_levels")
    if not isinstance(spread, Sequence) or len(spread) != 2:
        raise ValueError(f"config.yaml key 'engine.spread_levels' must be [low, high]; got {spread!r}")
    return levels, _check_spread_levels(spread, levels)


def _record_warnings(label: str, action: Callable[[], Any]) -> list[str]:
    """Run ``action`` under one BLAS thread and return every warning it raised as text."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _blas_single_thread():
            action()
    return [f"{label}: {item.category.__name__}: {item.message}" for item in caught]


@dataclass(frozen=True, eq=False)
class _ErrorSummary:
    """Summary of residual errors per outcome, added to a point prediction.

    mean (K,), median (K,), quantiles (K, Q) at the quantile levels, spread (K,) = half the
    width between the spread levels -- computed once, broadcast to every query row.
    """

    mean: np.ndarray
    median: np.ndarray
    quantiles: np.ndarray
    spread: np.ndarray


def _summarize_errors(errors: np.ndarray, levels: np.ndarray, spread_levels: tuple[float, float]) -> _ErrorSummary:
    """errors (n, K) -> per-outcome empirical mean, median, quantiles (type 7) and spread."""
    quantiles = np.quantile(errors, levels, axis=0, method=_EMPIRICAL_QUANTILE_METHOD).T
    low = int(np.flatnonzero(levels == spread_levels[0])[0])
    high = int(np.flatnonzero(levels == spread_levels[1])[0])
    return _ErrorSummary(
        mean=errors.mean(axis=0),
        median=np.quantile(errors, _MEDIAN_LEVEL, axis=0, method=_EMPIRICAL_QUANTILE_METHOD),
        quantiles=quantiles,
        spread=0.5 * (quantiles[:, high] - quantiles[:, low]),
    )


def _fold_assignment(n: int, folds: int, seed: int, stream_name: str) -> np.ndarray:
    """int64 (n,) fold index per attempt: a seeded permutation dealt round-robin into ``folds``."""
    order = stream_rng(seed, stream_name, 0).permutation(n)
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = np.arange(n) % folds
    return assignment


def _cross_validated_coverage(
    errors: np.ndarray,
    assignment: np.ndarray,
    folds: int,
    intervals: Sequence[tuple[float, float]],
    outcome_names: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Out-of-fold coverage of each interval on the training attempts.

    Row i of fold f is covered when its out-of-fold error lies inside the empirical
    [low, high] quantiles of the out-of-fold errors of the OTHER folds -- i.e. inside the
    interval the procedure would have given it without seeing it. errors (n, K).
    """
    n = errors.shape[0]
    records: list[dict[str, Any]] = []
    for column, outcome in enumerate(outcome_names):
        for low, high in intervals:
            covered = 0
            for fold in range(folds):
                held = assignment == fold
                bounds = np.quantile(errors[~held, column], [low, high], method=_EMPIRICAL_QUANTILE_METHOD)
                values = errors[held, column]
                covered += int(np.count_nonzero((values >= bounds[0]) & (values <= bounds[1])))
            records.append(
                {
                    "outcome": outcome,
                    "low_level": float(low),
                    "high_level": float(high),
                    "nominal": float(high - low),
                    "count": covered,
                    "n": int(n),
                    "coverage": covered / n,
                }
            )
    return tuple(records)


class _ResidualBaseline(ABC):
    """Shared input checks and output assembly. Not a model by itself."""

    method: ClassVar[str]
    quantile_method: ClassVar[str]

    def __init__(
        self,
        layout: FeatureLayout,
        quantile_levels: Sequence[float],
        spread_levels: Sequence[float],
        *,
        min_attempts: int,
        seed: int,
    ) -> None:
        if not isinstance(layout, FeatureLayout):
            raise TypeError(f"layout must be a FeatureLayout; got {type(layout).__name__}")
        self.layout = layout
        self.quantile_levels = _check_levels(quantile_levels)
        self.spread_levels = _check_spread_levels(spread_levels, self.quantile_levels)
        self.min_attempts = _check_count("min_attempts", min_attempts, 1)
        self.seed = _check_seed(seed)
        self.n_train = 0
        self.fit_warnings: tuple[str, ...] = ()
        self._fitted = False

    # ----- checks ----------------------------------------------------------------------

    def _check_training(self, features: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(residuals, dtype=np.float64)
        f, k = self.layout.n_features, self.layout.n_outcomes
        if x.ndim != 2 or x.shape[1] != f:
            raise ValueError(f"{self.method}: features must have shape (n, F={f}); got {x.shape}")
        if y.shape != (x.shape[0], k):
            raise ValueError(f"{self.method}: residuals must have shape (n={x.shape[0]}, K={k}); got {y.shape}")
        if x.shape[0] < self.min_attempts:
            raise ValueError(
                f"{self.method} needs at least min_attempts={self.min_attempts} attempts; got {x.shape[0]}"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError(f"{self.method}: features and residuals must be finite")
        return x, y

    def _check_query(self, query_features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.method}: call fit() before predicting")
        x = np.asarray(query_features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.layout.n_features:
            raise ValueError(f"{self.method}: query_features must have shape (n, F={self.layout.n_features}); got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{self.method}: query_features must be finite")
        return x

    def _finish_fit(self, n_train: int, fit_warnings: Sequence[str]) -> None:
        self.n_train = int(n_train)
        self.fit_warnings = tuple(fit_warnings)
        self._fitted = True

    def _distribution(self, mean: np.ndarray, median: np.ndarray, quantiles: np.ndarray) -> ResidualDistribution:
        return ResidualDistribution.from_quantiles(
            mean=mean,
            median=median,
            quantile_levels=self.quantile_levels,
            quantiles=quantiles,
            spread_levels=self.spread_levels,
        )

    def _homoscedastic_distribution(self, point: np.ndarray, summary: _ErrorSummary) -> ResidualDistribution:
        """point (n, K) + one error summary per outcome; the (K,) spread is broadcast exactly."""
        n, k = point.shape
        return ResidualDistribution(
            mean=point + summary.mean[None, :],
            median=point + summary.median[None, :],
            quantile_levels=self.quantile_levels,
            quantiles=point[:, :, None] + summary.quantiles[None, :, :],
            spread=np.broadcast_to(summary.spread, (n, k)).copy(),
            spread_levels=self.spread_levels,
        )

    # ----- ResidualModel surface ------------------------------------------------------

    @abstractmethod
    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "_ResidualBaseline":
        """features (n, F), residuals (n, K) -> self."""

    @abstractmethod
    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """query_features (n, F) -> ResidualDistribution with (n, K) / (n, K, Q) arrays."""

    def predict_residuals(self, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(median_residual, predictive_spread)``, each float64 (n_query, K)."""
        distribution = self.predict_residual_distribution(query_features)
        return distribution.median, distribution.spread

    def _settings(self) -> dict[str, Any]:
        return {}

    def _fit_details(self) -> dict[str, Any]:
        return {}

    def describe(self) -> dict[str, Any]:
        """JSON-serializable record of the method, its settings, how its quantiles are
        produced, and the last fit."""
        return {
            "method": self.method,
            "quantile_method": self.quantile_method,
            "layout": self.layout.describe(),
            "quantile_levels": self.quantile_levels.tolist(),
            "spread_levels": list(self.spread_levels),
            "min_attempts": self.min_attempts,
            "seed": self.seed,
            "settings": self._settings(),
            "n_train": self.n_train,
            "fit_warnings": list(self.fit_warnings),
            "fit_details": self._fit_details(),
        }


# --------------------------------------------------------------------------------------
# B2 constant bias
# --------------------------------------------------------------------------------------


class ConstantBiasResidual(_ResidualBaseline):
    """B2: physics + a constant residual per outcome.

    fit:     mean_k = mean of the training residuals of outcome k; the predictive quantiles
             are the empirical quantiles (numpy "linear", type 7) of those residuals and the
             median is their empirical 0.5 quantile. The features are ignored.
    predict: the same (K,) mean / median / (K, Q) quantiles for every query row.
    Works from 1 attempt (then every quantile equals that single residual, spread 0).
    """

    method = "constant_bias"
    quantile_method = (
        "in-sample empirical quantiles (numpy 'linear', type 7) of the training residuals of each outcome; "
        "the same quantiles, median and spread for every query row; features are ignored"
    )

    def __init__(
        self,
        layout: FeatureLayout,
        quantile_levels: Sequence[float],
        spread_levels: Sequence[float],
        *,
        seed: int,
    ) -> None:
        super().__init__(layout, quantile_levels, spread_levels, min_attempts=1, seed=seed)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], layout: FeatureLayout, *, seed: int) -> "ConstantBiasResidual":
        levels, spread = _config_levels(config)
        return cls(layout, levels, spread, seed=seed)

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "ConstantBiasResidual":
        """features (n, F), residuals (n, K) with n >= 1."""
        _, y = self._check_training(features, residuals)
        mean = y.mean(axis=0)
        median = np.quantile(y, _MEDIAN_LEVEL, axis=0, method=_EMPIRICAL_QUANTILE_METHOD)
        quantiles = np.quantile(y, self.quantile_levels, axis=0, method=_EMPIRICAL_QUANTILE_METHOD).T
        self._mean, self._median, self._quantiles = mean, median, quantiles
        self._finish_fit(y.shape[0], ())
        return self

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """query_features (n, F) -> mean/median (n, K), quantiles (n, K, Q), identical per row."""
        n = self._check_query(query_features).shape[0]
        k, q = self.layout.n_outcomes, self.quantile_levels.size
        return self._distribution(
            mean=np.broadcast_to(self._mean, (n, k)).copy(),
            median=np.broadcast_to(self._median, (n, k)).copy(),
            quantiles=np.broadcast_to(self._quantiles, (n, k, q)).copy(),
        )


# --------------------------------------------------------------------------------------
# B3 Gaussian process
# --------------------------------------------------------------------------------------


class GaussianProcessResidual(_ResidualBaseline):
    """B3: one ``GaussianProcessRegressor`` per outcome on standardized features.

    Kernel: ``ConstantKernel * RBF(ARD, one length scale per feature) + WhiteKernel`` with the
    configured bounds; initial values are the scikit-learn defaults (constant 1.0, length
    scales 1.0, noise 1.0). ``normalize_y=True``; ``n_restarts_optimizer`` and the seed are
    passed to the regressor. Categorical columns enter as their standardized level index.

    Predictive distribution per outcome: Normal(mean, std) from ``predict(return_std=True)``;
    quantile at level p = mean + std * Phi^-1(p); median = mean.

    Refuses fewer than ``min_attempts`` rows (proof p04: 0 rows raise inside scikit-learn and
    1 row gives an overconfident std). Every warning raised while fitting -- including
    ``ConvergenceWarning`` when a hyperparameter hits a bound -- is kept in ``fit_warnings``.
    """

    method = "gaussian_process"
    quantile_method = (
        "Normal(mean, std) of each outcome's Gaussian process (predict with return_std=True, std includes the "
        "fitted WhiteKernel noise); quantile = mean + std * Phi^-1(level); heteroscedastic per query row"
    )

    def __init__(
        self,
        layout: FeatureLayout,
        quantile_levels: Sequence[float],
        spread_levels: Sequence[float],
        *,
        n_restarts_optimizer: int,
        constant_value_bounds: Sequence[float],
        length_scale_bounds: Sequence[float],
        noise_level_bounds: Sequence[float],
        min_attempts: int,
        seed: int,
    ) -> None:
        super().__init__(layout, quantile_levels, spread_levels, min_attempts=min_attempts, seed=seed)
        self.n_restarts_optimizer = _check_count("n_restarts_optimizer", n_restarts_optimizer, 0)
        self.constant_value_bounds = _check_bounds("constant_value_bounds", constant_value_bounds)
        self.length_scale_bounds = _check_bounds("length_scale_bounds", length_scale_bounds)
        self.noise_level_bounds = _check_bounds("noise_level_bounds", noise_level_bounds)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], layout: FeatureLayout, *, seed: int) -> "GaussianProcessResidual":
        levels, spread = _config_levels(config)
        key = "baselines.gaussian_process."
        return cls(
            layout,
            levels,
            spread,
            n_restarts_optimizer=require(config, key + "n_restarts_optimizer"),
            constant_value_bounds=require(config, key + "constant_value_bounds"),
            length_scale_bounds=require(config, key + "length_scale_bounds"),
            noise_level_bounds=require(config, key + "noise_level_bounds"),
            min_attempts=require(config, key + "min_attempts"),
            seed=seed,
        )

    def _kernel(self) -> Any:
        return ConstantKernel(constant_value_bounds=self.constant_value_bounds) * RBF(
            length_scale=np.ones(self.layout.n_features), length_scale_bounds=self.length_scale_bounds
        ) + WhiteKernel(noise_level_bounds=self.noise_level_bounds)

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "GaussianProcessResidual":
        """features (n, F), residuals (n, K) with n >= min_attempts. All-or-nothing."""
        x, y = self._check_training(features, residuals)
        with _blas_single_thread():
            scaler = StandardScaler().fit(x)
            x_scaled = scaler.transform(x)
        models: list[GaussianProcessRegressor] = []
        recorded: list[str] = []
        for column, outcome in enumerate(self.layout.outcome_names):
            model = GaussianProcessRegressor(
                kernel=self._kernel(),
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.seed,
            )
            recorded += _record_warnings(outcome, lambda: model.fit(x_scaled, y[:, column]))
            models.append(model)
        self._scaler, self._models = scaler, models
        self._finish_fit(x.shape[0], recorded)
        return self

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """query_features (n, F) -> mean/median (n, K), quantiles (n, K, Q) from Normal(mean, std)."""
        x = self._check_query(query_features)
        n, k = x.shape[0], self.layout.n_outcomes
        mean, std = np.empty((n, k)), np.empty((n, k))
        with _blas_single_thread():
            x_scaled = self._scaler.transform(x)
            for column, model in enumerate(self._models):
                mu, sd = model.predict(x_scaled, return_std=True)
                mean[:, column], std[:, column] = mu, sd
        z = norm.ppf(self.quantile_levels)
        quantiles = mean[:, :, None] + std[:, :, None] * z[None, None, :]
        return self._distribution(mean=mean, median=mean.copy(), quantiles=quantiles)

    def fitted_kernels(self) -> dict[str, str]:
        """Outcome name -> fitted kernel text (for run records; compare kernels by theta, not ==)."""
        if not self._fitted:
            raise RuntimeError(f"{self.method}: call fit() first")
        return {name: str(model.kernel_) for name, model in zip(self.layout.outcome_names, self._models)}

    def _settings(self) -> dict[str, Any]:
        return {
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "constant_value_bounds": list(self.constant_value_bounds),
            "length_scale_bounds": list(self.length_scale_bounds),
            "noise_level_bounds": list(self.noise_level_bounds),
            "normalize_y": True,
            "kernel": str(self._kernel()),
        }


# --------------------------------------------------------------------------------------
# Out-of-fold homoscedastic baselines (B4, B5)
# --------------------------------------------------------------------------------------


class _OutOfFoldBaseline(_ResidualBaseline):
    """Point model fitted on all attempts plus a homoscedastic error distribution from
    out-of-fold errors of a K-fold refit of the same model.

    With p(x) the point prediction (n, K) and e (n_train, K) the out-of-fold errors
    y_i - p_(-fold(i))(x_i):
        mean     = p(x) + mean(e)
        median   = p(x) + empirical 0.5 quantile of e
        quantile = p(x) + empirical quantile of e at each level (numpy "linear", type 7)
        spread   = half the width between the spread levels of e, one value per outcome,
                   broadcast to every query row (exactly constant)
    K = min(interval_folds, n_train) >= 2; folds from the stream
    ``baselines/<method>/interval_folds`` of ``seed``; every fold model uses the same settings
    and seed as the point model.
    """

    quantile_method_template: ClassVar[str] = (
        "homoscedastic, out-of-fold: point prediction of the {point} fitted on all attempts, plus the empirical "
        "quantiles (numpy 'linear', type 7) of OUT-OF-FOLD errors from a K-fold refit of the same model with the "
        "same settings (K = min(interval_folds, number of attempts), folds from the stream "
        "'baselines/{method}/interval_folds' of the seed); the spread is one value per outcome, identical for "
        "every query row"
    )

    def __init__(
        self,
        layout: FeatureLayout,
        quantile_levels: Sequence[float],
        spread_levels: Sequence[float],
        *,
        interval_folds: int,
        coverage_intervals: Sequence[Sequence[float]],
        min_attempts: int,
        seed: int,
    ) -> None:
        super().__init__(layout, quantile_levels, spread_levels, min_attempts=min_attempts, seed=seed)
        if self.min_attempts < _MIN_FOLDS:
            raise ValueError(f"{self.method}: out-of-fold errors need min_attempts >= {_MIN_FOLDS}; got {min_attempts}")
        self.interval_folds = _check_count("interval_folds", interval_folds, _MIN_FOLDS)
        self.coverage_intervals = _check_intervals(coverage_intervals, self.quantile_levels)
        self.folds_used = 0
        self.oof_interval_coverage: tuple[dict[str, Any], ...] = ()

    @property
    def fold_stream(self) -> str:
        """Name of the stream (under ``seed``) that assigns attempts to folds."""
        return f"baselines/{self.method}/interval_folds"

    @abstractmethod
    def _fit_point(self, x: np.ndarray, y: np.ndarray) -> tuple[Any, list[str]]:
        """Fit the point model on x (m, F), y (m, K) -> (fitted state, warning texts). No side effects."""

    @abstractmethod
    def _predict_point_with(self, state: Any, x: np.ndarray) -> np.ndarray:
        """Point prediction (m, K) of a fitted state for x (m, F)."""

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "_OutOfFoldBaseline":
        """features (n, F), residuals (n, K) with n >= min_attempts. All-or-nothing."""
        x, y = self._check_training(features, residuals)
        n = x.shape[0]
        folds = min(self.interval_folds, n)
        assignment = _fold_assignment(n, folds, self.seed, self.fold_stream)
        state, point_warnings = self._fit_point(x, y)
        recorded = [f"all_attempts/{text}" for text in point_warnings]
        out_of_fold = np.empty_like(y)
        for fold in range(folds):
            held = assignment == fold
            fold_state, fold_warnings = self._fit_point(x[~held], y[~held])
            recorded += [f"fold{fold}/{text}" for text in fold_warnings]
            out_of_fold[held] = self._predict_point_with(fold_state, x[held])
        errors = y - out_of_fold
        summary = _summarize_errors(errors, self.quantile_levels, self.spread_levels)
        coverage = _cross_validated_coverage(
            errors, assignment, folds, self.coverage_intervals, self.layout.outcome_names
        )
        self._point_state, self._error_summary = state, summary
        self.folds_used, self.oof_interval_coverage = folds, coverage
        self._finish_fit(n, recorded)
        return self

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """query_features (n, F) -> point prediction (n, K) plus the out-of-fold error summary."""
        x = self._check_query(query_features)
        return self._homoscedastic_distribution(self._predict_point_with(self._point_state, x), self._error_summary)

    def _fit_details(self) -> dict[str, Any]:
        if not self._fitted:
            return {}
        return {
            "folds_used": self.folds_used,
            "fold_stream": self.fold_stream,
            "out_of_fold_interval_coverage": [dict(record) for record in self.oof_interval_coverage],
            "out_of_fold_note": "cross-validated coverage on the training attempts: an honesty indicator, not a result",
        }

    def _oof_settings(self) -> dict[str, Any]:
        return {
            "interval_folds": self.interval_folds,
            "coverage_intervals": [list(pair) for pair in self.coverage_intervals],
        }


# --------------------------------------------------------------------------------------
# B4 gradient boosting
# --------------------------------------------------------------------------------------


class GradientBoostingResidual(_OutOfFoldBaseline):
    """B4: one ``HistGradientBoostingRegressor(loss="squared_error")`` per outcome.

    Its predictive spread is HOMOSCEDASTIC and OUT-OF-FOLD: the quantiles are the point
    prediction plus the empirical quantiles of out-of-fold errors (see
    :class:`_OutOfFoldBaseline`); no per-level quantile models are fitted (review finding:
    in-sample quantile models covered 20-60% at a nominal 80%).

    ``early_stopping=False`` always (the library's "auto" mode would switch on above 10 000
    rows) and the small-sample settings come from config: the library default
    ``min_samples_leaf=20`` gives a constant model below 40 rows (proof p05).
    Categorical columns of the layout are passed as ``categorical_features``.
    """

    method = "gradient_boosting"
    quantile_method = _OutOfFoldBaseline.quantile_method_template.format(
        point="squared-error HistGradientBoostingRegressor of each outcome", method="gradient_boosting"
    )

    def __init__(
        self,
        layout: FeatureLayout,
        quantile_levels: Sequence[float],
        spread_levels: Sequence[float],
        *,
        max_iter: int,
        learning_rate: float,
        max_leaf_nodes: int,
        min_samples_leaf: int,
        interval_folds: int,
        coverage_intervals: Sequence[Sequence[float]],
        min_attempts: int,
        seed: int,
    ) -> None:
        super().__init__(
            layout,
            quantile_levels,
            spread_levels,
            interval_folds=interval_folds,
            coverage_intervals=coverage_intervals,
            min_attempts=min_attempts,
            seed=seed,
        )
        self.max_iter = _check_count("max_iter", max_iter, 1)
        self.learning_rate = _check_positive("learning_rate", learning_rate)
        self.max_leaf_nodes = _check_count("max_leaf_nodes", max_leaf_nodes, 2)
        self.min_samples_leaf = _check_count("min_samples_leaf", min_samples_leaf, 1)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], layout: FeatureLayout, *, seed: int) -> "GradientBoostingResidual":
        levels, spread = _config_levels(config)
        key = "baselines.gradient_boosting."
        return cls(
            layout,
            levels,
            spread,
            max_iter=require(config, key + "max_iter"),
            learning_rate=require(config, key + "learning_rate"),
            max_leaf_nodes=require(config, key + "max_leaf_nodes"),
            min_samples_leaf=require(config, key + "min_samples_leaf"),
            interval_folds=require(config, key + "interval_folds"),
            coverage_intervals=require(config, "evaluation.coverage_intervals"),
            min_attempts=require(config, key + "min_attempts"),
            seed=seed,
        )

    def _estimator(self) -> HistGradientBoostingRegressor:
        categorical = list(self.layout.categorical_indices)
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=False,
            categorical_features=categorical if categorical else None,
            random_state=self.seed,
        )

    def _fit_point(self, x: np.ndarray, y: np.ndarray) -> tuple[list[HistGradientBoostingRegressor], list[str]]:
        models: list[HistGradientBoostingRegressor] = []
        recorded: list[str] = []
        for column, outcome in enumerate(self.layout.outcome_names):
            target = y[:, column]
            model = self._estimator()
            recorded += _record_warnings(f"{outcome}/mean", lambda: model.fit(x, target))
            models.append(model)
        return models, recorded

    def _predict_point_with(self, state: list[HistGradientBoostingRegressor], x: np.ndarray) -> np.ndarray:
        with _blas_single_thread():
            return np.column_stack([model.predict(x) for model in state]).astype(np.float64, copy=False)

    def _settings(self) -> dict[str, Any]:
        params = self._estimator().get_params()
        settings = {key: (list(value) if isinstance(value, tuple) else value) for key, value in params.items()}
        settings.update(self._oof_settings())
        return settings


# --------------------------------------------------------------------------------------
# B5 MLP on actual attempts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _MLPState:
    """A fitted network with the feature and target scalers it was trained with."""

    x_scaler: StandardScaler
    y_scaler: StandardScaler
    network: MLPRegressor


class MLPAttemptsResidual(_OutOfFoldBaseline):
    """B5: an ``MLPRegressor`` (solver "lbfgs", proof p01) on standardized features and
    standardized residual targets, trained on the ACTUAL attempts. A baseline, not a student.

    Predictive distribution, HOMOSCEDASTIC and OUT-OF-FOLD (see :class:`_OutOfFoldBaseline`):
    the network fitted on all attempts gives p(x); the error quantiles come from out-of-fold
    refits of the same network, because the in-sample errors of a network that interpolates
    its attempts are about 0 (review finding: the in-sample 80% interval covered 0-0.2%).
    """

    method = "mlp_on_attempts"
    quantile_method = _OutOfFoldBaseline.quantile_method_template.format(
        point="lbfgs MLPRegressor (standardized features and targets)", method="mlp_on_attempts"
    )

    def __init__(
        self,
        layout: FeatureLayout,
        quantile_levels: Sequence[float],
        spread_levels: Sequence[float],
        *,
        hidden_layer_sizes: Sequence[int],
        alpha: float,
        max_iter: int,
        interval_folds: int,
        coverage_intervals: Sequence[Sequence[float]],
        min_attempts: int,
        seed: int,
    ) -> None:
        super().__init__(
            layout,
            quantile_levels,
            spread_levels,
            interval_folds=interval_folds,
            coverage_intervals=coverage_intervals,
            min_attempts=min_attempts,
            seed=seed,
        )
        sizes = tuple(hidden_layer_sizes)
        if not sizes:
            raise ValueError("hidden_layer_sizes must list at least one layer")
        self.hidden_layer_sizes = tuple(_check_count("hidden_layer_sizes[i]", size, 1) for size in sizes)
        alpha_value = float(alpha)
        if not np.isfinite(alpha_value) or alpha_value < 0.0:
            raise ValueError(f"alpha must be a finite number >= 0; got {alpha!r}")
        self.alpha = alpha_value
        self.max_iter = _check_count("max_iter", max_iter, 1)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], layout: FeatureLayout, *, seed: int) -> "MLPAttemptsResidual":
        levels, spread = _config_levels(config)
        key = "baselines.mlp_on_attempts."
        return cls(
            layout,
            levels,
            spread,
            hidden_layer_sizes=require(config, key + "hidden_layer_sizes"),
            alpha=require(config, key + "alpha"),
            max_iter=require(config, key + "max_iter"),
            interval_folds=require(config, key + "interval_folds"),
            coverage_intervals=require(config, "evaluation.coverage_intervals"),
            min_attempts=require(config, key + "min_attempts"),
            seed=seed,
        )

    def _network(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            solver="lbfgs",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.seed,
        )

    def _fit_point(self, x: np.ndarray, y: np.ndarray) -> tuple[_MLPState, list[str]]:
        with _blas_single_thread():
            x_scaler = StandardScaler().fit(x)
            y_scaler = StandardScaler().fit(y)
            x_scaled = x_scaler.transform(x)
            y_scaled = y_scaler.transform(y)
        target = y_scaled[:, 0] if y_scaled.shape[1] == 1 else y_scaled
        network = self._network()
        recorded = _record_warnings("all_outcomes", lambda: network.fit(x_scaled, target))
        return _MLPState(x_scaler=x_scaler, y_scaler=y_scaler, network=network), recorded

    def _predict_point_with(self, state: _MLPState, x: np.ndarray) -> np.ndarray:
        """x (m, F) -> network prediction in residual units, (m, K)."""
        with _blas_single_thread():
            scaled = state.network.predict(state.x_scaler.transform(x))
            return state.y_scaler.inverse_transform(np.asarray(scaled).reshape(x.shape[0], self.layout.n_outcomes))

    def _settings(self) -> dict[str, Any]:
        params = self._network().get_params()
        settings = {key: (list(value) if isinstance(value, tuple) else value) for key, value in params.items()}
        settings.update(self._oof_settings())
        return settings


# --------------------------------------------------------------------------------------
# B6 direct prediction without the physics feature
# --------------------------------------------------------------------------------------


class DirectTargetAdapter:
    """B6 wrapper "direct, no physics feature" around any :class:`ResidualModel`.

    The wrapped model sees a layout with NO physics block and learns the measured outcome
    itself. The adapter keeps the engine's residual interface:

        fit(features, residuals):  measured = residuals + spec.physics_guess(situation, action)
                                   (reconstructed up to float rounding), inner.fit(features, measured)
        predict:                   returned residual = inner predicted outcome - physics guess

    so the shared decide step (physics + residual) scores the directly predicted outcome.
    The situation and action blocks are read from the first ``n_situation + n_action``
    columns of the layout. ``last_request_hash`` and ``last_request_hashes`` are forwarded from the inner model so a
    provider-backed inner model keeps its cache key on each Decision. The quantiles are the
    inner model's (see its ``describe()["quantile_method"]``) shifted by -physics.
    """

    quantile_method = "the inner model's predictive quantiles of the outcome, shifted by minus the physics guess"

    def __init__(self, inner: ResidualModel, *, spec: SkillSpec, layout: FeatureLayout) -> None:
        if not isinstance(inner, ResidualModel):
            raise TypeError(f"inner must implement ResidualModel; got {type(inner).__name__}")
        if not isinstance(layout, FeatureLayout):
            raise TypeError(f"layout must be a FeatureLayout; got {type(layout).__name__}")
        if layout.n_physics != 0:
            raise ValueError(
                f"DirectTargetAdapter needs a layout without the physics block; this one has {layout.n_physics} "
                "physics columns (build it with build_layout(spec, include_physics=False))"
            )
        if (
            layout.skill != spec.name
            or layout.n_situation != spec.n_situation
            or layout.n_action != spec.n_action
            or layout.outcome_names != spec.outcome_names
        ):
            raise ValueError(f"layout of skill {layout.skill!r} does not match spec {spec.name!r}")
        inner_layout = getattr(inner, "layout", None)
        if inner_layout is not None and inner_layout != layout:
            raise ValueError("the inner model was built for a different FeatureLayout")
        self.inner = inner
        self.spec = spec
        self.layout = layout
        self.method = f"direct_{getattr(inner, 'method', type(inner).__name__)}"

    @property
    def last_request_hash(self) -> str | None:
        return getattr(self.inner, "last_request_hash", None)

    @property
    def last_request_hashes(self) -> tuple[str, ...]:
        """Every provider request key behind the inner model's latest prediction."""
        return tuple(getattr(self.inner, "last_request_hashes", None) or ())

    def _check_features(self, label: str, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.layout.n_features:
            raise ValueError(f"{label} must have shape (n, F={self.layout.n_features}); got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{label} must be finite")
        return x

    def physics(self, features: np.ndarray) -> np.ndarray:
        """features (n, F) -> spec.physics_guess of their situation/action blocks, (n, K)."""
        x = self._check_features("features", features)
        s, a = self.layout.n_situation, self.layout.n_action
        guess = np.asarray(self.spec.physics_guess(x[:, :s], x[:, s : s + a]), dtype=np.float64)
        if guess.shape != (x.shape[0], self.spec.n_outcome):
            raise ValueError(f"physics_guess must return ({x.shape[0]}, {self.spec.n_outcome}); got {guess.shape}")
        return guess

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "DirectTargetAdapter":
        """features (n, F) without physics columns, residuals (n, K) -> fits inner on (n, K) outcomes."""
        x = self._check_features("features", features)
        y = np.asarray(residuals, dtype=np.float64)
        if y.shape != (x.shape[0], self.spec.n_outcome):
            raise ValueError(f"residuals must have shape ({x.shape[0]}, {self.spec.n_outcome}); got {y.shape}")
        self.inner.fit(x, y + self.physics(x))
        return self

    def predict_residual_distribution(self, query_features: np.ndarray) -> ResidualDistribution:
        """query_features (n, F) -> inner outcome distribution shifted by -physics, (n, K[, Q])."""
        x = self._check_features("query_features", query_features)
        outcome = self.inner.predict_residual_distribution(x)
        guess = self.physics(x)
        if outcome.mean.shape != guess.shape:
            raise ValueError(f"the inner model returned {outcome.mean.shape}; expected {guess.shape}")
        return ResidualDistribution(
            mean=outcome.mean - guess,
            median=outcome.median - guess,
            quantile_levels=outcome.quantile_levels,
            quantiles=outcome.quantiles - guess[:, :, None],
            spread=outcome.spread,
            spread_levels=outcome.spread_levels,
        )

    def predict_residuals(self, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(median_residual, predictive_spread)``, each float64 (n_query, K)."""
        distribution = self.predict_residual_distribution(query_features)
        return distribution.median, distribution.spread

    def describe(self) -> dict[str, Any]:
        inner_describe = getattr(self.inner, "describe", None)
        return {
            "method": self.method,
            "quantile_method": self.quantile_method,
            "layout": self.layout.describe(),
            "inner": inner_describe() if callable(inner_describe) else type(self.inner).__name__,
        }
