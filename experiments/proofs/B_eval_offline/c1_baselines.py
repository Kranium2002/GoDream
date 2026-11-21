"""Check 1: residual baselines B2-B5 -- shapes, monotone quantiles, determinism, small-sample rules.

Fixture arrays only (fixture_spec.py). Prints shapes/booleans/warning text, never an error metric.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import json  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import Checks, FixtureSpec, fixture_rows  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.engine.types import ResidualModel  # noqa: E402
from learned_judgment.eval.baselines import (  # noqa: E402
    ConstantBiasResidual,
    GaussianProcessResidual,
    GradientBoostingResidual,
    MLPAttemptsResidual,
)

config = load_config()  # read only: engine.* and baselines.* settings
spec = FixtureSpec()
layout = build_layout(spec)
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
K, Q = spec.n_outcome, levels.size
seed = stream_seed(int(require(config, "runtime.seed")), "fixture/baselines/check1")
print(f"layout F={layout.n_features} K={K} Q={Q} categorical={layout.categorical_indices} seed={seed}")

s, a, p, r = fixture_rows(spec, 12, seed=1)
X = build_features(layout, s, a, physics=p)
sq, aq, pq, _ = fixture_rows(spec, 50, seed=2)
Xq = build_features(layout, sq, aq, physics=pq)

checks = Checks()
expect = checks.expect


classes = [ConstantBiasResidual, GaussianProcessResidual, GradientBoostingResidual, MLPAttemptsResidual]
for cls in classes:
    print(f"\n== {cls.__name__} (from_config) ==")
    first = cls.from_config(config, layout, seed=seed).fit(X, r)
    second = cls.from_config(config, layout, seed=seed).fit(X, r)
    d1 = first.predict_residual_distribution(Xq)
    d2 = second.predict_residual_distribution(Xq)
    median, spread = first.predict_residuals(Xq)
    expect(isinstance(first, ResidualModel), "implements the ResidualModel protocol")
    expect(d1.mean.shape == (50, K) and d1.median.shape == (50, K), f"mean/median shape (50, {K}): {d1.mean.shape}")
    expect(d1.quantiles.shape == (50, K, Q), f"quantiles shape (50, {K}, {Q}): {d1.quantiles.shape}")
    expect(median.shape == (50, K) and spread.shape == (50, K), "predict_residuals -> (median, spread) each (50, K)")
    expect(bool(np.all(np.diff(d1.quantiles, axis=-1) >= 0.0)), "quantiles non-decreasing along Q")
    expect(bool(np.all(d1.spread >= 0.0)), "spread >= 0")
    expect(np.array_equal(d1.quantile_levels, levels), "quantile levels equal config engine.quantile_levels")
    same = all(np.array_equal(x, y) for x, y in ((d1.mean, d2.mean), (d1.median, d2.median), (d1.quantiles, d2.quantiles)))
    expect(same, "two fits with the same seed give bitwise-identical predictions")
    json.dumps(first.describe())
    print(f"     describe() JSON-serializable; n_train={first.n_train}; fit_warnings={len(first.fit_warnings)}")
    for line in first.fit_warnings[:2]:
        print("     warning:", line[:160])

print("\n== small-sample behaviour ==")
s1, a1, p1, r1 = fixture_rows(spec, 1, seed=3)
X1 = build_features(layout, s1, a1, physics=p1)
cb = ConstantBiasResidual.from_config(config, layout, seed=seed).fit(X1, r1)
d = cb.predict_residual_distribution(Xq[:4])
expect(bool(np.all(d.quantiles == r1[0][None, :, None])), "constant bias with 1 attempt: every quantile equals the single residual")
expect(bool(np.all(d.spread == 0.0)), "constant bias with 1 attempt: spread 0")
expect(bool(np.all(d.mean == r1[0][None, :])), "constant bias with 1 attempt: mean equals that residual")

gp_min = int(require(config, "baselines.gaussian_process.min_attempts"))
try:
    GaussianProcessResidual.from_config(config, layout, seed=seed).fit(X[: gp_min - 1], r[: gp_min - 1])
    expect(False, f"GP refuses {gp_min - 1} rows (< min_attempts={gp_min})")
except ValueError as error:
    expect(True, f"GP refuses {gp_min - 1} rows: ValueError: {error}")
gp = GaussianProcessResidual.from_config(config, layout, seed=seed).fit(X[:gp_min], r[:gp_min])
expect(gp.predict_residual_distribution(Xq).quantiles.shape == (50, K, Q), f"GP fits at min_attempts={gp_min}")
print(f"     GP fit_warnings at n={gp_min}: {len(gp.fit_warnings)} (captured on the object, not silenced)")

gb_min = int(require(config, "baselines.gradient_boosting.min_attempts"))
try:
    GradientBoostingResidual.from_config(config, layout, seed=seed).fit(X[: gb_min - 1], r[: gb_min - 1])
    expect(False, f"GB refuses {gb_min - 1} rows")
except ValueError as error:
    expect(True, f"GB refuses {gb_min - 1} rows: ValueError: {error}")
gb5 = GradientBoostingResidual.from_config(config, layout, seed=seed).fit(X[:5], r[:5])
d5 = gb5.predict_residual_distribution(Xq)
unique_mean = [int(np.unique(d5.mean[:, k]).size) for k in range(K)]
expect(unique_mean[0] > 1, f"GB at 5 rows with config settings is NOT constant (unique mean predictions per outcome: {unique_mean})")
leaves = max(predictor[0].get_n_leaf_nodes() for predictor in gb5._point_state[0]._predictors)
print(f"     GB at 5 rows: max leaves in any iteration of the first point model = {leaves}")

mlp_min = int(require(config, "baselines.mlp_on_attempts.min_attempts"))
mlp = MLPAttemptsResidual.from_config(config, layout, seed=seed).fit(X[:mlp_min], r[:mlp_min])
dm = mlp.predict_residual_distribution(Xq)
expect(dm.quantiles.shape == (50, K, Q), f"MLP fits at min_attempts={mlp_min}")
offsets = dm.quantiles - dm.median[:, :, None]
row_spread = float(np.ptp(offsets, axis=0).max())
rounding = 8 * np.finfo(np.float64).eps * float(np.abs(dm.quantiles).max())
expect(
    row_spread <= rounding,
    f"MLP quantile - median offsets equal for every query row up to float rounding (homoscedastic): "
    f"max deviation {row_spread:.3g} <= {rounding:.3g}",
)
expect(bool(np.all(dm.spread == dm.spread[:1, :])), "MLP spread is exactly the same value for every query row (broadcast)")
expect(mlp.folds_used == min(int(require(config, "baselines.mlp_on_attempts.interval_folds")), mlp_min),
       f"MLP used K = min(interval_folds, n) = {mlp.folds_used} folds for its out-of-fold errors")

print("\n== constructor checks ==")
for label, build in (
    ("spread level not a member", lambda: ConstantBiasResidual(layout, levels, (0.2, 0.9), seed=seed)),
    ("negative seed", lambda: ConstantBiasResidual(layout, levels, (0.1, 0.9), seed=-1)),
    ("wrong feature width", lambda: ConstantBiasResidual.from_config(config, layout, seed=seed).fit(X[:, :-1], r)),
):
    try:
        build()
        expect(False, f"rejects {label}")
    except (ValueError, TypeError) as error:
        expect(True, f"rejects {label}: {type(error).__name__}: {error}")

checks.finish()
