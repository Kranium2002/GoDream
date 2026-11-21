"""Check 2: DirectTargetAdapter (B6 "direct, no physics feature") round trip.

physics + returned residual must equal the inner model's predicted OUTCOME (up to float
rounding of one subtraction and one addition), and the inner model must have been fitted on
the outcome itself. Fixture arrays only.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import numpy as np  # noqa: E402

from fixture_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSpec, fixture_rows  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import decide_batch  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.engine.types import ResidualModel  # noqa: E402
from learned_judgment.eval.baselines import (  # noqa: E402
    ConstantBiasResidual,
    DirectTargetAdapter,
    GradientBoostingResidual,
    MLPAttemptsResidual,
)

checks = Checks()
expect = checks.expect
config = load_config()
spec = FixtureSpec()
direct_layout = build_layout(spec, include_physics=False)
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
spread_levels = tuple(require(config, "engine.spread_levels"))
seed = stream_seed(int(require(config, "runtime.seed")), "fixture/baselines/check2")

s, a, p, r = fixture_rows(spec, 15, seed=11)
X = build_features(direct_layout, s, a)
measured = r + p
sq, aq, _, _ = fixture_rows(spec, 40, seed=12)
Xq = build_features(direct_layout, sq, aq)
physics_q = spec.physics_guess(sq, aq)
print(f"direct layout F={direct_layout.n_features} names={direct_layout.names} n_physics={direct_layout.n_physics}")

for cls in (ConstantBiasResidual, GradientBoostingResidual, MLPAttemptsResidual):
    print(f"\n== DirectTargetAdapter({cls.__name__}) ==")
    adapter = DirectTargetAdapter(cls.from_config(config, direct_layout, seed=seed), spec=spec, layout=direct_layout)
    adapter.fit(X, r)
    expect(isinstance(adapter, ResidualModel), "adapter implements the ResidualModel protocol")
    returned = adapter.predict_residual_distribution(Xq)
    inner_outcome = adapter.inner.predict_residual_distribution(Xq)
    pairs = (
        ("mean", physics_q + returned.mean, inner_outcome.mean),
        ("median", physics_q + returned.median, inner_outcome.median),
        ("quantiles", physics_q[:, :, None] + returned.quantiles, inner_outcome.quantiles),
    )
    for label, rebuilt, target in pairs:
        error = float(np.max(np.abs(rebuilt - target)))
        bound = 4 * np.finfo(np.float64).eps * max(float(np.abs(target).max()), float(np.abs(physics_q).max()))
        expect(error <= bound, f"physics + returned {label} == inner predicted outcome {label} (max |diff| {error:.3g} <= {bound:.3g})")
    expect(np.array_equal(returned.spread, inner_outcome.spread), "spread passed through unchanged")
    reference = cls.from_config(config, direct_layout, seed=seed).fit(X, measured).predict_residual_distribution(Xq)
    expect(
        np.array_equal(reference.mean, inner_outcome.mean) and np.array_equal(reference.quantiles, inner_outcome.quantiles),
        "inner model was fitted on the outcome: bitwise equal to a separate fit on residual + physics",
    )
    expect(adapter.method == f"direct_{cls.method}", f"method label {adapter.method!r}")

print("\n== decide_batch with the adapter ==")
adapter = DirectTargetAdapter(GradientBoostingResidual.from_config(config, direct_layout, seed=seed), spec=spec, layout=direct_layout).fit(X, r)
decisions = decide_batch(
    spec, sq[:3], model=adapter, layout=direct_layout, points_per_dim=FIXTURE_POINTS_PER_DIM,
    quantile_levels=levels, spread_levels=spread_levels, method=adapter.method,
)
for decision in decisions:
    situations = np.repeat(decision.situation[None, :], decision.candidates.shape[0], axis=0)
    direct = adapter.inner.predict_residual_distribution(build_features(direct_layout, situations, decision.candidates))
    expected_mean = np.clip(direct.mean, spec.outcome_low, spec.outcome_high)
    error = float(np.max(np.abs(decision.predicted.mean - expected_mean)))
    bound = 4 * np.finfo(np.float64).eps * float(np.abs(direct.mean).max() + np.abs(decision.predicted.physics).max())
    expect(error <= bound, f"decided outcome mean = clip(inner direct prediction) (max |diff| {error:.3g} <= {bound:.3g})")

print("\n== refusals ==")
try:
    DirectTargetAdapter(ConstantBiasResidual.from_config(config, build_layout(spec), seed=seed), spec=spec, layout=build_layout(spec))
    expect(False, "refuses a layout with the physics block")
except ValueError as error:
    expect(True, f"refuses a layout with the physics block: {error}")
try:
    DirectTargetAdapter(ConstantBiasResidual.from_config(config, build_layout(spec), seed=seed), spec=spec, layout=direct_layout)
    expect(False, "refuses an inner model built for another layout")
except ValueError as error:
    expect(True, f"refuses an inner model built for another layout: {error}")

checks.finish()
