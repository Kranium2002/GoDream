"""Check 3: every baseline plugs into learned_judgment.engine.decide.decide_batch.

Each decision must list its candidates with the physics action among them, and the physics
index must equal the physics-only choice on the same grid. Fixture spec and arrays only.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import numpy as np  # noqa: E402

from fixture_spec import FIXTURE_COMMAND_SCHEMA, FIXTURE_POINTS_PER_DIM, Checks, FixtureSpec, fixture_rows  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.eval.baselines import (  # noqa: E402
    ConstantBiasResidual,
    DirectTargetAdapter,
    GaussianProcessResidual,
    GradientBoostingResidual,
    MLPAttemptsResidual,
)
from learned_judgment.skills.base import validate_spec  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()
spec = FixtureSpec()
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
spread_levels = tuple(require(config, "engine.spread_levels"))
validate_spec(
    spec,
    FIXTURE_COMMAND_SCHEMA,
    hidden_value_names=("payload_mass_kg",),
    quantile_levels=levels,
    probe_seed=0,
    probe_situations=int(require(config, "engine.spec_validation.probe_situations")),
    probe_points_per_dim=int(require(config, "engine.spec_validation.probe_points_per_dim")),
)
print("fixture spec passes validate_spec against a FIXTURE command schema (not the Go2 schema)")
seed = stream_seed(int(require(config, "runtime.seed")), "fixture/baselines/check3")

layout = build_layout(spec)
direct_layout = build_layout(spec, include_physics=False)
s, a, p, r = fixture_rows(spec, 12, seed=21)
X = build_features(layout, s, a, physics=p)
X_direct = build_features(direct_layout, s, a)
situations, _, _, _ = fixture_rows(spec, 4, seed=22)

physics_only = decide_batch(
    spec, situations, model=None, layout=layout, points_per_dim=FIXTURE_POINTS_PER_DIM,
    quantile_levels=levels, spread_levels=spread_levels, method=PHYSICS_ONLY,
)

models = [cls.from_config(config, layout, seed=seed).fit(X, r) for cls in
          (ConstantBiasResidual, GaussianProcessResidual, GradientBoostingResidual, MLPAttemptsResidual)]
models.append(
    DirectTargetAdapter(GradientBoostingResidual.from_config(config, direct_layout, seed=seed), spec=spec, layout=direct_layout).fit(X_direct, r)
)

for model in models:
    model_layout = model.layout
    decisions = decide_batch(
        spec, situations, model=model, layout=model_layout, points_per_dim=FIXTURE_POINTS_PER_DIM,
        quantile_levels=levels, spread_levels=spread_levels, method=model.method,
    )
    print(f"\n== {model.method} ==")
    expect(len(decisions) == situations.shape[0], f"{len(decisions)} decisions for {situations.shape[0]} situations")
    for decision, reference in zip(decisions, physics_only):
        n_candidates = decision.candidates.shape[0]
        in_set = bool(np.any(np.all(decision.candidates == decision.physics_action[None, :], axis=1)))
        expect(in_set, f"physics action {decision.physics_action.tolist()} is one of the {n_candidates} candidates")
        expect(
            np.array_equal(decision.candidates, reference.candidates) and decision.physics_index == reference.physics_index,
            "same candidate grid and physics index as the physics-only decision",
        )
        expect(
            decision.predicted.quantiles.shape == (n_candidates, spec.n_outcome, levels.size)
            and decision.scores.shape == (n_candidates,)
            and 0 <= decision.chosen_index < n_candidates,
            f"predicted ({n_candidates}, K, Q), scores ({n_candidates},), chosen index {decision.chosen_index}",
        )
        expect(decision.method == model.method, f"decision.method == {model.method!r}")

checks.finish()
