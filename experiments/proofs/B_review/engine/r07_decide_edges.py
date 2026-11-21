"""R07: decide_batch / OutcomeDistribution edge cases."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, TS, ReviewSpec  # noqa: E402

from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.learn import fit_residual_model  # noqa: E402
from learned_judgment.engine.rows import build_real_row  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution, ResidualDistribution, cdf_from_quantiles  # noqa: E402
from learned_judgment.eval.baselines import ConstantBiasResidual  # noqa: E402

spec = ReviewSpec()
layout = build_layout(spec)
rng = np.random.default_rng(3)
sit = spec.sample_situations(rng, 6)
act = np.column_stack([rng.uniform(-1, 1, 6), rng.uniform(0, 0.4, 6)])
rows = [build_real_row(spec, run_id="rv", seed=1, world_preset="default", situation=sit[i], action=act[i],
                       measured_outcome={"rv_sep": 0.2 + 0.1 * i, "rv_hit": float(i % 2)}, timestamp_utc=TS) for i in range(6)]
model = fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=0), rows, layout)
kw = dict(layout=layout, points_per_dim=[3, 3], quantile_levels=LEVELS, spread_levels=SPREAD)

# 1. alignment: per-situation slices equal a one-at-a-time decision
batch = decide_batch(spec, sit, model=model, method="constant_bias", **kw)
single = [decide_batch(spec, s[None, :], model=model, method="constant_bias", **kw)[0] for s in sit]
print("1. batched == one-at-a-time (candidates, scores, quantiles):",
      all(np.array_equal(b.candidates, s.candidates) and np.array_equal(b.scores, s.scores)
          and np.array_equal(b.predicted.quantiles, s.predicted.quantiles) for b, s in zip(batch, single)))
print("   physics action is one of the candidates in every decision:",
      all(any(np.array_equal(b.physics_action, c) for c in b.candidates) for b in batch))

# 2. zero situations
print("2. zero situations, physics only ->", decide_batch(spec, np.empty((0, 2)), model=None, method=PHYSICS_ONLY, **kw))
print("   zero situations, constant_bias ->", decide_batch(spec, np.empty((0, 2)), model=model, method="constant_bias", **kw))

# 3. physics-only objective that is NaN on a point mass (e.g. divides by spread): checked only without a model
class SpreadRatioSpec(ReviewSpec):
    def objective(self, predicted):
        with np.errstate(divide="ignore", invalid="ignore"):
            return predicted.median_of("rv_sep") / predicted.spread_of("rv_sep") - predicted.mean_of("rv_hit")


ratio = SpreadRatioSpec()
try:
    decide_batch(ratio, sit[:1], model=None, method=PHYSICS_ONLY, **kw)
except ValueError as error:
    print("3. physics only with a NaN point-mass objective ->", error)
d = decide_batch(ratio, sit[:1], model=model, method="constant_bias", **kw)[0]
with np.errstate(divide="ignore", invalid="ignore"):
    phys_scores = ratio.objective(OutcomeDistribution.point(ratio.outcome_names, d.predicted.physics, np.array(LEVELS)))
print("   with a model: no error; physics_scores =", np.round(phys_scores, 3).tolist())
print("   recorded physics_index =", d.physics_index, "(argmax over all-non-finite scores: an arbitrary candidate)")

# 4. prob_below: tails, point mass, near point mass
levels = np.array(LEVELS)
q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0, 3.0 + 1e-12]])
for t in (0.0, 1.0, 2.5, 5.0, 9.0):
    print(f"4. cdf t={t}:", cdf_from_quantiles(q, levels, t).round(4).tolist(), " rows=[spread, point mass at 3, near-point mass at 3]")

# 5. crossing quantiles accepted when a model builds ResidualDistribution directly
crossing = ResidualDistribution(mean=np.zeros((1, 1)), median=np.zeros((1, 1)), quantile_levels=levels,
                                quantiles=np.array([[[0.0, 1.0, 0.5, 2.0, 3.0]]]), spread=np.ones((1, 1)), spread_levels=(0.1, 0.9))
print("5. ResidualDistribution(...) accepts crossing quantiles:", crossing.quantiles[0, 0].tolist())
print("   cdf at t=0.6, 0.9, 1.0 ->", [float(cdf_from_quantiles(crossing.quantiles[:, 0, :], levels, t)[0]) for t in (0.6, 0.9, 1.0)])

# 6. binary clipping: physics 0.9 + residual quantiles up to +0.5 -> quantiles clipped at 1, mean clipped at 1
res = ResidualDistribution.from_quantiles(np.array([[0.0, 0.3]]), np.array([[0.0, 0.2]]), levels,
                                          np.array([[[0, 0, 0, 0, 0], [-0.2, 0.0, 0.2, 0.4, 0.5]]], dtype=float), (0.1, 0.9))
od = OutcomeDistribution.from_physics_and_residual(spec.outcome_names, np.array([[0.5, 0.9]]), res, spec.outcome_low, spec.outcome_high)
print("6. binary: mean", od.mean[0, 1], "quantiles", od.quantiles[0, 1].round(3).tolist(), "spread (unclipped residual)", od.spread[0, 1])
