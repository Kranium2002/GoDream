"""R03: dream provenance.
(a) Request-hash attribution: TabPFNTeacher in multi_outcome_mode "per_outcome" makes one
    request per active outcome and exposes every key in ``last_request_hashes`` while
    ``last_request_hash`` is only the LAST key (teacher.py:929-930). decide_batch reads only
    ``last_request_hash``. A duck-typed fixture model sets both attributes the same way the
    teacher does (no provider, no request).
(b) Spread levels: decide_batch does not compare the model's spread_levels with the ones the
    caller passes, and dreams_from_decisions records the caller's.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, TS, ReviewSpec  # noqa: E402

from learned_judgment.engine.dream import dream, dream_rows, keep_confident  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.learn import fit_residual_model  # noqa: E402
from learned_judgment.engine.rows import build_real_row  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.types import ResidualDistribution  # noqa: E402
from learned_judgment.eval.baselines import ConstantBiasResidual  # noqa: E402

spec = ReviewSpec()
layout = build_layout(spec)


class PerOutcomeLikeModel:
    """FIXTURE: wraps ConstantBiasResidual; records one fake key per outcome like the teacher's
    per_outcome mode (keys list, last_request_hash = keys[-1]). No provider is involved."""

    def __init__(self, inner):
        self.inner = inner
        self.last_request_hash = None
        self.last_request_hashes = ()

    def fit(self, features, residuals):
        self.inner.fit(features, residuals)
        return self

    def predict_residuals(self, q):
        d = self.predict_residual_distribution(q)
        return d.median, d.spread

    def predict_residual_distribution(self, q):
        self.last_request_hash, self.last_request_hashes = None, ()
        out = self.inner.predict_residual_distribution(q)
        keys = [f"fixturekey_outcome{k}" for k in range(layout.n_outcomes)]
        self.last_request_hashes = tuple(keys)
        self.last_request_hash = keys[-1]
        return out


rng = np.random.default_rng(0)
sit = spec.sample_situations(rng, 8)
act = np.column_stack([rng.uniform(-1, 1, 8), rng.uniform(0, 0.4, 8)])
phys = spec.physics_guess(sit, act)
rows = []
for i in range(8):
    measured = {"rv_sep": float(phys[i, 0] + rng.normal(0, 0.1)) if phys[i, 0] > 0.2 else 0.3, "rv_hit": float(i % 2)}
    rows.append(build_real_row(spec, run_id="rv", seed=1, world_preset="default", situation=sit[i], action=act[i],
                               measured_outcome=measured, timestamp_utc=TS))

model = fit_residual_model(spec, PerOutcomeLikeModel(ConstantBiasResidual(layout, LEVELS, SPREAD, seed=0)), rows, layout)
ds = dream(spec, model, layout, n_dreams=5, rng=stream_rng(1, "rv/dreams"), points_per_dim=[3, 3],
           quantile_levels=LEVELS, spread_levels=SPREAD, method="teacher")
print("A. keys the model used for this ONE predict call:", list(model.last_request_hashes))
print("   DreamSet.request_hashes recorded              :", list(ds.request_hashes))
r = dream_rows(spec, ds, run_id="rv", seed=1, world_preset="default", timestamp_utc=TS)[0]
print("   dream row teacher.request_hash                :", r.teacher["request_hash"])

# (b) model built with spread levels (0.25, 0.75); caller passes (0.1, 0.9)
inner = ConstantBiasResidual(layout, LEVELS, (0.25, 0.75), seed=0)
model_b = fit_residual_model(spec, inner, rows, layout)
ds_b = dream(spec, model_b, layout, n_dreams=5, rng=stream_rng(1, "rv/dreams"), points_per_dim=[3, 3],
             quantile_levels=LEVELS, spread_levels=(0.1, 0.9), method="constant_bias")
q = inner._quantiles[0]  # residual quantiles of rv_sep (K, Q)[0]
print("B. DreamSet.spread_levels recorded:", ds_b.spread_levels)
print("   chosen_spread[0] =", ds_b.chosen_spread[0], "| half(q0.75-q0.25) =", 0.5 * (q[3] - q[1]),
      "| half(q0.9-q0.1) =", 0.5 * (q[4] - q[0]))
print("   keep_confident(threshold chosen for (0.1,0.9) spreads) keeps", keep_confident(ds_b, float(0.5 * (q[4] - q[0])) * 0.99).n_rows,
      "of", ds_b.n_rows)
