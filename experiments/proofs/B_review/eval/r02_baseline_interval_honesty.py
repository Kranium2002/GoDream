"""Review reproduction: held-out coverage of each residual baseline's predictive interval at the
attempt counts of the learning curve, on fixture residuals from a KNOWN process
(residual = 0.2*surface - 0.3*side_speed + 0.05*N(0,1), from fixture_rows of B_eval_offline).

A baseline whose 80% interval [q0.1, q0.9] is honest covers about 80% of fresh draws. These
quantiles enter spec.objective in decide_batch, so an interval that is far too narrow makes
the method's decisions ignore risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "proofs" / "B_eval_offline"))

from fixture_spec import FixtureSpec, fixture_rows  # noqa: E402

from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.eval.baselines import (  # noqa: E402
    ConstantBiasResidual,
    GaussianProcessResidual,
    GradientBoostingResidual,
    MLPAttemptsResidual,
)

print("REVIEW REPRODUCTION - fixture, not evidence")

config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
spec = FixtureSpec()
layout = build_layout(spec)
levels = np.asarray(config["engine"]["quantile_levels"])
lo, hi = list(levels).index(0.1), list(levels).index(0.9)

# held-out set from the same process
s_te, a_te, p_te, r_te = fixture_rows(spec, 2000, seed=999)
x_te = build_features(layout, s_te, a_te, physics=p_te)

classes = {
    "constant_bias": ConstantBiasResidual,
    "gaussian_process": GaussianProcessResidual,
    "gradient_boosting": GradientBoostingResidual,
    "mlp_on_attempts": MLPAttemptsResidual,
}
print(f"{'method':18s} {'n':>3s}  cov80(separation)  mean spread  (nominal 0.80; mean over 5 fixture seeds)")
for name, cls in classes.items():
    for n in (5, 10, 20, 40):
        covs, spreads = [], []
        for seed in range(5):
            s, a, p, r = fixture_rows(spec, n, seed=100 + seed)
            x = build_features(layout, s, a, physics=p)
            model = cls.from_config(config, layout, seed=seed).fit(x, r)
            dist = model.predict_residual_distribution(x_te)
            q = dist.quantiles[:, 0, :]
            covs.append(np.mean((r_te[:, 0] >= q[:, lo]) & (r_te[:, 0] <= q[:, hi])))
            spreads.append(dist.spread[:, 0].mean())
        print(f"{name:18s} {n:3d}  {np.mean(covs):.3f}              {np.mean(spreads):.4f}")
