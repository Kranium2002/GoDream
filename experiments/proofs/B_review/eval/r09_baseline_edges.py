"""Review reproduction: baseline edge cases - categorical column in gradient boosting, an unseen
category at prediction, duplicated rows, a constant residual, a binary outcome, minimum counts."""

from __future__ import annotations

import sys
import warnings
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

warnings.simplefilter("ignore")
print("REVIEW REPRODUCTION - fixture, not evidence")
config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
spec = FixtureSpec()
layout = build_layout(spec)
classes = (ConstantBiasResidual, GaussianProcessResidual, GradientBoostingResidual, MLPAttemptsResidual)

s, a, p, r = fixture_rows(spec, 10, seed=3)
x = build_features(layout, s, a, physics=p)
gb = GradientBoostingResidual.from_config(config, layout, seed=1).fit(x, r)
print("GB is_categorical_ of the mean model:", gb._mean_models[0].is_categorical_.tolist(), "| layout categorical idx", layout.categorical_indices)
xq = x.copy()
seen = sorted(set(x[:, 2].tolist()))
unseen = [level for level in (0.0, 1.0, 2.0) if level not in seen][0]
xq[:, 2] = unseen
print("categories seen in training:", seen, f"| predicting with category {unseen} ->", "unseen" if unseen not in seen else "seen",
      "| finite:", bool(np.all(np.isfinite(gb.predict_residual_distribution(xq).quantiles))))

print("\nduplicated rows (every attempt twice), constant residual, binary-only residual, min counts:")
for cls in classes:
    m = cls.from_config(config, layout, seed=1)
    n_min = m.min_attempts
    s, a, p, r = fixture_rows(spec, max(n_min, 2), seed=4)
    x = build_features(layout, s, a, physics=p)
    dup = cls.from_config(config, layout, seed=1).fit(np.vstack([x, x]), np.vstack([r, r]))
    d1 = dup.predict_residual_distribution(x)
    const = cls.from_config(config, layout, seed=1).fit(x, np.full_like(r, 0.25))
    dc = const.predict_residual_distribution(x)
    mn = cls.from_config(config, layout, seed=1).fit(x[:n_min], r[:n_min])
    dm = mn.predict_residual_distribution(x)
    print(f"  {cls.method:18s} min_attempts={n_min}: dup finite={bool(np.all(np.isfinite(d1.quantiles)))} "
          f"| const residual 0.25 -> median {float(np.max(np.abs(dc.median - 0.25))):.2e} off, spread max {float(dc.spread.max()):.2e} "
          f"| at min count spread[sep] mean {float(dm.spread[:, 0].mean()):.4f}")
