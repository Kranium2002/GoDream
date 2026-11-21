"""Review reproduction: (a) baseline determinism across thread settings the module does not pin
(OpenMP for HistGradientBoosting); (b) model state after a refit that raises part-way;
(c) ratio CI on a sign-crossing denominator without an exact zero."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "proofs" / "B_eval_offline"))

from fixture_spec import FixtureSpec, fixture_rows  # noqa: E402

from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.eval import baselines as B  # noqa: E402
from learned_judgment.eval.bootstrap import paired_ratio_ci  # noqa: E402

print("REVIEW REPRODUCTION - fixture, not evidence")
config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
spec = FixtureSpec()
layout = build_layout(spec)
s, a, p, r = fixture_rows(spec, 400, seed=1)
x = build_features(layout, s, a, physics=p)
sq, aq, pq, _ = fixture_rows(spec, 300, seed=2)
xq = build_features(layout, sq, aq, physics=pq)

print("\n(a) threadpools:", [(i["user_api"], i["num_threads"]) for i in threadpool_info()])
for name, cls in (("gradient_boosting", B.GradientBoostingResidual), ("gaussian_process", B.GaussianProcessResidual), ("mlp_on_attempts", B.MLPAttemptsResidual)):
    n = 400 if name == "gradient_boosting" else 40
    preds = []
    for limit in (None, 1, 4):
        if limit is None:
            m = cls.from_config(config, layout, seed=5).fit(x[:n], r[:n])
            preds.append(m.predict_residual_distribution(xq).quantiles)
        else:
            with threadpool_limits(limits=limit, user_api="openmp"):
                m = cls.from_config(config, layout, seed=5).fit(x[:n], r[:n])
                preds.append(m.predict_residual_distribution(xq).quantiles)
    print(f"   {name:18s} n={n}: openmp default vs 1 bitwise {np.array_equal(preds[0], preds[1])}, default vs 4 bitwise {np.array_equal(preds[0], preds[2])}, max|diff| {float(np.max(np.abs(preds[0]-preds[1]))):.3g}")

print("\n(b) refit that raises part-way (failure injected into the 2nd outcome's GP fit)")
gp = B.GaussianProcessResidual.from_config(config, layout, seed=5).fit(x[:20], r[:20])
before = gp.predict_residual_distribution(xq).mean.copy()
real_fit = B.GaussianProcessRegressor.fit
calls = {"n": 0}


def failing_fit(self, X, y):  # injected failure, stands for any exception inside scikit-learn
    calls["n"] += 1
    if calls["n"] == 2:
        raise np.linalg.LinAlgError("injected: kernel not positive definite")
    return real_fit(self, X, y)


B.GaussianProcessRegressor.fit = failing_fit
try:
    gp.fit(x[200:240] * 3.0 + 5.0, r[200:240])  # new attempts with a very different feature scale
except np.linalg.LinAlgError as error:
    print("   refit raised:", error)
finally:
    B.GaussianProcessRegressor.fit = real_fit
after = gp.predict_residual_distribution(xq).mean
print("   model still predicts after the failed refit:", True, "| n_train reported:", gp.n_train)
print("   predictions changed although no new model was installed:", not np.array_equal(before, after),
      "max|diff|", float(np.max(np.abs(before - after))))

print("\n(c) ratio CI when the denominator mean is near 0 and changes sign across resamples")
rng = np.random.default_rng(4)
T = rng.normal(0.02, 0.2, (25, 3))
S = np.abs(rng.normal(0.3, 0.1, (25, 3)))
res = paired_ratio_ci(T, S, n_resamples=4000, confidence_level=0.95, method="BCa", seed=1)
print("   estimate", res["estimate"], "low", res["low"], "high", res["high"], "method", res["method_used"], "undefined_reason", res["undefined_reason"])
