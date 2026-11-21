"""Review reproduction: hand checks of metric formulas and bootstrap / ratio / Spearman edge cases."""

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
from learned_judgment.eval.baselines import MLPAttemptsResidual  # noqa: E402
from learned_judgment.eval.bootstrap import paired_difference_ci, paired_ratio_ci  # noqa: E402
from learned_judgment.eval.calibration import spread_error_spearman  # noqa: E402
from learned_judgment.eval.metrics import (  # noqa: E402
    attempts_to_tolerance,
    learning_curve_area,
    regression_summary,
    regret,
    student_teacher_ratio,
)

print("REVIEW REPRODUCTION - fixture, not evidence")
spec = FixtureSpec()
config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
levels = config["engine"]["quantile_levels"]

print("\n== hand checks ==")
counts, errors = np.array([0, 5, 10, 20, 40]), np.array([1.0, 0.8, 0.5, 0.4, 0.3])
hand = 2.5 * (1.0 + 0.8) + 2.5 * (0.8 + 0.5) + 5 * (0.5 + 0.4) + 10 * (0.4 + 0.3)
print("area", learning_curve_area(counts, errors), "hand", hand)
print("attempts_to_tolerance(0.5)", attempts_to_tolerance(counts, errors, 0.5), "| (0.2)", attempts_to_tolerance(counts, errors, 0.2))
print("non-monotone curve [1, .4, .9, .9, .9] tol .5 ->", attempts_to_tolerance(counts, [1, .4, .9, .9, .9], 0.5), "(first reach, even though it leaves tolerance)")
print("p90 of 1..5", regression_summary([1, 2, 3, 4, 5])["p90"], "hand 4.6")
m_meth = np.array([[0.5, 0.0], [0.1, 1.0]])
m_orc = np.array([[0.7, 0.0], [0.4, 0.0]])
# fixture objective = q_low(separation) - 2*mean(contact); point mass => sep - 2*contact
hand_regret = (m_orc[:, 0] - 2 * m_orc[:, 1]) - (m_meth[:, 0] - 2 * m_meth[:, 1])
print("regret", regret(spec, m_meth, m_orc, levels).tolist(), "hand", hand_regret.tolist())

print("\n== cluster bootstrap equals resampling whole situations (percentile, same seed) ==")
rng = np.random.default_rng(3)
A = rng.gamma(2.0, 1.0, (30, 3))
B = A * 1.1 + rng.normal(0, 0.2, (30, 3))
res = paired_difference_ci(A, B, n_resamples=9999, confidence_level=0.95, method="percentile", seed=11)
idx = np.random.default_rng(11).integers(0, 30, (9999, 30))
boots = (B[idx] - A[idx]).mean(axis=(1, 2))
print("module", (res["low"], res["high"]), "| whole-cluster numpy", tuple(np.quantile(boots, [0.025, 0.975])))
r = paired_ratio_ci(A, B, n_resamples=9999, confidence_level=0.95, method="percentile", seed=11)
rb = B[idx].mean(axis=(1, 2)) / A[idx].mean(axis=(1, 2))
print("ratio module", (r["estimate"], r["low"], r["high"]), "| numpy", (B.mean() / A.mean(), *np.quantile(rb, [0.025, 0.975])))

print("\n== ratio with losses that are not non-negative ==")
teacher = np.array([-0.2, -0.1, -0.3, -0.2])  # e.g. a loss defined as -utility
student = np.array([-0.4, -0.2, -0.6, -0.4])  # lower loss = better: the student is BETTER everywhere
print("student_teacher_ratio:", student_teacher_ratio(student, teacher))
T = np.tile(teacher[:, None], (1, 3)) + np.random.default_rng(1).normal(0, 0.01, (4, 3))
S = np.tile(student[:, None], (1, 3)) + np.random.default_rng(2).normal(0, 0.01, (4, 3))
r = paired_ratio_ci(T, S, n_resamples=2000, confidence_level=0.95, method="BCa", seed=1)
print("paired_ratio_ci estimate/low/high:", r["estimate"], r["low"], r["high"], "| undefined_reason:", r["undefined_reason"])
# sign-crossing denominator: teacher mean near 0 but not exactly 0
T2 = np.array([[0.3], [-0.29], [0.02], [-0.02], [0.01], [-0.015]])
S2 = np.abs(T2) + 0.05
r2 = paired_ratio_ci(T2, S2, n_resamples=2000, confidence_level=0.95, method="BCa", seed=1)
print("sign-crossing denominator -> estimate", r2["estimate"], "CI", (r2["low"], r2["high"]), "method_used", r2["method_used"])

print("\n== Spearman on a homoscedastic-by-construction spread (MLP baseline) ==")
layout = build_layout(spec)
s, a, p, res_ = fixture_rows(spec, 20, seed=5)
mlp = MLPAttemptsResidual.from_config(config, layout, seed=3).fit(build_features(layout, s, a, physics=p), res_)
s2, a2, p2, r2_ = fixture_rows(spec, 200, seed=6)
dist = mlp.predict_residual_distribution(build_features(layout, s2, a2, physics=p2))
spread = dist.spread[:, 0]
abs_err = np.abs(r2_[:, 0] - dist.median[:, 0])
print("distinct spread values:", np.unique(spread).size, "range", float(spread.max() - spread.min()))
print("spread_error_spearman:", spread_error_spearman(spread, abs_err))
