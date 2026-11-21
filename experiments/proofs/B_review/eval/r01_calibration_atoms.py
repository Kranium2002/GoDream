"""Review reproduction: calibration functions on PERFECTLY calibrated predictions of outcomes
that have atoms (a binary outcome, and a regression outcome clipped at a physical bound).

Every quantile below is the exact quantile of the true generating distribution, so a correct
calibration check must NOT reject and coverage must match its nominal level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from learned_judgment.eval.calibration import (  # noqa: E402
    interval_coverage,
    pit_bin_counts,
    pit_chisquare,
    randomized_pit_ks,
)

print("REVIEW REPRODUCTION - fixture, not evidence")

LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])  # config.yaml engine.quantile_levels
N = 5000
rng = np.random.default_rng(0)

# ---- (A) continuous control: Gaussian, exact quantiles ------------------------------------
mu = rng.uniform(-1, 1, N)
y = mu + rng.normal(size=N)
q = mu[:, None] + stats.norm.ppf(LEVELS)[None, :]
print("\n(A) continuous Gaussian, exact quantiles (control)")
print("  cov80", interval_coverage(q, LEVELS, y, 0.1, 0.9)["coverage"], "nominal 0.8")
print("  chi2 p", round(pit_chisquare(q, LEVELS, y)["pvalue"], 4), "| rand-PIT KS p", round(randomized_pit_ks(q, LEVELS, y, seed=1)["pvalue"], 4))

# ---- (B) regression outcome clipped at a bound (e.g. a duration or distance >= 0) ----------
# True outcome Y = max(0, Z), Z ~ N(mu, 1). Exact quantiles of Y: max(0, mu + z_L).
# This is exactly what OutcomeDistribution.from_physics_and_residual produces for an outcome with
# low=0 when the residual quantiles are right: clip(physics + residual quantile, low, high).
y_c = np.maximum(0.0, mu + rng.normal(size=N))
q_c = np.maximum(0.0, mu[:, None] + stats.norm.ppf(LEVELS)[None, :])
print("\n(B) clipped-at-0 outcome, exact quantiles of the clipped truth")
print("  share of y exactly 0:", float(np.mean(y_c == 0.0)))
cov = interval_coverage(q_c, LEVELS, y_c, 0.1, 0.9)
print("  cov80", cov["coverage"], "nominal", cov["nominal"])
chi = pit_chisquare(q_c, LEVELS, y_c)
print("  bin counts", chi["counts"], "expected", [round(v) for v in chi["expected"]])
print("  chi2 p", chi["pvalue"], "| rand-PIT KS p", randomized_pit_ks(q_c, LEVELS, y_c, seed=1)["pvalue"])

# ---- (C) binary outcome (collision / fall), exact Bernoulli quantiles ----------------------
# Quantile function of Bernoulli(p): Q(L) = 0 if L <= 1 - p else 1.
p = rng.uniform(0.05, 0.95, N)
y_b = (rng.uniform(size=N) < p).astype(np.float64)
q_b = (LEVELS[None, :] > (1.0 - p)[:, None]).astype(np.float64)
print("\n(C) binary outcome, exact Bernoulli quantiles")
cov = interval_coverage(q_b, LEVELS, y_b, 0.1, 0.9)
print("  cov80", cov["coverage"], "nominal", cov["nominal"])
cov50 = interval_coverage(q_b, LEVELS, y_b, 0.25, 0.75)
print("  cov50", cov50["coverage"], "nominal", cov50["nominal"])
print("  bin counts", pit_bin_counts(q_b, LEVELS, y_b).tolist())
print("  chi2 p", pit_chisquare(q_b, LEVELS, y_b)["pvalue"], "| rand-PIT KS p", randomized_pit_ks(q_b, LEVELS, y_b, seed=1)["pvalue"])

# ---- (D) binary outcome as the engine would present it: continuous residual quantiles added
# to a physics probability and clipped to [0, 1]. y = 0 can never exceed a knot (all knots >= 0)
print("\n(D) binary outcome, quantiles = clip(p + residual quantiles, 0, 1)")
r_q = (np.array([0.0, 0.0, 0.0, 1.0, 1.0])[None, :] - p[:, None]) + rng.normal(0, 0.05, (N, 5))
q_d = np.clip(p[:, None] + np.sort(r_q, axis=1), 0.0, 1.0)
counts = pit_bin_counts(q_d, LEVELS, y_b)
print("  bin counts", counts.tolist(), "(y=0 rows can only land in bin 0, y=1 rows mostly in bin 5)")
print("  rows with y=0 in bin 0:", int(np.sum((y_b == 0) & ((y_b[:, None] > q_d).sum(1) == 0))), "of", int(np.sum(y_b == 0)))
