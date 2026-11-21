"""Check 5: metric functions, paired bootstrap and calibration on fixture arrays.

- metrics: hand-written arrays with answers worked out by hand in the comments
- bootstrap: percentile CI equals a 10-line numpy percentile bootstrap for the same seed;
  identical inputs make BCa degenerate and the fallback is recorded; zero denominators give
  "undefined"
- calibration: a fixture with a KNOWN Gaussian truth (so the right coverage is known);
  levels are read exactly at the knots and nothing is interpolated
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import inspect  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from fixture_spec import Checks, FixtureSpec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.eval import bootstrap, calibration, metrics  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()
spec = FixtureSpec()
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
n_resamples = int(require(config, "evaluation.bootstrap_resamples"))
confidence = float(require(config, "evaluation.bootstrap_confidence_level"))
method = require(config, "evaluation.bootstrap_method")

print("== metrics ==")
summary = metrics.regression_summary([0.0, 1.0, 2.0, 3.0, 4.0])
# mean 2, median 2, p90 (type 7) = 3 + 0.6 * (4 - 3) = 3.6
expect(summary == {"mean": 2.0, "median": 2.0, "p90": 3.6, "n": 5}, f"regression_summary {summary}")
rate = metrics.rate_with_count(np.array([True, False, True, True]))
expect(rate == {"rate": 0.75, "count": 3, "n": 4}, f"rate_with_count {rate}")
measured = np.array([[0.5, 0.0], [0.1, 1.0], [0.3, 0.0], [0.29, 0.0]])
tolerance = metrics.tolerance_rate(spec, measured)  # success: separation >= 0.3 and no contact -> rows 0, 2
expect(tolerance == {"rate": 0.5, "count": 2, "n": 4}, f"tolerance_rate {tolerance}")
oracle = np.array([[0.6, 0.0], [0.4, 0.0], [0.3, 0.0], [0.5, 0.0]])
regret = metrics.regret(spec, measured, oracle, levels)
# fixture objective on a point mass = separation - 2 * contact
by_hand = (oracle[:, 0] - 2 * oracle[:, 1]) - (measured[:, 0] - 2 * measured[:, 1])
expect(np.array_equal(regret, by_hand), f"regret per episode {regret.tolist()} == hand value {by_hand.tolist()}")
counts = np.array([0, 5, 10, 20, 40])
errors = np.array([4.0, 3.0, 2.0, 1.0, 1.0])
area = metrics.learning_curve_area(counts, errors)  # 17.5 + 12.5 + 15 + 20 = 65
expect(area == 65.0, f"learning_curve_area {area} == 65.0")
expect(metrics.attempts_to_tolerance(counts, errors, 2.0) == 10, "attempts_to_tolerance(2.0) == 10 (exact <=)")
expect(metrics.attempts_to_tolerance(counts, errors, 0.5) is None, "attempts_to_tolerance(0.5) is None (not reached)")
ratio = metrics.student_teacher_ratio([1.0, 2.0], [0.0, 0.0])
expect(ratio["ratio"] == "undefined", f"student_teacher_ratio with zero teacher error -> {ratio}")
expect(metrics.student_teacher_ratio([1.0, 3.0], [1.0, 1.0])["ratio"] == 2.0, "student_teacher_ratio 2.0")
agreement = metrics.decision_agreement(np.array([[0.1, 0.0], [0.2, 0.1]]), np.array([[0.1, 0.0], [0.2, 0.2]]))
expect(agreement == {"rate": 0.5, "count": 1, "n": 2}, f"decision_agreement {agreement}")

print("\n== bootstrap ==")
rng = np.random.default_rng(11)
loss_a = np.abs(rng.normal(1.0, 0.3, (30, 3)))
loss_b = loss_a + rng.normal(0.1, 0.05, (30, 3))
seed = 5
percentile = bootstrap.paired_difference_ci(loss_a, loss_b, n_resamples=n_resamples, confidence_level=confidence, method="percentile", seed=seed)


def numpy_percentile_bootstrap(a, b, n_resamples, seed, level):  # the 10-line reference (p06)
    a_s, b_s = a.mean(axis=1), b.mean(axis=1)
    d = b_s - a_s
    generator = np.random.default_rng(seed)
    index = generator.integers(0, d.size, size=(n_resamples, d.size))
    boots = d[index].mean(axis=1)
    low, high = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return d.mean(), low, high


ref_estimate, ref_low, ref_high = numpy_percentile_bootstrap(loss_a, loss_b, n_resamples, seed, confidence)
print(f"     module:    estimate={percentile['estimate']!r} low={percentile['low']!r} high={percentile['high']!r}")
print(f"     reference: estimate={ref_estimate!r} low={ref_low!r} high={ref_high!r}")
expect(percentile["low"] == ref_low and percentile["high"] == ref_high and percentile["estimate"] == ref_estimate,
       "percentile CI equals the numpy reference bitwise for the same seed")
bca = bootstrap.paired_difference_ci(loss_a, loss_b, n_resamples=n_resamples, confidence_level=confidence, method=method, seed=seed)
expect(bca["method_used"] == method and bca["fallback_reason"] is None and bca["low"] < bca["estimate"] < bca["high"],
       f"{method} on non-degenerate data: method_used={bca['method_used']} CI=({bca['low']:.6f}, {bca['high']:.6f})")
expect(len(bca["per_repeat_estimates"]) == 3 and (bca["n_situations"], bca["n_repeats"]) == (30, 3),
       "per-repeat point estimates and (n_situations, n_repeats) recorded")
again = bootstrap.paired_difference_ci(loss_a, loss_b, n_resamples=n_resamples, confidence_level=confidence, method=method, seed=seed)
expect(again["low"] == bca["low"] and again["high"] == bca["high"], "same seed -> identical interval")
degenerate = bootstrap.paired_difference_ci(loss_a, loss_a.copy(), n_resamples=n_resamples, confidence_level=confidence, method="BCa", seed=seed)
expect(degenerate["method_used"] == "percentile" and degenerate["fallback_reason"] is not None
       and degenerate["low"] == 0.0 and degenerate["high"] == 0.0,
       f"identical methods: BCa degenerate -> {degenerate['method_used']} recorded; reason: {degenerate['fallback_reason']}")
print(f"     recorded warnings: {[w[:80] for w in degenerate['warnings']]}")
ratio_ci = bootstrap.paired_ratio_ci(loss_a, loss_b, n_resamples=n_resamples, confidence_level=confidence, method=method, seed=seed)
expect(isinstance(ratio_ci["estimate"], float) and ratio_ci["low"] < ratio_ci["estimate"] < ratio_ci["high"],
       f"ratio CI defined: {ratio_ci['estimate']:.6f} ({ratio_ci['low']:.6f}, {ratio_ci['high']:.6f}) via {ratio_ci['method_used']}")
zero = bootstrap.paired_ratio_ci(np.zeros((10, 2)), loss_b[:10, :2], n_resamples=n_resamples, confidence_level=confidence, method=method, seed=seed)
expect(zero["estimate"] == "undefined" and zero["low"] == "undefined", f"zero denominator -> undefined: {zero['undefined_reason']}")
sparse_a = np.zeros((10, 2))
sparse_a[0, :] = 1.0
sparse = bootstrap.paired_ratio_ci(sparse_a, loss_b[:10, :2], n_resamples=n_resamples, confidence_level=confidence, method=method, seed=seed)
expect(sparse["estimate"] == "undefined", f"zero denominator in some resamples -> undefined: {sparse['undefined_reason']}")
for bad in (dict(method="bca"), dict(seed=-1)):
    try:
        bootstrap.paired_difference_ci(loss_a, loss_b, **{"n_resamples": n_resamples, "confidence_level": confidence, "method": method, "seed": seed, **bad})
        expect(False, f"rejects {bad}")
    except ValueError as error:
        expect(True, f"rejects {bad}: {error}")

print("\n== calibration (known Gaussian truth, unbounded regression outcome) ==")
n = 20000
mu = rng.normal(size=n)
sigma = np.exp(rng.normal(-0.5, 0.3, size=n))
quantiles = mu[:, None] + sigma[:, None] * stats.norm.ppf(levels)[None, :]
truth = {
    "calibrated": mu + sigma * rng.normal(size=n),
    "too narrow": mu + 2.0 * sigma * rng.normal(size=n),
}
REG = {"kind": "regression", "bounds": (-np.inf, np.inf)}
for low_level, high_level in require(config, "evaluation.coverage_intervals"):
    nominal = high_level - low_level
    result = calibration.interval_coverage(quantiles, levels, truth["calibrated"], low_level, high_level, **REG)
    band = 4 * np.sqrt(nominal * (1 - nominal) / n)  # 4 binomial standard errors
    expect(result["at_bound_n"] == 0 and result["interior_n"] == n, "unbounded outcome: every interval is interior")
    expect(abs(result["interior_coverage"] - nominal) <= band,
           f"calibrated coverage [{low_level}, {high_level}] = {result['interior_coverage']:.4f} within {band:.4f} of {nominal}")
    narrow = calibration.interval_coverage(quantiles, levels, truth["too narrow"], low_level, high_level, **REG)
    expect(narrow["interior_coverage"] < nominal - band, f"too-narrow coverage {narrow['interior_coverage']:.4f} is below {nominal}")
chi_ok = calibration.pit_chisquare(quantiles, levels, truth["calibrated"], seed=7, **REG)
chi_bad = calibration.pit_chisquare(quantiles, levels, truth["too narrow"], seed=7, **REG)
print(f"     chi-square calibrated p={chi_ok['pvalue']:.3g}, too narrow p={chi_bad['pvalue']:.3g}; counts {chi_ok['counts']}")
expect(chi_ok["pvalue"] > 1e-3 and chi_bad["pvalue"] < 1e-6, "PIT bin chi-square separates calibrated from too narrow")
expect(chi_ok["tied_rows"] == 0 and chi_ok["merged_rows"] == 0, "continuous outcome: no tied and no merged rows")
expect(int(calibration.pit_bin_counts(quantiles, levels, truth["calibrated"], seed=7, **REG).sum()) == n, "bin counts sum to n")
ks_ok = calibration.randomized_pit_ks(quantiles, levels, truth["calibrated"], seed=7, **REG)
ks_bad = calibration.randomized_pit_ks(quantiles, levels, truth["too narrow"], seed=7, **REG)
print(f"     randomized-PIT KS calibrated p={ks_ok['pvalue']:.3g}, too narrow p={ks_bad['pvalue']:.3g}")
expect(ks_ok["pvalue"] > 1e-3 and ks_bad["pvalue"] < 1e-6, "randomized-PIT KS separates calibrated from too narrow")
expect(np.array_equal(calibration.randomized_pit(quantiles, levels, truth["calibrated"], seed=7, **REG),
                      calibration.randomized_pit(quantiles, levels, truth["calibrated"], seed=7, **REG)), "randomized PIT is seeded")
# Without ties the randomized PIT is the plain per-bin draw of proof p07 (same seed, same values).
q_sorted = np.sort(quantiles, axis=1)
edges = np.concatenate(([0.0], levels, [1.0]))
plain_bins = (truth["calibrated"][:, None] > q_sorted).sum(axis=1)
plain = np.random.default_rng(7).uniform(edges[plain_bins], edges[plain_bins + 1])
expect(np.array_equal(plain, calibration.randomized_pit(quantiles, levels, truth["calibrated"], seed=7, **REG)),
       "without ties the randomized PIT equals the plain per-bin draw bitwise")
for level in (0.2, 0.1 + 1e-12):
    try:
        calibration.interval_coverage(quantiles, levels, truth["calibrated"], level, 0.9, **REG)
        expect(False, f"level {level!r} rejected")
    except ValueError as error:
        expect(True, f"level {level!r} is not an exact member -> rejected (no interpolation): {error}")
source = inspect.getsource(calibration)
expect("interp(" not in source, "calibration module source contains no interpolation call")
const = calibration.spread_error_spearman(np.ones(20), np.arange(20.0))
expect(const["rho"] == "undefined", f"constant spread -> {const}")
rho = calibration.spread_error_spearman(np.arange(20.0), np.arange(20.0) ** 2)
expect(rho["rho"] == 1.0, f"monotone relation -> rho {rho['rho']}")

checks.finish()
