"""Check 8 (review finding E1): calibration with atoms -- binary outcomes and outcomes clipped at a bound.

Every "calibrated" fixture below uses the EXACT quantiles / probabilities of the distribution
that generated y, so the applicable calibration functions must not flag it; every
"mis-specified" fixture is wrong on purpose and must be flagged. Thresholds: a calibrated
fixture passes when its test p-value is above 1e-3 and its coverage lies within 4 binomial
standard errors of nominal; a mis-specified one is flagged when p < 1e-6 (fixture sizes of
20 000 rows). Fixture arrays only -- nothing here is evidence about any model.

On the code before this fix the clipped fixture gave chi-square / KS p = 0.0 and the binary
calls were accepted (cov80 0.994 against 0.8), so the assertions below fail there.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from fixture_spec import Checks  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.eval import calibration  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()  # read only
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
intervals = require(config, "evaluation.coverage_intervals")
n_bins = int(require(config, "evaluation.reliability_bins"))
z = stats.norm.ppf(levels)
N = 20000
rng = np.random.default_rng(20261121)
P_OK, P_FLAG = 1e-3, 1e-6


def judge_regression(
    label: str, q: np.ndarray, y: np.ndarray, bounds: tuple[float, float], calibrated: bool, coverage_flags: bool = False
) -> None:
    """Calibrated: coverage near nominal and both PIT tests pass. Mis-specified: both PIT tests
    flag it, and -- only for a scale error (``coverage_flags``) -- the interior coverage is off
    nominal too (a small location shift barely moves a central interval's coverage)."""
    kw = {"kind": "regression", "bounds": bounds}
    chi = calibration.pit_chisquare(q, levels, y, seed=11, **kw)
    ks = calibration.randomized_pit_ks(q, levels, y, seed=11, **kw)
    print(f"     {label}: chi2 p={chi['pvalue']:.3g} KS p={ks['pvalue']:.3g} counts={chi['counts']} "
          f"tied share={chi['tied_share']:.3f} merged share={chi['merged_share']:.3f}")
    for low_level, high_level in intervals:
        cov = calibration.interval_coverage(q, levels, y, low_level, high_level, **kw)
        nominal = cov["nominal"]
        interior = cov["interior_coverage"]
        band = 4 * np.sqrt(nominal * (1 - nominal) / max(cov["interior_n"], 1))
        print(f"       [{low_level}, {high_level}] interior n={cov['interior_n']} coverage={interior} | at bound "
              f"n={cov['at_bound_n']} share={cov['at_bound_share']:.3f} coverage={cov['at_bound_coverage']}")
        if calibrated:
            expect(cov["interior_n"] > 0 and abs(interior - nominal) <= band,
                   f"{label}: interior coverage [{low_level}, {high_level}] within {band:.4f} of nominal {nominal}")
            if cov["at_bound_n"]:
                expect(cov["at_bound_coverage"] > nominal and "above high_level - low_level" in cov["at_bound_note"],
                       f"{label}: at-bound rows reported separately; their coverage {cov['at_bound_coverage']:.4f} is above "
                       f"{nominal} as stated")
        elif coverage_flags:
            if cov["interior_n"] > 100:
                expect(interior < nominal - band or interior > nominal + band,
                       f"{label}: interior coverage [{low_level}, {high_level}] = {interior:.4f} is off nominal {nominal}")
    if calibrated:
        expect(chi["pvalue"] > P_OK and ks["pvalue"] > P_OK, f"{label}: judged calibrated by both PIT tests (p > {P_OK})")
    else:
        expect(chi["pvalue"] < P_FLAG and ks["pvalue"] < P_FLAG, f"{label}: flagged by both PIT tests (p < {P_FLAG})")


print("== (A) continuous Gaussian, exact quantiles (control) ==")
mu = rng.uniform(-1.0, 1.0, N)
y = mu + rng.normal(size=N)
judge_regression("continuous calibrated", mu[:, None] + z[None, :], y, (-np.inf, np.inf), calibrated=True)
judge_regression("continuous too narrow", mu[:, None] + 0.6 * z[None, :], y, (-np.inf, np.inf), calibrated=False,
                 coverage_flags=True)

print("\n== (B) Gaussian clipped at the lower bound 0, exact quantiles of the clipped truth ==")
# y = max(0, Z), Z ~ N(mu, 1): exactly what clip(physics + exact residual quantile, low, high) gives.
for name, low_mu, high_mu in (("reviewer case, mu in (-1, 1)", -1.0, 1.0), ("mu in (-1, 3)", -1.0, 3.0)):
    mu_c = rng.uniform(low_mu, high_mu, N)
    y_c = np.maximum(0.0, mu_c + rng.normal(size=N))
    q_c = np.maximum(0.0, mu_c[:, None] + z[None, :])
    print(f"   {name}: share of y exactly 0 = {np.mean(y_c == 0.0):.3f}")
    if high_mu > 1.5:
        judge_regression(f"clipped calibrated ({name})", q_c, y_c, (0.0, np.inf), calibrated=True)
    else:
        kw = {"kind": "regression", "bounds": (0.0, np.inf)}
        chi = calibration.pit_chisquare(q_c, levels, y_c, seed=11, **kw)
        ks = calibration.randomized_pit_ks(q_c, levels, y_c, seed=11, **kw)
        print(f"     chi2 p={chi['pvalue']:.3g} KS p={ks['pvalue']:.3g} counts={chi['counts']} tied share={chi['tied_share']:.3f}")
        expect(chi["pvalue"] > P_OK and ks["pvalue"] > P_OK, f"clipped calibrated ({name}): judged calibrated by both PIT tests")
        expect(chi["tied_rows"] > 0 and chi["tied_share"] == float(np.mean(np.any(np.sort(q_c, 1) == y_c[:, None], 1))),
               f"clipped calibrated ({name}): share of tied rows reported ({chi['tied_share']:.3f})")
    # Why the library merges level bins across tied / bound knots: drawing the tied range
    # [edges[#{q<y}], edges[#{q<=y}+1]] for the tied rows ALONE (neighbours keep their own bin)
    # is not exact. Re-implemented here only to show it rejects this calibrated fixture.
    q_sorted = np.sort(q_c, axis=1)
    knot_edges = np.concatenate(([0.0], levels, [1.0]))
    below_count = (q_sorted < y_c[:, None]).sum(axis=1)
    upto_count = (q_sorted <= y_c[:, None]).sum(axis=1)
    tied_only = np.random.default_rng(11).uniform(knot_edges[below_count], knot_edges[upto_count + 1])
    tied_only_counts = np.bincount(np.searchsorted(levels, tied_only, side="right"), minlength=levels.size + 1)
    tied_only_p = stats.chisquare(tied_only_counts, np.diff(knot_edges) * N).pvalue
    print(f"     tied-range-for-tied-rows-only rule (not used): chi2 p={tied_only_p:.3g} counts={tied_only_counts.tolist()}")
    expect(tied_only_p < P_FLAG, f"({name}) the tied-range-only rule would reject this calibrated fixture (p={tied_only_p:.3g}); "
           "the merged-bin rule above does not")
    q_shift = np.maximum(0.0, (mu_c + 0.3)[:, None] + z[None, :])
    judge_regression(f"clipped mis-specified, mean +0.3 ({name})", q_shift, y_c, (0.0, np.inf), calibrated=False)
    q_narrow = np.maximum(0.0, mu_c[:, None] + 0.6 * z[None, :])
    judge_regression(f"clipped mis-specified, too narrow ({name})", q_narrow, y_c, (0.0, np.inf), calibrated=False,
                     coverage_flags=True)

print("\n== (B2) Gaussian clipped at the upper bound 0 ==")
mu_u = rng.uniform(-3.0, 1.0, N)
y_u = np.minimum(0.0, mu_u + rng.normal(size=N))
judge_regression("upper-clipped calibrated", np.minimum(0.0, mu_u[:, None] + z[None, :]), y_u, (-np.inf, 0.0), calibrated=True)

print("\n== (C) binary outcome, exact Bernoulli probabilities ==")
p = rng.uniform(0.05, 0.95, N)
y_b = (rng.uniform(size=N) < p).astype(np.float64)
q_b = (levels[None, :] > (1.0 - p)[:, None]).astype(np.float64)  # exact Bernoulli quantiles
quantile_calls = {
    "interval_coverage": lambda: calibration.interval_coverage(q_b, levels, y_b, 0.1, 0.9, kind="binary", bounds=(0.0, 1.0)),
    "pit_bin_counts": lambda: calibration.pit_bin_counts(q_b, levels, y_b, kind="binary", bounds=(0.0, 1.0), seed=1),
    "pit_chisquare": lambda: calibration.pit_chisquare(q_b, levels, y_b, kind="binary", bounds=(0.0, 1.0), seed=1),
    "randomized_pit": lambda: calibration.randomized_pit(q_b, levels, y_b, kind="binary", bounds=(0.0, 1.0), seed=1),
    "randomized_pit_ks": lambda: calibration.randomized_pit_ks(q_b, levels, y_b, kind="binary", bounds=(0.0, 1.0), seed=1),
}
for name, call in quantile_calls.items():
    try:
        call()
        expect(False, f"{name} refuses kind='binary'")
    except ValueError as error:
        expect("binary" in str(error) and "brier_score" in str(error), f"{name} refuses kind='binary': {error}")
brier = calibration.brier_score(p, y_b)
print(f"     brier={brier['brier']:.5f} (expected value under calibration mean p(1-p) = {np.mean(p * (1 - p)):.5f})")
expect(abs(brier["brier"] - float(np.mean((p - y_b) ** 2))) == 0.0 and brier["n"] == N, "brier_score is mean((p - y)^2)")
table = calibration.reliability_table(p, y_b, n_bins)
test = table["binned_chisquare"]
for row in table["bins"]:
    if row["count"]:
        print(f"       bin [{row['low']:.1f}, {row['high']:.1f}) count={row['count']:5d} mean_predicted={row['mean_predicted']:.4f} "
              f"observed_rate={row['observed_rate']:.4f}")
print(f"     binned chi-square: statistic={test['statistic']:.3f} df={test['df']} p={test['pvalue']:.3g}")
expect(sum(row["count"] for row in table["bins"]) == N and len(table["bins"]) == n_bins, f"{n_bins} equal-width bins hold all rows")
expect(test["pvalue"] > P_OK and test["zero_variance_mismatch"] == 0.0, f"exact Bernoulli probabilities judged calibrated (p > {P_OK})")
bad = calibration.reliability_table(np.clip(p + 0.1, 0.0, 1.0), y_b, n_bins)["binned_chisquare"]
print(f"     mis-specified (p + 0.1): binned chi-square p={bad['pvalue']:.3g}")
expect(bad["pvalue"] < P_FLAG, f"mis-specified probabilities flagged (p < {P_FLAG})")
edge = calibration.reliability_table(np.array([0.0, 0.3, 1.0]), np.array([0.0, 1.0, 1.0]), n_bins)
expect(edge["bins"][-1]["count"] == 1 and edge["bins"][0]["count"] == 1, "p = 1.0 falls in the last bin, p = 0.0 in the first")

print("\n== input refusals ==")
for label, call in (
    ("y below the lower bound", lambda: calibration.interval_coverage(np.zeros((2, levels.size)), levels, np.array([-0.1, 0.0]), 0.1, 0.9, kind="regression", bounds=(0.0, np.inf))),
    ("quantiles below the lower bound", lambda: calibration.randomized_pit(-np.ones((2, levels.size)), levels, np.array([0.1, 0.0]), kind="regression", bounds=(0.0, np.inf), seed=1)),
    ("an unknown kind", lambda: calibration.pit_chisquare(np.zeros((2, levels.size)), levels, np.zeros(2), kind="count", bounds=(0.0, 1.0), seed=1)),
    ("a binary y that is not 0/1", lambda: calibration.brier_score(np.array([0.5]), np.array([0.5]))),
):
    try:
        call()
        expect(False, f"refuses {label}")
    except ValueError as error:
        expect(True, f"refuses {label}: {error}")

checks.finish()
