"""Check 9 (review findings E2, E8, E9): honest baseline intervals, Spearman on a constant spread,
all-or-nothing refits.

E2  B4 / B5 predictive quantiles come from out-of-fold errors. On a noisy FIXTURE process
    (fixture_rows: residual = 0.2*surface - 0.3*side_speed + 0.05*N(0,1)) the held-out
    coverage is printed as fixture mechanics only. Asserted: the spread is strictly positive,
    exactly one value per outcome for every query row, and above the old in-sample value
    (B5: >= 10x the spread of its own in-sample errors, which is exactly what the old code
    used; B4: on average per attempt count, above the old per-level in-sample quantile models,
    rebuilt here with scikit-learn and printed next to it). On the old code the B5 spread was
    ~1e-4 (in-sample), so the 10x assertion fails there.
E8  spread_error_spearman returns "undefined" for the exactly-constant B5 spread (the old
    code returned rho = -0.031 from rounding noise).
E9  a failure injected part-way through a refit leaves the previous fit intact: predictions
    bitwise unchanged and n_train unchanged (the old GP changed its predictions by up to 1.3).
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import json  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from fixture_spec import Checks, FixtureSpec, fixture_rows  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.eval import baselines as B  # noqa: E402
from learned_judgment.eval.calibration import spread_error_spearman  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()  # read only
spec = FixtureSpec()
layout = build_layout(spec)
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
spread_levels = tuple(require(config, "engine.spread_levels"))
lo, hi = int(np.flatnonzero(levels == spread_levels[0])[0]), int(np.flatnonzero(levels == spread_levels[1])[0])
gb_key = "baselines.gradient_boosting."
s_te, a_te, p_te, r_te = fixture_rows(spec, 2000, seed=999)
x_te = build_features(layout, s_te, a_te, physics=p_te)


def in_sample_spread(model: B._OutOfFoldBaseline, x: np.ndarray, r: np.ndarray) -> float:
    """Half the [q_lo, q_hi] width of the in-sample errors of the all-attempts point model (outcome 0)."""
    errors = r[:, 0] - model._predict_point_with(model._point_state, x)[:, 0]
    q = np.quantile(errors, [spread_levels[0], spread_levels[1]], method="linear")
    return float(0.5 * (q[1] - q[0]))


def old_gb_quantile_spread(x: np.ndarray, r: np.ndarray, seed: int) -> float:
    """The OLD B4 construction, rebuilt here: one in-sample quantile model per level (plus 0.5),
    sorted per row, spread between the spread levels; mean over the held-out rows (outcome 0)."""
    fit_levels = np.union1d(levels, [0.5])
    predictions = []
    with threadpool_limits(limits=1, user_api="blas"):
        for level in fit_levels:
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=float(level), max_iter=require(config, gb_key + "max_iter"),
                learning_rate=require(config, gb_key + "learning_rate"),
                max_leaf_nodes=require(config, gb_key + "max_leaf_nodes"),
                min_samples_leaf=require(config, gb_key + "min_samples_leaf"), early_stopping=False,
                categorical_features=list(layout.categorical_indices), random_state=seed,
            ).fit(x, r[:, 0])
            predictions.append(model.predict(x_te))
    fitted = np.sort(np.column_stack(predictions), axis=1)
    low_pos = int(np.flatnonzero(fit_levels == spread_levels[0])[0])
    high_pos = int(np.flatnonzero(fit_levels == spread_levels[1])[0])
    return float(np.mean(0.5 * (fitted[:, high_pos] - fitted[:, low_pos])))


print("== E2: out-of-fold intervals of B4 / B5 on a noisy fixture (numbers are fixture mechanics) ==")
print(f"   held-out rows: 2000; true 80% half-width of the fixture noise = {1.2816 * 0.05:.4f}")
for cls in (B.GradientBoostingResidual, B.MLPAttemptsResidual):
    for n in (5, 10, 20, 40):
        coverages, spreads, ratios, old_spreads = [], [], [], []
        for seed in range(5):
            s, a, p, r = fixture_rows(spec, n, seed=100 + seed)
            x = build_features(layout, s, a, physics=p)
            model = cls.from_config(config, layout, seed=seed).fit(x, r)
            d = model.predict_residual_distribution(x_te)
            q = d.quantiles[:, 0, :]
            coverages.append(float(np.mean((r_te[:, 0] >= q[:, lo]) & (r_te[:, 0] <= q[:, hi]))))
            spread = float(d.spread[0, 0])
            spreads.append(spread)
            constant = bool(np.all(d.spread == d.spread[:1, :]))
            old = in_sample_spread(model, x, r)
            ratios.append(spread / old if old > 0 else np.inf)
            if cls is B.MLPAttemptsResidual:  # the old B5 spread WAS the in-sample error spread
                expect(spread > 0.0 and constant and spread >= 10.0 * old,
                       f"{cls.method} n={n} seed={seed}: spread {spread:.4f} > 0, identical on all rows, "
                       f">= 10x the old in-sample spread {old:.2e}")
            else:  # the old B4 spread came from in-sample quantile models, compared per n below
                expect(spread > 0.0 and constant, f"{cls.method} n={n} seed={seed}: spread {spread:.4f} > 0, identical on all rows")
                old_spreads.append(old_gb_quantile_spread(x, r, seed))
        line = (f"   {cls.method:18s} n={n:2d}: held-out cov80 mean {np.mean(coverages):.3f} | spread mean {np.mean(spreads):.4f} "
                f"| spread / in-sample spread min {min(ratios):.1f}")
        if old_spreads:
            line += f" | old in-sample quantile-model spread mean {np.mean(old_spreads):.4f}"
            expect(np.mean(spreads) > np.mean(old_spreads),
                   f"{cls.method} n={n}: mean spread {np.mean(spreads):.4f} above the old in-sample quantile models {np.mean(old_spreads):.4f}")
        print(line)

print("\n== E2: describe() states how the quantiles are produced; out-of-fold coverage exposed ==")
s, a, p, r = fixture_rows(spec, 20, seed=5)
x = build_features(layout, s, a, physics=p)
intervals = [tuple(pair) for pair in require(config, "evaluation.coverage_intervals")]
for cls in (B.ConstantBiasResidual, B.GaussianProcessResidual, B.GradientBoostingResidual, B.MLPAttemptsResidual):
    model = cls.from_config(config, layout, seed=3).fit(x, r)
    record = model.describe()
    json.dumps(record)
    expect(isinstance(record.get("quantile_method"), str) and len(record["quantile_method"]) > 40,
           f"{cls.method}: describe()['quantile_method'] = {record['quantile_method'][:90]}...")
    if isinstance(model, B._OutOfFoldBaseline):
        coverage = model.oof_interval_coverage
        pairs = {(c["outcome"], (c["low_level"], c["high_level"])) for c in coverage}
        expect(pairs == {(o, i) for o in layout.outcome_names for i in intervals}
               and record["fit_details"]["out_of_fold_interval_coverage"] == [dict(c) for c in coverage],
               f"{cls.method}: out-of-fold coverage of {intervals} per outcome on the training attempts "
               f"(honesty indicator): {[(c['outcome'], c['low_level'], round(c['coverage'], 3)) for c in coverage]}")
        expect(model.folds_used == min(int(require(config, f"baselines.{cls.method}.interval_folds")), 20)
               and "out-of-fold" in record["quantile_method"], f"{cls.method}: K = {model.folds_used} folds recorded")
adapter_layout = build_layout(spec, include_physics=False)
adapter = B.DirectTargetAdapter(B.MLPAttemptsResidual.from_config(config, adapter_layout, seed=3), spec=spec, layout=adapter_layout)
expect("quantile_method" in adapter.describe() and "quantile_method" in adapter.describe()["inner"],
       "DirectTargetAdapter.describe() records its own and the inner model's quantile method")

print("\n== E8: Spearman on the exactly-constant B5 spread (reviewer case: 20 attempts, 200 query rows) ==")
s, a, p, r = fixture_rows(spec, 20, seed=5)
mlp = B.MLPAttemptsResidual.from_config(config, layout, seed=3).fit(build_features(layout, s, a, physics=p), r)
s2, a2, p2, r2 = fixture_rows(spec, 200, seed=6)
dist = mlp.predict_residual_distribution(build_features(layout, s2, a2, physics=p2))
spread = dist.spread[:, 0]
abs_err = np.abs(r2[:, 0] - dist.median[:, 0])
result = spread_error_spearman(spread, abs_err)
print(f"   distinct spread values: {np.unique(spread).size}; spread_error_spearman -> {result}")
expect(np.unique(spread).size == 1 and result["rho"] == "undefined", "constant spread -> rho 'undefined' (not a number from rounding)")
gp = B.GaussianProcessResidual.from_config(config, layout, seed=3).fit(build_features(layout, s, a, physics=p), r)
gp_dist = gp.predict_residual_distribution(build_features(layout, s2, a2, physics=p2))
gp_result = spread_error_spearman(gp_dist.spread[:, 0], np.abs(r2[:, 0] - gp_dist.median[:, 0]))
expect(isinstance(gp_result["rho"], float), f"heteroscedastic GP spread -> a numeric rho ({gp_result['rho']:.3f}; fixture)")

print("\n== E9: a refit that raises part-way leaves the previous fit intact ==")
s, a, p, r = fixture_rows(spec, 400, seed=1)
x_all = build_features(layout, s, a, physics=p)
sq, aq, pq, _ = fixture_rows(spec, 300, seed=2)
xq = build_features(layout, sq, aq, physics=pq)


def inject(owner: type, failing_call: int):
    """Patch owner.fit so that call number ``failing_call`` raises; returns (restore, counter)."""
    real = owner.fit
    calls = {"n": 0}

    def failing(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == failing_call:
            raise np.linalg.LinAlgError("injected failure (check 9)")
        return real(self, *args, **kwargs)

    owner.fit = failing
    return (lambda: setattr(owner, "fit", real)), calls


for label, cls, owner, failing_call in (
    ("GP, 2nd outcome's fit", B.GaussianProcessResidual, B.GaussianProcessRegressor, 2),
    ("GB, a fold refit", B.GradientBoostingResidual, B.HistGradientBoostingRegressor, 4),
    ("MLP, a fold refit", B.MLPAttemptsResidual, B.MLPRegressor, 3),
):
    model = cls.from_config(config, layout, seed=5).fit(x_all[:20], r[:20])
    before = model.predict_residual_distribution(xq)
    before_describe = json.dumps(model.describe(), sort_keys=True)
    restore, calls = inject(owner, failing_call)
    try:
        model.fit(x_all[200:240] * 3.0 + 5.0, r[200:240])  # new attempts, very different feature scale
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    finally:
        restore()
    after = model.predict_residual_distribution(xq)
    same = all(np.array_equal(getattr(before, f), getattr(after, f)) for f in ("mean", "median", "quantiles", "spread"))
    expect(raised and same and model.n_train == 20 and json.dumps(model.describe(), sort_keys=True) == before_describe,
           f"{label}: refit raised, predictions bitwise unchanged, n_train still {model.n_train}, describe() unchanged")
    model.fit(x_all[200:240], r[200:240])
    expect(model.n_train == 40, f"{label}: a later successful refit installs the new fit (n_train {model.n_train})")

checks.finish()
