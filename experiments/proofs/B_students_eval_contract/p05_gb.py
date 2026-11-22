"""p05: GradientBoostingRegressor / HistGradientBoostingRegressor small-n behaviour. FIXTURE data only (API/failure modes)."""
import inspect, warnings
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

print("SIGNATURE GBR", inspect.signature(GradientBoostingRegressor.__init__))
print("SIGNATURE HGBR", inspect.signature(HistGradientBoostingRegressor.__init__))

def fixture(n, seed=0, d=4, k=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    Y = np.c_[2.0 * (X[:, 0] > 0) + X[:, 1], X[:, 2] - X[:, 3]] + 0.05 * rng.normal(size=(n, 2))
    return X, (Y[:, 0] if k == 1 else Y[:, :k])

Xq = np.random.default_rng(9).uniform(-1, 1, size=(200, 4))

def describe(est, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            est.fit(X, y)
        except Exception as e:
            return f"EXC {type(e).__name__}: {e}"
    p = est.predict(Xq)
    extra = ""
    if hasattr(est, "_predictors"):
        nodes = [pr[0].get_n_leaf_nodes() for pr in est._predictors]
        extra = f" n_iter_={est.n_iter_} max_leaves_any_iter={max(nodes) if nodes else 0} do_early_stopping_={est.do_early_stopping_}"
    elif hasattr(est, "estimators_"):
        leaves = [t[0].get_n_leaves() for t in est.estimators_]
        extra = f" n_estimators_={est.n_estimators_} max_leaves_any_tree={max(leaves)}"
    ww = "; ".join(f"{x.category.__name__}: {str(x.message)[:100]}" for x in w)
    return f"n_unique_pred={len(np.unique(np.round(p, 12)))} constant={np.ptp(p) < 1e-12}{extra}{' WARN ' + ww if ww else ''}"

print("\n== DEFAULTS at small n ==")
for n in [1, 2, 5, 10, 20, 39, 40, 41, 60, 200]:
    X, y = fixture(n)
    print(f"n={n:>3} GBR(default)  : {describe(GradientBoostingRegressor(random_state=0), X, y)}")
    print(f"n={n:>3} HGBR(default) : {describe(HistGradientBoostingRegressor(random_state=0), X, y)}")

print("\n== SMALL-N SETTINGS ==")
for n in [5, 10, 20, 40]:
    X, y = fixture(n)
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=2, subsample=1.0, random_state=0)
    hgb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=4, min_samples_leaf=2, early_stopping=False, l2_regularization=0.0, random_state=0)
    hgb_es = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=2, early_stopping=True, random_state=0)
    print(f"n={n:>3} GBR(min_samples_leaf=2,max_depth=2): {describe(gbr, X, y)}")
    print(f"n={n:>3} HGBR(min_samples_leaf=2,early_stopping=False): {describe(hgb, X, y)}")
    print(f"n={n:>3} HGBR(min_samples_leaf=2,early_stopping=True): {describe(hgb_es, X, y)}")

print("\n== early_stopping='auto' switch point (documented: enabled if n_samples > 10000) ==")
X, y = fixture(10001)
h = HistGradientBoostingRegressor(random_state=0).fit(X, y)
print("n=10001 do_early_stopping_ =", h.do_early_stopping_)
X, y = fixture(10000)
h = HistGradientBoostingRegressor(random_state=0).fit(X, y)
print("n=10000 do_early_stopping_ =", h.do_early_stopping_)

print("\n== quantile losses ==")
X, y = fixture(40)
for a in [0.1, 0.5, 0.9]:
    g = GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=2, random_state=0).fit(X, y)
    h = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=200, learning_rate=0.05, max_leaf_nodes=4, min_samples_leaf=2, early_stopping=False, random_state=0).fit(X, y)
    print(f"alpha/quantile={a}: GBR frac(y<=pred_train)={np.mean(y <= g.predict(X)):.3f}  HGBR frac(y<=pred_train)={np.mean(y <= h.predict(X)):.3f}")
try:
    HistGradientBoostingRegressor(loss="quantile").fit(X, y)
except Exception as e:
    print("HGBR loss='quantile' without quantile=: EXC", type(e).__name__, str(e)[:200])
try:
    GradientBoostingRegressor(loss="quantile", alpha=1.0).fit(X, y)
except Exception as e:
    print("GBR alpha=1.0: EXC", type(e).__name__, str(e)[:200])

print("\n== multi-output: 2-D y ==")
X, Y = fixture(40, k=2)
for est in [GradientBoostingRegressor(random_state=0), HistGradientBoostingRegressor(random_state=0)]:
    try:
        est.fit(X, Y); print(type(est).__name__, "2-D y OK pred shape", est.predict(Xq).shape)
    except Exception as e:
        print(type(est).__name__, "2-D y EXC", type(e).__name__, str(e)[:200])
mo = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=2, max_leaf_nodes=4, early_stopping=False, random_state=0)).fit(X, Y)
print("MultiOutputRegressor(HGBR) pred shape", mo.predict(Xq).shape)

print("\n== determinism ==")
X, y = fixture(40)
a = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=2, early_stopping=False, random_state=0).fit(X, y).predict(Xq)
b = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=2, early_stopping=False, random_state=0).fit(X, y).predict(Xq)
c = GradientBoostingRegressor(subsample=0.8, min_samples_leaf=2, random_state=0).fit(X, y).predict(Xq)
d = GradientBoostingRegressor(subsample=0.8, min_samples_leaf=2, random_state=0).fit(X, y).predict(Xq)
print("HGBR same random_state bitwise:", np.array_equal(a, b), "| GBR subsample=0.8 same random_state bitwise:", np.array_equal(c, d))
