"""p06: scipy.stats.bootstrap contract + numpy paired/cluster bootstrap. FIXTURE numbers only (API/semantics, not results)."""
import inspect, warnings
import numpy as np
import scipy
from scipy import stats

print("scipy", scipy.__version__)
print("SIGNATURE stats.bootstrap", inspect.signature(stats.bootstrap))
res_fields = None

rng0 = np.random.default_rng(0)
n_sit = 30
err_a = np.abs(rng0.normal(1.0, 0.3, n_sit))            # fixture: method A per-situation error
err_b = err_a + rng0.normal(0.1, 0.05, n_sit)           # fixture: method B paired on same situations

print("\n== paired=True with statistic on two samples ==")
def mean_diff(a, b, axis=-1):
    return np.mean(b - a, axis=axis)
r = stats.bootstrap((err_a, err_b), mean_diff, paired=True, vectorized=True, n_resamples=9999,
                    confidence_level=0.95, method="BCa", rng=np.random.default_rng(1))
print("result type", type(r).__name__, "fields", [f for f in dir(r) if not f.startswith("_")])
print("BCa CI", r.confidence_interval, "SE", r.standard_error, "bootstrap_distribution shape", r.bootstrap_distribution.shape)
r2 = stats.bootstrap((err_a, err_b), mean_diff, paired=True, vectorized=True, n_resamples=9999, method="BCa", rng=np.random.default_rng(1))
print("same rng seed -> identical CI:", r.confidence_interval == r2.confidence_interval)
print("paired=True == one-sample bootstrap of the differences (percentile, same rng)?")
p1 = stats.bootstrap((err_a, err_b), mean_diff, paired=True, vectorized=True, n_resamples=9999, method="percentile", rng=np.random.default_rng(5))
p2 = stats.bootstrap(((err_b - err_a),), np.mean, vectorized=True, n_resamples=9999, method="percentile", rng=np.random.default_rng(5))
print("  paired:", p1.confidence_interval, "\n  diffs :", p2.confidence_interval)

print("\n== seed argument name ==")
for kw in ["rng", "random_state"]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            stats.bootstrap(((err_b - err_a),), np.mean, n_resamples=999, **{kw: 3})
            print(f"  {kw}=3: OK", [f"{x.category.__name__}: {str(x.message)[:150]}" for x in w])
        except Exception as e:
            print(f"  {kw}=3: EXC {type(e).__name__}: {e}")

print("\n== degenerate: constant differences (all zero, all 0.5) ==")
for label, d in [("all-zero", np.zeros(20)), ("all-0.5", np.full(20, 0.5))]:
    for method in ["BCa", "percentile", "basic"]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                rr = stats.bootstrap((d,), np.mean, method=method, n_resamples=999, rng=np.random.default_rng(0))
                msg = f"CI={rr.confidence_interval} SE={rr.standard_error}"
            except Exception as e:
                msg = f"EXC {type(e).__name__}: {e}"
        print(f"  {label:8s} {method:10s}: {msg} | warnings={[f'{x.category.__name__}: {str(x.message)[:200]}' for x in w]}")

print("\n== degenerate: n=1 and n=2 ==")
for d in [np.array([0.3]), np.array([0.3, 0.4])]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            rr = stats.bootstrap((d,), np.mean, method="BCa", n_resamples=999, rng=np.random.default_rng(0))
            msg = f"CI={rr.confidence_interval}"
        except Exception as e:
            msg = f"EXC {type(e).__name__}: {e}"
    print(f"  n={len(d)} BCa: {msg} | warnings={[f'{x.category.__name__}: {str(x.message)[:150]}' for x in w]}")

print("\n== ten-line numpy percentile bootstrap of mean paired difference ==")
def paired_bootstrap_mean_diff(a, b, n_resamples=9999, seed=0, level=0.95):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return d.mean(), lo, hi, boots
m, lo, hi, boots = paired_bootstrap_mean_diff(err_a, err_b, seed=5)
print(f"  numpy: point={m:.6f} CI=({lo:.6f}, {hi:.6f}) boots.shape={boots.shape}")
print(f"  scipy percentile (rng=5): CI=({p1.confidence_interval.low:.6f}, {p1.confidence_interval.high:.6f})  |diff| low={abs(lo - p1.confidence_interval.low):.2e} high={abs(hi - p1.confidence_interval.high):.2e}")

print("\n== cluster bootstrap: situations x seeds, paired (resample situations, keep all seeds of a situation together) ==")
S, K = 30, 3
rng1 = np.random.default_rng(11)
A = np.abs(rng1.normal(1.0, 0.3, (S, K)))          # fixture errors: A[s, k] = method A, situation s, seed k
B = A + rng1.normal(0.1, 0.05, (S, K))
def cluster_bootstrap(A, B, stat, n_resamples=9999, seed=0, level=0.95):
    """A, B: (S, K) arrays on identical situations s and seeds/noise draws k. Resample situation rows with replacement."""
    rng = np.random.default_rng(seed)
    S = A.shape[0]
    idx = rng.integers(0, S, size=(n_resamples, S))
    boots = np.array([stat(A[i], B[i]) for i in idx])
    point = stat(A, B)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return point, lo, hi
diff_stat = lambda a, b: float(np.mean(b - a))
ratio_stat = lambda a, b: float(np.mean(b) / np.mean(a))        # ratio of means, e.g. student_err / teacher_err
print("  mean diff   :", tuple(round(v, 6) for v in cluster_bootstrap(A, B, diff_stat, seed=2)))
print("  ratio       :", tuple(round(v, 6) for v in cluster_bootstrap(A, B, ratio_stat, seed=2)))
print("  log-ratio   :", tuple(round(v, 6) for v in cluster_bootstrap(A, B, lambda a, b: float(np.log(np.mean(b) / np.mean(a))), seed=2)))
# scipy equivalent of the cluster bootstrap: pass per-situation aggregates (seed-mean) as paired samples
a_s, b_s = A.mean(axis=1), B.mean(axis=1)
rs = stats.bootstrap((a_s, b_s), lambda a, b, axis=-1: np.mean(b, axis=axis) / np.mean(a, axis=axis), paired=True, vectorized=True,
                     n_resamples=9999, method="BCa", rng=np.random.default_rng(2))
print("  scipy BCa ratio on per-situation seed-means:", rs.confidence_interval)
print("  NOTE: mean over seeds per situation then resampling situations == cluster bootstrap for mean statistics when K is equal for all situations")

print("\n== vectorized=False custom statistic (axis not accepted) ==")
rr = stats.bootstrap((a_s, b_s), lambda a, b: np.median(b - a), paired=True, vectorized=False, n_resamples=999, method="percentile", rng=np.random.default_rng(0))
print("  median paired diff CI:", rr.confidence_interval)
