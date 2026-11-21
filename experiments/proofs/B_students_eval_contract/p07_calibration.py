"""p07: calibration statistics contract (spearmanr, mean_pinball_loss, interval coverage, PIT from finite quantiles, kstest).
FIXTURE numbers only; validated against a known Gaussian where the true answer is known analytically."""
import inspect, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import mean_pinball_loss

print("SIGNATURE spearmanr", inspect.signature(stats.spearmanr))
print("SIGNATURE kstest", inspect.signature(stats.kstest))
print("SIGNATURE mean_pinball_loss", inspect.signature(mean_pinball_loss))

rng = np.random.default_rng(0)
x = rng.normal(size=50); y = x + rng.normal(size=50)
r = stats.spearmanr(x, y)
print("\nspearmanr result type", type(r).__name__, "statistic", r.statistic, "pvalue", r.pvalue, "| r[0]==statistic:", r[0] == r.statistic, "| has .correlation:", hasattr(r, "correlation"))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rc = stats.spearmanr(np.ones(10), y[:10])
print("spearmanr constant input ->", rc.statistic, rc.pvalue, "warnings:", [f"{x_.category.__name__}: {str(x_.message)[:160]}" for x_ in w])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        rn = stats.spearmanr([1.0, 2.0], [2.0, 1.0]); print("spearmanr n=2 ->", rn.statistic, rn.pvalue)
    except Exception as e:
        print("spearmanr n=2 EXC", type(e).__name__, e)
print("spearmanr with NaN, default nan_policy='propagate' ->", stats.spearmanr([1, 2, np.nan, 4], [1, 2, 3, 4]).statistic,
      "| nan_policy='omit' ->", stats.spearmanr([1, 2, np.nan, 4], [1, 2, 3, 4], nan_policy="omit").statistic)

print("\n== mean_pinball_loss ==")
yt = np.array([1.0, 2.0, 3.0]); yp = np.array([1.5, 1.5, 1.5])
for a in [0.1, 0.5, 0.9]:
    manual = np.mean(np.maximum(a * (yt - yp), (a - 1) * (yt - yp)))
    print(f"alpha={a}: sklearn={mean_pinball_loss(yt, yp, alpha=a):.6f} manual={manual:.6f}")
print("default alpha:", inspect.signature(mean_pinball_loss).parameters["alpha"].default,
      "| alpha=0.5 equals 0.5*MAE:", np.isclose(mean_pinball_loss(yt, yp, alpha=0.5), 0.5 * np.mean(np.abs(yt - yp))))
Y2 = np.c_[yt, yt * 2]; P2 = np.c_[yp, yp]
print("2-D multioutput default ('uniform_average'):", mean_pinball_loss(Y2, P2, alpha=0.9), "raw_values:", mean_pinball_loss(Y2, P2, alpha=0.9, multioutput="raw_values"))

print("\n== interval coverage + PIT from finite quantile levels ==")
LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

def central_interval_coverage(y, q, levels, central):
    """y: (n,), q: (n, L) predicted quantiles at `levels` (sorted ascending). central in {0.5, 0.8} -> uses levels (0.25,0.75) or (0.1,0.9) exactly; raises if not present."""
    lo_l, hi_l = (1 - central) / 2, 1 - (1 - central) / 2
    il = np.flatnonzero(np.isclose(levels, lo_l)); ih = np.flatnonzero(np.isclose(levels, hi_l))
    if il.size != 1 or ih.size != 1:
        raise ValueError(f"levels {levels} do not contain {lo_l} and {hi_l}")
    return float(np.mean((y >= q[:, il[0]]) & (y <= q[:, ih[0]])))

def pit_from_quantiles(y, q, levels, tails="linear"):
    """Piecewise-linear predictive CDF through (q_j, levels_j). Inside [q_1, q_L]: linear interpolation.
    Outside: 'clip' -> 0 / 1 at the extreme knots would put mass at PIT=0/1, so 'linear' extrapolates the CDF with the
    slope of the outermost segment and clips to [0, 1]; 'censor' returns levels[0]/2 and (1+levels[-1])/2 midpoints.
    Requires q non-decreasing per row (sort first if crossing)."""
    q = np.sort(q, axis=1)
    out = np.empty(y.shape[0])
    for i in range(y.shape[0]):
        qi, yi = q[i], y[i]
        if qi[0] <= yi <= qi[-1]:
            out[i] = np.interp(yi, qi, levels)
        elif tails == "censor":
            out[i] = levels[0] / 2 if yi < qi[0] else (1 + levels[-1]) / 2
        else:
            if yi < qi[0]:
                slope = (levels[1] - levels[0]) / max(qi[1] - qi[0], 1e-300)
                out[i] = max(0.0, levels[0] - slope * (qi[0] - yi))
            else:
                slope = (levels[-1] - levels[-2]) / max(qi[-1] - qi[-2], 1e-300)
                out[i] = min(1.0, levels[-1] + slope * (yi - qi[-1]))
    return out

n = 20000
mu = rng.normal(size=n); sig = np.exp(rng.normal(-0.5, 0.3, size=n))
y_cal = mu + sig * rng.normal(size=n)                      # calibrated fixture: truth ~ N(mu, sig)
y_over = mu + 0.5 * sig * rng.normal(size=n)               # predictive too wide (truth narrower)
y_under = mu + 2.0 * sig * rng.normal(size=n)              # predictive too narrow
Q = mu[:, None] + sig[:, None] * stats.norm.ppf(LEVELS)[None, :]
for label, yy in [("calibrated", y_cal), ("too-wide", y_over), ("too-narrow", y_under)]:
    c50 = central_interval_coverage(yy, Q, LEVELS, 0.5); c80 = central_interval_coverage(yy, Q, LEVELS, 0.8)
    pit = pit_from_quantiles(yy, Q, LEVELS, "linear")
    ks = stats.kstest(pit, "uniform")
    pinball = {a: round(mean_pinball_loss(yy, Q[:, j], alpha=a), 5) for j, a in enumerate(LEVELS)}
    print(f"{label:11s} cov50={c50:.4f} cov80={c80:.4f} PIT mean={pit.mean():.4f} frac(PIT in [.1,.9])={np.mean((pit>=.1)&(pit<=.9)):.4f} KS stat={ks.statistic:.4f} p={ks.pvalue:.3g} | pinball {pinball}")
print("kstest result fields:", [f for f in ("statistic", "pvalue", "statistic_location", "statistic_sign") if hasattr(ks, f)])
print("NOTE PIT from 5 quantiles is exact only at knots; linear interpolation between knots of a Gaussian is approximate, so KS on PIT for a perfectly calibrated model is NOT exactly uniform at large n (see calibrated KS above).")
pit_exact = stats.norm.cdf((y_cal - mu) / sig)
print("reference exact-CDF PIT for calibrated fixture KS:", stats.kstest(pit_exact, "uniform").statistic)
print("kstest alternative explicit form: stats.kstest(pit, stats.uniform.cdf) ->", stats.kstest(pit_exact, stats.uniform.cdf).statistic)
print("\n== exact alternatives that do not depend on interpolation ==")
EDGES = np.r_[0.0, LEVELS, 1.0]                 # probability mass per bin = diff(EDGES) = [.1,.15,.25,.25,.15,.1]
def bin_index(y, q):
    return (y[:, None] > np.sort(q, axis=1)).sum(axis=1)    # 0..L, bin j means q_{j-1} < y <= q_j
def randomized_pit(y, q, levels, rng):
    e = np.r_[0.0, levels, 1.0]; b = bin_index(y, q)
    return rng.uniform(e[b], e[b + 1])
print("SIGNATURE chisquare", inspect.signature(stats.chisquare))
for label, yy in [("calibrated", y_cal), ("too-wide", y_over), ("too-narrow", y_under)]:
    b = bin_index(yy, Q); obs = np.bincount(b, minlength=len(LEVELS) + 1); exp = np.diff(EDGES) * len(yy)
    cs = stats.chisquare(obs, exp)
    rp = randomized_pit(yy, Q, LEVELS, np.random.default_rng(7))
    ks = stats.kstest(rp, "uniform")
    print(f"{label:11s} bin fractions={np.round(obs / len(yy), 4)} expected={np.diff(EDGES)} chisq={cs.statistic:.2f} p={cs.pvalue:.3g} | randomized-PIT KS={ks.statistic:.4f} p={ks.pvalue:.3g}")

print("\ncrossing quantiles fixture: np.sort per row before use; monotonic check:", np.all(np.diff(np.sort(np.array([[1.0, 0.9, 1.2, 1.1, 1.5]]), axis=1)) >= 0))
