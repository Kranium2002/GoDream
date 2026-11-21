"""p04: GaussianProcessRegressor residual-baseline API contract. FIXTURE data only (API/shape/failure modes, not model quality)."""
import inspect, warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

print("SIGNATURE GPR", inspect.signature(GaussianProcessRegressor.__init__))
print("SIGNATURE GPR.predict", inspect.signature(GaussianProcessRegressor.predict))
print("SIGNATURE RBF", inspect.signature(RBF.__init__))
print("SIGNATURE WhiteKernel", inspect.signature(WhiteKernel.__init__))
print("SIGNATURE ConstantKernel", inspect.signature(ConstantKernel.__init__))

D = 4
def kernel():
    return (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-6, 1e1)))

def fixture(n, k=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, D))
    Y = np.c_[np.sin(3 * X[:, 0]) + 0.5 * X[:, 1], X[:, 2] ** 2 - X[:, 3]][:, :k] + 0.05 * rng.normal(size=(n, k))
    return X, (Y[:, 0] if k == 1 else Y)

Xq = np.random.default_rng(9).uniform(-1, 1, size=(7, D))
print("\n== n sweep, 1-D y, normalize_y=True, n_restarts_optimizer=5, random_state=0 ==")
for n in [0, 1, 2, 5, 40]:
    X, y = fixture(n)
    gp = GaussianProcessRegressor(kernel=kernel(), normalize_y=True, n_restarts_optimizer=5, random_state=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            gp.fit(X, y)
            mu, sd = gp.predict(Xq, return_std=True)
            msg = f"OK mu.shape={mu.shape} std.shape={sd.shape} std[:3]={np.round(sd[:3], 4)} kernel_={gp.kernel_} lml={gp.log_marginal_likelihood_value_:.4f}"
        except Exception as e:
            msg = f"EXC {type(e).__name__}: {e}"
    print(f"[n={n}] {msg}")
    for x in w:
        print(f"    WARN {x.category.__name__}: {str(x.message)[:220]}")

print("\n== unfitted GP predict (prior) ==")
gp0 = GaussianProcessRegressor(kernel=kernel(), normalize_y=True)
mu, sd = gp0.predict(Xq, return_std=True)
print("prior predict OK shapes", mu.shape, sd.shape, "std[:2]", np.round(sd[:2], 4))

print("\n== constant y (n=5), normalize_y=True ==")
X, _ = fixture(5)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        gp = GaussianProcessRegressor(kernel=kernel(), normalize_y=True, random_state=0).fit(X, np.full(5, 2.5))
        mu, sd = gp.predict(Xq, return_std=True)
        print("OK mu[:3]", mu[:3], "std[:3]", sd[:3], "_y_train_std", gp._y_train_std)
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}")
    for x in w:
        print(f"    WARN {x.category.__name__}: {str(x.message)[:200]}")

print("\n== multi-output: 2-D y with one shared kernel vs one GP per target ==")
X, Y = fixture(40, k=2)
gp2 = GaussianProcessRegressor(kernel=kernel(), normalize_y=True, n_restarts_optimizer=2, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    gp2.fit(X, Y)
mu, sd = gp2.predict(Xq, return_std=True)
print("2-D y: mu.shape", mu.shape, "std.shape", sd.shape, "kernel_ (SHARED hyperparameters across outputs):", gp2.kernel_)
mu, cov = gp2.predict(Xq, return_cov=True)
print("2-D y: return_cov shape", cov.shape)
per = []
for j in range(2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        g = GaussianProcessRegressor(kernel=kernel(), normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(X, Y[:, j])
    per.append(g)
    print(f"  per-target GP j={j}: kernel_={g.kernel_}")
print("  return_std and return_cov together:")
try:
    gp2.predict(Xq, return_std=True, return_cov=True)
except Exception as e:
    print(f"  EXC {type(e).__name__}: {e}")

print("\n== determinism via random_state (restarts sample from log-uniform within bounds) ==")
X, y = fixture(40)
def fit_once(rs, restarts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        g = GaussianProcessRegressor(kernel=kernel(), normalize_y=True, n_restarts_optimizer=restarts, random_state=rs).fit(X, y)
    return g.kernel_.theta.copy(), g.predict(Xq)
a = fit_once(0, 5); b = fit_once(0, 5); c = fit_once(1, 5)
print("rs=0 twice bitwise equal theta:", np.array_equal(a[0], b[0]), "pred:", np.array_equal(a[1], b[1]))
print("rs=0 vs rs=1 theta equal:", np.array_equal(a[0], c[0]), "max|dtheta|", np.max(np.abs(a[0] - c[0])))
print("theta is log-transformed hyperparameters, len", len(a[0]), "kernel_.bounds shape", GaussianProcessRegressor(kernel=kernel()).kernel.bounds.shape)

print("\n== bounds ConvergenceWarning trigger: tight bounds ==")
k_tight = ConstantKernel(1.0, (0.9, 1.1)) * RBF(np.ones(D), (0.5, 0.6)) + WhiteKernel(1e-1, (1e-2, 2e-1))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    GaussianProcessRegressor(kernel=k_tight, normalize_y=True, random_state=0).fit(X, y)
for x in w:
    print(f"    WARN {x.category.__name__}: {str(x.message)[:260]}")

print("\n== optimizer=None keeps given hyperparameters; alpha default ==")
g = GaussianProcessRegressor(kernel=kernel(), optimizer=None, normalize_y=True).fit(X, y)
print("optimizer=None kernel_ == kernel:", g.kernel_ == g.kernel, "| theta equal:", np.array_equal(g.kernel_.theta, g.kernel.theta), "| kernel_ is kernel:", g.kernel_ is g.kernel, " default alpha:", GaussianProcessRegressor().alpha)
print("normalize_y with n=1: _y_train_std =", GaussianProcessRegressor(kernel=kernel(), optimizer=None, normalize_y=True).fit(*fixture(1))._y_train_std)
print("sample_y shape (7 queries, 3 draws, 1-D y):", g.sample_y(Xq, n_samples=3, random_state=0).shape)
g2 = GaussianProcessRegressor(kernel=kernel(), optimizer=None, normalize_y=True).fit(*fixture(40, k=2))
print("sample_y shape (7 queries, 3 draws, 2-D y):", g2.sample_y(Xq, n_samples=3, random_state=0).shape)
