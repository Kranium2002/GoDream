"""p02b: latency of the full picklable student object TransformedTargetRegressor(Pipeline(StandardScaler, MLP), StandardScaler)
vs a numpy forward pass that folds both scalers in. FIXTURE data; this machine's CPU; API-overhead only."""
import time, warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

def bench(fn, n=1000, warm=50):
    for _ in range(warm):
        fn()
    t = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter_ns(); fn(); t[i] = time.perf_counter_ns() - t0
    return round(np.median(t) / 1000.0, 1), round(np.percentile(t, 95) / 1000.0, 1)

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 10)) * 7 + 3; Y = np.c_[np.sin(X[:, 0]) * 40 + 5, X[:, 1] * X[:, 2]]
model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100, random_state=0)), transformer=StandardScaler())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    model.fit(X, Y)
xs = model.regressor_.named_steps["standardscaler"]; mlp = model.regressor_.named_steps["mlpregressor"]; ys = model.transformer_
W = [w.copy() for w in mlp.coefs_]; B = [b.copy() for b in mlp.intercepts_]
mx, sx, my, sy = xs.mean_.copy(), xs.scale_.copy(), ys.mean_.copy(), ys.scale_.copy()

def fwd(x):
    h = (x - mx) / sx
    for i in range(len(W) - 1):
        h = np.maximum(h @ W[i] + B[i], 0.0)
    return (h @ W[-1] + B[-1]) * sy + my

d = np.max(np.abs(fwd(X) - model.predict(X)))
rel = np.max(np.abs(fwd(X) - model.predict(X)) / np.maximum(1e-12, np.abs(model.predict(X))))
print("max abs diff numpy-folded vs TTR.predict:", d, "max rel diff:", rel)
x1 = X[:1].copy()
print("TTR(Pipeline) predict 1 row  med/p95 us:", bench(lambda: model.predict(x1)))
print("numpy folded fwd 1 row      med/p95 us:", bench(lambda: fwd(x1)))
X64 = X[:64].copy()
print("TTR(Pipeline) predict 64 rows med/p95 us:", bench(lambda: model.predict(X64)))
print("numpy folded fwd 64 rows      med/p95 us:", bench(lambda: fwd(X64)))
