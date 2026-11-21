"""p02: single-row decision latency, MLPRegressor.predict vs pure-numpy forward pass; MAC count.
FIXTURE data only. Timings are THIS MACHINE's CPU, API-overhead measurements only (not robot/simulator/TabPFN evidence)."""
import os, platform, time, warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info

print("cpu:", platform.processor(), "os.cpu_count", os.cpu_count(), "threadpools", [(d['internal_api'], d['num_threads']) for d in threadpool_info()])

ACT = {
    "identity": lambda z: z,
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
}

def numpy_forward(x, coefs, intercepts, activation="relu", out_activation="identity"):
    h = x
    last = len(coefs) - 1
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        h = h @ W + b
        h = ACT[out_activation if i == last else activation](h)
    return h

def mac_count(coefs):
    # multiply-accumulates per single input row = sum over layers of fan_in * fan_out
    return int(sum(W.shape[0] * W.shape[1] for W in coefs))

def param_count(coefs, intercepts):
    return int(sum(W.size for W in coefs) + sum(b.size for b in intercepts))

def bench(fn, n=1000, warm=50):
    for _ in range(warm):
        fn()
    t = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter_ns(); fn(); t[i] = time.perf_counter_ns() - t0
    return np.median(t) / 1000.0, np.percentile(t, 95) / 1000.0

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 10)); Y = np.c_[np.sin(X[:, 0]), X[:, 1] * X[:, 2]]
x1 = X[:1].copy()
print(f"{'hidden':>10} {'act':>5} {'MACs':>6} {'params':>6} {'max|np-sk|':>11} {'sk_med_us':>9} {'sk_p95_us':>9} {'np_med_us':>9} {'np_p95_us':>9}")
for act in ["relu", "tanh"]:
    for hls in [(16,), (32,), (64,), (32, 32), (64, 64)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = MLPRegressor(hidden_layer_sizes=hls, activation=act, solver="adam", max_iter=50, random_state=0).fit(X, Y)
        coefs = [W.copy() for W in m.coefs_]; ints = [b.copy() for b in m.intercepts_]
        diff = np.max(np.abs(numpy_forward(X, coefs, ints, act, m.out_activation_) - m.predict(X)))
        sk = bench(lambda: m.predict(x1))
        nv = bench(lambda: numpy_forward(x1, coefs, ints, act, "identity"))
        print(f"{str(hls):>10} {act:>5} {mac_count(coefs):>6} {param_count(coefs, ints):>6} {diff:11.3e} {sk[0]:9.1f} {sk[1]:9.1f} {nv[0]:9.1f} {nv[1]:9.1f}")
print("MAC formula: sum_l coefs_[l].shape[0]*coefs_[l].shape[1]  (= d*h1 + h1*h2 + ... + hL*k); adds for biases = sum_l intercepts_[l].size; activations = sum hidden units")
print("batched predict of 64 candidate rows (relu (64,64)):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=50, random_state=0).fit(X, Y)
X64 = X[:64].copy()
print("  sk predict 64 rows med/p95 us:", tuple(round(v, 1) for v in bench(lambda: m.predict(X64))), "shape", m.predict(X64).shape)
