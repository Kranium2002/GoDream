"""p01b: cross-process determinism of MLPRegressor fits (FIXTURE data). Prints a digest of coefs_/intercepts_ and predictions.
Run it in several separate processes (optionally with different BLAS thread env) and compare digests."""
import hashlib, inspect, os, sys, warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info

info = [(d.get("internal_api"), d.get("num_threads")) for d in threadpool_info()]
print("threadpools:", info, "OPENBLAS_NUM_THREADS=", os.environ.get("OPENBLAS_NUM_THREADS"))
print("MLPRegressor.fit signature:", inspect.signature(MLPRegressor.fit))
rng = np.random.default_rng(123)
X = rng.normal(size=(2000, 10)); Y = np.c_[np.sin(X[:, 0]) + X[:, 1] ** 2, X[:, 2] * X[:, 3]]
for solver, hls in [("lbfgs", (64, 64)), ("adam", (64, 64)), ("adam", (32,))]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = MLPRegressor(hidden_layer_sizes=hls, solver=solver, random_state=11, max_iter=100).fit(X, Y)
    h = hashlib.sha256()
    for a in m.coefs_ + m.intercepts_:
        h.update(np.ascontiguousarray(a).tobytes())
    hp = hashlib.sha256(np.ascontiguousarray(m.predict(X)).tobytes()).hexdigest()[:16]
    print(f"{solver} {hls}: weights_sha256={h.hexdigest()[:16]} pred_sha256={hp} n_iter_={m.n_iter_}")
