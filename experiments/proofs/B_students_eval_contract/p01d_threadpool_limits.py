"""p01d: does threadpoolctl.threadpool_limits(limits=1) inside the process reproduce the OPENBLAS_NUM_THREADS=1 digests of p01b? FIXTURE."""
import hashlib, warnings
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

rng = np.random.default_rng(123)
X = rng.normal(size=(2000, 10)); Y = np.c_[np.sin(X[:, 0]) + X[:, 1] ** 2, X[:, 2] * X[:, 3]]
with threadpool_limits(limits=1, user_api="blas"):
    print("inside limits:", [(d["internal_api"], d["num_threads"]) for d in threadpool_info()])
    for solver, hls in [("lbfgs", (64, 64)), ("adam", (64, 64))]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = MLPRegressor(hidden_layer_sizes=hls, solver=solver, random_state=11, max_iter=100).fit(X, Y)
        h = hashlib.sha256()
        for a in m.coefs_ + m.intercepts_:
            h.update(np.ascontiguousarray(a).tobytes())
        print(f"{solver} {hls}: weights_sha256={h.hexdigest()[:16]}")
print("after context:", [(d["internal_api"], d["num_threads"]) for d in threadpool_info()])
