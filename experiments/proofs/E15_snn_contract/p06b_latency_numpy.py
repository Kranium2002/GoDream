"""p06b: single-decision latency of the pure-numpy forward (no torch import; asserted). Same protocol as p06a:
raw float64 row -> standardize -> forward -> argmax -> int; 50 warm-up + 1000 timed calls, perf_counter_ns.
Params exported by p06a (FIXTURE models, untrained, seed 0)."""
import sys
import time

import numpy as np

from np_forward import decide, load_params

try:
    from threadpoolctl import threadpool_info
    print("BLAS:", [(d.get("internal_api"), d.get("version"), d.get("num_threads")) for d in threadpool_info()])
except Exception as e:  # threadpoolctl comes with scikit-learn in the probe venv
    print("threadpoolctl unavailable:", e)
sys.path.insert(0, ".")
from snn_common_fixture_only import make_fixture  # noqa: E402  (numpy-only copy of the fixture generator)

X_q, _, _, _ = make_fixture(1050, seed=31337)


def stats(ns):
    a = np.array(ns) / 1e3
    return f"median {np.median(a):8.1f} us  p95 {np.percentile(a, 95):8.1f} us"


for hidden in ["32", "64", "64x64"]:
    for T in [5, 10, 25]:
        out = []
        for dt in ["f32", "f64"]:
            P = load_params(f"p06_params/h{hidden}_T{T}_{dt}.npz")
            ns = []
            for i in range(1050):
                t0 = time.perf_counter_ns()
                k = decide(X_q[i], P)
                t1 = time.perf_counter_ns()
                if i >= 50:
                    ns.append(t1 - t0)
            out.append(f"{dt}: {stats(ns)}")
        print(f"numpy h{hidden:6s} T{T:<3d} " + " | ".join(out))
print("'torch' in sys.modules:", "torch" in sys.modules)
assert "torch" not in sys.modules
