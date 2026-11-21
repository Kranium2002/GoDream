"""p06a: single-decision latency of the torch forward (FIXTURE models, untrained weights seed 0 - dense compute
cost does not depend on weight values). One raw feature row (numpy float64, 8 values) -> standardize -> tensor ->
model -> argmax -> python int. 1 torch thread, torch.no_grad(). 50 warm-up calls, then 1000 timed calls with
time.perf_counter_ns. Also exports each model's float32 and float64 .npz for p06b (numpy, no torch).
"""
import os
import platform
import time

import numpy as np
import torch

from snn_common import SpikingStudent, Standardizer, export_npz, make_fixture, N_IN, N_CLASSES

torch.set_num_threads(1)
os.makedirs("p06_params", exist_ok=True)
X, _, _, _ = make_fixture(2000, seed=1)
st = Standardizer(X)
X_q, _, _, _ = make_fixture(1050, seed=31337)
print("platform:", platform.platform(), "| processor:", platform.processor(), "| torch", torch.__version__, "threads", torch.get_num_threads())


def stats(ns):
    a = np.array(ns) / 1e3
    return f"median {np.median(a):8.1f} us  p95 {np.percentile(a, 95):8.1f} us"


for hidden in [(32,), (64,), (64, 64)]:
    for T in [5, 10, 25]:
        torch.manual_seed(0)
        m = SpikingStudent(N_IN, N_CLASSES, hidden=hidden, T=T).eval()
        tag = f"h{'x'.join(map(str, hidden))}_T{T}"
        export_npz(m, f"p06_params/{tag}_f32.npz", st.mean, st.std)
        m64 = SpikingStudent(N_IN, N_CLASSES, hidden=hidden, T=T)
        m64.load_state_dict(m.state_dict())
        export_npz(m64.double(), f"p06_params/{tag}_f64.npz", st.mean, st.std)
        mean32, std32 = st.mean, st.std
        res = {}
        for mode in ["no_grad", "inference_mode"]:
            ctx = torch.no_grad if mode == "no_grad" else torch.inference_mode
            ns = []
            with ctx():
                for i in range(1050):
                    t0 = time.perf_counter_ns()
                    xs = ((X_q[i] - mean32) / std32).astype(np.float32)
                    k = int(m(torch.from_numpy(xs)[None]).argmax(1).item())
                    t1 = time.perf_counter_ns()
                    if i >= 50:
                        ns.append(t1 - t0)
            res[mode] = stats(ns)
        print(f"torch {tag:10s} float32 1-thread  no_grad: {res['no_grad']} | inference_mode: {res['inference_mode']}")
