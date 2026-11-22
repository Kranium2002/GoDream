"""p03d: per-(input, neuron) firing-rate histogram at init for different input scales / thresholds (FIXTURE inputs).
'binary fraction' = share of (input, neuron) pairs whose rate over T steps is exactly 0 or exactly 1, i.e. the layer
transmits only a sign bit for that pair instead of a graded rate. hidden (64,), T=10, beta 0.9, default init, seed 0."""
import numpy as np, torch
from snn_common import SpikingStudent, Standardizer, make_fixture, N_IN, N_CLASSES
torch.set_num_threads(1)
X, _, _, _ = make_fixture(2000, seed=1); st = Standardizer(X); xs = st(X)
for scale, thr in [(0.1, 1.0), (1.0, 1.0), (3.0, 1.0), (10.0, 1.0), (1.0, 0.1), (1.0, 5.0)]:
    torch.manual_seed(0)
    m = SpikingStudent(N_IN, N_CLASSES, hidden=(64,), T=10, threshold=thr)
    with torch.no_grad():
        _, rec = m(torch.as_tensor(xs * scale, dtype=torch.float32), record=True)
    r = rec[0].mean(0).numpy()  # (B, H) rate per input/neuron
    print(f"scale={scale:5.1f} thr={thr:4.1f}: mean rate {r.mean():.3f}; rate==0 {np.mean(r == 0):.3f}; rate==1 {np.mean(r == 1):.3f}; "
          f"binary {np.mean((r == 0) | (r == 1)):.3f}; graded (0<r<1) {np.mean((r > 0) & (r < 1)):.3f}")
