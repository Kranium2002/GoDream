"""p05b: numpy-only deployment parity vs snnTorch outputs saved by p05a. Never imports torch (asserted)."""
import sys

import numpy as np

from np_forward import load_params, snn_forward_np

ref = dict(np.load("p05_torch_outputs.npz"))
xs = ref["xs"]
for tag, dt in [("64", np.float64), ("32", np.float32)]:
    P = load_params(f"p05_params_f{tag}.npz")
    print(f"=== float{tag}: params dtype {P['Wo'].dtype}, beta0 {P['beta0']!r}, thr0 {P['thr0']!r}, T {int(P['T'])} ===")
    lg, rec = snn_forward_np(xs.astype(dt), P, record=True)
    tl = ref[f"logits{tag}"]
    L = int(P["n_hidden_layers"])
    mism = [int((rec[i] != ref[f"spk{tag}_{i}"]).sum()) for i in range(L)]
    tot = [int(rec[i].size) for i in range(L)]
    print(f"  batched (1000 inputs): max|logit diff| = {np.abs(lg - tl).max():.3e}; spike mismatches per layer {mism} of {tot}; "
          f"argmax agreement {(lg.argmax(1) == tl.argmax(1)).mean():.4f}")
    lg1 = np.concatenate([snn_forward_np(xs[i:i + 1].astype(dt), P) for i in range(len(xs))])
    print(f"  single-row numpy vs torch batched: max|logit diff| = {np.abs(lg1 - tl).max():.3e}; argmax agreement {(lg1.argmax(1) == tl.argmax(1)).mean():.4f}")
    if tag == "32":
        ts = ref["single32"]
        print(f"  single-row numpy vs single-row torch: max|logit diff| = {np.abs(lg1 - ts).max():.3e}; argmax agreement {(lg1.argmax(1) == ts.argmax(1)).mean():.4f}")
print("'torch' in sys.modules:", "torch" in sys.modules)
assert "torch" not in sys.modules
print("numpy", np.__version__)
