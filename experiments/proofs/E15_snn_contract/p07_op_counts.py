"""p07: operation counts per decision for the p05 FIXTURE-trained model (hidden (64,64), T=10, C=64), numpy only.
Counting code: np_forward.snn_forward_np(..., count=True) + np_forward.op_counts. Mechanics numbers on a fixture."""
import json, sys
import numpy as np
from np_forward import load_params, op_counts, snn_forward_np
P = load_params("p05_params_f64.npz")
xs = np.load("p05_torch_outputs.npz")["xs"]
logits, counts = snn_forward_np(xs, P, count=True)  # counts: (L, T, B)
one = op_counts(P, counts[:, :, 0])
print("ONE decision (fixture input #0):")
print(json.dumps(one, indent=1))
per = [op_counts(P, counts[:, :, b]) for b in range(xs.shape[0])]
for key in ["total_spikes", "synops_total"]:
    v = np.array([p[key] for p in per])
    print(f"over {len(per)} fixture inputs: {key} mean {v.mean():.1f} min {v.min()} max {v.max()}")
fr = np.array([p["mean_firing_rate_per_layer"] for p in per])
print("mean firing rate per layer over inputs:", fr.mean(0).round(4).tolist())
print("constants: first_layer_macs_once", one["first_layer_macs_once"], "| if recomputed per step", one["first_layer_macs_if_recomputed_each_step"],
      "| neuron_state_updates", one["neuron_state_updates"], "| dense MLP MACs reference", one["dense_mlp_macs_reference"])
assert "torch" not in sys.modules
