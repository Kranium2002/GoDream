"""R10: snn_core operation counts and recorded paths against a hand count (1 and 2 hidden layers).

  * forward_with_spike_trains summed over neurons == forward_with_spikes counts; logits identical
    across forward / forward_with_spikes / forward_with_spike_trains;
  * count_operations fields against a hand count from the spike trains;
  * save/load round trip in a temporary directory: digest, decisions, no pickle needed.
No torch import.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import random_network  # noqa: E402
from learned_judgment.students.snn_core import count_operations, load_network, save_network  # noqa: E402

rng = np.random.default_rng(2)
for sizes in ((5, 12, 7), (5, 12, 9, 7)):
    net = random_network(rng, sizes, time_steps=6)
    x = rng.normal(size=(40, sizes[0])) * 2
    l0 = net.forward(x)
    l1, counts = net.forward_with_spikes(x)
    l2, trains = net.forward_with_spike_trains(x)
    same_logits = bool(np.array_equal(l0, l1) and np.array_equal(l0, l2))
    counts_from_trains = np.stack([t.sum(axis=2).astype(np.int64) for t in trains])
    ops = count_operations(net, x)
    hidden = sizes[1:-1]
    fan_out = list(sizes[2:])
    hand_spikes = np.stack([t.sum(axis=(0, 2)) for t in trains], axis=1).astype(float)  # (n, L)
    hand_synops = hand_spikes @ np.array(fan_out, dtype=float)
    hand_dense = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    print(
        f"sizes {sizes}: logits identical across paths {same_logits}; trains->counts equal {bool(np.array_equal(counts, counts_from_trains))}; "
        f"spikes/decision equal {bool(np.array_equal(ops.spikes_per_decision, hand_spikes))}; synops equal {bool(np.array_equal(ops.synaptic_ops_per_decision, hand_synops))}; "
        f"first_layer_macs {ops.first_layer_macs} (hand {sizes[0] * sizes[1]}); neuron_updates {ops.neuron_updates} (hand {(sum(hidden) + sizes[-1]) * 6}); "
        f"dense_reference_macs {ops.dense_reference_macs} (hand {hand_dense}); firing rate {ops.firing_rate_per_layer.round(4).tolist()} "
        f"(hand {(hand_spikes.mean(0) / (np.array(hidden) * 6)).round(4).tolist()})"
    )
    scratch = Path(tempfile.mkdtemp(prefix="review_r10_"))
    try:
        digest = save_network(scratch / "net", net, {"note": "fixture"})
        loaded, meta = load_network(scratch / "net.npz")
        print(f"  save/load: digest equal {digest == loaded.digest() == meta['network_sha256']}; decisions equal {bool(np.array_equal(loaded.decide(x), net.decide(x)))}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
print(f"torch imported: {'torch' in sys.modules}")
