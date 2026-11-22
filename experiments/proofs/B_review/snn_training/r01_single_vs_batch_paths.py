"""R01: is the per-decision path (encode_one + decide_one) bitwise the batched path (encode + forward)?

snn_core._simulate's docstring claims "a batch, a single decision and a recorded run use the same
arithmetic in the same order"; encode_one claims bitwise equality with encode. measure.py is planned
to switch to decide_one / encode_one, while check_parity only compares torch with the BATCHED numpy
path. This script compares, on fixture networks (random torch-like init, 1 and 2 hidden layers):
  * logits of row i from forward(X) vs _simulate(X[i:i+1]) (what decide_one runs), bitwise;
  * spike counts and decisions of both paths;
  * the first-layer current of one row computed in a batch vs alone;
  * encode_one vs encode on in-range, range-end and out-of-range situations.
No torch import.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import numpy as np  # noqa: E402

from _review_common import random_network  # noqa: E402
from fixture_snn_spec import FixtureSnnSpec  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402

rng = np.random.default_rng(20261121)
for sizes in ((4, 64, 35), (4, 64, 64, 35), (16, 64, 35), (40, 128, 64, 35)):
    network = random_network(rng, sizes, time_steps=10)
    x = rng.normal(size=(2000, sizes[0])) * 1.2
    batch_logits, batch_counts = network.forward_with_spikes(x)
    single_logits = np.empty_like(batch_logits)
    single_counts = np.empty_like(batch_counts)
    for i in range(x.shape[0]):
        logits, counts = network.forward_with_spikes(x[i : i + 1])
        single_logits[i] = logits[0]
        single_counts[:, :, i] = counts[:, :, 0]
    decide_one = np.array([network.decide_one(row) for row in x])
    batch_dec = network.decide(x)
    first_batch = x @ network.hidden_weights[0].T + network.hidden_biases[0]
    first_single = np.stack([(x[i : i + 1] @ network.hidden_weights[0].T + network.hidden_biases[0])[0] for i in range(x.shape[0])])
    # batch-size dependence: the same rows inside batches of different sizes
    half = network.forward(x[:777])
    print(
        f"sizes {sizes}: rows with logits not bitwise equal (batch vs single) = "
        f"{int(np.sum(np.any(batch_logits != single_logits, axis=1)))} / {x.shape[0]}, "
        f"max |diff| = {float(np.max(np.abs(batch_logits - single_logits))):.3e}; "
        f"first-layer current rows differing = {int(np.sum(np.any(first_batch != first_single, axis=1)))}; "
        f"spike-count cells differing = {int(np.sum(batch_counts != single_counts))}; "
        f"decide_one vs decide differing = {int(np.sum(decide_one != batch_dec))}; "
        f"batch 777 vs 2000 rows differing = {int(np.sum(np.any(half != batch_logits[:777], axis=1)))}"
    )

print("\n== encode_one vs encode (fixture spec: 2 floats + 1 categorical) ==")
spec = FixtureSnnSpec()
encoder = InputEncoder.from_variables(spec.situation)
situations = spec.sample_situations(stream_rng(3, "review/encode"), 20000)
ends = np.array([[-1.0, 2.0, 0.0], [1.0, 6.0, 1.0], [-1.0, 6.0, 1.0], [1.0, 2.0, 0.0]])
outside = np.array([[-5.0, 30.0, 1.0], [7.5, -2.0, 0.0], [1e6, 1e-9, 1.0]])
for label, block in (("sampled", situations), ("range ends", ends), ("outside range", outside)):
    batch = encoder.encode(block)
    single = np.stack([encoder.encode_one(row) for row in block])
    print(f"  {label}: rows not bitwise equal = {int(np.sum(np.any(batch != single, axis=1)))} / {block.shape[0]}")
print(f"  range ends encode to: {encoder.encode(ends)[:2].round(6).tolist()}")
print(f"  out-of-range situations are encoded without refusal: {encoder.encode(outside).round(3).tolist()}")
