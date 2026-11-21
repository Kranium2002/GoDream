"""R01b: a constructed network on which decide_one and the batched decide disagree.

R01 showed the first-layer current of one row differs in the last bit between the batched matmul
(n rows) and the single-row matmul decide_one runs. This script builds a FIXTURE network whose
threshold sits between those two values for one neuron of one input row, and whose readout turns
that one spike into the decision, to show that the "same arithmetic in the same order" claim of
snn_core._simulate does not hold and that check_parity (batched path only) does not cover
decide_one. Contrived on purpose; for trained networks R01 saw no decision difference.
No torch import.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import numpy as np  # noqa: E402

from _review_common import random_network  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

rng = np.random.default_rng(5)
base = random_network(rng, (16, 64, 2), time_steps=1, beta=0.0)
x = rng.normal(size=(500, 16))
batch_current = x @ base.hidden_weights[0].T + base.hidden_biases[0]
single_current = np.stack([(x[i : i + 1] @ base.hidden_weights[0].T + base.hidden_biases[0])[0] for i in range(500)])
row, neuron = map(int, np.argwhere((batch_current != single_current) & (batch_current > 0.1))[0])  # positive: no reset at t0
low, high = sorted((batch_current[row, neuron], single_current[row, neuron]))
print(f"row {row} neuron {neuron}: batched current {batch_current[row, neuron]!r}, single-row current {single_current[row, neuron]!r}")
readout = np.zeros((2, 64))
readout[1, neuron] = 1.0
network = SpikingNetwork(
    hidden_weights=base.hidden_weights,
    hidden_biases=base.hidden_biases,
    betas=(0.0,),
    thresholds=(float(low),),  # strict '>' : the larger of the two values fires, the smaller does not
    readout_weight=readout,
    readout_bias=np.array([0.5, 0.0]),
    readout_beta=0.0,
    time_steps=1,
)
batched = network.decide(x)[row]
single = network.decide_one(x[row])
print(f"decide(batch of 500)[{row}] = {batched}; decide_one(row {row}) = {single}; differ = {batched != single}")
_, batch_counts = network.forward_with_spikes(x)
_, single_counts = network.forward_with_spikes(x[row : row + 1])
print(f"spikes of row {row}: batched {int(batch_counts[0, 0, row])}, single {int(single_counts[0, 0, 0])}")
