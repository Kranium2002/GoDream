"""Check 9 (decisions S3/S4): parity compares spike trains, logits and the per-decision path.

FIXTURE CHECK - mechanics only. Old behaviour (review findings on parity, reproductions
experiments/proofs/B_review/snn_training/r09 and r01b): check_parity compared per-step spike
COUNTS and argmax decisions on the validation rows only (sometimes 1 row), never bounded the
logit difference, and never ran the deployed per-decision path. A network with a mis-exported
readout_beta was accepted on 1 and on 10 rows; a network whose hidden neurons are permuted has
equal counts but different spike trains; and decide_one can differ from the batched decision in
the last bit of the first-layer current. Now check_parity compares every neuron's spike train,
the decisions, logits within students.snn.parity_logit_tolerance, and decide_one on every row
against the batched and the torch decisions; train_in_dream runs it on every training and
early-stopping row; the student decides single situations through encoder.encode_one +
network.decide_one (SNNStudent.decide_index_one / decide_one). Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c9_parity.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import ParityError, SNNSettings  # noqa: E402

BASE_SEED = 13  # fixture run seed

checks = Checks()
config = load_config()
settings = SNNSettings.from_config(config)
tolerance = settings.parity_logit_tolerance
print(f"\nparity_logit_tolerance = {tolerance} (config students.snn.parity_logit_tolerance)")
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 600)
encoder = InputEncoder.from_variables(spec.situation)
inputs = encoder.encode(dreams.situations)
model = snn_train.build_model(encoder.width, {"h": dreams.n_candidates}, settings, seed=21)
exported = snn_train.export_network(model, "h")

print("\n== the model's own export passes, and the report covers every comparison ==")
report = snn_train.check_parity(model, "h", exported, inputs, logit_tolerance=tolerance)
print(f"  {report}")
checks.expect(
    report["spike_train_mismatches_per_layer"] == [0]
    and report["spike_train_cells_per_layer"] == [settings.time_steps * inputs.shape[0] * settings.hidden_sizes[0]],
    "spike trains compared cell by cell (T x n x H) with zero mismatches",
)
checks.expect(
    report["decide_one_vs_batch_differing"] == 0 and report["decide_one_vs_torch_differing"] == 0,
    "decide_one on every row = batched decisions = torch decisions",
)

print("\n== readout_beta mis-exported (0.5 instead of the model's value) ==")
wrong_readout = dataclasses.replace(exported, readout_beta=0.5)
for rows in (1, 10):
    old_criterion = bool(np.array_equal(np.argmax(model(torch.as_tensor(inputs[:rows]), "h").detach().numpy(), axis=1), wrong_readout.decide(inputs[:rows])))
    print(f"  {rows} row(s): decisions equal = {old_criterion} (the old check compared decisions and counts only)")
    checks.raises(
        f"check_parity on {rows} row(s)", ParityError,
        lambda rows=rows: snn_train.check_parity(model, "h", wrong_readout, inputs[:rows], logit_tolerance=tolerance),
        "exceeds the tolerance",
    )

print("\n== hidden neurons permuted: equal spike counts, different spike trains ==")
permutation = np.random.default_rng(0).permutation(settings.hidden_sizes[0])
permuted = dataclasses.replace(
    exported,
    hidden_weights=(exported.hidden_weights[0][permutation],),
    hidden_biases=(exported.hidden_biases[0][permutation],),
    readout_weight=exported.readout_weight[:, permutation],
)
_, permuted_counts = permuted.forward_with_spikes(inputs)
_, own_counts = exported.forward_with_spikes(inputs)
checks.expect(bool(np.array_equal(permuted_counts, own_counts)), "(the permuted network emits the same spike count at every step: the old count comparison is blind to it)")
error = checks.raises(
    "check_parity with permuted hidden neurons", ParityError,
    lambda: snn_train.check_parity(model, "h", permuted, inputs, logit_tolerance=tolerance),
    "spike-train mismatches",
)
if isinstance(error, ParityError):
    print(f"  report: {error.report}")

print("\n== a constructed network on which decide_one differs from the batched path ==")
# Fixture construction after review r01b: one row's first-layer current differs in the last bit
# between the batched matmul and the single-row matmul; the threshold is set between the two
# values and the readout turns that one spike into the decision. T = 1, beta = 0, readout_beta = 0.
width, hidden = 16, 64
flip_settings = dataclasses.replace(settings, hidden_sizes=(hidden,), time_steps=1, beta=0.0, readout_beta=0.0)
flip_model = snn_train.build_model(width, {"h": 2}, flip_settings, seed=5)
x = np.random.default_rng(5).normal(size=(500, width))
weights = flip_model.trunk_linears[0].weight.detach().numpy()
bias = flip_model.trunk_linears[0].bias.detach().numpy()
with torch.no_grad():
    torch_current = flip_model.trunk_linears[0](torch.as_tensor(x)).numpy()
batch_current = x @ weights.T + bias
single_current = np.stack([(x[i : i + 1] @ weights.T + bias)[0] for i in range(x.shape[0])])
candidates = np.argwhere((batch_current != single_current) & (torch_current == batch_current) & (batch_current > 0.1))
print(f"  (row, neuron) pairs whose single-row current differs from the batched (and torch) current: {len(candidates)}")
checks.expect(len(candidates) > 0, "a row whose single-row current differs in the last bit exists")
if len(candidates):
    row, neuron = (int(value) for value in candidates[0])
    low = float(min(batch_current[row, neuron], single_current[row, neuron]))  # strict '>': the larger value fires
    rebuilt = snn_train.build_model(width, {"h": 2}, dataclasses.replace(flip_settings, threshold=low), seed=5)
    with torch.no_grad():
        rebuilt.trunk_linears[0].weight.copy_(flip_model.trunk_linears[0].weight)
        rebuilt.trunk_linears[0].bias.copy_(flip_model.trunk_linears[0].bias)
        rebuilt.head_linears["h"].weight.zero_()
        rebuilt.head_linears["h"].weight[1, neuron] = 1.0
        rebuilt.head_linears["h"].bias.copy_(torch.tensor([0.5, 0.0], dtype=torch.float64))
    flip_network = snn_train.export_network(rebuilt, "h")
    batched, single = int(flip_network.decide(x)[row]), flip_network.decide_one(x[row])
    print(f"  row {row} neuron {neuron}: batched current {batch_current[row, neuron]!r}, single-row {single_current[row, neuron]!r}; "
          f"threshold {low!r}; decide (batch of {x.shape[0]}) = {batched}, decide_one = {single}")
    checks.expect(batched != single, "decide_one and the batched decide disagree on that row")
    error = checks.raises(
        "check_parity on the constructed network", ParityError,
        lambda: snn_train.check_parity(rebuilt, "h", flip_network, x, logit_tolerance=tolerance),
        "the per-decision path (decide_one) differs",
    )
    if isinstance(error, ParityError):
        report = error.report
        print(f"  report: spike-train mismatches {report['spike_train_mismatches_per_layer']}, batched decisions differing "
              f"{report['decisions_differing']}, decide_one vs batch {report['decide_one_vs_batch_differing']}")
        checks.expect(
            report["spike_train_mismatches_per_layer"] == [0] and report["decisions_differing"] == 0,
            "(torch and the batched numpy path agree; only the per-decision path differs, which the old check never ran)",
        )

print("\n== train_in_dream: parity on every row; the student decides single situations via decide_one ==")
student = SNNStudent.train_in_dream(
    spec, dreams, settings=dataclasses.replace(settings, max_steps=150, evaluate_every_steps=50),
    points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED,
)
parity = student.meta["parity"]
checks.expect(
    parity["n_inputs_checked"] == dreams.n_rows == student.meta["dream"]["n_rows_train"] + student.meta["dream"]["n_rows_early_stopping"],
    f"parity checked on all {dreams.n_rows} rows = training rows + early-stopping rows",
)
situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 500)
batch_index = student.decide_index(situations)
single_index = np.array([student.decide_index_one(situation) for situation in situations])
checks.expect(bool(np.array_equal(single_index, batch_index)), "SNNStudent.decide_index_one (encode_one + decide_one) = decide_index on 500 situations")
direct = np.array([student.network.decide_one(student.encoder.encode_one(situation)) for situation in situations])
checks.expect(bool(np.array_equal(single_index, direct)), "decide_index_one is encoder.encode_one + network.decide_one")
actions = student.decide(spec, situations)
single_actions = np.stack([student.decide_one(spec, situation) for situation in situations])
checks.expect(single_actions.shape == (500, 2) and bool(np.array_equal(single_actions, actions)), "SNNStudent.decide_one (A,) = decide on every situation")
checks.raises("decide_index_one with a (1, S) array", ValueError, lambda: student.decide_index_one(situations[:1]), "shape (3,)")
checks.finish()
