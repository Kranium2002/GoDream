"""Check 15: a two-hidden-layer student trained through engine.train_in_dream.

FIXTURE CHECK - mechanics only. The config is config.yaml loaded in memory with three in-memory
changes for the fixture (config.yaml is not modified): students.snn.hidden_sizes = [32, 24],
students.snn.max_steps reduced, and skills.fixture_snn.candidate_points_per_dim = the fixture
grid. Checked: the engine step builds two spiking hidden layers; neither is silent; parity
(spike trains of both layers, decisions, logits, decide_one) holds on every dream row; the
deployed numpy network has the layer sizes and operation counts of two layers; save / load keeps
the decisions. Files go to a temporary directory that is removed at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c15_two_layers_engine.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import copy  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402

HIDDEN = [32, 24]  # fixture architecture: two hidden layers
MAX_STEPS = 400  # reduced from config students.snn.max_steps for a short mechanics check
BASE_SEED = 41  # fixture run seed

checks = Checks()
config = copy.deepcopy(load_config())
config["students"]["snn"]["hidden_sizes"] = HIDDEN
config["students"]["snn"]["max_steps"] = MAX_STEPS
config["skills"]["fixture_snn"] = {"candidate_points_per_dim": list(FIXTURE_POINTS_PER_DIM)}
print(f"\nin-memory config: students.snn.hidden_sizes={HIDDEN} max_steps={MAX_STEPS}; skills.fixture_snn.candidate_points_per_dim={list(FIXTURE_POINTS_PER_DIM)}")

spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 800)
student = train_in_dream(spec, dreams, config, base_seed=BASE_SEED)
training, parity = student.meta["training"], student.meta["parity"]
print(f"  steps_run {training['steps_run']} best_step {training['best_step']}; firing rate per layer {training['firing_rate_per_layer']}; "
      f"dead {training['dead_fraction_per_layer']}; validation loss {training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}")
print(f"  parity: {parity}")
checks.expect(student.network.hidden_sizes == tuple(HIDDEN) and student.network.layer_sizes == (4, 32, 24, dreams.n_candidates), f"deployed network layers {student.network.layer_sizes}")
checks.expect(student.meta["settings"]["hidden_sizes"] == HIDDEN, "the settings record both hidden layers")
checks.expect(training["silent_layers"] == [] and all(rate > 0.0 for rate in training["firing_rate_per_layer"]), "both hidden layers fire on the early-stopping rows")
checks.expect(
    parity["n_inputs_checked"] == dreams.n_rows and parity["spike_train_mismatches_per_layer"] == [0, 0]
    and parity["spike_train_cells_per_layer"] == [10 * dreams.n_rows * 32, 10 * dreams.n_rows * 24],
    "parity: spike trains of both layers compared on every row, zero mismatches",
)
checks.expect(
    parity["decisions_differing"] == 0 and parity["decide_one_vs_batch_differing"] == 0
    and parity["max_abs_logit_difference"] <= parity["logit_tolerance"],
    "parity: decisions, decide_one and logits within the tolerance",
)

situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 300)
counts = student.operation_counts(situations)
print(f"  operation counts: layer sizes {counts.layer_sizes}, fan-out {counts.fan_out}, mean spikes per layer "
      f"{[round(float(v), 1) for v in counts.mean_spikes_per_layer]}, dense reference MACs {counts.dense_reference_macs}")
checks.expect(counts.fan_out == (24, dreams.n_candidates) and counts.dense_reference_macs == 4 * 32 + 32 * 24 + 24 * dreams.n_candidates, "operation counts follow the two-layer structure")
index = student.decide_index(situations)
checks.expect(bool(np.array_equal(np.array([student.decide_index_one(s) for s in situations]), index)), "decide_index_one = decide_index on 300 situations")

scratch = Path(tempfile.mkdtemp(prefix="b_snn_offline_c15_"))
checks.redactions[str(scratch)] = "<tmp>"
try:
    student.save(scratch / "two_layer")
    loaded = SNNStudent.load(scratch / "two_layer")
    checks.expect(loaded.digest == student.digest and bool(np.array_equal(loaded.decide_index(situations), index)), "save / load: same digest and decisions")
    checks.expect(bool(np.array_equal(loaded.decide(spec, situations), student.decide(spec, situations))), "save / load: same commands through the grid checks")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
