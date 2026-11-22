"""Check 5: guard rails of the per-skill spiking student and its training module.

FIXTURE CHECK - mechanics only. Refused with clear errors: a DreamSet from another skill, a
wrong points_per_dim, too few or flat dream rows, invalid settings (including target_spread,
out_of_range, the Adam settings and parity_logit_tolerance), a missing config key,
non-finite inputs (to targets, fit, check_parity, decide_index, decide, operation_counts), a
spec whose name or situation encoding differs from the trained one, a wrong saved-file kind.
Also exercised: ParityError on a deliberately altered network and TrainingError when training
cannot move the weights. Students here are trained for a few steps only (printed below).
Files go to a temporary directory that is removed at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c5_guard_rails.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import copy  # noqa: E402
import dataclasses  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import ConfigError, load_config  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork, save_network  # noqa: E402
from learned_judgment.students.snn_train import ParityError, SNNSettings, TrainingError  # noqa: E402
from learned_judgment.students.targets import soft_targets  # noqa: E402

SHORT_STEPS = 40  # a few steps: these students only exist to exercise the guard rails
BASE_SEED = 5  # fixture run seed

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=SHORT_STEPS, evaluate_every_steps=20)
print(f"\nsettings: config.yaml students.snn with max_steps={SHORT_STEPS} evaluate_every_steps=20")
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 300)
other = FixtureSnnSpec(name="fixture_snn_other")
other_dreams = fixture_dreams(other, config, 300)


def train(spec_=spec, dreams_=dreams, points=FIXTURE_POINTS_PER_DIM, settings_=settings):  # noqa: ANN001, ANN202
    return SNNStudent.train_in_dream(spec_, dreams_, settings=settings_, points_per_dim=points, base_seed=BASE_SEED)


print("\n== dream / grid / config refusals (before any training) ==")
checks.raises("DreamSet of another skill", ValueError, lambda: train(dreams_=other_dreams), "DreamSet.skill 'fixture_snn_other'")
checks.raises("points_per_dim (5, 7) instead of (7, 5)", ValueError, lambda: train(points=(5, 7)), "are not the grid")
checks.raises("points_per_dim (7, 6): 42 instead of 35 candidates", ValueError, lambda: train(points=(7, 6)), "(42 candidates)")
checks.raises("points_per_dim with one entry", ValueError, lambda: train(points=(7,)), "points_per_dim must be 2 integers")
checks.raises("points_per_dim with a float", ValueError, lambda: train(points=(7.0, 5)), "points_per_dim must be 2 integers")
flat = dataclasses.replace(dreams, scores=np.zeros_like(dreams.scores))
checks.raises("flat dream scores", ValueError, lambda: train(dreams_=flat), "score spread is 0")
single = fixture_dreams(spec, config, 1)
checks.raises("a single dream row", ValueError, lambda: train(dreams_=single), "leaves no training rows")
checks.raises("negative base seed", ValueError, lambda: SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=-1), "base_seed")
checks.raises(
    "engine step with config.yaml as is (no skills.fixture_snn)",
    ConfigError,
    lambda: train_in_dream(spec, dreams, config, base_seed=BASE_SEED),
    "skills.fixture_snn.candidate_points_per_dim",
)
no_snn = copy.deepcopy(config)
no_snn["students"]["per_skill_snn"] = False
checks.raises("engine step with students.per_skill_snn false", ValueError, lambda: train_in_dream(spec, dreams, no_snn, base_seed=BASE_SEED), "per_skill_snn")
missing = copy.deepcopy(config)
del missing["students"]["snn"]["beta"]
checks.raises("SNNSettings.from_config without students.snn.beta", ConfigError, lambda: SNNSettings.from_config(missing), "students.snn.beta")
for field, value, needle in (
    ("surrogate", "relu", "students.snn.surrogate"),
    ("threshold", 0.0, "students.snn.threshold"),
    ("beta", 1.5, "students.snn.beta"),
    ("validation_fraction", 1.0, "students.snn.validation_fraction"),
    ("batch_size", 0, "students.snn.batch_size"),
    ("hidden_sizes", (), "students.snn.hidden_sizes"),
    ("learn_beta", "yes", "students.snn.learn_beta"),
    ("max_steps", 10.5, "students.snn.max_steps"),
    ("spike_rate_penalty", float("nan"), "students.snn.spike_rate_penalty"),
    ("target_spread", "median", "students.snn.target_spread"),
    ("out_of_range", "ignore", "students.snn.out_of_range"),
    ("adam_beta1", 1.0, "students.snn.adam_beta1"),
    ("adam_beta2", -0.1, "students.snn.adam_beta2"),
    ("adam_eps", 0.0, "students.snn.adam_eps"),
    ("weight_decay", -1e-4, "students.snn.weight_decay"),
    ("parity_logit_tolerance", -1e-9, "students.snn.parity_logit_tolerance"),
    ("parity_logit_tolerance", float("inf"), "students.snn.parity_logit_tolerance"),
):
    checks.raises(f"SNNSettings {field}={value!r}", ValueError, lambda f=field, v=value: dataclasses.replace(settings, **{f: v}), needle)

print("\n== targets and training-data refusals ==")
checks.raises("soft_targets with inf", ValueError, lambda: soft_targets(np.array([[1.0, np.inf]]), temperature=1.0, target_spread="global_median", spread=1.0), "non-finite")
encoder = InputEncoder.from_variables(spec.situation)
inputs = encoder.encode(dreams.situations)
targets = soft_targets(dreams.scores, temperature=settings.target_temperature, target_spread="global_median", spread=1.0)
model = snn_train.build_model(encoder.width, {spec.name: dreams.n_candidates}, settings, seed=1)
bad_inputs = inputs.copy()
bad_inputs[3, 1] = np.nan


def fit(tasks, validation=None, sampling=None, settings_=settings):  # noqa: ANN001, ANN202
    return snn_train.fit(
        model, tasks, validation=validation or {spec.name: (inputs[:30], targets[:30])},
        sampling=sampling or {spec.name: 1.0}, settings=settings_, seed=1,
    )


checks.raises("fit with a NaN input", ValueError, lambda: fit({spec.name: (bad_inputs, targets)}), "non-finite values (first at row 3, column 1)")
checks.raises("fit with targets not summing to 1", ValueError, lambda: fit({spec.name: (inputs, targets * 2.0)}), "summing to 1")
checks.raises("fit with sampling summing to 0.5", ValueError, lambda: fit({spec.name: (inputs, targets)}, sampling={spec.name: 0.5}), "sum to 1")
checks.raises("fit with an unknown head", KeyError, lambda: fit({"nobody": (inputs, targets)}), "unknown head 'nobody'")
checks.raises("fit with wrong input width", ValueError, lambda: fit({spec.name: (inputs[:, :3], targets)}), "shape (n >= 1, 4)")
checks.raises("build_model with a dotted head name", ValueError, lambda: snn_train.build_model(4, {"a.b": 3}, settings, seed=1), "identifier")
frozen = dataclasses.replace(settings, learning_rate=1e-300)
try:
    snn_train.fit(model, {spec.name: (inputs, targets)}, validation={spec.name: (inputs[:30], targets[:30])},
                  sampling={spec.name: 1.0}, settings=frozen, seed=1)
    checks.expect(False, "learning_rate 1e-300 (weights cannot move) did not raise TrainingError")
except TrainingError as error:
    checks.expect(
        error.report is not None and error.report.best_step == 0,
        f"learning_rate 1e-300 -> TrainingError with report (best_step {error.report.best_step if error.report else None}): {str(error)[:140]}",
    )

print(f"\n== a {SHORT_STEPS}-step student for the decision guard rails ==")
student = train()
print(f"  trained: digest {student.digest[:16]}..., parity {student.meta['parity']['decisions_differing']} differing decisions")
situations = dreams.situations[:20].copy()
nan_situations = situations.copy()
nan_situations[4, 1] = np.nan
inf_situations = situations.copy()
inf_situations[2, 0] = np.inf
checks.raises("decide_index with NaN", ValueError, lambda: student.decide_index(nan_situations), "non-finite values (first at row 4, feature 'fx_speed')")
checks.raises("decide with inf", ValueError, lambda: student.decide(spec, inf_situations), "non-finite values (first at row 2, feature 'fx_lateral')")
checks.raises("operation_counts with NaN", ValueError, lambda: student.operation_counts(nan_situations), "non-finite")
checks.raises("decide_index with 2 columns", ValueError, lambda: student.decide_index(situations[:, :2]), "shape (n, 3)")
checks.raises("decide with another skill's spec", ValueError, lambda: student.decide(other, situations), "trained for skill 'fixture_snn', not 'fixture_snn_other'")
checks.raises(
    "decide with a spec whose situation range changed",
    ValueError,
    lambda: student.decide(FixtureSnnSpec(speed_high=8.0), situations),
    "encodes its situation differently",
)

network = student.network
altered = SpikingNetwork(
    hidden_weights=network.hidden_weights,
    hidden_biases=network.hidden_biases,
    betas=network.betas,
    thresholds=tuple(value * 1.5 for value in network.thresholds),
    readout_weight=network.readout_weight,
    readout_bias=network.readout_bias,
    readout_beta=network.readout_beta,
    time_steps=network.time_steps,
)
parity_model = snn_train.build_model(encoder.width, {spec.name: dreams.n_candidates}, settings, seed=2)
exported = snn_train.export_network(parity_model, spec.name)
tolerance = settings.parity_logit_tolerance
checks.expect(snn_train.check_parity(parity_model, spec.name, exported, inputs[:50], logit_tolerance=tolerance)["decisions_differing"] == 0, "check_parity accepts the model's own export")
altered_export = dataclasses.replace(exported, thresholds=tuple(value * 1.5 for value in exported.thresholds))
checks.raises("check_parity with thresholds x 1.5", ParityError, lambda: snn_train.check_parity(parity_model, spec.name, altered_export, inputs[:50], logit_tolerance=tolerance), "spike-train mismatches")
checks.raises("check_parity with NaN inputs", ValueError, lambda: snn_train.check_parity(parity_model, spec.name, exported, bad_inputs, logit_tolerance=tolerance), "non-finite")
checks.raises("check_parity with another network's shape", ValueError, lambda: snn_train.check_parity(parity_model, spec.name, altered.__class__(
    hidden_weights=(np.zeros((8, 4)),), hidden_biases=(np.zeros(8),), betas=(0.9,), thresholds=(1.0,),
    readout_weight=np.zeros((35, 8)), readout_bias=np.zeros(35), readout_beta=0.9, time_steps=10), inputs[:5], logit_tolerance=tolerance), "not the model's trunk")
checks.raises("check_parity with a negative logit tolerance", ValueError, lambda: snn_train.check_parity(parity_model, spec.name, exported, inputs[:5], logit_tolerance=-1.0), "logit_tolerance")
checks.raises("check_parity without a logit tolerance", TypeError, lambda: snn_train.check_parity(parity_model, spec.name, exported, inputs[:5]), "logit_tolerance")

scratch = Path(tempfile.mkdtemp(prefix="b_snn_offline_c5_"))
checks.redactions[str(scratch)] = "<tmp>"
try:
    save_network(scratch / "not_a_student", network, {"kind": "something_else"})
    checks.raises("load a network saved without the student kind", ValueError, lambda: SNNStudent.load(scratch / "not_a_student"), "is not a per_skill_snn file")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.finish()
