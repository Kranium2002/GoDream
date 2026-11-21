"""Check 8 (decision S2): training refuses a silent or input-independent network.

FIXTURE CHECK - mechanics only. Old behaviour (review finding "TrainingError guard accepts a
silent, input-independent network", reproduction experiments/proofs/B_review/snn_training/r03):
fit raised only when the restored validation loss was not below its loss at initialisation, so
a network whose hidden layer never spikes still passed by moving its readout bias toward the mean
target, and train_in_dream returned a student that picks the same candidate for every situation.
Now fit raises TrainingError when the restored validation loss of a head is not below that of the
best constant prediction (the head's mean training target evaluated on its validation rows), and
when any hidden layer emits no spike on the validation rows. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c8_training_guard.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError, constant_predictor_loss  # noqa: E402
from learned_judgment.students.targets import score_spread, soft_targets  # noqa: E402

BASE_SEED = 9  # fixture run seed
SILENT_THRESHOLD = 1000.0  # fixture: a firing threshold the encoded inputs never reach (review r03)

checks = Checks()
config = load_config()
base = SNNSettings.from_config(config)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 500)

print("\n== constant_predictor_loss (pure function) ==")
hand = constant_predictor_loss(np.array([[0.5, 0.5], [1.0, 0.0]]), np.array([[1.0, 0.0]]))
checks.expect(abs(hand - math.log(1.0 / 0.75)) < 1e-15, f"mean target [0.75, 0.25], validation [1, 0]: {hand:.15f} = log(4/3)")
infinite = constant_predictor_loss(np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([[0.5, 0.5]]))
checks.expect(infinite == math.inf, "a mean target with zero mass where a validation row has mass -> inf")
encoder = InputEncoder.from_variables(spec.situation)
targets = soft_targets(dreams.scores, temperature=base.target_temperature, target_spread="global_median", spread=score_spread(dreams.scores[:400]))
mean = targets[:400].mean(axis=0)
torch_kl = float(functional.kl_div(torch.log(torch.as_tensor(mean)).expand(100, -1), torch.as_tensor(targets[400:]), reduction="batchmean"))
ours = constant_predictor_loss(targets[:400], targets[400:])
checks.expect(abs(ours - torch_kl) < 1e-12, f"equals torch's batch-mean KL of the constant log(mean) prediction ({ours:.12f} vs {torch_kl:.12f})")

print(f"\n== a silent hidden layer (threshold {SILENT_THRESHOLD}) ==")
silent_settings = dataclasses.replace(base, threshold=SILENT_THRESHOLD, max_steps=300, evaluate_every_steps=50)
error = checks.raises(
    "train_in_dream with a threshold no input reaches",
    TrainingError,
    lambda: SNNStudent.train_in_dream(spec, dreams, settings=silent_settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED),
    "emit no spike on the validation rows",
)
if isinstance(error, TrainingError) and error.report is not None:
    report = error.report
    print(
        f"  report: best_step {report.best_step}, validation loss {report.initial_validation_loss:.4f} -> "
        f"{report.best_validation_loss:.4f}, constant prediction {report.constant_predictor_validation_loss}, "
        f"silent layers {report.silent_layers}, firing rate {report.firing_rate_per_layer}"
    )
    checks.expect(report.silent_layers == (0,) and report.firing_rate_per_layer == (0.0,), "the report names hidden layer 0 as silent (firing rate 0)")
    checks.expect(
        report.best_validation_loss < report.initial_validation_loss,
        "the old guard alone (restored loss below initialisation) would have accepted this network",
    )
    restored, constant = report.best_validation_loss_per_head[spec.name], report.constant_predictor_validation_loss[spec.name]
    print(
        f"  (the constant-prediction guard alone {'would' if not restored < constant else 'would NOT'} refuse it here: "
        f"restored {restored:.6f} vs constant {constant:.6f} on the early-stopping rows; the silent-layer guard does)"
    )

print("\n== an input-independent network (every encoded input identical) ==")
inputs = np.zeros((300, encoder.width))
model = snn_train.build_model(encoder.width, {spec.name: dreams.n_candidates}, base, seed=4)
constant_settings = dataclasses.replace(base, max_steps=300, evaluate_every_steps=50)
error = checks.raises(
    "fit on identical inputs, validated on its own training targets",
    TrainingError,
    lambda: snn_train.fit(
        model, {spec.name: (inputs, targets[:300])}, validation={spec.name: (inputs, targets[:300])},
        sampling={spec.name: 1.0}, settings=constant_settings, seed=4,
    ),
    "the loss of the best constant prediction",
)
if isinstance(error, TrainingError) and error.report is not None:
    report = error.report
    print(
        f"  report: validation loss {report.initial_validation_loss:.6f} -> {report.best_validation_loss:.6f} "
        f"(best_step {report.best_step}); constant prediction {report.constant_predictor_validation_loss}; "
        f"silent layers {report.silent_layers}"
    )
    checks.expect(report.best_validation_loss < report.initial_validation_loss, "the old guard alone would have accepted it (loss fell below initialisation)")
    checks.expect(
        report.best_validation_loss_per_head[spec.name] >= report.constant_predictor_validation_loss[spec.name],
        "restored loss >= constant-prediction loss: refused by the new guard",
    )

print("\n== a network that uses the situation passes ==")
good = SNNStudent.train_in_dream(
    spec, dreams, settings=dataclasses.replace(base, max_steps=150, evaluate_every_steps=50), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED
)
training = good.meta["training"]
print(
    f"  per head: restored {training['best_validation_loss_per_head']} < constant {training['constant_predictor_validation_loss']}; "
    f"silent layers {training['silent_layers']}"
)
checks.expect(
    training["best_validation_loss_per_head"][spec.name] < training["constant_predictor_validation_loss"][spec.name]
    and training["silent_layers"] == [],
    "the stored training report shows the restored loss below the constant prediction and no silent layer",
)
checks.expect("acceptance" in training and "constant prediction" in training["acceptance"], "the report states the acceptance rule")
checks.finish()
