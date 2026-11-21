"""Check 17: one fit over two heads (a shared trunk), with the per-head acceptance rule.

FIXTURE CHECK - mechanics only. Two fixture specs with the same situation layout and different
candidate grids (35 and 15 candidates) give two heads on one spiking trunk; fit draws each batch
from both heads in the sampling shares. Checked: the report covers both heads (rows, shares,
validation curve per head, restored loss below each head's constant prediction); each head's
batch share follows the sampling; a head whose restored loss is not below its constant prediction
is refused by name even when the total improves; each head exports to a numpy network that passes
parity. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c17_multi_head_fit.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402
from learned_judgment.students.targets import score_spread, soft_targets  # noqa: E402

GRIDS = {"fx_a": (7, 5), "fx_b": (5, 3)}  # fixture grids: 35 and 15 candidates
SAMPLING = {"fx_a": 0.7, "fx_b": 0.3}  # fixture batch shares
SEED = 43  # fixture seed

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=300, evaluate_every_steps=50)
encoder = None
tasks, validation = {}, {}
for name, grid in GRIDS.items():
    spec = FixtureSnnSpec(name=name)
    dreams = fixture_dreams(spec, config, 400, points_per_dim=grid)
    encoder = InputEncoder.from_variables(spec.situation)
    encoded = encoder.encode(dreams.situations)
    targets = soft_targets(dreams.scores, temperature=settings.target_temperature, target_spread="global_median", spread=score_spread(dreams.scores[:360]))
    tasks[name], validation[name] = (encoded[:360], targets[:360]), (encoded[360:], targets[360:])
heads = {name: tasks[name][1].shape[1] for name in GRIDS}
print(f"\nheads {heads}; sampling {SAMPLING}; 360 training and 40 validation rows per head")

model = snn_train.build_model(encoder.width, heads, settings, seed=SEED)
report = snn_train.fit(model, tasks, validation=validation, sampling=SAMPLING, settings=settings, seed=SEED)
print(f"  steps_run {report.steps_run} best_step {report.best_step}; restored per head {report.best_validation_loss_per_head}; "
      f"constant prediction {report.constant_predictor_validation_loss}; firing rate {report.firing_rate_per_layer}")
checks.expect(report.heads == tuple(GRIDS) and report.sampling == SAMPLING, "the report names both heads and their batch shares")
checks.expect(report.n_train_rows == {"fx_a": 360, "fx_b": 360} and report.n_validation_rows == {"fx_a": 40, "fx_b": 40}, "rows per head")
checks.expect(all(set(losses) == set(GRIDS) for _, losses in report.validation_loss_curve), "the validation curve holds a loss per head at every evaluation")
total = sum(SAMPLING[h] * report.best_validation_loss_per_head[h] for h in GRIDS)
checks.expect(abs(total - report.best_validation_loss) < 1e-12, "the restored total = sampling-weighted sum of the per-head losses")
checks.expect(
    all(report.best_validation_loss_per_head[h] < report.constant_predictor_validation_loss[h] for h in GRIDS),
    "each head's restored loss is below its own constant prediction",
)

import torch  # noqa: E402

generator = torch.Generator().manual_seed(0)
counts = np.stack([snn_train._batch_counts(np.array([0.7, 0.3]), settings.batch_size, generator) for _ in range(2000)])
print(f"  rows per batch over 2000 draws: mean {counts.mean(axis=0)} (batch {settings.batch_size}); min {counts.min(axis=0)} max {counts.max(axis=0)}")
checks.expect(
    bool(np.all(counts.sum(axis=1) == settings.batch_size)) and bool(np.all(np.abs(counts.mean(axis=0) - np.array([0.7, 0.3]) * settings.batch_size) < 0.05)),
    "each batch has batch_size rows and the mean share per head follows the sampling",
)

for name in GRIDS:
    network = snn_train.export_network(model, name)
    rows = np.vstack([tasks[name][0], validation[name][0]])
    parity = snn_train.check_parity(model, name, network, rows, logit_tolerance=settings.parity_logit_tolerance)
    checks.expect(
        network.n_outputs == heads[name] and parity["spike_train_mismatches_per_layer"] == [0] and parity["decide_one_vs_batch_differing"] == 0,
        f"head {name}: exported numpy network ({network.n_outputs} outputs) passes parity on all 400 rows",
    )
trunk_a, trunk_b = snn_train.export_network(model, "fx_a"), snn_train.export_network(model, "fx_b")
checks.expect(all(np.array_equal(a, b) for a, b in zip(trunk_a.hidden_weights, trunk_b.hidden_weights)), "both heads' networks share the same trunk weights")

print("\n== a head that cannot learn is refused by name ==")
# FIXTURE: head fx_b gets identical inputs on every row, so it can only learn a constant; fx_a
# learns, so the total loss improves and the old total-only guard would accept the fit.
blind = {**tasks, "fx_b": (np.zeros_like(tasks["fx_b"][0]), tasks["fx_b"][1])}
blind_validation = {**validation, "fx_b": (np.zeros_like(tasks["fx_b"][0]), tasks["fx_b"][1])}
model_2 = snn_train.build_model(encoder.width, heads, settings, seed=SEED)
error = checks.raises(
    "fit where head fx_b sees no situation", TrainingError,
    lambda: snn_train.fit(model_2, blind, validation=blind_validation, sampling=SAMPLING, settings=settings, seed=SEED),
    "head 'fx_b': the restored validation loss",
)
if isinstance(error, TrainingError) and error.report is not None:
    checks.expect(
        error.report.best_validation_loss < error.report.initial_validation_loss
        and error.report.best_validation_loss_per_head["fx_a"] < error.report.constant_predictor_validation_loss["fx_a"],
        "(the total improved and head fx_a learned: only the per-head rule refuses the fit)",
    )
checks.finish()
