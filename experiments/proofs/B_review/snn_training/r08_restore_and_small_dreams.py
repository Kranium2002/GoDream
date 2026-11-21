"""R08: is the restored network the best one; does training run on very few dreams / extreme splits?

  (a) snn_train.fit directly with early stopping forced (patience 1, evaluation every 10 steps):
      the restored model's validation loss equals report.best_validation_loss, best_step is not
      the last step, and the report's firing rate equals the numpy network's on the same rows;
  (b) train_in_dream on 2, 3 and 5 FIXTURE dream rows and with validation_fraction 0.999 / 1e-6:
      what happens (crash, refusal, or a student; how many rows parity was checked on).
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

import _review_common  # noqa: E402,F401
from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_core import count_operations  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402
from learned_judgment.students.targets import score_spread, soft_targets  # noqa: E402

config = load_config()
base = SNNSettings.from_config(config)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 600)
encoder = InputEncoder.from_variables(spec.situation)
x = encoder.encode(dreams.situations)
q = soft_targets(dreams.scores, temperature=base.target_temperature, spread=score_spread(dreams.scores[:500]))

print("== (a) restored = best ==")
settings = dataclasses.replace(base, max_steps=400, evaluate_every_steps=10, patience_evaluations=1, learning_rate=0.05)
model = snn_train.build_model(encoder.width, {"h": dreams.n_candidates}, settings, seed=1)
report = snn_train.fit(model, {"h": (x[:500], q[:500])}, validation={"h": (x[500:], q[500:])}, sampling={"h": 1.0}, settings=settings, seed=2)
import torch  # noqa: E402

threads = torch.get_num_threads()
restored_default_threads = snn_train.validation_losses(model, {"h": (x[500:], q[500:])})["h"]
torch.set_num_threads(settings.torch_threads)
restored = snn_train.validation_losses(model, {"h": (x[500:], q[500:])})["h"]
torch.set_num_threads(threads)
print(f"  restored loss evaluated with {threads} threads: {restored_default_threads!r}; with the training's {settings.torch_threads} thread(s): {restored!r}")
curve = report.validation_total_curve
print(f"  steps_run {report.steps_run} best_step {report.best_step} stopped_early {report.stopped_early}")
print(f"  curve {[(s, round(v, 4)) for s, v in curve]}")
print(f"  restored validation loss {restored!r} == best_validation_loss {report.best_validation_loss!r}: {restored == report.best_validation_loss}")
network = snn_train.export_network(model, "h")
ops = count_operations(network, x[500:])
print(f"  report firing rate {report.firing_rate_per_layer} vs numpy {ops.firing_rate_per_layer.tolist()}")

print("== (b) very few dreams / extreme validation fractions ==")
short = dataclasses.replace(base, max_steps=60, evaluate_every_steps=20)
for label, rows, fraction in (("2 rows", 2, 0.1), ("3 rows", 3, 0.1), ("5 rows", 5, 0.1), ("600 rows, fraction 0.999", 600, 0.999), ("600 rows, fraction 1e-6", 600, 1e-6)):
    subset = fixture_dreams(spec, config, rows)
    try:
        student = SNNStudent.train_in_dream(spec, subset, settings=dataclasses.replace(short, validation_fraction=fraction),
                                            points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=4)
        meta = student.meta
        print(f"  {label}: student; train {meta['dream']['n_rows_train']} validation {meta['dream']['n_rows_validation']}; "
              f"parity checked on {meta['parity']['n_inputs_checked']} rows; best_step {meta['training']['best_step']}; "
              f"agreement {meta['validation_in_dream']['agreement_with_dream_choice']}")
    except Exception as error:  # noqa: BLE001
        print(f"  {label}: {type(error).__name__}: {str(error)[:220]}")
