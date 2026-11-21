"""Check 16: a spike_rate_penalty above 0 lowers the firing rate (mechanics on the fixture).

FIXTURE CHECK - mechanics only. The same fixture dream, seed and settings are trained with
spike_rate_penalty 0 (config value) and with two fixture penalties above 0 (not tuned values).
Checked: the firing rate of the restored network falls below that of the unpenalised one, both
on the early-stopping rows (training report, torch) and on new fixture situations (deployed
numpy network, operation_counts), and no layer goes silent. Whether a penalty is worth its cost
in decision quality is a dev-data question this check does not answer. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c16_spike_rate_penalty.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

PENALTIES = (0.5, 1.0)  # fixture penalties above 0, chosen for a visible effect, not tuned
MAX_STEPS = 300  # reduced from config students.snn.max_steps for a short mechanics check
BASE_SEED = 37  # fixture run seed

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=MAX_STEPS, evaluate_every_steps=50)
print(f"\nconfig spike_rate_penalty {require(config, 'students.snn.spike_rate_penalty')}; fixture penalties {PENALTIES}; max_steps {MAX_STEPS}")
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 500)
situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 400)

rates = {}
for penalty in (0.0, *PENALTIES):
    student = SNNStudent.train_in_dream(
        spec, dreams, settings=dataclasses.replace(settings, spike_rate_penalty=penalty), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED
    )
    training = student.meta["training"]
    numpy_rate = float(student.operation_counts(situations).firing_rate_per_layer[0])
    rates[penalty] = (training["firing_rate_per_layer"][0], numpy_rate, training["silent_layers"])
    print(f"  penalty {penalty}: firing rate on early-stopping rows {training['firing_rate_per_layer'][0]:.4f}, on 400 new situations "
          f"{numpy_rate:.4f}; validation KL {training['best_validation_loss']:.4f}; settings record {student.meta['settings']['spike_rate_penalty']}")
for penalty in PENALTIES:
    checks.expect(rates[penalty][0] < rates[0.0][0], f"penalty {penalty}: lower firing rate than penalty 0 on the early-stopping rows (torch report)")
    checks.expect(rates[penalty][1] < rates[0.0][1], f"penalty {penalty}: lower firing rate than penalty 0 on new situations (deployed numpy network)")
    checks.expect(rates[penalty][2] == [], f"penalty {penalty}: no silent layer")
checks.expect(rates[PENALTIES[1]][1] < rates[PENALTIES[0]][1], f"the larger penalty {PENALTIES[1]} fires less than {PENALTIES[0]}")
checks.finish()
