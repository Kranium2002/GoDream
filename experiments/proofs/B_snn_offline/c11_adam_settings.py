"""Check 11 (decision S6): Adam's settings come from config.yaml and reach the optimizer.

FIXTURE CHECK - mechanics only. Old behaviour (review finding "Adam settings are hidden
defaults", reproduction experiments/proofs/B_review/snn_training/r11): fit built
torch.optim.Adam(params, lr=...), so betas, eps and weight_decay were torch's defaults, absent
from config.yaml and only recorded after training. Now SNNSettings reads students.snn.adam_beta1,
adam_beta2, adam_eps and weight_decay (nothing defaulted), validates them, and fit passes them to
Adam. Fixture students are trained for a few steps (printed below). Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c11_adam_settings.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import copy  # noqa: E402
import dataclasses  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import ConfigError, load_config, require  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

BASE_SEED = 19  # fixture run seed
KEYS = ("adam_beta1", "adam_beta2", "adam_eps", "weight_decay")
FIXTURE_ADAM = {"adam_beta1": 0.8, "adam_beta2": 0.99, "adam_eps": 1e-6, "weight_decay": 1e-3}  # fixture values, not tuned

checks = Checks()
config = load_config()
settings = SNNSettings.from_config(config)
print(f"\nconfig.yaml: {[(key, require(config, f'students.snn.{key}')) for key in KEYS]}")
checks.expect(all(getattr(settings, key) == float(require(config, f"students.snn.{key}")) for key in KEYS), "SNNSettings carries the four values from config.yaml")
for key in KEYS:
    missing = copy.deepcopy(config)
    del missing["students"]["snn"][key]
    checks.raises(f"config without students.snn.{key}", ConfigError, lambda m=missing: SNNSettings.from_config(m), f"students.snn.{key}")

print("\n== the values reach torch.optim.Adam ==")
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 300)
short = dataclasses.replace(settings, max_steps=100, evaluate_every_steps=50)
students = {}
for label, values in (("config values", {}), ("fixture values", FIXTURE_ADAM)):
    run = dataclasses.replace(short, **values)
    students[label] = SNNStudent.train_in_dream(spec, dreams, settings=run, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
    optimizer = students[label].meta["training"]["optimizer"]
    print(f"  {label}: optimizer record {optimizer}; network {students[label].digest[:16]}...")
    checks.expect(
        optimizer["betas"] == [run.adam_beta1, run.adam_beta2] and optimizer["eps"] == run.adam_eps
        and optimizer["weight_decay"] == run.weight_decay and optimizer["lr"] == run.learning_rate,
        f"{label}: the optimizer ran with betas ({run.adam_beta1}, {run.adam_beta2}), eps {run.adam_eps}, weight_decay {run.weight_decay}",
    )
    checks.expect(
        all(students[label].meta["settings"][key] == getattr(run, key) for key in KEYS),
        f"{label}: the student's stored settings hold the four values",
    )
checks.expect(students["config values"].digest != students["fixture values"].digest, "other Adam values train a different network from the same seed")
for key, value in FIXTURE_ADAM.items():
    one = SNNStudent.train_in_dream(
        spec, dreams, settings=dataclasses.replace(short, **{key: value}), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED
    )
    checks.expect(one.digest != students["config values"].digest, f"changing only {key} to {value} changes the trained network")
checks.finish()
