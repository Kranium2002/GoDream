"""R11: optimizer settings that are not in config.yaml.

fit() builds torch.optim.Adam(model.parameters(), lr=settings.learning_rate); betas, eps and
weight_decay come from torch's defaults and are only recorded after the fact in the training
report. This prints the recorded optimizer of a 20-step FIXTURE training and the students.snn keys
config.yaml actually has.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import _review_common  # noqa: E402,F401
from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=20, evaluate_every_steps=10)
spec = FixtureSnnSpec()
student = SNNStudent.train_in_dream(spec, fixture_dreams(spec, config, 200), settings=settings,
                                    points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=1)
print(f"optimizer recorded in the student's meta: {student.meta['training']['optimizer']}")
keys = sorted(config["students"]["snn"])
print(f"config.yaml students.snn keys: {keys}")
optimizer_keys = [k for k in keys if k.startswith(("adam", "optimizer", "weight_decay")) or k in ("eps", "betas", "adam_betas", "adam_eps")]
print(f"optimizer keys (adam*/optimizer*/weight_decay/eps/betas) in config: {optimizer_keys}")
