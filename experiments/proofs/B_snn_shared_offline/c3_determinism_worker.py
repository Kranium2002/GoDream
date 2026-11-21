"""Check 3 worker: add fx_alpha -> fx_beta -> fx_gamma (rehearsal=True) in this process and print
the per-skill network digests after every add.

FIXTURE CHECK - mechanics only. Usage: python c3_determinism_worker.py <base_seed> <max_steps>
Config: config.yaml loaded in memory with the fixture shared_feature_width and fixture skill names
(fixture_shared_specs.fixture_config); settings from it with max_steps replaced by <max_steps>.
config.yaml is not modified. Writes no file.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_shared_specs import POINTS, SEQUENCE, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

base_seed, max_steps = int(sys.argv[1]), int(sys.argv[2])
config = fixture_config(load_config())
student = SharedSNNStudent(
    skills=require(config, "students.snn.shared_skills"),
    feature_width=require(config, "students.snn.shared_feature_width"),
    settings=dataclasses.replace(SNNSettings.from_config(config), max_steps=max_steps),
    rehearsal_fraction=require(config, "students.snn.rehearsal_fraction"),
)
after_each_add = []
for name in SEQUENCE:
    spec = fixture_spec(name)
    record = student.add_skill(spec, fixture_dreams(spec, config), points_per_dim=POINTS[name], base_seed=base_seed, rehearsal=True)
    after_each_add.append(
        {
            "skill": name,
            "network_digests": record["network_digests"],
            "steps_run": record["steps_run"],
            "best_step": record["training"]["best_step"],
            "best_validation_loss": record["training"]["best_validation_loss"],
            "torch_threads": record["training"]["torch_threads"],
        }
    )
print("RESULT_JSON " + json.dumps({"base_seed": base_seed, "after_each_add": after_each_add, "final": student.network_digests()}))
