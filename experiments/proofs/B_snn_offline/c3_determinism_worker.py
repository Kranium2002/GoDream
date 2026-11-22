"""Check 3 worker: train a student through the engine step in this process; print its digest.

FIXTURE CHECK - mechanics only. Usage: python c3_determinism_worker.py <base_seed> <max_steps>
The config is config.yaml loaded in memory with two in-memory changes for the fixture:
students.snn.max_steps = <max_steps> (reduced for a short check) and
skills.fixture_snn.candidate_points_per_dim = the fixture grid. config.yaml is not modified.
Writes no file.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import copy  # noqa: E402
import sys  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402

base_seed, max_steps = int(sys.argv[1]), int(sys.argv[2])
config = copy.deepcopy(load_config())
config["students"]["snn"]["max_steps"] = max_steps
config["skills"]["fixture_snn"] = {"candidate_points_per_dim": list(FIXTURE_POINTS_PER_DIM)}
spec = FixtureSnnSpec()
student = train_in_dream(spec, fixture_dreams(spec, config), config, base_seed=base_seed)
training = student.meta["training"]
print(
    f"RESULT base_seed={base_seed} digest={student.digest} steps_run={training['steps_run']} "
    f"best_step={training['best_step']} best_validation_loss={training['best_validation_loss']!r} "
    f"init_seed={student.meta['seeds']['init_seed']} train_seed={student.meta['seeds']['train_seed']}"
)
