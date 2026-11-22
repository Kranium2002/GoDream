"""Check (c) worker: add fx_alpha -> fx_beta -> fx_gamma (rehearsal=True) in this process and print
the per-skill network digests after every add, the digests of every stage kept at the end, the
split rows and the retention numbers.

FIXTURE CHECK - mechanics only. Usage: python c3_determinism_worker.py <base_seed> <max_steps>
Config: config.yaml loaded in memory with the fixture shared_feature_width and max_steps replaced by
<max_steps> (fixture_shared_specs.fixture_config); config.yaml is not modified. Writes no file.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_shared_specs import POINTS, SEQUENCE, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402

base_seed, max_steps = int(sys.argv[1]), int(sys.argv[2])
config = fixture_config(load_config())
config["students"]["snn"]["max_steps"] = max_steps
student = SharedSNNStudent.create(config)
after_each_add = []
for name in SEQUENCE:
    spec = fixture_spec(name)
    record = student.add_skill(spec, fixture_dreams(spec, config), points_per_dim=POINTS[name], base_seed=base_seed, rehearsal=True)
    after_each_add.append(
        {
            "skill": name,
            "slot": record["slot"],
            "network_digests": record["network_digests"],
            "steps_run": record["steps_run"],
            "best_step": record["training"]["best_step"],
            "best_validation_loss": record["training"]["best_validation_loss"],
            "torch_threads": record["training"]["torch_threads"],
            "retention_in_dream": record["retention_in_dream"],
        }
    )
rows = {
    name: hashlib.sha256(
        b"".join(getattr(student, part)(name).tobytes() for part in ("training_rows", "early_stopping_rows", "retention_rows"))
    ).hexdigest()
    for name in SEQUENCE
}
print(
    "RESULT_JSON "
    + json.dumps(
        {
            "base_seed": base_seed,
            "after_each_add": after_each_add,
            "stages": [student.network_digests(stage) for stage in range(student.n_stages)],
            "split_rows_sha256": rows,
            "final": student.network_digests(),
        }
    )
)
