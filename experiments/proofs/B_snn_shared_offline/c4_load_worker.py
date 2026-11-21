"""Check 4 worker: load a saved shared student in a FRESH process, decide for every skill, report
whether torch was imported, then add one more fixture skill (fx_delta) to the loaded student.
Started by c1_sequence.py with paths inside its temporary directory.

FIXTURE CHECK - mechanics only. Usage:
    python c4_load_worker.py <student directory> <scratch directory> <base_seed>
Prints RESULT_JSON with the network digests after the add. Exit code 0 when the loaded digests and
decisions are identical and neither torch nor snntorch was imported before the add.
Writes nothing (the add happens in memory only).
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_shared_specs import POINTS, SEQUENCE, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402

directory, scratch, base_seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
student = SharedSNNStudent.load(directory)
expected_digests = json.loads((scratch / "expected_digests.json").read_text(encoding="utf-8"))
digests_ok = student.network_digests() == expected_digests
identical = True
for name in SEQUENCE:
    situations = np.load(scratch / f"{name}_situations.npy", allow_pickle=False)
    expected = np.load(scratch / f"{name}_expected.npy", allow_pickle=False)
    index = student.decide_index(name, situations)
    actions = student.decide(fixture_spec(name), situations)
    counts = student.operation_counts(name, situations)
    identical = identical and bool(np.array_equal(index, expected)) and actions.shape[0] == counts.n_decisions == index.shape[0]
torch_loaded, snntorch_loaded = "torch" in sys.modules, "snntorch" in sys.modules
print(f"worker: loaded skills_held={list(student.skills_held)} digests_match={digests_ok}")
print(f"worker: decisions_identical={identical}")
print(f"worker: before adding a skill: torch_in_sys_modules={torch_loaded} snntorch_in_sys_modules={snntorch_loaded}")

spec = fixture_spec("fx_delta")
record = student.add_skill(spec, fixture_dreams(spec, load_config()), points_per_dim=POINTS["fx_delta"], base_seed=base_seed, rehearsal=True)
print(f"worker: after add_skill(fx_delta): torch_in_sys_modules={'torch' in sys.modules}")
parity_clean = all(
    all(m == 0 for m in report["spike_count_mismatches_per_layer"]) and report["decisions_differing"] == 0
    for report in record["parity"].values()
)
print(
    "RESULT_JSON "
    + json.dumps(
        {
            "skills_held": list(student.skills_held),
            "digests": student.network_digests(),
            "steps_run": record["steps_run"],
            "parity_clean": parity_clean,
        }
    )
)
raise SystemExit(0 if digests_ok and identical and not torch_loaded and not snntorch_loaded else 1)
