"""Check 4 worker: load a saved student in a FRESH process, decide, and report whether torch
was imported. Started by c2_end_to_end.py with paths inside its temporary directory.

FIXTURE CHECK - mechanics only. Usage:
    python c4_load_worker.py <student stem> <situations.npy> <expected_indices.npy> <expected_digest>
Exit code 0 when the decisions and digest are identical and neither torch nor snntorch was imported.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root

from learned_judgment.students.snn import SNNStudent  # noqa: E402

stem, situations_path, expected_path, expected_digest = sys.argv[1:5]
student = SNNStudent.load(stem)
situations = np.load(situations_path, allow_pickle=False)
expected = np.load(expected_path, allow_pickle=False)
decisions = student.decide_index(situations)
identical = bool(np.array_equal(decisions, expected))
digest_ok = student.digest == expected_digest
torch_loaded = "torch" in sys.modules
snntorch_loaded = "snntorch" in sys.modules
print(f"worker: loaded skill={student.skill} digest_matches={digest_ok} decisions={decisions.shape[0]}")
print(f"worker: decisions_identical={identical}")
print(f"worker: torch_in_sys_modules={torch_loaded} snntorch_in_sys_modules={snntorch_loaded}")
raise SystemExit(0 if identical and digest_ok and not torch_loaded and not snntorch_loaded else 1)
