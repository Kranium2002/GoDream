"""Check 1 worker: latency of a saved student's deployed decision path in a FRESH process.

FIXTURE CHECK - mechanics only. Started by c1_end_to_end.py with paths inside its temporary
directory. Loads the saved student (numpy only), runs measure_decision_latency with
evaluation.latency_decisions from config.yaml, prints the result as one JSON line and whether
torch or snntorch was imported. Exit code 0 only when the process stayed torch-free.

Usage: python c1_latency_worker.py <student stem> <situations.npy>
"""

from _checks import BANNER

print(BANNER)

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.students.measure import measure_decision_latency  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402

stem, situations_path = sys.argv[1:3]
student = SNNStudent.load(stem)
situations = np.load(situations_path, allow_pickle=False)
latency = measure_decision_latency(student, situations, n_decisions=require(load_config(), "evaluation.latency_decisions"))
torch_loaded, snntorch_loaded = "torch" in sys.modules, "snntorch" in sys.modules
print("LATENCY_JSON " + json.dumps(latency, sort_keys=True, allow_nan=False))
print(f"worker: torch_in_sys_modules={torch_loaded} snntorch_in_sys_modules={snntorch_loaded}")
raise SystemExit(0 if latency["torch_free_process"] and not torch_loaded and not snntorch_loaded else 1)
