"""Check worker: latency of a saved student's deployed decision path in a FRESH process.

FIXTURE CHECK - mechanics only. Started by c1_end_to_end.py and c5_latency_bound.py with paths
inside their temporary directories. Loads the saved per-skill student (numpy only), builds the
named FIXTURE spec, runs measure_decision_latency (which binds the student to the spec once and
times the full path and the index-only path) with evaluation.latency_decisions from config.yaml,
prints the result as one JSON line and whether torch or snntorch was imported. Exit code 0 only
when the process stayed torch-free.

Usage: python c1_latency_worker.py <student stem> <situations.npy> <fixture spec: fixture_measure | fixture_hand>
"""

from _checks import BANNER

print(BANNER)

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import FixtureHandSpec, FixtureMeasureSpec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.students.measure import measure_decision_latency  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402

SPECS = {"fixture_measure": lambda: FixtureMeasureSpec(), "fixture_hand": lambda: FixtureHandSpec("fixture_hand", 2)}

stem, situations_path, spec_name = sys.argv[1:4]
student = SNNStudent.load(stem)
spec = SPECS[spec_name]()
situations = np.load(situations_path, allow_pickle=False)
latency = measure_decision_latency(student, spec, situations, n_decisions=require(load_config(), "evaluation.latency_decisions"))
torch_loaded, snntorch_loaded = "torch" in sys.modules, "snntorch" in sys.modules
print("LATENCY_JSON " + json.dumps(latency, sort_keys=True, allow_nan=False))
print(f"worker: torch_in_sys_modules={torch_loaded} snntorch_in_sys_modules={snntorch_loaded}")
raise SystemExit(0 if latency["torch_free_process"] and not torch_loaded and not snntorch_loaded else 1)
