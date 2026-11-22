"""R07: does a second training in the same process give the fresh-process result, whatever the
global torch / numpy state? Are the caller's thread count and deterministic flag restored?

Uses the existing c3 worker (B_snn_offline/c3_determinism_worker.py: engine.train_in_dream on the
FIXTURE dream, config values with max_steps reduced) in a fresh process, and the same call three
times in this process: plain; after perturbing the global torch and numpy RNGs, the thread count
and the deterministic-algorithms flag; after training another seed in between.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import copy  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import FIXTURE_DIR, REPO_ROOT  # noqa: E402
from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402

SEED, STEPS = 11, 150
config = copy.deepcopy(load_config())
config["students"]["snn"]["max_steps"] = STEPS
config["skills"]["fixture_snn"] = {"candidate_points_per_dim": list(FIXTURE_POINTS_PER_DIM)}
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config)

fresh = subprocess.run(
    [sys.executable, str(FIXTURE_DIR / "c3_determinism_worker.py"), str(SEED), str(STEPS)],
    cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
)
match = re.search(r"digest=([0-9a-f]{64})", fresh.stdout)
fresh_digest = match.group(1) if match else None
print(f"fresh process: exit {fresh.returncode} digest {fresh_digest}")

import torch  # noqa: E402

first = train_in_dream(spec, dreams, config, base_seed=SEED).digest
torch.manual_seed(999)
torch.randn(1000)
np.random.seed(1)
np.random.rand(1000)
torch.set_num_threads(3)
torch.use_deterministic_algorithms(False)
second = train_in_dream(spec, dreams, config, base_seed=SEED).digest
print(f"after training: torch threads {torch.get_num_threads()} (caller set 3), deterministic flag {torch.are_deterministic_algorithms_enabled()} (caller set False)")
other = train_in_dream(spec, dreams, config, base_seed=SEED + 1).digest
third = train_in_dream(spec, dreams, config, base_seed=SEED).digest
print(f"in-process run 1 digest {first}")
print(f"in-process run 2 (perturbed global state) digest {second}")
print(f"in-process run 3 (after training seed {SEED + 1} -> {other[:16]}...) digest {third}")
print(f"all equal to the fresh process: {first == second == third == fresh_digest}")
