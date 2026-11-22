"""Check 3: training is reproducible across processes and depends on the seed.

FIXTURE CHECK - mechanics only. Starts three separate processes of c3_determinism_worker.py,
which trains a student on the fixture dream through engine.train_in_dream (config threads:
students.snn.torch_threads; max_steps reduced, printed below): base seed A twice and base seed
B once. The same seed must give the same network digest; a different seed a different digest.
Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c3_determinism.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from fixture_snn_spec import REPO_ROOT, Checks  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402

MAX_STEPS = 150  # reduced from config students.snn.max_steps for a short mechanics check
SEED_A, SEED_B = 11, 12  # fixture run seeds

checks = Checks()
print(
    f"\nworkers: max_steps={MAX_STEPS} (config {require(load_config(), 'students.snn.max_steps')}), "
    f"torch_threads={require(load_config(), 'students.snn.torch_threads')} (config); seeds {SEED_A}, {SEED_A}, {SEED_B}"
)
worker = Path(__file__).resolve().parent / "c3_determinism_worker.py"
runs = [SEED_A, SEED_A, SEED_B]
processes = [
    subprocess.Popen(
        [sys.executable, str(worker), str(seed), str(MAX_STEPS)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for seed in runs
]
digests = []
for seed, process in zip(runs, processes):
    stdout, stderr = process.communicate(timeout=900)
    result = [line for line in stdout.splitlines() if line.startswith("RESULT")]
    print(f"  process pid={process.pid} exit={process.returncode}: {result[0] if result else stdout.strip()[-300:]}")
    if stderr.strip():
        print(f"    stderr tail: {stderr.strip()[-300:]}")
    checks.expect(process.returncode == 0 and bool(result), f"worker for seed {seed} finished")
    match = re.search(r"digest=([0-9a-f]{64})", result[0]) if result else None
    digests.append(match.group(1) if match else None)

checks.expect(digests[0] is not None and digests[0] == digests[1], "same seed in two separate processes -> identical network digest")
checks.expect(digests[2] is not None and digests[2] != digests[0], "different seed -> different network digest")
checks.finish()
