"""Check 3: the same skill sequence in two separate processes gives identical per-skill digests.

FIXTURE CHECK - mechanics only. Starts three separate processes of c3_determinism_worker.py,
each adding fx_alpha -> fx_beta -> fx_gamma with rehearsal=True (config threads:
students.snn.torch_threads; max_steps reduced, printed below): base seed A twice and base seed B
once. The same seed must give identical network digests for every skill after every add; a
different seed must give different digests. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c3_determinism.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from fixture_shared_specs import REPO_ROOT, Checks  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402

MAX_STEPS = 150  # reduced from config students.snn.max_steps for a short mechanics check
SEED_A, SEED_B = 11, 12  # fixture run seeds

checks = Checks()
config = load_config()
print(
    f"\nworkers: max_steps={MAX_STEPS} (config {require(config, 'students.snn.max_steps')}), "
    f"torch_threads={require(config, 'students.snn.torch_threads')} (config); seeds {SEED_A}, {SEED_A}, {SEED_B}"
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
results = []
for seed, process in zip(runs, processes):
    stdout, stderr = process.communicate(timeout=1800)
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT_JSON ")]
    result = json.loads(lines[0][len("RESULT_JSON "):]) if lines else None
    print(f"  process pid={process.pid} seed={seed} exit={process.returncode}")
    if result is not None:
        for step in result["after_each_add"]:
            digests = {skill: digest[:16] for skill, digest in step["network_digests"].items()}
            print(
                f"    after {step['skill']}: steps_run={step['steps_run']} best_step={step['best_step']} "
                f"best_validation_loss={step['best_validation_loss']!r} threads={step['torch_threads']} digests(16)={digests}"
            )
    if stderr.strip():
        print(f"    stderr tail: {stderr.strip()[-300:]}")
    checks.expect(process.returncode == 0 and result is not None, f"worker for seed {seed} finished")
    results.append(result)

same = results[0] is not None and results[1] is not None and results[0]["after_each_add"] == results[1]["after_each_add"]
checks.expect(same, "same seed in two separate processes -> identical digests of every skill's view after every add")
checks.expect(
    results[0] is not None and results[2] is not None
    and all(results[0]["final"][skill] != results[2]["final"][skill] for skill in results[0]["final"]),
    "different seed -> different digests for every skill",
)
checks.finish()
