"""Check (c): the same skill sequence in separate processes gives identical networks at every stage.

FIXTURE CHECK - mechanics only. Starts three separate processes of c3_determinism_worker.py, each
adding fx_alpha -> fx_beta -> fx_gamma with rehearsal=True (config threads:
students.snn.torch_threads; max_steps reduced, printed below): base seed A twice and base seed B
once. The same seed must give identical slots, network digests of every skill after every add,
digests of every stage kept at the end, split rows and retention numbers; a different seed must give
different digests and different splits. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c3_determinism.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

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
    print(f"  process seed={seed} exit={process.returncode}")
    if result is not None:
        for step in result["after_each_add"]:
            digests = {skill: digest[:16] for skill, digest in step["network_digests"].items()}
            print(
                f"    after {step['skill']} (slot {step['slot']}): steps_run={step['steps_run']} best_step={step['best_step']} "
                f"best_validation_loss={step['best_validation_loss']!r} threads={step['torch_threads']} digests(16)={digests}"
            )
        print(f"    split rows sha256(16): { {k: v[:16] for k, v in result['split_rows_sha256'].items()} }")
    if stderr.strip():
        print(f"    stderr tail: {stderr.strip()[-300:]}")
    checks.expect(process.returncode == 0 and result is not None, f"worker for seed {seed} finished")
    results.append(result)

first, second, other = results
checks.expect(
    first is not None and second is not None and first["after_each_add"] == second["after_each_add"],
    "same seed in two separate processes -> identical slots, digests of every view after every add and retention numbers",
)
checks.expect(
    first is not None and second is not None and first["stages"] == second["stages"] and first["split_rows_sha256"] == second["split_rows_sha256"],
    "same seed -> identical digests of every stage kept and identical training / early-stopping / retention rows",
)
checks.expect(
    first is not None and first["stages"][0] == first["after_each_add"][0]["network_digests"]
    and first["stages"][1] == first["after_each_add"][1]["network_digests"],
    "the stages kept at the end are the views right after each add",
)
checks.expect(
    first is not None and other is not None
    and all(first["final"][skill] != other["final"][skill] for skill in first["final"])
    and all(first["split_rows_sha256"][skill] != other["split_rows_sha256"][skill] for skill in first["split_rows_sha256"]),
    "different seed -> different digests for every skill and different splits",
)
checks.finish()
