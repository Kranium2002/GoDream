"""Check 18: the decision checks (c7-c14) fail on the OLD code of this lane, which accepts each defect.

FIXTURE CHECK - mechanics only. Builds a temporary copy of the repository's learned_judgment
package, config.yaml and this proof directory, replaces the four files of this lane
(students/snn.py, students/snn_train.py, students/targets.py, engine/train_in_dream.py) by their
committed versions (git show OLD_COMMIT:<path>, read only), and runs each check there with the same
Python. Every other module is the current working tree. A decision check must exit non-zero
there; the reason printed is the first failing line (a missing name counts: the old code had no
such guard). The three capability checks (c15-c17) are run and reported, not asserted, because
they test mechanics, not a fix. Writes only inside the temporary directory, which is removed.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c18_old_behaviour.py
"""

# The commit that holds the old lane code. The check first read HEAD; after the fixed code was
# committed, HEAD holds the fix, so the old code is named by its commit.
OLD_COMMIT = "e077b7924728d252f14d5877c818063b4d1c819a"

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from fixture_snn_spec import REPO_ROOT, Checks  # noqa: E402

LANE_FILES = (
    "learned_judgment/students/snn.py",
    "learned_judgment/students/snn_train.py",
    "learned_judgment/students/targets.py",
    "learned_judgment/engine/train_in_dream.py",
)
DECISION_CHECKS = {
    "c7_grid_mismatch.py": "S1 grid mismatch",
    "c8_training_guard.py": "S2 training guard",
    "c9_parity.py": "S3/S4 parity and decide_one",
    "c10_target_spread.py": "S5 target spread",
    "c11_adam_settings.py": "S6 Adam settings",
    "c12_out_of_range.py": "S7 out of range",
    "c13_split_and_digests.py": "M2/M4 split by situation, digests",
    "c14_deployed_student.py": "M1 DeployedStudent, early-stopping labels",
}
CAPABILITY_CHECKS = ("c15_two_layers_engine.py", "c16_spike_rate_penalty.py", "c17_multi_head_fit.py")
HERE = Path(__file__).resolve().parent

checks = Checks()
overlay = Path(tempfile.mkdtemp(prefix="b_snn_offline_c18_"))
checks.redactions[str(overlay)] = "<tmp>"


def first_failure(output: str) -> str:
    lines = output.splitlines()
    for line in lines:
        if "[FAILED]" in line:
            return line.strip()
    errors = [line for line in lines if line and not line.startswith(" ") and ("Error" in line or "Exception" in line)]
    return errors[-1].strip() if errors else (lines[-1].strip() if lines else "(no output)")


try:
    shutil.copytree(REPO_ROOT / "learned_judgment", overlay / "learned_judgment", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO_ROOT / "config.yaml", overlay / "config.yaml")
    proof_dir = overlay / "experiments" / "proofs" / "B_snn_offline"
    proof_dir.mkdir(parents=True)
    for script in HERE.glob("*.py"):
        shutil.copy2(script, proof_dir / script.name)
    for path in LANE_FILES:
        committed = subprocess.run(["git", "show", f"{OLD_COMMIT}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
        (overlay / path).write_bytes(committed)
        current = (REPO_ROOT / path).read_bytes()
        print(f"  overlay {path}: committed version ({len(committed)} bytes; working tree {len(current)} bytes)")
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import learned_judgment.students.snn as m; print(m.__file__)", str(overlay)],
        cwd=overlay, capture_output=True, text=True, timeout=300,
    )
    checks.expect(probe.returncode == 0 and probe.stdout.strip().startswith(str(overlay)), f"the overlay imports its own (committed) snn.py: {probe.stdout.strip() or probe.stderr.strip()[-200:]}")

    print("\n== decision checks on the committed (old) lane code ==")
    for script, label in DECISION_CHECKS.items():
        result = subprocess.run([sys.executable, str(proof_dir / script)], cwd=overlay, capture_output=True, text=True, timeout=1800)
        reason = first_failure(result.stdout + result.stderr).replace(str(overlay), "<tmp>")
        checks.expect(result.returncode != 0, f"{script} ({label}) fails on the old code (exit {result.returncode}): {reason[:200]}")

    print("\n== the same scenarios through the OLD API (c18_old_probe.py): the old code accepts each defect ==")
    result = subprocess.run([sys.executable, str(proof_dir / "c18_old_probe.py")], cwd=overlay, capture_output=True, text=True, timeout=1800)
    lines = {line.split(":", 1)[0][4:]: line for line in result.stdout.splitlines() if line.startswith("OLD ")}
    checks.expect(result.returncode == 0, f"the old-API probe ran (exit {result.returncode}){'' if result.returncode == 0 else ': ' + result.stderr.strip()[-300:]}")
    for item in ("S1", "S1b", "S2", "S3", "S3b", "S3c", "S4", "S5", "S6", "S7", "M2", "M4", "M1", "labels"):
        line = lines.get(item, f"OLD {item}: (no line printed)")
        checks.expect(line.startswith(f"OLD {item}: DEFECT"), line.replace(str(overlay), "<tmp>")[:260])

    print("\n== capability checks on the committed (old) lane code (reported, not asserted) ==")
    for script in CAPABILITY_CHECKS:
        result = subprocess.run([sys.executable, str(proof_dir / script)], cwd=overlay, capture_output=True, text=True, timeout=1800)
        reason = first_failure(result.stdout + result.stderr).replace(str(overlay), "<tmp>")
        print(f"  {script}: exit {result.returncode}; {'first failure: ' + reason[:200] if result.returncode else 'passes on the old code too'}")
finally:
    shutil.rmtree(overlay, ignore_errors=True)
checks.expect(not overlay.exists(), "temporary directory removed")
checks.finish()
