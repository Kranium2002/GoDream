"""Check 8: the decision checks of this lane fail on the OLD measure.py, which shows each defect.

FIXTURE CHECK - mechanics only. Builds a temporary copy of the repository's learned_judgment
package, config.yaml and this proof directory, and replaces learned_judgment/students/measure.py by
its committed version (git show OLD_COMMIT:<path>, read only). Every other module is the current working
tree, including snn.py with this lane's BoundStudent addition (an addition the old measure.py never
calls). Then, with the same Python, inside that copy:

  * each decision check (c2 M3/M6/M7, c4 M2, c5 M5, c6 M1, c7 M4) and the adapted c1 must exit
    non-zero; the first failing line is printed (a missing name counts: the old code had no such
    guard or field);
  * c8_old_probe.py runs the scenarios through the OLD API and must show every defect.

Writes only inside the temporary directory, which is removed.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c8_old_behaviour.py
"""

from _checks import BANNER, REPO_ROOT, Checks

# The commit that holds the old lane code. The check first read HEAD; after the fixed code was
# committed, HEAD holds the fix, so the old code is named by its commit.
OLD_COMMIT = "0eced4653eeb95f99dc91795c4b7297cadac352f"

print(BANNER)

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

LANE_FILE = "learned_judgment/students/measure.py"
DECISION_CHECKS = {
    "c1_end_to_end.py": "existing end-to-end check, adapted (M1-M5 fields)",
    "c2_hand_network.py": "M3 both estimates and the verdict rule, M6 spike trains, M7 hand-built DeployedStudent",
    "c4_overlap.py": "M2 held-out overlap",
    "c5_latency_bound.py": "M5 latency through a bound student",
    "c6_shared_layout.py": "M1 shared-layout DeployedStudent",
    "c7_distinct_situations.py": "M4 distinct situations",
}
HERE = Path(__file__).resolve().parent

checks = Checks()
overlay = Path(tempfile.mkdtemp(prefix="b_snn_measure_c8_"))


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
    proof_dir = overlay / "experiments" / "proofs" / HERE.name
    proof_dir.mkdir(parents=True)
    for script in HERE.glob("*.py"):
        shutil.copy2(script, proof_dir / script.name)
    committed = subprocess.run(["git", "show", f"{OLD_COMMIT}:{LANE_FILE}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
    (overlay / LANE_FILE).write_bytes(committed)
    print(f"  overlay {LANE_FILE}: committed version ({len(committed)} bytes; working tree {len((REPO_ROOT / LANE_FILE).read_bytes())} bytes)")
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import learned_judgment.students.measure as m; "
         "print(m.__file__, hasattr(m, 'HeldOutOverlapError'))", str(overlay)],
        cwd=overlay, capture_output=True, text=True, timeout=300,
    )
    checks.expect(
        probe.returncode == 0 and probe.stdout.strip().startswith(str(overlay)) and probe.stdout.strip().endswith("False"),
        f"the overlay imports its own committed measure.py: {probe.stdout.strip().replace(str(overlay), '<tmp>') or probe.stderr.strip()[-200:]}",
    )

    print("\n== decision checks on the committed (old) measure.py ==")
    for script, label in DECISION_CHECKS.items():
        result = subprocess.run([sys.executable, str(proof_dir / script)], cwd=overlay, capture_output=True, text=True, timeout=1800)
        reason = first_failure(result.stdout + result.stderr).replace(str(overlay), "<tmp>")
        print(f"  {script} ({label}): exit {result.returncode}; {reason[:240]}")
        checks.expect(result.returncode != 0, f"{script} fails on the old code")

    print("\n== the old API, scenario by scenario (c8_old_probe.py inside the overlay) ==")
    result = subprocess.run([sys.executable, str(proof_dir / "c8_old_probe.py")], cwd=overlay, capture_output=True, text=True, timeout=1800)
    for line in (result.stdout + result.stderr[-2000:]).splitlines():
        print(f"    {line.replace(str(overlay), '<tmp>')}")
    checks.expect(result.returncode == 0, f"c8_old_probe.py reproduced every defect on the old code (exit {result.returncode})")
finally:
    shutil.rmtree(overlay, ignore_errors=True)
checks.expect(not overlay.exists(), "temporary directory removed")
checks.finish()
