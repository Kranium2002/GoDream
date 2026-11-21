"""Check 7: checks (a)-(e) fail on the OLD code of this lane, and the old code shows each defect.

FIXTURE CHECK - mechanics only. Two temporary overlays of the repository's learned_judgment package,
config.yaml and this proof directory, run with the same Python:

  overlay A  only this lane's file, students/shared_snn.py, is the committed version (git show
             OLD_COMMIT:<path>, read only); every other module is the current working tree. Checks c1-c5
             must exit non-zero there; the reason printed is the first failing line.
  overlay B  the student package (shared_snn.py and the modules it ran with) is the version of the
             commit that last changed the committed shared_snn.py, so the old code runs as it did;
             c7_old_probe.py runs the scenarios of (a)-(e) through the OLD API and must report a
             defect for each decision (H1 slots, H3 retention rows and row split, H4 out-of-range
             and grid checks, H5 deployed views and kept stages).

Writes only inside the temporary directories, which are removed.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c7_old_behaviour.py
"""

# The commit that holds the old lane code. The check first read HEAD; after the fixed code was
# committed, HEAD holds the fix, so the old code is named by its commit.
OLD_COMMIT = "0eced4653eeb95f99dc91795c4b7297cadac352f"

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from fixture_shared_specs import REPO_ROOT, Checks  # noqa: E402

LANE_FILE = "learned_judgment/students/shared_snn.py"
OLD_PACKAGE = (
    "learned_judgment/students/shared_snn.py",
    "learned_judgment/students/encoding.py",
    "learned_judgment/students/snn.py",
    "learned_judgment/students/snn_core.py",
    "learned_judgment/students/snn_train.py",
    "learned_judgment/students/targets.py",
    "learned_judgment/engine/train_in_dream.py",
)
DECISION_CHECKS = {
    "c2_one_skill.py": "(a) one skill with empty slots, second skill, stage-0 view (H1, H2, H5)",
    "c1_sequence.py": "(b) retention rows never trained on or used for early stopping (H3)",
    "c3_determinism.py": "(c) determinism of every stage across processes (H6)",
    "c4_save_load.py": "(d) save, load without torch, every skill and stage, add a skill (H5, H6)",
    "c5_guard_rails.py": "(e) guard rails: slots, width, grid, ranges (H1, H4)",
}
PROBE_ITEMS = ("H1", "H3", "H3b", "H5a", "H5b", "H4a", "H4b", "H4c")
HERE = Path(__file__).resolve().parent

checks = Checks()


def git(*arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, check=True).stdout


def first_failure(output: str) -> str:
    lines = output.splitlines()
    for line in lines:
        if "[FAILED]" in line:
            return line.strip()
    errors = [line for line in lines if line and not line.startswith(" ") and ("Error" in line or "Exception" in line)]
    return errors[-1].strip() if errors else (lines[-1].strip() if lines else "(no output)")


def make_overlay(prefix: str) -> tuple[Path, Path]:
    overlay = Path(tempfile.mkdtemp(prefix=prefix))
    checks.redactions[str(overlay)] = "<tmp>"
    shutil.copytree(REPO_ROOT / "learned_judgment", overlay / "learned_judgment", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO_ROOT / "config.yaml", overlay / "config.yaml")
    proof_dir = overlay / "experiments" / "proofs" / "B_snn_shared_offline"
    proof_dir.mkdir(parents=True)
    for script in HERE.glob("*.py"):
        shutil.copy2(script, proof_dir / script.name)
    return overlay, proof_dir


def imported_from(overlay: Path) -> str:
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import learned_judgment.students.shared_snn as m; print(m.__file__)", str(overlay)],
        cwd=overlay, capture_output=True, text=True, timeout=300,
    )
    return probe.stdout.strip() or probe.stderr.strip()[-200:]


overlays: list[Path] = []
try:
    print("\n== overlay A: the committed shared_snn.py with the current modules ==")
    overlay_a, proof_a = make_overlay("b_snn_shared_c7a_")
    overlays.append(overlay_a)
    committed = git("show", f"{OLD_COMMIT}:{LANE_FILE}")
    (overlay_a / LANE_FILE).write_bytes(committed)
    print(f"  overlay {LANE_FILE}: committed version ({len(committed)} bytes; working tree {len((REPO_ROOT / LANE_FILE).read_bytes())} bytes)")
    location = imported_from(overlay_a)
    checks.expect(location.startswith(str(overlay_a)), f"overlay A imports its own (committed) shared_snn.py: {checks.clean(location)}")
    for script, label in DECISION_CHECKS.items():
        result = subprocess.run([sys.executable, str(proof_a / script)], cwd=overlay_a, capture_output=True, text=True, timeout=1800)
        reason = checks.clean(first_failure(result.stdout + result.stderr))
        checks.expect(result.returncode != 0, f"{script} {label} fails on the committed lane file (exit {result.returncode}): {reason[:220]}")

    print("\n== overlay B: the student package of the commit that last changed the committed shared_snn.py ==")
    commit = git("log", "-1", "--format=%H", OLD_COMMIT, "--", LANE_FILE).decode().strip()
    print(f"  commit {commit[:12]} ({git('log', '-1', '--format=%s', commit).decode().strip()})")
    checks.expect(
        git("show", f"{commit}:{LANE_FILE}") == committed,
        "the committed shared_snn.py is the one of that commit (so overlay B runs the committed lane code)",
    )
    overlay_b, proof_b = make_overlay("b_snn_shared_c7b_")
    overlays.append(overlay_b)
    for path in OLD_PACKAGE:
        old = git("show", f"{commit}:{path}")
        (overlay_b / path).write_bytes(old)
        print(f"  overlay {path}: version of {commit[:12]} ({len(old)} bytes)")
    location = imported_from(overlay_b)
    checks.expect(location.startswith(str(overlay_b)), f"overlay B imports its own (old) shared_snn.py: {checks.clean(location)}")
    result = subprocess.run([sys.executable, str(proof_b / "c7_old_probe.py")], cwd=overlay_b, capture_output=True, text=True, timeout=1800)
    lines = {line.split(":", 1)[0][4:]: line for line in result.stdout.splitlines() if line.startswith("OLD ")}
    checks.expect(
        result.returncode == 0,
        f"the old-API probe ran (exit {result.returncode}){'' if result.returncode == 0 else ': ' + checks.clean(result.stderr.strip()[-400:])}",
    )
    for item in PROBE_ITEMS:
        line = lines.get(item, f"OLD {item}: (no line printed)")
        checks.expect(line.startswith(f"OLD {item}: DEFECT"), checks.clean(line)[:300])
finally:
    for overlay in overlays:
        shutil.rmtree(overlay, ignore_errors=True)
checks.expect(not any(overlay.exists() for overlay in overlays), "temporary directories removed")
checks.finish()
