"""c9 PLUMBING CHECK: a run never adopts files it did not write, and collect always checks its plan.

Review finding (integrator, 2026-11-21). The run identity (skill, preset, base seed, world, config
digest) is enforced only through the run manifest. Before the fix, when the manifest of a run_id
was missing while its attempt, dream or student files existed (manifest deleted, manifests_dir
changed, or another tool wrote a file of that name), SkillRun created a fresh manifest under a NEW
identity and then:
  * collect_attempts(n) returned the old rows without comparing them with the new plan whenever
    n <= rows already stored, so the manifest recorded base seed B while every row came from
    base seed A (other situations, actions and attempt seeds);
  * train_student() trained on the old identity's dreams, and student() / evaluate(snn_per_skill)
    used the old identity's student.

Reproduction: run "orphan" with base seed A collects 5 attempts, dreams (physics_only) and trains
a student; its manifest is deleted; the same run_id is opened with base seed B.
Expected after the fix: opening refuses (RunStateError naming the files) and writes nothing, and
collect_attempts compares the stored rows with the plan even when no new attempt is needed.

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only; nothing here is evidence
about the robot, the simulator, TabPFN or any model. Every file goes to a temporary directory.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c9_orphan_files.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, BASE_SEED, Checks, make_config, open_fixture_run  # noqa: E402

print(BANNER)

import numpy as np  # noqa: E402

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.engine.rows import append_rows, read_rows  # noqa: E402

checks = Checks()
expect = checks.expect
MAX_STEPS = 100
OTHER_SEED = BASE_SEED + 1


def digest_tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp, max_steps=MAX_STEPS)
    print(f"settings: config.yaml copy with outputs under <tmp>; students.snn.max_steps={MAX_STEPS}")

    print("\n== run 'orphan' with base seed A: collect 5, dream, train ==")
    first = open_fixture_run(config, run_id="orphan", base_seed=BASE_SEED)
    rows_a = first.collect_attempts(5)
    first.dream("physics_only", 40)
    student_a = first.train_student()
    expect(len(rows_a) == 5 and first.paths.student_npz.exists(), "run A: 5 attempts, a dream set and a student exist")

    print("\n== manifest deleted; the same run_id opened with base seed B ==")
    first.paths.manifest.unlink()
    before = digest_tree(tmp)
    try:
        second = open_fixture_run(config, run_id="orphan", base_seed=OTHER_SEED)
    except P.RunStateError as error:
        second = None
        message = str(error)
        expect("without a manifest" in message and "attempts" in message, f"opening refused: RunStateError: {checks.redact(message)[:200]}")
    else:
        expect(False, "opening refused (it was not: a fresh manifest was created over run A's files)")
        rows_b = second.collect_attempts(5)
        plan_seeds = [int(seed) for seed in second._plan_context(5)[2]]  # noqa: SLF001
        expect([row.seed for row in rows_b] == plan_seeds,
               f"rows returned under base seed B carry base seed B's attempt seeds (got {[row.seed for row in rows_b][:2]}..., "
               f"plan {plan_seeds[:2]}...)")
        expect(second.student().digest != student_a.digest, "the run under base seed B has no student of base seed A")
    if second is None:
        expect(not first.paths.manifest.exists(), "no manifest was created for the refused identity")
    expect(digest_tree(tmp) == before, "nothing was written or changed by the refused open")

    print("\n== collect compares stored rows with the plan even when no attempt is missing ==")
    third = open_fixture_run(config, run_id="planned", base_seed=BASE_SEED)
    third.collect_attempts(4)
    rows = read_rows(third.paths.attempts, spec=third.spec)
    # Rewrite the attempt file with row 2's attempt seed changed (a seed is not part of the physics
    # guess, so the row stays a valid row): a file that no longer matches the plan.
    third.paths.attempts.unlink()
    other_seed = max(row.seed for row in rows) + 1
    append_rows(third.paths.attempts, rows[:2] + [dataclasses.replace(rows[2], seed=other_seed)] + rows[3:], spec=third.spec)
    checks.raises("collect_attempts(4) with 4 stored rows of which one differs from the plan", P.RunStateError,
                  lambda: third.collect_attempts(4), "differ from the planned attempts")
    checks.raises("collect_attempts(3) (fewer than stored) with the same file", P.RunStateError,
                  lambda: third.collect_attempts(3), "differ from the planned attempts")
    expect(third.step_state("collect") == "failed", "the step ends 'failed', not 'done'")
    situations_ok = np.array_equal(third.spec.encode_situation(rows[0].situation), third._plan_context(1)[0][0])  # noqa: SLF001
    expect(situations_ok, "an unchanged row still equals the plan (the comparison is exact, not loose)")

checks.finish()
