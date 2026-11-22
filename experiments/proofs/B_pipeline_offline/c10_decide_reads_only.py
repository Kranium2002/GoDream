"""c10 PLUMBING CHECK: decide_with and the cost previews never write the run manifest.

Review finding (integrator, 2026-11-21). SkillRun.decide_with is documented as "Nothing is written
to the manifest", and the interface uses it for Play while another process may drive the same run.
Before the fix, decide_with with a residual brain fitted the model through residual_model(), which
records a manifest step "teacher" and rewrites the whole manifest from the calling object's
in-memory copy. A Play object opened before the driver's last step therefore overwrote the
driver's newer records (a lost update): here the driver's evaluation disappears from the manifest.
cost_preview and dream_cost_preview of a billed method fitted the same way.

Reproduction: object "driver" collects 6 attempts; object "play" opens the same run; the driver
evaluates (manifest lists eval001-dev); play.decide_with("gradient_boosting", ...).
Expected after the fix: the manifest bytes are unchanged by decide_with, the driver's evaluation
stays listed, and a later recorded step of the same object (dream) still records its fit as step
"teacher".

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only; nothing here is evidence
about the robot, the simulator, TabPFN or any model. Every file goes to a temporary directory.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c10_decide_reads_only.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, Checks, make_config, open_fixture_run  # noqa: E402

print(BANNER)

from learned_judgment import pipeline as P  # noqa: E402

checks = Checks()
expect = checks.expect
SITUATION = {"fx_lateral": 0.25, "fx_speed": 4.0, "fx_surface": "slip"}
BRAINS = ("constant_bias", "gaussian_process", "gradient_boosting", "mlp_on_attempts")

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp, max_steps=100)

    print("\n== a driver and a Play object on the same run ==")
    driver = open_fixture_run(config, run_id="shared")
    driver.collect_attempts(6)
    play = open_fixture_run(config, run_id="shared")
    driver.evaluate(["physics_only"], split="dev", n_situations=3, n_repeats=1)
    expect([e["evaluation_id"] for e in P.read_run_manifest(driver.paths.manifest)["evaluations"]] == ["eval001-dev"],
           "the driver's evaluation is in the manifest")
    before = driver.paths.manifest.read_bytes()

    for brain in BRAINS:
        answer = play.decide_with(brain, SITUATION)
        expect(answer["brain"] == brain and answer["n_context"] == 6, f"play.decide_with({brain!r}) answered (chosen {answer['chosen_index']})")
    after = driver.paths.manifest.read_bytes()
    expect(after == before, "the manifest bytes are unchanged by decide_with")
    expect([e["evaluation_id"] for e in P.read_run_manifest(driver.paths.manifest)["evaluations"]] == ["eval001-dev"],
           "the driver's evaluation is still listed (no lost update)")
    expect(play.dream_cost_preview("gradient_boosting", 5)["billed"] is False
           and driver.paths.manifest.read_bytes() == before, "a local dream cost preview writes nothing")

    print("\n== the same object: an unrecorded fit never hides a recorded step ==")
    answer = driver.decide_with("gradient_boosting", SITUATION)
    expect(driver.paths.manifest.read_bytes() == before, "driver.decide_with wrote nothing either")
    history_before = len(driver.manifest["history"])
    driver.dream("gradient_boosting", 30)
    steps = [entry["step"] for entry in driver.manifest["history"][history_before:]]
    teacher = [entry for entry in driver.manifest["history"][history_before:] if entry["step"] == "teacher"]
    expect(steps == ["teacher", "dream"] and teacher and teacher[0]["settings"]["method"] == "gradient_boosting"
           and teacher[0]["state"] == "done", f"dream recorded its fit as step 'teacher' first (history {steps})")
    again = driver.decide_with("gradient_boosting", SITUATION)
    expect(again == answer, "the recorded fit and the unrecorded fit decide identically (same seed, same attempts)")

checks.finish()
