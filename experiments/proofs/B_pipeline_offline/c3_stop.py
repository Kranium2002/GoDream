"""c3 PLUMBING CHECK: a stop during collect or evaluate leaves a usable run.

1. should_stop() turns true after the 3rd finished episode of collect_attempts(8): the call
   raises Stopped, the manifest says collect "stopped" with the reason, 3 rows stay on disk. A new
   SkillRun on the same run_id then collects the remaining 5, and the file equals a one-shot
   collection of 8 (fixed clock).
2. should_stop() turns true after the 7th episode of an evaluation of two methods: the call
   raises Stopped, the manifest says evaluate "stopped", the evaluation is listed as stopped and no
   run-result file was written for it. The next evaluate gets a new evaluation id and completes.
3. should_stop() already true: the step records "stopped" before doing anything.

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only; outputs in a temporary
directory; nothing here is evidence about the robot, the simulator, TabPFN or any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c3_stop.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, Checks, fixed_clock, make_config, open_fixture_run  # noqa: E402

print(BANNER)

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.eval.report import read_run_result  # noqa: E402

checks = Checks()
expect = checks.expect


class StopAfter:
    """should_stop() / progress pair: stop once `limit` episodes of `step` have finished."""

    def __init__(self, step: str, limit: int) -> None:
        self.step, self.limit, self.armed, self.flag = step, limit, True, False

    def progress(self, event: dict) -> None:
        if self.armed and event["step"] == self.step and event["event"] == "progress" and event["done"] >= self.limit:
            self.flag = True

    def should_stop(self) -> bool:
        return self.flag


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp / "run")

    print("\n== 1. stop during collect ==")
    stopper = StopAfter("collect", 3)
    run = open_fixture_run(config, run_id="stopped", clock=fixed_clock, progress=stopper.progress, should_stop=stopper.should_stop)
    checks.raises("collect_attempts(8) stops", P.Stopped, lambda: run.collect_attempts(8), "should_stop")
    record = run.manifest["steps"]["collect"]
    expect(record["state"] == "stopped" and "should_stop" in (record["reason"] or ""), f"manifest: collect stopped ({record['reason']})")
    expect(record["counts"]["rows_after"] == 3 and len(run.attempts()) == 3, "3 measured rows kept on disk and counted")
    expect(P.read_run_manifest(run.paths.manifest)["steps"]["collect"]["state"] == "stopped", "the state on disk is 'stopped'")
    resumed = open_fixture_run(config, run_id="stopped", clock=fixed_clock)
    rows = resumed.collect_attempts(8)
    expect(len(rows) == 8 and resumed.step_state("collect") == "done", "a new SkillRun on the run collects the remaining 5")
    one_shot = open_fixture_run(make_config(tmp / "oneshot"), run_id="stopped", clock=fixed_clock)
    one_shot.collect_attempts(8)
    expect(resumed.paths.attempts.read_bytes() == one_shot.paths.attempts.read_bytes(),
           "stopped + resumed attempt file is byte-identical to one call with 8")

    print("\n== 2. stop during evaluate ==")
    stopper = StopAfter("evaluate", 7)
    run = open_fixture_run(config, run_id="stopped", clock=fixed_clock, progress=stopper.progress, should_stop=stopper.should_stop)
    checks.raises("evaluate(2 methods x 5 situations x 2 repeats) stops", P.Stopped,
                  lambda: run.evaluate(["physics_only", "constant_bias"], split="dev", n_situations=5, n_repeats=2), "should_stop")
    manifest = run.manifest
    expect(manifest["steps"]["evaluate"]["state"] == "stopped", "manifest: evaluate stopped")
    expect(manifest["evaluations"][-1]["evaluation_id"] == "eval001-dev" and manifest["evaluations"][-1]["state"] == "stopped",
           "the evaluation is listed with state stopped")
    expect(manifest["steps"]["evaluate"]["counts"]["done"] == 7, "7 episodes were finished before the stop")
    stopped_results = run.paths.run_results("eval001-dev")
    expect(not stopped_results.exists() or not any(stopped_results.iterdir()), "no run-result file was written for the stopped evaluation")
    stopper.armed, stopper.flag = False, False
    result = run.evaluate(["physics_only", "constant_bias"], split="dev", n_situations=5, n_repeats=2)
    expect(result["evaluation_id"] == "eval002-dev" and run.step_state("evaluate") == "done", "the next evaluate gets eval002-dev and completes")
    expect(all(read_run_result(item["run_result"])["metrics"]["n_episodes"] == 10 for item in result["methods"].values()),
           "its run results read back with 10 episodes each")

    print("\n== 3. stop before a step starts ==")
    run = open_fixture_run(config, run_id="stopped", clock=fixed_clock, should_stop=lambda: True)
    checks.raises("residual_model with should_stop() true", P.Stopped, lambda: run.residual_model("constant_bias"), "before the step started")
    expect(run.step_state("teacher") == "stopped" and run.manifest["history"][-1]["step"] == "teacher", "manifest: teacher stopped, in history")
    later = open_fixture_run(config, run_id="stopped", clock=fixed_clock)
    expect(later.residual_model("constant_bias") is not None and later.step_state("teacher") == "done", "the run stays usable afterwards")

checks.finish()
