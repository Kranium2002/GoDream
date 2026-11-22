"""c11 PLUMBING CHECK: the sealed test split is read only by evaluate(..., split="test", final_evaluation=True).

The pipeline builder listed the final-evaluation path as written but NOT RUN. This check runs it on
the FIXTURE skill and world with the test-access log redirected into a temporary directory, so no
project test split and no project log is touched:
  * every spelling of the test split ("test", " Test", "TEST\\n", "tEsT") without
    final_evaluation=True is refused with SealedSplitError, before any episode and without an
    access-log line; final_evaluation must be exactly True or False; True with another split is
    refused; a non-string split is refused;
  * dream situations, context situations and dev situations share no row with the test situations;
  * evaluate(split="test", final_evaluation=True) appends exactly one access-log line, runs the
    test situations and seeds that eval.splits.test_situations returns, and its run results say
    split "test".

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only; nothing here is evidence
about the robot, the simulator, TabPFN or any model. Every file goes to a temporary directory.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c11_sealed_split.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, Checks, fixed_clock, make_config, open_fixture_run  # noqa: E402

print(BANNER)

import numpy as np  # noqa: E402

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.eval.harness import read_manifest  # noqa: E402
from learned_judgment.eval.report import read_run_result  # noqa: E402
from learned_judgment.eval.splits import SealedSplitError, check_disjoint, split_situations, test_situations  # noqa: E402

checks = Checks()
expect = checks.expect
N = 6

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp, max_steps=100)
    log = Path(config["evaluation"]["test_access_log"])
    run = open_fixture_run(config, run_id="sealed")
    run.collect_attempts(5)

    print("\n== refused without final_evaluation=True ==")
    for spelling in ("test", " Test", "TEST\n", "tEsT"):
        checks.raises(f"evaluate(split={spelling!r})", SealedSplitError,
                      lambda s=spelling: run.evaluate(["physics_only"], split=s, n_situations=N, n_repeats=1), "sealed")
    checks.raises("final_evaluation=1 (not exactly True)", ValueError,
                  lambda: run.evaluate(["physics_only"], split="test", n_situations=N, n_repeats=1, final_evaluation=1), "True or False")
    checks.raises("final_evaluation=True with split 'dev'", ValueError,
                  lambda: run.evaluate(["physics_only"], split="dev", n_situations=N, n_repeats=1, final_evaluation=True), "test")
    checks.raises("a non-string split", ValueError, lambda: run.evaluate(["physics_only"], split=None, n_situations=N, n_repeats=1), "split")
    expect(not log.exists(), "no access-log line was written by any refused call")
    expect(not run.paths.evaluation_rows_root.exists(), "no evaluation directory was created by any refused call")

    print("\n== the test split is disjoint from every split the run reads ==")
    sealed = test_situations(run.spec, config, base_seed=run.base_seed, n=N, n_repeats=1, world_preset=run.world_preset,
                             final_evaluation=True, clock=fixed_clock)
    log.unlink()  # this read was the check's own; the evaluation below must write exactly one line
    others = {
        "context": run._plan_context(5)[0],  # noqa: SLF001
        "dev": split_situations(run.spec, base_seed=run.base_seed, split="dev", n=N, world_preset=run.world_preset),
        "dreams": run._dream_situations(50),  # noqa: SLF001
        "test": sealed.situations,
    }
    try:
        check_disjoint(others)
        expect(True, "context, dev, dream and test situations share no row")
    except ValueError as error:
        expect(False, f"splits share rows: {error}")

    print("\n== final evaluation on the test split ==")
    result = run.evaluate(["physics_only", "constant_bias"], split="test", n_situations=N, n_repeats=1, final_evaluation=True)
    lines = log.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0]) if lines else {}
    expect(len(lines) == 1 and entry.get("event") == "test_split_access" and entry.get("skill") == run.spec.name
           and entry.get("n") == N, f"exactly one access-log line: {lines}")
    expect(result["evaluation_id"] == "eval001-test" and result["final_evaluation"] is True, f"evaluation id {result['evaluation_id']}")
    for method, item in result["methods"].items():
        harness = read_manifest(item["harness_manifest"])
        seeds = [episode[2] for episode in harness["episodes"]]
        expect(seeds == [int(seed) for seed in sealed.noise_seeds[:, 0]] and harness["split"] == "test",
               f"{method}: the harness ran the sealed test seeds")
        expect(read_run_result(item["run_result"])["split"] == "test", f"{method}: run result says split 'test'")
    rows = [json.loads(line) for line in Path(result["methods"]["physics_only"]["attempts"]).read_text(encoding="utf-8").splitlines()]
    ran = np.array([run.spec.encode_situation(row["situation"]) for row in rows])
    expect(np.array_equal(ran, sealed.situations), "the episodes ran exactly the sealed test situations")

checks.finish()
