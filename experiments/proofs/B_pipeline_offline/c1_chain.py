"""c1 PLUMBING CHECK: the whole pipeline chain on the fixture world.

collect -> residual model (local baseline constant_bias) -> dream -> train -> evaluate on dev ->
decide_with for every local brain and the student; a repeated evaluate overwrites nothing; the
manifest and every result reload from disk in a fresh process (c1_reload_worker.py).

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only. The spiking student is
trained with students.snn.max_steps shortened (printed below). Nothing here is evidence about the
robot, the simulator, TabPFN or any model; every file goes to a temporary directory.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c1_chain.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, ROOT, Checks, PRESET, make_config, open_fixture_run  # noqa: E402

print(BANNER)

import numpy as np  # noqa: E402

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.eval.harness import read_manifest  # noqa: E402
from learned_judgment.eval.report import read_run_result  # noqa: E402
from learned_judgment.eval.splits import SealedSplitError  # noqa: E402

checks = Checks()
expect = checks.expect
MAX_STEPS = 300
N_ATTEMPTS = 12
N_DREAMS = 800
LOCAL_BRAINS = ("physics_only", "constant_bias", "gaussian_process", "gradient_boosting", "mlp_on_attempts")
SITUATION = {"fx_lateral": 0.25, "fx_speed": 4.0, "fx_surface": "slip"}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp, max_steps=MAX_STEPS)
    print(f"settings: config.yaml copy with outputs under <tmp>; students.snn.max_steps={MAX_STEPS} "
          f"(config value {P.load_config()['students']['snn']['max_steps']})")
    events: list[dict] = []
    run = open_fixture_run(config, run_id="chain", clock=P.utc_now, progress=events.append)

    print("\n== manifest at creation ==")
    manifest = P.read_run_manifest(run.paths.manifest)
    expect(manifest["format"] == P.MANIFEST_FORMAT, "manifest format tag")
    expect(all(manifest["steps"][step]["state"] == "not_run" for step in P.STEPS), "every step not_run")
    expect(manifest["world_seed"] == P.world_seed_for(run.base_seed, PRESET), "world seed = stream_seed(base, 'pipeline/world/default')")
    expect(manifest["config_digest"] == P.config_digest(config), "config digest recorded")
    expect("PLUMBING FIXTURE" in manifest["world"], f"world description recorded: {manifest['world'][:60]}...")
    expect(str(run.paths.manifest).startswith(tmp_name), "manifest written under the temporary directory")

    print("\n== collect ==")
    rows = run.collect_attempts(N_ATTEMPTS)
    expect(len(rows) == N_ATTEMPTS, f"{N_ATTEMPTS} attempt rows")
    expect(run.step_state("collect") == "done", "collect done")
    record = run.manifest["steps"]["collect"]
    expect(record["counts"]["new_rows"] == N_ATTEMPTS and record["counts"]["rows_after"] == N_ATTEMPTS, f"counts {record['counts']}")
    expect(record["started_utc"] and record["finished_utc"] and record["wall_s"] >= 0.0, "times and wall seconds recorded")
    collect_events = [e for e in events if e["step"] == "collect"]
    expect([e["event"] for e in collect_events][0] == "start" and collect_events[-1]["event"] == "finish", "collect events start ... finish")
    expect(sum(e["event"] == "progress" for e in collect_events) == N_ATTEMPTS, "one progress event per episode")
    expect(all(set(e) == {"run_id", "step", "event", "done", "total", "message"} for e in events), "event keys")

    print("\n== residual model (constant_bias) ==")
    model = run.residual_model("constant_bias")
    expect(type(model).__name__ == "ConstantBiasResidual", "constant_bias fitted")
    teacher_record = run.manifest["steps"]["teacher"]
    expect(teacher_record["state"] == "done" and teacher_record["settings"]["method"] == "constant_bias"
           and teacher_record["settings"]["n_context"] == N_ATTEMPTS, "manifest step 'teacher' records method and n_context")
    expect(run.residual_model("constant_bias") is model, "same method and attempts: no refit")
    expect(run.residual_model("physics_only") is None, "physics_only: no model")
    preview = run.cost_preview("constant_bias", 100)
    expect(preview["billed"] is False and "nothing is billed" in preview["message"], "local cost preview: nothing billed")

    print("\n== dream ==")
    dreams = run.dream("constant_bias", N_DREAMS)
    dream_record = run.manifest["steps"]["dream"]
    expect(dreams.n_rows == N_DREAMS and dreams.n_candidates == 35, f"dream set {dreams.n_rows} x {dreams.n_candidates}")
    expect(dream_record["settings"]["confidence_threshold"] is None and "null" in dream_record["settings"]["confidence_threshold_reason"],
           "no threshold configured: null and the reason recorded")
    expect(run.paths.dreams.exists() and run.paths.dreams_meta.exists(), "dream files saved")
    checks.raises("second dream refused", P.RunStateError, lambda: run.dream("constant_bias", 10), "never overwritten")

    print("\n== train ==")
    student = run.train_student()
    train_record = run.manifest["steps"]["train"]
    expect(train_record["state"] == "done" and train_record["network_sha256"] == student.digest, "network digest recorded")
    expect(isinstance(train_record["training_report"], dict) and "steps_run" in train_record["training_report"], "training report recorded")
    expect([e["event"] for e in events if e["step"] == "train"] == ["start", "finish"], "train emits start and finish only")
    checks.raises("second train refused", P.RunStateError, lambda: run.train_student(), "never overwritten")

    print("\n== evaluate on dev ==")
    methods = ["physics_only", "constant_bias", "snn_per_skill"]
    first = run.evaluate(methods, split="dev", n_situations=8, n_repeats=2)
    expect(first["evaluation_id"] == "eval001-dev", f"evaluation id {first['evaluation_id']}")
    for method in methods:
        item = first["methods"][method]
        result = read_run_result(item["run_result"])
        harness = read_manifest(item["harness_manifest"])
        expect(result["method"] == method and result["split"] == "dev" and harness["completed"] is True,
               f"{method}: run result readable, harness manifest completed")
        expect(harness["world_seed"] == run.world_seed, f"{method}: harness used the run's world seed")
        expect(result["metrics"]["n_episodes"] == 16, f"{method}: 16 episodes")
        expect("PLUMBING FIXTURE" in result["provenance"]["world"], f"{method}: result names the fixture world")
    seeds = {m: [e[2] for e in read_manifest(first["methods"][m]["harness_manifest"])["episodes"]] for m in methods}
    expect(seeds["physics_only"] == seeds["constant_bias"] == seeds["snn_per_skill"], "every method met the same attempt seeds")
    print("  metrics (fixture, mechanics only):", json.dumps(first["methods"]["constant_bias"]["metrics"]))
    before = {m: sha(first["methods"][m]["run_result"]) for m in methods}
    second = run.evaluate(methods, split="dev", n_situations=8, n_repeats=2)
    expect(second["evaluation_id"] == "eval002-dev", f"repeat evaluate gets a new id {second['evaluation_id']}")
    expect(all(sha(first["methods"][m]["run_result"]) == before[m] for m in methods), "first evaluation's files unchanged")
    expect(all(read_run_result(second["methods"][m]["run_result"])["metrics"] == first["methods"][m]["metrics"] for m in methods),
           "same situations, seeds and world: identical metrics")
    expect([e["evaluation_id"] for e in run.manifest["evaluations"]] == ["eval001-dev", "eval002-dev"], "evaluations listed in the manifest")
    checks.raises("oracle is not built", P.MethodNotBuiltError, lambda: run.evaluate(["oracle"], split="dev", n_situations=2, n_repeats=1), "not built")
    checks.raises("snn_shared is not part of a single-skill run", P.MethodNotBuiltError,
                  lambda: run.evaluate(["snn_shared"], split="dev", n_situations=2, n_repeats=1), "snn_shared")
    checks.raises("test split without final_evaluation", SealedSplitError,
                  lambda: run.evaluate(["physics_only"], split="test", n_situations=2, n_repeats=1), "sealed")
    checks.raises("final_evaluation with another split", ValueError,
                  lambda: run.evaluate(["physics_only"], split="dev", n_situations=2, n_repeats=1, final_evaluation=True), "test")
    expect(not (tmp / "test_access_log.jsonl").exists(), "the sealed test split was never read (no access-log line)")

    print("\n== decide_with every brain ==")
    answers = {}
    for brain in (*LOCAL_BRAINS, "snn_per_skill"):
        answer = run.decide_with(brain, SITUATION)
        answers[brain] = answer
        text = json.dumps(answer)
        candidates = np.asarray(answer["candidates"])
        chosen = [answer["action"][name] for name in answer["action_names"]]
        ok = (
            candidates.shape == (35, 2)
            and len(answer["scores"]) == 35
            and np.array_equal(candidates[answer["chosen_index"]], chosen)
            and set(answer["physics_guess"]) == {"fx_separation", "fx_fall", "fx_effort"}
            and answer["request_hashes"] == []
        )
        expect(ok and len(text) > 0, f"{brain}: JSON, chosen index {answer['chosen_index']} -> {answer['action']}, {answer['score_kind']}")
        if brain == "snn_per_skill":
            expect(answer["predicted_outcome"] is None and answer["score_kind"] == "snn_readout"
                   and int(np.argmax(answer["scores"])) == answer["chosen_index"], "student: readout per candidate, no predicted outcome")
        else:
            expect(answer["score_kind"] == "objective" and int(np.argmax(answer["scores"])) == answer["chosen_index"]
                   and set(answer["predicted_outcome"]) == {"mean", "median", "spread"}, f"{brain}: objective scores and predicted outcome")
    encoded = run.spec.encode_situation(SITUATION)
    expect(run.decide_with("constant_bias", encoded) == answers["constant_bias"], "an encoded row gives the same answer as the mapping")
    checks.raises("oracle brain", P.MethodNotBuiltError, lambda: run.decide_with("oracle", SITUATION), "not built")

    print("\n== reload in a fresh process ==")
    paths = {
        "manifest": run.paths.manifest.as_posix(),
        "attempts": run.paths.attempts.as_posix(),
        "dreams": run.paths.dreams.as_posix(),
        "student": run.paths.student_stem.as_posix(),
        "results": [item["run_result"] for evaluation in (first, second) for item in evaluation["methods"].values()],
    }
    paths_file = tmp / "paths.json"
    paths_file.write_text(json.dumps(paths), encoding="utf-8")
    worker = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("c1_reload_worker.py")), str(paths_file)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    expect(worker.returncode == 0, f"worker (fresh process) exit code {worker.returncode} {worker.stderr[-400:]}")
    loaded = json.loads(worker.stdout.strip().splitlines()[-1]) if worker.returncode == 0 else {}
    manifest = run.manifest
    expect(loaded.get("steps") == {step: manifest["steps"][step]["state"] for step in P.STEPS}, f"step states {loaded.get('steps')}")
    expect(loaded.get("history_steps") == [entry["step"] for entry in manifest["history"]], "history reloads")
    expect(loaded.get("evaluations") == ["eval001-dev", "eval002-dev"], "evaluations reload")
    expect(loaded.get("n_attempts") == N_ATTEMPTS and loaded.get("attempt_seeds") == [row.seed for row in run.attempts()], "attempt rows reload")
    expect(loaded.get("n_dreams") == N_DREAMS, "dream set reloads")
    expect(loaded.get("student_digest") == student.digest, "student reloads with the same network digest")
    expected_metrics = [item["metrics"] for evaluation in (first, second) for item in evaluation["methods"].values()]
    expect(loaded.get("metrics") == json.loads(json.dumps(expected_metrics)), "every run result reloads (digests re-checked) with the same metrics")
    expect(loaded.get("student_decision") == answers["snn_per_skill"]["chosen_index"], "reloaded student makes the same decision")

checks.finish()
