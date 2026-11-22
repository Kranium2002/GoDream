"""c5 PLUMBING CHECK (LIVE, FREE): the pipeline's teacher path is refused before any prediction
request, and the step ends "refused".

Pattern of experiments/proofs/B_teacher_offline/c5_live_free_gate.py:
  * Every client entry point that can create a fit, prediction or upload job is replaced by a
    tripwire that raises before anything is connected. The tripwires produce no response; they
    make a billable call impossible in this process.
  * The config copy sets tabpfn.project_token_budget to null, so the teacher's gate refuses every
    uncached request whatever config.yaml says; the script aborts before connecting if the
    settings the pipeline will use carry a budget. The cache and the call log are redirected into
    a temporary directory.
  * Only free provider calls are made: authentication (prior_labs.connect), usage reads and cost
    quotes (estimate_cost).

Checked: the teacher fit sends nothing; cost_preview and dream_cost_preview read usage and quotes
and replay the gate; dream("teacher") ends "refused" with the usage and quote written into the
manifest and no dream file; a second SkillRun object on the same run (a reader, as the interface's
Play would be) previews and is refused without writing the manifest (integrator review fix);
decide_with("teacher") and evaluate([... "teacher"]) are refused with no episode run; the provider's prediction counter stays 0, no tripwire is hit, the call log holds
refusals only, nothing is cached and the account's token counters are unchanged.

The attempts come from the fixture world (fixture_pipeline.py). PLUMBING CHECK: mechanics only;
nothing here is evidence about the robot, the simulator, TabPFN or any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c5_teacher_refused.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, Checks, fixed_clock, make_config, open_fixture_run  # noqa: E402

print(BANNER)
print("LIVE FREE CALLS ONLY: authentication, usage and estimate_cost; every billable entry point is tripwired")

import tabpfn_client  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402
from tabpfn_client.service_wrapper import InferenceClient  # noqa: E402

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.teacher import TeacherSettings  # noqa: E402

checks = Checks()
expect = checks.expect
TRIPWIRE_HITS: list[str] = []
SITUATION = {"fx_lateral": 0.25, "fx_speed": 4.0, "fx_surface": "slip"}
NUMERIC_USAGE = ("current_usage", "usage_limit", "daily_tokens_used", "daily_token_limit", "monthly_tokens_used", "monthly_token_limit")


def install_tripwire(owner, name: str) -> None:  # noqa: ANN001
    label = f"{owner.__name__}.{name}"

    def tripwire(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        TRIPWIRE_HITS.append(label)
        raise AssertionError(f"billable entry point {label} was called")

    setattr(owner, name, tripwire)


installed = []
for owner, names in (
    (tabpfn_client.TabPFNRegressor, ("fit", "_fit", "predict", "_predict", "submit_fit", "submit_predict", "wait_for_predict",
                                     "cross_val_predict", "submit_cross_val_predict", "cross_val_predict_splits")),
    (InferenceClient, ("fit", "fit_with_result", "predict", "predict_chunks", "submit_predict")),
    (ServiceClient, ("fit", "fit_with_result", "_fit", "_submit_fit_job", "_upload_train_set", "_upload_fold_ids", "predict",
                     "predict_chunks", "submit_predict", "_predict", "_start_predict", "_sync_predict", "_async_predict",
                     "_submit_predict_job", "_upload_test_set", "_upload_to_gcs", "_upload_single_chunk",
                     "_submit_cross_validate_job")),
):
    for name in names:
        if hasattr(owner, name):
            install_tripwire(owner, name)
            installed.append(f"{owner.__name__}.{name}")
print(f"tripwires installed on {len(installed)} entry points")


def numbers(status: dict) -> dict:
    return {name: (status.get("usage") or {}).get(name) for name in NUMERIC_USAGE}


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp, budget_none=True)
    used = TeacherSettings.from_config(config)
    print(f"settings the pipeline uses: project_token_budget {used.project_token_budget} (forced null), "
          f"cache and call log under <tmp>, mode {used.multi_outcome_mode}, max rows/request {used.max_query_rows_per_request}")
    if used.project_token_budget is not None:
        print("ABORT: the budget is not null; nothing was connected")
        raise SystemExit(1)

    run = open_fixture_run(config, run_id="teacher_refused", clock=P.utc_now)
    run.collect_attempts(10)

    print("\n== teacher fit (free) ==")
    teacher = run.residual_model("teacher")
    expect(type(teacher).__name__ == "TabPFNTeacher" and run.step_state("teacher") == "done", "teacher fitted; step 'teacher' done")
    expect(run.manifest["steps"]["teacher"]["settings"]["method"] == "teacher", "manifest step 'teacher' records the method")

    print("\n== free status and previews ==")
    before = P.tabpfn_status(config)
    expect(before.get("reachable") is True, f"tabpfn_status reachable: {before.get('reachable')} {before.get('error', '')}")
    expect(before.get("project_token_budget") is None and before.get("tokens_quoted_for_billed_calls") == 0,
           "status: budget null, nothing quoted in the (temporary) call log")
    print("  usage before:", numbers(before))
    preview = run.cost_preview("teacher", 700)
    expect(preview.get("reachable") is True and isinstance(preview.get("total_quoted_tokens"), int),
           f"cost_preview: reachable, total quoted tokens {preview.get('total_quoted_tokens')}")
    expect(preview["n_requests"] >= 1 and all(isinstance(r.get("quote_tokens"), int) for r in preview["requests"]),
           f"cost_preview: {preview['n_requests']} request(s), each quoted ({[(r['n_train'], r['n_query'], r['n_features']) for r in preview['requests']]})")
    expect(preview["gate_preview"]["would_refuse"] is True
           and any("project_token_budget is not set" in reason for reason in preview["gate_preview"]["reasons"]),
           "gate replay: would refuse (budget not set)")
    dream_preview = run.dream_cost_preview("teacher", 20)
    expect(dream_preview["cache_checked"] is True and dream_preview["n_to_send"] == dream_preview["n_requests"] >= 1
           and all(len(r["key"]) == 64 for r in dream_preview["requests"]),
           f"dream_cost_preview: {dream_preview['n_requests']} keyed request(s), none cached, {dream_preview['n_query_rows']} query rows")
    direct_preview = run.dream_cost_preview("direct_tabpfn", 5)
    expect(direct_preview.get("reachable") is True and direct_preview["n_requests"] >= 1,
           f"direct_tabpfn preview reachable, {direct_preview['n_requests']} request(s)")

    print("\n== dream with the teacher ==")
    checks.raises("dream('teacher', 20)", P.StepRefused, lambda: run.dream("teacher", 20), "project_token_budget is not set")
    record = run.manifest["steps"]["dream"]
    expect(record["state"] == "refused" and "project_token_budget is not set" in (record["reason"] or ""), "manifest: dream refused with the reason")
    written = record.get("cost_preview") or {}
    expect(written.get("reachable") is True and isinstance(written.get("usage"), dict)
           and isinstance(written.get("total_quoted_tokens"), int), "the usage and the quote were written into the manifest first")
    expect(not run.paths.dreams.exists() and not run.paths.dreams_meta.exists(), "no dream file")

    print("\n== a second object (a reader) on the same run: billed previews and decisions write nothing ==")
    viewer = open_fixture_run(config, run_id="teacher_refused", clock=P.utc_now)
    manifest_bytes = run.paths.manifest.read_bytes()
    viewer_preview = viewer.dream_cost_preview("teacher", 20)
    expect(viewer_preview.get("reachable") is True and [r["key"] for r in viewer_preview["requests"]]
           == [r["key"] for r in dream_preview["requests"]], "the reader's dream preview has the same request keys")
    checks.raises("reader decide_with('teacher')", P.StepRefused, lambda: viewer.decide_with("teacher", SITUATION),
                  "project_token_budget is not set")
    expect(run.paths.manifest.read_bytes() == manifest_bytes, "the reader's teacher fit, preview and refused decision wrote nothing to the manifest")
    expect(viewer._provider is not None and viewer._provider.prediction_calls_started == 0,  # noqa: SLF001
           "the reader's provider prediction counter stayed 0")

    print("\n== decide_with and evaluate with the teacher ==")
    checks.raises("decide_with('teacher')", P.StepRefused, lambda: run.decide_with("teacher", SITUATION), "project_token_budget is not set")
    checks.raises("evaluate(['physics_only', 'teacher'])", P.StepRefused,
                  lambda: run.evaluate(["physics_only", "teacher"], split="dev", n_situations=3, n_repeats=1), "project_token_budget is not set")
    expect(run.step_state("evaluate") == "refused" and run.manifest["evaluations"][-1]["state"] == "refused", "manifest: evaluate refused")
    rows_dir = run.paths.evaluation_rows("eval001-dev")
    expect(not rows_dir.exists() or not list(rows_dir.glob("*.jsonl")), "no episode was run (no attempt file)")

    print("\n== nothing was sent ==")
    provider = run._provider  # the one provider object the run used  # noqa: SLF001
    expect(provider is not None and provider.prediction_calls_started == 0, "the provider's prediction counter stayed 0")
    expect(TRIPWIRE_HITS == [], f"no tripwire was hit {TRIPWIRE_HITS}")
    records = run._call_log().read()  # noqa: SLF001
    events = sorted({record.get("event") for record in records})
    expect(records and events == ["refused"] and not any(record.get("billable_dispatch") for record in records),
           f"call log: {len(records)} record(s), events {events}, no dispatch")
    dream_record = records[0] if records else {}
    planned = dream_preview["requests"][0]
    expect([r["key"] for r in dream_preview["requests"]] == [dream_record.get("key")]
           and (planned["n_train"], planned["n_query"], planned["n_features"])
           == (dream_record.get("n_train"), dream_record.get("n_query"), dream_record.get("n_features")),
           "the preview's request key and shape equal the request the teacher built and refused for the dream")
    expect(run._call_log().tokens_quoted_for_billed_calls() == 0, "zero tokens counted toward the project budget")  # noqa: SLF001
    cache_dir = Path(config["tabpfn"]["cache_dir"])
    expect(not any(cache_dir.glob("*.npz")), "nothing was cached")
    after = P.tabpfn_status(config)
    print("  usage after: ", numbers(after))
    expect(numbers(before) == numbers(after), "account token counters unchanged")

checks.finish()
