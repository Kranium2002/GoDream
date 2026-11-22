"""The teacher path of the pipeline on real Go2 attempts, refused before any prediction request.

DEV RUN on the real Go2 simulator for the attempts (Person A's Go2World through
pipeline.go2_world_factory, preset default, no ball; the sidestep_reach dev skill of
experiments/proofs/B_sidestep_dev/). LIVE FREE CALLS ONLY for the provider: authentication
(prior_labs.connect), usage reads and cost quotes (estimate_cost). Refusal tripwire pattern of
experiments/proofs/B_teacher_offline/c5_live_free_gate.py:
  * every tabpfn_client entry point that can create a fit, prediction or upload job is replaced by
    a tripwire that raises before anything is connected;
  * the config copy sets tabpfn.project_token_budget to null, so the teacher's own gate refuses every
    uncached request; the script aborts before connecting if the settings carry a budget;
  * the teacher cache and call log are redirected into the out directory (empty), so the project's
    cache and call log are not touched.

Steps: collect pipeline.default_attempts real attempts; fit the teacher (free: nothing is sent at
fit); read account usage (free); dream_cost_preview("teacher", engine.dreams_per_skill_initial);
SkillRun.dream("teacher", same) must end "refused" with the usage and quote written into the
manifest first and no dream file; decide_with("teacher") and evaluate([physics_only, teacher]) must
be refused with no episode; read account usage again (free).

Checks (exit 1 if any fails): the states and files above; the preview's first request key and
shape equal the request the teacher built and refused; the call log holds refusals only and no
dispatch; nothing was cached; the provider's prediction counter is 0; no tripwire was hit; the
account's token counters are the same before and after. If another process of the project
dispatched a billed request in between, the counters move without this script: the script then
lists the dispatch lines of results/tabpfn_calls.jsonl (read only) in that window.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_real\\teacher_refused_real.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SIDESTEP_DIR = ROOT / "experiments" / "proofs" / "B_sidestep_dev"
for entry in (ROOT, SIDESTEP_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

print("DEV RUN on the real Go2 simulator (attempts only); the teacher path is checked under the refusal tripwire; "
      "LIVE FREE CALLS ONLY: authentication, usage and estimate_cost; every billable entry point is tripwired", flush=True)

import tabpfn_client  # noqa: E402
import yaml  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402
from tabpfn_client.service_wrapper import InferenceClient  # noqa: E402

TRIPWIRE_HITS: list[str] = []


def install_tripwire(owner, name: str) -> None:  # noqa: ANN001
    label = f"{owner.__name__}.{name}"

    def tripwire(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        TRIPWIRE_HITS.append(label)
        raise AssertionError(f"billable entry point {label} was called")

    setattr(owner, name, tripwire)


installed = 0
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
            installed += 1
print(f"tripwires installed on {installed} billable tabpfn_client entry points", flush=True)

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.eval.splits import split_situations  # noqa: E402
from learned_judgment.teacher import TeacherSettings  # noqa: E402
from sidestep_reach_spec import ROOT as SPEC_ROOT  # noqa: E402
from sidestep_reach_spec import for_world  # noqa: E402

FAILURES: list[str] = []
PASSED = [0]
NUMERIC_USAGE = ("current_usage", "usage_limit", "daily_tokens_used", "daily_token_limit", "monthly_tokens_used", "monthly_token_limit")


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED[0] += 1
    else:
        FAILURES.append(label)
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""), flush=True)


def numbers(status: dict) -> dict:
    return {name: (status.get("usage") or {}).get(name) for name in NUMERIC_USAGE}


def refused(label: str, function) -> P.StepRefused | None:  # noqa: ANN001
    try:
        function()
    except P.StepRefused as error:
        check(f"{label} -> StepRefused", "project_token_budget is not set" in error.reason, error.reason[:200])
        return error
    except Exception as error:  # noqa: BLE001
        check(f"{label} -> StepRefused", False, f"raised {type(error).__name__}: {error}")
        return None
    check(f"{label} -> StepRefused", False, "did not raise")
    return None


stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = HERE / "out" / f"{stamp}_teacher_refused"
out.mkdir(parents=True, exist_ok=False)
base = load_config()
config = copy.deepcopy(base)
for dotted, value in {
    "engine.attempts_dir": (out / "attempts").as_posix(),
    "engine.dreams_dir": (out / "dreams").as_posix(),
    "pipeline.manifests_dir": (out / "manifests").as_posix(),
    "pipeline.students_dir": (out / "students").as_posix(),
    "pipeline.evaluation_rows_dir": (out / "evaluation_rows").as_posix(),
    "pipeline.run_results_dir": (out / "run_results").as_posix(),
    "tabpfn.cache_dir": (out / "tabpfn_cache").as_posix(),
    "tabpfn.call_log": (out / "tabpfn_calls.jsonl").as_posix(),
    "evaluation.test_access_log": (out / "test_access_log.jsonl").as_posix(),
    "tabpfn.project_token_budget": None,
    "world.save_traces": require(base, f"{SPEC_ROOT}.save_traces"),
}.items():
    section, key = dotted.split(".")
    config[section][key] = value
config["skills"]["sidestep_reach"] = {"candidate_points_per_dim": list(require(base, f"{SPEC_ROOT}.candidate_points_per_dim"))}
(out / "config_copy.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
settings = TeacherSettings.from_config(config)
print(f"output directory {out.relative_to(ROOT).as_posix()}; settings the pipeline uses: project_token_budget "
      f"{settings.project_token_budget} (forced null), cache and call log in the output directory, mode "
      f"{settings.multi_outcome_mode}, max rows/request {settings.max_query_rows_per_request}")
if settings.project_token_budget is not None:
    print("ABORT: the budget is not null; nothing was connected")
    raise SystemExit(1)

base_seed = int(require(config, "runtime.seed"))
preset = str(require(config, f"{SPEC_ROOT}.world_preset"))
n_attempts = int(require(config, "pipeline.default_attempts"))
n_dreams = int(require(config, "engine.dreams_per_skill_initial"))
factory = P.go2_world_factory(config)
spec = for_world(config, factory(P.world_seed_for(base_seed, preset), preset))
run = P.SkillRun(config, spec, world_factory=factory, run_id="sidestep-teacher-refused", world_preset=preset,
                 base_seed=base_seed, clock=P.utc_now)

print(f"\n== collect {n_attempts} real attempts ==", flush=True)
began = time.perf_counter()
run.collect_attempts(n_attempts)
print(f"  {n_attempts} episodes in {time.perf_counter() - began:.1f} s")

print("\n== teacher fit (free: nothing is sent at fit) ==", flush=True)
teacher = run.residual_model("teacher")
check("teacher fitted, step 'teacher' done", type(teacher).__name__ == "TabPFNTeacher" and run.step_state("teacher") == "done")

print("\n== usage before (free) ==", flush=True)
window_start = P.utc_now()
before = P.tabpfn_status(config)
check("tabpfn_status reachable", before.get("reachable") is True, before.get("error", ""))
print(f"  usage before: {numbers(before)}")

print(f"\n== dream_cost_preview('teacher', {n_dreams}) ==", flush=True)
preview = run.dream_cost_preview("teacher", n_dreams)
print(f"  {preview['n_requests']} request(s), {preview['n_cached']} cached, {preview['n_query_rows']} query rows, active outcomes "
      f"{preview['active_outcomes']}, constant {preview['constant_outcomes']}")
for request in preview["requests"]:
    print(f"    chunk {request['chunk']}/{request['n_chunks']}: train {request['n_train']} x query {request['n_query']} x "
          f"{request['n_features']} columns, quote {request.get('quote_tokens')} tokens ({request.get('pricing_version')}), "
          f"key {request['key'][:16]}...")
print(f"  total quoted {preview.get('total_quoted_tokens')} tokens; gate replay: would_refuse "
      f"{preview['gate_preview']['would_refuse']} {preview['gate_preview']['reasons']}")
check("preview reachable, every request keyed and quoted", preview.get("reachable") is True and preview["n_requests"] >= 1
      and all(len(r["key"]) == 64 and isinstance(r.get("quote_tokens"), int) for r in preview["requests"]))

print(f"\n== dream('teacher', {n_dreams}) ==", flush=True)
refused(f"dream('teacher', {n_dreams})", lambda: run.dream("teacher", n_dreams))
record = run.manifest["steps"]["dream"]
written = record.get("cost_preview") or {}
check("manifest: dream 'refused' with the reason", record["state"] == "refused" and "project_token_budget is not set" in (record["reason"] or ""))
check("manifest: the usage and the quote were written before the refusal", written.get("reachable") is True
      and isinstance(written.get("usage"), dict) and isinstance(written.get("total_quoted_tokens"), int),
      f"usage current_usage {(written.get('usage') or {}).get('current_usage')}, quoted {written.get('total_quoted_tokens')}")
check("no dream file", not run.paths.dreams.exists() and not run.paths.dreams_meta.exists())

print("\n== decide_with and evaluate with the teacher ==", flush=True)
dev0 = spec.decode_situation(split_situations(spec, base_seed=base_seed, split="dev", n=1, world_preset=preset)[0])
manifest_bytes = run.paths.manifest.read_bytes()
refused("decide_with('teacher', dev situation 0)", lambda: run.decide_with("teacher", dev0))
check("decide_with wrote nothing to the manifest", run.paths.manifest.read_bytes() == manifest_bytes)
refused("evaluate(['physics_only', 'teacher'], dev, 3 situations)",
        lambda: run.evaluate(["physics_only", "teacher"], split="dev", n_situations=3, n_repeats=1))
check("manifest: evaluate 'refused'", run.step_state("evaluate") == "refused" and run.manifest["evaluations"][-1]["state"] == "refused")
rows_dir = run.paths.evaluation_rows("eval001-dev")
check("no evaluation episode was run (no attempt file)", not rows_dir.exists() or not list(rows_dir.glob("*.jsonl")))

print("\n== usage after (free) ==", flush=True)
after = P.tabpfn_status(config)
window_end = P.utc_now()
print(f"  usage after:  {numbers(after)}")

print("\n== nothing was sent ==", flush=True)
provider = run._provider  # noqa: SLF001
check("the provider's prediction counter stayed 0", provider is not None and provider.prediction_calls_started == 0)
check("no tripwire was hit", TRIPWIRE_HITS == [], str(TRIPWIRE_HITS))
records = run._call_log().read()  # noqa: SLF001
events = sorted({r.get("event") for r in records})
check("call log: refusals only, no dispatch", bool(records) and events == ["refused"] and not any(r.get("billable_dispatch") for r in records),
      f"{len(records)} record(s), events {events}")
first = records[0] if records else {}
planned = preview["requests"][0]
check("the preview's first request equals the request the teacher built and refused (key and shape)",
      planned["key"] == first.get("key") and (planned["n_train"], planned["n_query"], planned["n_features"])
      == (first.get("n_train"), first.get("n_query"), first.get("n_features")))
print(f"  the teacher's own gate logged before refusing: quote {first.get('quote_tokens')} tokens, usage_before current_usage "
      f"{(first.get('usage_before') or {}).get('current_usage')}, reasons {first.get('reasons')}")
check("zero tokens counted toward the project budget", run._call_log().tokens_quoted_for_billed_calls() == 0)  # noqa: SLF001
check("nothing was cached", not any(Path(config["tabpfn"]["cache_dir"]).glob("**/*.npz")))
unchanged = numbers(before) == numbers(after)
if not unchanged:
    project_log = ROOT / "results" / "tabpfn_calls.jsonl"
    others = []
    if project_log.exists():
        for line in project_log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line) if line.strip() else {}
            stamp_utc = entry.get("timestamp_utc") or entry.get("time_utc") or ""
            if entry.get("billable_dispatch") and window_start <= str(stamp_utc) <= window_end:
                others.append({"time": stamp_utc, "key": str(entry.get("key"))[:16], "quote_tokens": entry.get("quote_tokens")})
    print(f"  account counters moved; dispatches by other processes in results/tabpfn_calls.jsonl between {window_start} and "
          f"{window_end}: {others}")
check("account token counters unchanged before and after", unchanged,
      f"current_usage {numbers(before)['current_usage']} -> {numbers(after)['current_usage']}")

print(f"\n{PASSED[0]} passed, {len(FAILURES)} failed {FAILURES or ''}")
sys.exit(1 if FAILURES else 0)
