"""Check 5 (LIVE, FREE): the real provider gate refuses an uncached request before any billable call.

Uses the real TabPFNProvider and the free endpoints only: authentication (prior_labs.connect), usage
read and cost estimate. The teacher's context is a small FIXTURE (layout and arrays below), so
nothing here is evidence about the robot, the simulator or the model. The cache directory and call
log are redirected to a temporary directory; nothing is written to results/ or .cache/.

Safety (this script never relies on config.yaml to stop a paid request):
  * The settings are built with dataclasses.replace(settings, project_token_budget=None), so the
    gate must refuse whatever config.yaml says; the script exits before connecting if the budget
    used is not None.
  * Before connecting, every client entry point that can create a fit, prediction or upload job is
    replaced by a tripwire that raises. The tripwires produce no response; they only make a
    billable call impossible in this script.
  * After the refusal, the provider's prediction counter must be 0, no tripwire may have been hit
    and the account's token counters (read before and after) must be unchanged.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c5_live_free_gate.py
"""

import dataclasses
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tabpfn_client  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402
from tabpfn_client.service_wrapper import InferenceClient  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.features import FeatureLayout  # noqa: E402
from learned_judgment.teacher import BudgetError, TabPFNProvider, TabPFNTeacher, TeacherSettings  # noqa: E402

FAILURES: list[str] = []
TRIPWIRE_HITS: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


def install_tripwire(owner, name: str) -> None:
    label = f"{owner.__name__}.{name}"

    def tripwire(*args, **kwargs):
        TRIPWIRE_HITS.append(label)
        raise AssertionError(f"billable entry point {label} was called")

    setattr(owner, name, tripwire)
    print(f"  tripwire on {label}")


print("== tripwires on billable / job-creating / upload entry points ==")
for owner, names in (
    (
        tabpfn_client.TabPFNRegressor,
        ("fit", "_fit", "predict", "_predict", "submit_fit", "submit_predict", "wait_for_predict",
         "cross_val_predict", "submit_cross_val_predict", "cross_val_predict_splits"),
    ),
    (InferenceClient, ("fit", "fit_with_result", "predict", "predict_chunks", "submit_predict")),
    (
        ServiceClient,
        ("fit", "fit_with_result", "_fit", "_submit_fit_job", "_upload_train_set", "_upload_fold_ids",
         "predict", "predict_chunks", "submit_predict", "_predict", "_start_predict", "_sync_predict",
         "_async_predict", "_submit_predict_job", "_upload_test_set", "_upload_to_gcs", "_upload_single_chunk",
         "_submit_cross_validate_job"),
    ),
):
    for name in names:
        if hasattr(owner, name):
            install_tripwire(owner, name)

NUMERIC_USAGE = (
    "current_usage", "usage_limit", "daily_tokens_used", "daily_token_limit", "monthly_tokens_used", "monthly_token_limit",
)


def numbers(usage: dict) -> dict:
    return {name: usage.get(name) for name in NUMERIC_USAGE}


LAYOUT = FeatureLayout(
    skill="fixture_gate_check",
    names=("sit_x", "sit_side", "act_t", "physics_r0", "physics_r1"),
    categorical=((1, ("left", "right")),),
    n_situation=2,
    n_action=1,
    n_physics=2,
    n_extra=0,
    outcome_names=("r0", "r1"),
    outcome_kinds=("regression", "regression"),
)
rng = np.random.default_rng(5)
FEATURES = rng.normal(size=(6, 5))
FEATURES[:, 1] = [0, 1, 0, 1, 1, 0]
RESIDUALS = rng.normal(size=(6, 2))
QUERY = rng.normal(size=(4, 5))
QUERY[:, 1] = [1, 0, 1, 0]

real_settings = TeacherSettings.from_config(load_config())
print("\n== settings ==")
print("  from config.yaml: client_version", real_settings.client_version, "| model_path", real_settings.model_path,
      "| n_estimators", real_settings.n_estimators, "| mode", real_settings.multi_outcome_mode,
      "| max_tokens_per_request", real_settings.max_tokens_per_request,
      "| project_token_budget (config, NOT used here)", real_settings.project_token_budget)

with tempfile.TemporaryDirectory() as tmp:
    # The refusal never depends on config.yaml: the budget is forced to None here.
    settings = dataclasses.replace(
        real_settings, project_token_budget=None, cache_dir=Path(tmp) / "cache", call_log=Path(tmp) / "calls.jsonl"
    )
    print("  used by this check: project_token_budget", settings.project_token_budget, "(forced, refuses every uncached request)")
    if settings.project_token_budget is not None:
        print("ABORT: the forced budget is not None; nothing was connected")
        sys.exit(1)
    clock = lambda: datetime.now(timezone.utc)  # noqa: E731
    provider = TabPFNProvider(settings)
    provider.connect()
    print("\n== live free calls ==")
    print("  connected; installed tabpfn_client", tabpfn_client.__version__)
    usage_before = numbers(provider.usage())
    print("  usage before:", usage_before)

    teacher = TabPFNTeacher.from_settings(LAYOUT, settings, provider=provider, clock=clock)
    teacher.fit(FEATURES, RESIDUALS)
    refusal = None
    try:
        teacher.predict_residual_distribution(QUERY)
    except BudgetError as exc:
        refusal = exc
    print("  BudgetError:", refusal)
    usage_after = numbers(provider.usage())
    print("  usage after: ", usage_after)

    print("\n== refusal evidence ==")
    check("predict_residual_distribution raised BudgetError", refusal is not None)
    check("the provider's prediction counter stayed 0", provider.prediction_calls_started == 0)
    check("no tripwire was hit", TRIPWIRE_HITS == [])
    records = teacher.call_log.read()
    for record in records:
        shown = {k: v for k, v in record.items() if k not in ("usage_before", "timestamp_utc")}
        print("  call-log record:", shown)
    check("exactly one call-log record", len(records) == 1)
    record = records[0] if records else {}
    check("it is a refusal, not a dispatch", record.get("event") == "refused" and record.get("billable_dispatch") is False)
    check("the refusal carries the integer quote", isinstance(record.get("quote_tokens"), int))
    check("the quote resolved the pinned model and n_estimators",
          record.get("quote", {}).get("inputs", {}).get("model_version") == "v3.5"
          and record.get("quote", {}).get("inputs", {}).get("n_estimators") == settings.n_estimators)
    check("the stacked request was quoted as 12 train rows x 8 query rows x 6 columns",
          (record.get("n_train"), record.get("n_query"), record.get("n_features")) == (12, 8, 6))
    check("a reason names the unset project budget",
          any("project_token_budget is not set" in reason for reason in record.get("reasons", [])))
    check("the refused request counts zero tokens toward the project budget", teacher.call_log.tokens_quoted_for_billed_calls() == 0)
    check("nothing was cached (no entry, no unvalidated response)", not teacher.cache.has_entries())
    check("the cache lock was released", not teacher.cache.lock_path.exists())
    check("no request hash recorded on the teacher", teacher.last_request_hash is None and teacher.last_request_hashes == ())
    check("account token counters unchanged", usage_before == usage_after)

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
