"""Check 7: budget-gate rules, usage arithmetic and settings validation (pure functions, no network).

FIXTURE usage dictionaries only; they exercise remaining_tokens and budget_refusal_reasons of
learned_judgment.teacher, plus TeacherSettings.from_config on edited copies of config.yaml's dict.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c7_gate_rules.py
"""

import copy
import sys
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from learned_judgment.config import ConfigError, load_config  # noqa: E402
from learned_judgment.teacher import (  # noqa: E402
    ProviderResponseError,
    TeacherSettings,
    budget_refusal_reasons,
    remaining_tokens,
)

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


def raises(label: str, exc_type, call) -> None:
    try:
        call()
        check(f"{label} raises {exc_type.__name__}", False)
    except exc_type as exc:
        check(f"{label} raises {exc_type.__name__}: {exc}", True)


print("== remaining_tokens ==")
usage = {
    "current_usage": 40000, "usage_limit": 35000000,
    "daily_tokens_used": 40000, "daily_token_limit": 35000000,
    "monthly_tokens_used": 90000, "monthly_token_limit": 100000,
}
check("minimum over the three limit pairs", remaining_tokens(usage) == 10000)
check("unreported token pairs are skipped", remaining_tokens({"current_usage": 5, "usage_limit": 25}) == 20)
check("usage_limit == -1 is unlimited", remaining_tokens({"current_usage": 5, "usage_limit": -1}) is None)
check("unlimited usage_limit still bounded by a daily pair",
      remaining_tokens({"current_usage": 5, "usage_limit": -1, "daily_tokens_used": 7, "daily_token_limit": 9}) == 2)
raises("no limit reported", ProviderResponseError, lambda: remaining_tokens({"reset_time": "x"}))
raises("negative daily limit", ProviderResponseError, lambda: remaining_tokens({"daily_token_limit": -1, "daily_tokens_used": 0}))
raises("non-integer counter", ProviderResponseError, lambda: remaining_tokens({"current_usage": 1.5, "usage_limit": 10}))
raises("bool counter", ProviderResponseError, lambda: remaining_tokens({"current_usage": True, "usage_limit": 10}))

print("\n== budget_refusal_reasons (exact integer boundaries) ==")
ok_args = dict(quote_tokens=10000, max_tokens_per_request=10000, project_token_budget=50000,
               tokens_already_quoted=40000, remaining_account_tokens=10000)
check("all limits met exactly -> no reason", budget_refusal_reasons(**ok_args) == [])
for label, change in {
    "quote one above max_tokens_per_request": {"quote_tokens": 10001, "remaining_account_tokens": 20000, "project_token_budget": 60000},
    "project budget not set (None)": {"project_token_budget": None},
    "already quoted + quote one above the project budget": {"tokens_already_quoted": 40001},
    "account remaining one below the quote": {"remaining_account_tokens": 9999},
}.items():
    reasons = budget_refusal_reasons(**{**ok_args, **change})
    check(f"{label} -> refused: {reasons}", len(reasons) == 1)
check("unlimited account (None) never refuses on remaining tokens",
      budget_refusal_reasons(**{**ok_args, "remaining_account_tokens": None}) == [])
check("several violations are all reported",
      len(budget_refusal_reasons(quote_tokens=20000, max_tokens_per_request=10000, project_token_budget=None,
                                 tokens_already_quoted=0, remaining_account_tokens=5)) == 3)

print("\n== TeacherSettings.from_config ==")
config = load_config()
settings = TeacherSettings.from_config(config)
print("  ", settings)
# No check here relies on the value config.yaml holds for the budget (it is 35000000 since 2026-11-21).
check("relative paths resolve against the repository root", settings.cache_dir == ROOT / ".cache" / "tabpfn")
check("lock_poll_s is persistence.replace_pause_s", settings.lock_poll_s == float(config["persistence"]["replace_pause_s"]))


def edited(path: str, value):
    data = copy.deepcopy(config)
    node = data
    *parents, last = path.split(".")
    for part in parents:
        node = node[part]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return data


for label, path, value in [
    ("missing tabpfn.model_path", "tabpfn.model_path", KeyError),
    ("null tabpfn.n_estimators", "tabpfn.n_estimators", None),
    ("float random_state 0.0", "tabpfn.random_state", 0.0),
    ("model_path 'auto' (server chooses)", "tabpfn.model_path", "auto"),
    ("spread level not a quantile level", "engine.spread_levels", [0.05, 0.9]),
    ("decreasing quantile levels", "engine.quantile_levels", [0.5, 0.25]),
    ("unknown multi_outcome_mode", "engine.multi_outcome_mode", "joint"),
    ("non-integer project budget", "tabpfn.project_token_budget", 1.5),
    ("missing tabpfn.max_query_rows_per_request", "tabpfn.max_query_rows_per_request", KeyError),
    ("row cap 5 (not above the 5 quantile levels)", "tabpfn.max_query_rows_per_request", 5),
    ("null tabpfn.lock_timeout_s", "tabpfn.lock_timeout_s", None),
    ("zero tabpfn.lock_timeout_s", "tabpfn.lock_timeout_s", 0),
    ("missing persistence.replace_pause_s", "persistence.replace_pause_s", KeyError),
]:
    raises(label, ConfigError, lambda p=path, v=value: TeacherSettings.from_config(edited(p, v)))
check("an integer project budget is read", TeacherSettings.from_config(edited("tabpfn.project_token_budget", 30000)).project_token_budget == 30000)
check("a null project budget reads as None (refuses every uncached request)",
      TeacherSettings.from_config(edited("tabpfn.project_token_budget", None)).project_token_budget is None)
check("a missing project budget reads as None", TeacherSettings.from_config(edited("tabpfn.project_token_budget", KeyError)).project_token_budget is None)

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
