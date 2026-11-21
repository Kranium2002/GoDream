"""Check 10 (review T3): a missing call log never restarts the project budget at zero.

When the call log (the spend ledger) does not exist while the cache directory or its unvalidated
folder holds any entry, every teacher request (uncached or cached) and every operator decision
raises LedgerMissingError and nothing is written. With an empty cache a missing log is the first
use and is allowed.
Old behaviour (before the fix): after deleting the log, the 4th request was sent although the
project budget was spent (review r04 part 1).

FixtureProvider (see _fixture_provider.py); temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c10_ledger_missing.py
"""

import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _fixture_provider as fx  # noqa: E402
from learned_judgment.teacher import BudgetError, LedgerMissingError, ProviderResponseError  # noqa: E402

check = fx.Checks()


def run(teacher, seed: int):
    features, residuals, query = fx.arrays(seed=seed)
    teacher.fit(features, residuals)
    return teacher.predict_residual_distribution(query)


print("== first use: empty cache and no call log is allowed ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp), project_token_budget=30_000)
    check("no call log and no cache before the first request", not settings.call_log.exists() and not teacher.cache.has_entries())
    run(teacher, 1)
    check("the first request is sent and logged", account.billed_predicts == 1 and settings.call_log.exists())

    print("\n== budget spent, then the call log is deleted ==")
    run(teacher, 2)
    run(teacher, 3)
    check("3 requests billed, 30,000 tokens quoted", account.billed_predicts == 3 and teacher.call_log.tokens_quoted_for_billed_calls() == 30_000)
    check.raises("the 4th request (budget spent)", BudgetError, lambda: run(teacher, 4))
    settings.call_log.unlink()  # e.g. results/ cleaned before a fresh evaluation run
    error = check.raises("the 4th request after the log was deleted", LedgerMissingError, lambda: run(teacher, 4))
    check("the message names the missing log and the cache, and warns against an empty log",
          error is not None and str(settings.call_log) in str(error) and str(settings.cache_dir) in str(error)
          and "empty" in str(error))
    check("no billable call: still 3", account.billed_predicts == 3)
    check.raises("a cached request after the log was deleted", LedgerMissingError, lambda: run(teacher, 1))
    check("the call log was not re-created (the budget did not restart at zero)", not settings.call_log.exists())

print("\n== an unvalidated response alone also counts as a cache entry ==")
with tempfile.TemporaryDirectory() as tmp:
    def nan_mean(raw):
        raw["mean"] = raw["mean"].copy()
        raw["mean"][0] = np.nan
        return raw

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=nan_mean)
    check.raises("a request whose response is rejected", ProviderResponseError, lambda: run(teacher, 1))
    key = teacher.call_log.read()[0]["key"]
    entries = sorted(p.name for p in settings.cache_dir.glob("*.npz"))
    check("only unvalidated/<key>.npz exists", entries == [] and teacher.cache.unvalidated_path(key).exists())
    settings.call_log.unlink()
    check.raises("another request after the log was deleted", LedgerMissingError, lambda: run(teacher, 2))
    check.raises("an operator decision after the log was deleted", LedgerMissingError,
                 lambda: teacher.resolve_unresolved(key, action="abandon"))
    check("still 1 billable call and no log", account.billed_predicts == 1 and not settings.call_log.exists())

check.finish()
