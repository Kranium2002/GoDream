"""Check 13 (review T6): persistence settings and every write location are checked BEFORE the gate
lets a request through; a failure is a refusal before any billable call.

Old behaviour (before the fix): a null persistence.replace_attempts raised only inside the cache
write, after the billable call, and the response was lost (review r07 part 2).

FixtureProvider (see _fixture_provider.py); config.yaml is never written (a modified copy lives in
memory); temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c13_persistence_probe.py
"""

import copy
import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fixture_provider as fx  # noqa: E402
import learned_judgment.fileio as fileio  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import BudgetError  # noqa: E402

check = fx.Checks()
features, residuals, query = fx.arrays()


def nothing_sent(provider, account, label: str) -> None:
    check(f"{label}: no billable call, no usage read, no quote", account.billed_predicts == 0
          and provider.prediction_calls_started == 0 and provider.usage_reads == 0 and provider.quote_reads == 0)


print("== a. persistence.replace_attempts is null ==")
broken = copy.deepcopy(load_config())
broken["persistence"]["replace_attempts"] = None
real_load = fileio.load_config
fileio.load_config = lambda *args, **kwargs: broken
fileio._replace_settings.cache_clear()
try:
    with tempfile.TemporaryDirectory() as tmp:
        teacher, provider, account, settings = fx.make_teacher(Path(tmp))
        teacher.fit(features, residuals)
        error = check.raises("a request", BudgetError, lambda: teacher.predict_residual_distribution(query))
        check("the reason names persistence.replace_attempts", error is not None and "persistence.replace_attempts" in str(error))
        nothing_sent(provider, account, "a")
        records = teacher.call_log.read()
        check("one refused record, no dispatch record", [r["event"] for r in records] == ["refused"] and fx.dispatch_count(teacher) == 0)
finally:
    fileio.load_config = real_load
    fileio._replace_settings.cache_clear()

print("\n== b. the unvalidated folder cannot be created ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    settings.cache_dir.mkdir(parents=True)
    teacher.cache.unvalidated_dir.write_text("a file where the folder should be", encoding="ascii")
    error = check.raises("a request", BudgetError, lambda: teacher.predict_residual_distribution(query))
    check("the reason names the cache directory or its unvalidated folder", error is not None and "unvalidated folder" in str(error))
    nothing_sent(provider, account, "b")
    check("one refused record", fx.events(teacher) == ["refused"])

print("\n== c. the call-log directory cannot be created ==")
with tempfile.TemporaryDirectory() as tmp:
    blocker = Path(tmp) / "results_is_a_file"
    blocker.write_text("a file where the call-log directory should be", encoding="ascii")
    teacher, provider, account, settings = fx.make_teacher(Path(tmp), call_log=blocker / "calls.jsonl")
    teacher.fit(features, residuals)
    error = check.raises("a request", BudgetError, lambda: teacher.predict_residual_distribution(query))
    check("the message names the call-log directory and says the refusal could not be logged",
          error is not None and "call-log directory" in str(error) and "could not be logged" in str(error))
    nothing_sent(provider, account, "c")

print("\n== d. healthy settings and directories ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    teacher.predict_residual_distribution(query)
    check("the request completes", fx.events(teacher) == ["dispatch", "completed"] and account.billed_predicts == 1)
    leftovers = [p.name for p in settings.cache_dir.rglob("*") if p.name.startswith(".write-probe")]
    leftovers += [p.name for p in Path(tmp).glob(".write-probe*")]
    check("no probe file is left behind", leftovers == [])

check.finish()
