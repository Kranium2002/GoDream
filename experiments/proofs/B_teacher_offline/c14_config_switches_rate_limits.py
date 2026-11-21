"""Check 14 (review T7): the tabpfn safety switches are read and must be true; the verified request
limits are enforced from the dispatch timestamps in the call log when set, and recorded as not
configured when null.

Old behaviour (before the fix): flipping the switches to false changed nothing and no rate limit
was ever read (review r07 part 1).

Edited copies of config.yaml's dict live in memory only. FixtureProvider (see
_fixture_provider.py); temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c14_config_switches_rate_limits.py
"""

import copy
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fixture_provider as fx  # noqa: E402
from learned_judgment.config import ConfigError, load_config  # noqa: E402
from learned_judgment.teacher import BudgetError, TeacherSettings  # noqa: E402

check = fx.Checks()
CONFIG = load_config()


def edited(path: str, value):
    data = copy.deepcopy(CONFIG)
    node = data
    *parents, last = path.split(".")
    for part in parents:
        node = node[part]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return data


print("== the three switches must be true ==")
for key in ("check_usage_before_uncached_request", "estimate_cost_before_uncached_request", "cache_requests_and_responses"):
    for value in (False, None, "yes", KeyError):
        error = check.raises(f"tabpfn.{key} = {value if value is not KeyError else '<missing>'}", ConfigError,
                             lambda k=key, v=value: TeacherSettings.from_config(edited(f"tabpfn.{k}", v)))
        if error is not None and value is False:
            check(f"  the message names tabpfn.{key} and says it cannot be switched off",
                  f"tabpfn.{key}" in str(error) and "cannot be switched off" in str(error))
check("all true (config.yaml as it is) is accepted", isinstance(TeacherSettings.from_config(CONFIG), TeacherSettings))

print("\n== verified request limits are read with optional() ==")
settings = TeacherSettings.from_config(CONFIG)
print(f"  config.yaml: per_minute={settings.verified_request_limit_per_minute!r}, per_hour={settings.verified_request_limit_per_hour!r}")
check("per_minute 60 is read", TeacherSettings.from_config(edited("tabpfn.verified_request_limit_per_minute", 60)).verified_request_limit_per_minute == 60)
check("per_hour 500 is read", TeacherSettings.from_config(edited("tabpfn.verified_request_limit_per_hour", 500)).verified_request_limit_per_hour == 500)
check("a missing key reads as None", TeacherSettings.from_config(edited("tabpfn.verified_request_limit_per_hour", KeyError)).verified_request_limit_per_hour is None)
for label, value in (("0", 0), ("'60'", "60"), ("1.5", 1.5)):
    check.raises(f"per_minute {label}", ConfigError,
                 lambda v=value: TeacherSettings.from_config(edited("tabpfn.verified_request_limit_per_minute", v)))


class SetClock:
    """A UTC clock that stays where it is put (fixture)."""

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def run(teacher, seed: int):
    features, residuals, query = fx.arrays(seed=seed)
    teacher.fit(features, residuals)
    return teacher.predict_residual_distribution(query)


print("\n== per-minute limit 2 ==")
with tempfile.TemporaryDirectory() as tmp:
    clock = SetClock()
    teacher, provider, account, s = fx.make_teacher(Path(tmp), clock=clock, verified_request_limit_per_minute=2)
    for seed in (1, 2):
        clock.advance(1)
        run(teacher, seed)
    clock.advance(1)
    error = check.raises("a 3rd dispatch 3 s after the first", BudgetError, lambda: run(teacher, 3))
    check("the reason names the per-minute limit", error is not None and "verified_request_limit_per_minute" in str(error))
    refused = teacher.call_log.read()[-1]
    check("the refused record carries the window counts",
          refused["event"] == "refused" and refused["rate_limit"]["dispatches_last_minute"] == 2)
    check("2 billable calls", account.billed_predicts == 2)
    clock.advance(60)
    run(teacher, 3)
    check("61 s later the 3rd request is sent", account.billed_predicts == 3)
    dispatch = [r for r in teacher.call_log.read() if r.get("billable_dispatch") is True][-1]
    check("the dispatch record says rate_limit_configured true with its counts",
          dispatch["rate_limit_configured"] is True and dispatch["rate_limit"]["per_minute"] == 2
          and dispatch["rate_limit"]["dispatches_last_minute"] == 0)

print("\n== per-hour limit 2 ==")
with tempfile.TemporaryDirectory() as tmp:
    clock = SetClock()
    teacher, provider, account, s = fx.make_teacher(Path(tmp), clock=clock, verified_request_limit_per_hour=2)
    for seed in (1, 2):
        clock.advance(120)
        run(teacher, seed)
    clock.advance(120)
    check.raises("a 3rd dispatch within the hour", BudgetError, lambda: run(teacher, 3))
    clock.advance(3600)
    run(teacher, 3)
    check("an hour later it is sent; 3 billable calls in total", account.billed_predicts == 3)

print("\n== limits null (config.yaml today) ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, s = fx.make_teacher(Path(tmp), verified_request_limit_per_minute=None,
                                                     verified_request_limit_per_hour=None)
    for seed in (1, 2, 3):
        run(teacher, seed)
    dispatches = [r for r in teacher.call_log.read() if r.get("billable_dispatch") is True]
    check("every dispatch record says rate_limit_configured false and carries no limit",
          len(dispatches) == 3 and all(r["rate_limit_configured"] is False and "rate_limit" not in r for r in dispatches))

check.finish()
