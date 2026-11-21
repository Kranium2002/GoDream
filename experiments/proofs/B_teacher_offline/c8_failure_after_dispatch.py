"""Check 8 (review T1): a failure after the billable call never leads to a second dispatch; the raw
response is on disk before validation; adopt / abandon are explicit, logged operator decisions.

Scenarios (FixtureProvider, see _fixture_provider.py; the real teacher, cache and call log run):
  A. the validated-entry write fails (OSError) after a valid billed response
  B. the process dies (KeyboardInterrupt) during the one atomic write of <key>.npz
  C. the response fails validation (one NaN) -> adopt is rejected -> abandon -> a new dispatch
  D. the provider raises after the network stage (no response at all) -> abandon
  E. resolve_unresolved argument rules
Old behaviour (before the fix): each plain re-run dispatched and billed the same request again,
and nothing of the billed response was kept in A and C.

Temporary directory only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c8_failure_after_dispatch.py
"""

import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _fixture_provider as fx  # noqa: E402
import learned_judgment.teacher_cache as teacher_cache  # noqa: E402
from learned_judgment.teacher import ProviderResponseError, UnresolvedDispatchError  # noqa: E402

check = fx.Checks()
features, residuals, query = fx.arrays()


def the_key(teacher) -> str:
    keys = [r["key"] for r in teacher.call_log.read() if r.get("billable_dispatch") is True]
    return keys[-1]


def expected_answer(teacher_factory) -> np.ndarray:
    """Median residuals of the same prediction made by a fresh teacher with no failure."""
    with tempfile.TemporaryDirectory() as tmp:
        teacher, *_ = teacher_factory(Path(tmp))
        teacher.fit(features, residuals)
        return teacher.predict_residual_distribution(query).median


REFERENCE = expected_answer(fx.make_teacher)

print("== A. the validated-entry write fails after a valid billed response ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    real_put = teacher.cache.put
    state = {"failed": False}

    def failing_put(*args, **kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise OSError(28, "No space left on device (injected)")
        return real_put(*args, **kwargs)

    teacher.cache.put = failing_put
    check.raises("first prediction", OSError, lambda: teacher.predict_residual_distribution(query))
    key = the_key(teacher)
    stored = teacher.cache.get_unvalidated(key)
    check("the raw billed response is on disk in unvalidated/<key>.npz", stored is not None)
    sent_raw_q = np.asarray(account.sent_queries[0])
    check("its request is the query exactly as sent", stored is not None and np.array_equal(stored.request_arrays["x_query"], sent_raw_q))
    check("its quantiles keep the received (Q, n) layout", stored is not None and stored.raw_response["quantiles"].shape == (5, sent_raw_q.shape[0]))
    check("no validated entry was written", teacher.cache.get(key) is None)
    check("events: dispatch, cache_write_failed", fx.events(teacher) == ["dispatch", "cache_write_failed"])
    reads_before = (provider.usage_reads, provider.quote_reads)
    error = check.raises("plain re-run", UnresolvedDispatchError, lambda: teacher.predict_residual_distribution(query))
    check("the error names the key and the unvalidated file", error is not None and key in str(error) and "unvalidated" in str(error))
    check("re-run made no billable call (1 in total)", account.billed_predicts == 1 and provider.prediction_calls_started == 1)
    check("re-run read neither usage nor quote", (provider.usage_reads, provider.quote_reads) == reads_before)
    check("exactly one dispatch record", fx.dispatch_count(teacher) == 1)
    record = teacher.resolve_unresolved(key, action="adopt")
    check("adopt logs completed with adopted true", record["event"] == "completed" and record["adopted"] is True)
    check("adopt made no billable call and no usage/quote read",
          account.billed_predicts == 1 and (provider.usage_reads, provider.quote_reads) == reads_before)
    check("the unvalidated file is gone after the validated entry exists",
          teacher.cache.get(key) is not None and not teacher.cache.unvalidated_path(key).exists())
    answer = teacher.predict_residual_distribution(query).median
    check("the next run is a cache hit (still 1 billable call)", account.billed_predicts == 1 and fx.events(teacher)[-1] == "cache_hit")
    check("the adopted answer equals the answer of an undisturbed run", np.array_equal(answer, REFERENCE))

print("\n== B. killed during the single atomic write of <key>.npz ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    real_write = teacher_cache.atomic_write_bytes

    def killed_on_entry(path, payload):
        path = Path(path)
        if path.parent == settings.cache_dir and path.suffix == ".npz":
            raise KeyboardInterrupt("simulated kill during the validated-entry write")
        return real_write(path, payload)

    teacher_cache.atomic_write_bytes = killed_on_entry
    try:
        check.raises("first prediction", KeyboardInterrupt, lambda: teacher.predict_residual_distribution(query))
    finally:
        teacher_cache.atomic_write_bytes = real_write
    key = the_key(teacher)
    entries = sorted(p.name for p in settings.cache_dir.glob("*.npz"))
    check("no half-written entry: no <key>.npz and get() is None", entries == [] and teacher.cache.get(key) is None)
    check("the raw response is on disk", teacher.cache.unvalidated_path(key).exists())
    check.raises("plain re-run", UnresolvedDispatchError, lambda: teacher.predict_residual_distribution(query))
    check("still exactly 1 billable call and 1 dispatch record", account.billed_predicts == 1 and fx.dispatch_count(teacher) == 1)
    teacher.resolve_unresolved(key, action="adopt")
    answer = teacher.predict_residual_distribution(query).median
    check("after adopt: cache hit, still 1 billable call, same answer",
          account.billed_predicts == 1 and fx.events(teacher)[-1] == "cache_hit" and np.array_equal(answer, REFERENCE))

print("\n== C. the response fails validation; adopt is rejected; abandon allows a new dispatch ==")
with tempfile.TemporaryDirectory() as tmp:
    calls = {"n": 0}

    def one_bad_value(raw):
        calls["n"] += 1
        if calls["n"] == 1:
            raw["mean"] = raw["mean"].copy()
            raw["mean"][0] = np.nan
        return raw

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=one_bad_value)
    teacher.fit(features, residuals)
    error = check.raises("first prediction", ProviderResponseError, lambda: teacher.predict_residual_distribution(query))
    key = the_key(teacher)
    check("the error says where the billed response is kept", error is not None and str(teacher.cache.unvalidated_path(key)) in str(error))
    stored = teacher.cache.get_unvalidated(key)
    check("the raw response is stored exactly as received (NaN included)",
          stored is not None and np.isnan(stored.raw_response["mean"][0]) and stored.raw_response["mean"].dtype == np.float64)
    check("events: dispatch, response_rejected", fx.events(teacher) == ["dispatch", "response_rejected"])
    check.raises("plain re-run", UnresolvedDispatchError, lambda: teacher.predict_residual_distribution(query))
    check("still 1 billable call", account.billed_predicts == 1)
    check.raises("adopt of the invalid response", ProviderResponseError, lambda: teacher.resolve_unresolved(key, action="adopt"))
    check("adopt_rejected is logged and the key stays unresolved",
          fx.events(teacher)[-1] == "adopt_rejected"
          and check.raises("re-run after a rejected adopt", UnresolvedDispatchError,
                           lambda: teacher.predict_residual_distribution(query)) is not None)
    record = teacher.resolve_unresolved(key, action="abandon")
    moved = record.get("unvalidated_moved_to")
    check("abandon logs abandoned and keeps the raw response as evidence under unvalidated/abandoned/",
          record["event"] == "abandoned" and moved is not None and Path(moved).exists()
          and Path(moved).parent == teacher.cache.abandoned_dir and not teacher.cache.unvalidated_path(key).exists())
    answer = teacher.predict_residual_distribution(query).median
    check("after abandon the request is dispatched again (2 billable calls, 2 dispatch records)",
          account.billed_predicts == 2 and fx.dispatch_count(teacher) == 2)
    check("events end with dispatch, completed", fx.events(teacher)[-2:] == ["dispatch", "completed"])
    check("the new answer equals an undisturbed run", np.array_equal(answer, REFERENCE))

print("\n== D. the provider fails after the network stage (no response) ==")
with tempfile.TemporaryDirectory() as tmp:
    calls = {"n": 0}

    def timeout_once(raw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("read timed out (injected)")
        return raw

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=timeout_once)
    teacher.fit(features, residuals)
    check.raises("first prediction", TimeoutError, lambda: teacher.predict_residual_distribution(query))
    key = the_key(teacher)
    last = teacher.call_log.read()[-1]
    check("provider_error is logged as possibly billed in phase 'predict'",
          last["event"] == "provider_error" and last["possibly_billed"] is True and last["phase"] == "predict")
    check.raises("plain re-run", UnresolvedDispatchError, lambda: teacher.predict_residual_distribution(query))
    check.raises("adopt without a stored response", ValueError, lambda: teacher.resolve_unresolved(key, action="adopt"))
    check("still 1 billable call", account.billed_predicts == 1)
    teacher.resolve_unresolved(key, action="abandon")
    teacher.predict_residual_distribution(query)
    check("after abandon: dispatched once more and completed", account.billed_predicts == 2 and fx.events(teacher)[-1] == "completed")

print("\n== E. resolve_unresolved rules ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    teacher.predict_residual_distribution(query)
    key = teacher.last_request_hash
    check.raises("a key whose dispatch completed", ValueError, lambda: teacher.resolve_unresolved(key, action="abandon"))
    check.raises("an unknown action", ValueError, lambda: teacher.resolve_unresolved(key, action="retry"))
    check.raises("a malformed key", ValueError, lambda: teacher.resolve_unresolved("abc", action="adopt"))
    check("nothing was billed by these calls", account.billed_predicts == 1)

check.finish()
