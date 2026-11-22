"""Review r02: the same request is dispatched (billed) again after a failure that follows the billable call.

Three post-dispatch failures, each followed by a plain re-run of the same prediction:
  A. cache.put raises (e.g. disk full / permission) after a valid, billed response
  B. the process dies between writing <key>.npz and <key>.json (simulated by failing the .json write)
  C. the response fails validation (one non-finite value) after the billable predict
FixtureProvider (see _fixture.py) replaces only the network; the real teacher gate, cache and
call log run. Temporary directory only.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r02_redispatch_after_failure.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import collections
import tempfile
from pathlib import Path

import numpy as np

import _fixture as fx
import learned_judgment.teacher_cache as teacher_cache


def summary(log, account, provider, label):
    records = log.read()
    dispatch_keys = collections.Counter(r["key"] for r in records if r.get("billable_dispatch") is True)
    print(f"  [{label}] billable predicts made: {account.billed_predicts}; provider.prediction_calls_started: "
          f"{provider.prediction_calls_started}")
    print(f"  [{label}] events: {[r['event'] for r in records]}")
    print(f"  [{label}] dispatch records per key: {sorted(dispatch_keys.values())}; "
          f"tokens_quoted_for_billed_calls: {log.tokens_quoted_for_billed_calls()}")


features, residuals, query = fx.arrays()

print("== A. cache write fails after a valid billed response ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    real_put = teacher.cache.put
    state = {"failed": False}

    def failing_put(*args, **kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise OSError(28, "No space left on device")
        return real_put(*args, **kwargs)

    teacher.cache.put = failing_put
    try:
        teacher.predict_residual_distribution(query)
    except OSError as exc:
        print("  first run raised:", type(exc).__name__, exc)
    print("  files in cache dir after first run:", sorted(p.name for p in settings.cache_dir.glob("*")) if settings.cache_dir.exists() else "(none)")
    teacher.predict_residual_distribution(query)  # same inputs, disk healthy again
    summary(log, account, provider, "A")

print("\n== B. crash between <key>.npz and <key>.json ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    real_write = teacher_cache.atomic_write_bytes

    def crash_on_json(path, payload):
        if str(path).endswith(".json"):
            raise KeyboardInterrupt("simulated kill after the .npz was written")
        return real_write(path, payload)

    teacher_cache.atomic_write_bytes = crash_on_json
    try:
        teacher.predict_residual_distribution(query)
    except KeyboardInterrupt as exc:
        print("  first run interrupted:", exc)
    teacher_cache.atomic_write_bytes = real_write
    npz = sorted(settings.cache_dir.glob("*.npz"))
    print("  orphan files:", [p.name[:12] + "..." + p.suffix for p in settings.cache_dir.iterdir()])
    with np.load(npz[0]) as archive:
        print("  orphan .npz members (complete billed response is on disk):", sorted(archive.files))
    key = npz[0].stem
    print("  RequestCache.get(key) ->", teacher.cache.get(key))
    teacher.predict_residual_distribution(query)  # a plain re-run
    summary(log, account, provider, "B")

print("\n== C. response rejected by validation after the billable predict ==")
with tempfile.TemporaryDirectory() as tmp:
    calls = {"n": 0}

    def one_bad_value(raw):
        calls["n"] += 1
        if calls["n"] == 1:
            raw["mean"] = raw["mean"].copy()
            raw["mean"][0] = np.nan
        return raw

    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp), response_hook=one_bad_value)
    teacher.fit(features, residuals)
    try:
        teacher.predict_residual_distribution(query)
    except Exception as exc:  # ProviderResponseError
        print("  first run raised:", type(exc).__name__, exc)
    print("  anything saved from the billed response:",
          sorted(p.name for p in settings.cache_dir.glob("*")) if settings.cache_dir.exists() else "(cache dir not even created)")
    teacher.predict_residual_distribution(query)
    summary(log, account, provider, "C")
