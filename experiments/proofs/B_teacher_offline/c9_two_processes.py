"""Check 9 (review T2): inter-process exclusion on the shared cache directory and call log.

Part 1: two processes ask for the SAME uncached request at the same moment (released together by a
        barrier; the fixture's predict takes 0.5 s like a round trip; project budget 10,000 = one
        minimum charge). Exactly one dispatch record may exist; the other process must get the
        answer from the cache after waiting for the lock.
Part 2: four processes append 300 records each through the real CallLog.append at the same time:
        no line may be lost, merged or blank.
Part 3: a stale <cache_dir>/.lock (its process is gone) makes a request wait lock_timeout_s and
        raise CacheLockedError; the lock is never broken automatically.
Old behaviour (before the fix): part 1 gave two dispatch records (20,000 tokens against a 10,000
budget) and part 2 lost 269 of 1,200 lines (review r03).

FixtureProvider (see _fixture_provider.py); temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c9_two_processes.py
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def same_request_worker(tmp: str, barrier, results) -> None:
    import _fixture_provider as fx

    def slow(raw):
        time.sleep(0.5)
        return raw

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=slow, project_token_budget=10_000)
    features, residuals, query = fx.arrays()
    teacher.fit(features, residuals)
    provider.base_url()  # import the client before the start line, so both processes start together
    barrier.wait()
    try:
        median = teacher.predict_residual_distribution(query).median
        results.put(("ok", os.getpid(), account.billed_predicts, median.tolist()))
    except Exception as exc:
        results.put(("error", os.getpid(), account.billed_predicts, f"{type(exc).__name__}: {exc}"))


def append_worker(tmp: str, worker: int, count: int, barrier) -> None:
    import _fixture_provider as fx
    from learned_judgment.teacher_cache import CacheLock, CallLog

    settings = fx.settings_for(Path(tmp))
    lock = CacheLock(settings.cache_dir / ".lock", timeout_s=settings.lock_timeout_s, poll_s=settings.lock_poll_s)
    log = CallLog(settings.call_log, clock=fx.make_clock(), lock=lock)
    barrier.wait()
    for index in range(count):
        log.append({"event": "dispatch", "billable_dispatch": True, "quote_tokens": 10_000, "key": f"w{worker}",
                    "worker": worker, "index": index, "pad": "x" * 200})


if __name__ == "__main__":
    print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model", flush=True)
    import _fixture_provider as fx
    from learned_judgment.teacher import CacheLockedError

    check = fx.Checks()

    print("== Part 1: the same uncached request from two processes, project budget 10,000 ==", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        barrier, results = mp.Barrier(2), mp.Queue()
        workers = [mp.Process(target=same_request_worker, args=(tmp, barrier, results)) for _ in range(2)]
        for worker in workers:
            worker.start()
        outcomes = [results.get(timeout=300) for _ in workers]
        for worker in workers:
            worker.join(timeout=300)
        for status, pid, billed, payload in outcomes:
            print(f"  process: {status}, billable predicts in it = {billed}", flush=True)
        teacher, *_ = fx.make_teacher(Path(tmp), project_token_budget=10_000)
        records = teacher.call_log.read()
        dispatches = [r for r in records if r.get("billable_dispatch") is True]
        print("  call-log events:", [r["event"] for r in records], flush=True)
        check("both processes got an answer", all(o[0] == "ok" for o in outcomes))
        check("exactly one dispatch record", len(dispatches) == 1)
        check("billable predicts over both processes: 1", sum(o[2] for o in outcomes) == 1)
        check("events: dispatch, completed, cache_hit", [r["event"] for r in records] == ["dispatch", "completed", "cache_hit"])
        check("the cache hit was found after waiting for the lock", records[-1].get("found_after_lock_wait") is True)
        check("both answers are identical", len(outcomes) == 2 and outcomes[0][3] == outcomes[1][3])
        check("tokens quoted for billed calls: 10,000 (within the budget)", teacher.call_log.tokens_quoted_for_billed_calls() == 10_000)
        check("the lock file is gone", not teacher.cache.lock_path.exists())
        check("both processes exited 0", all(worker.exitcode == 0 for worker in workers))

    print("\n== Part 2: 4 processes x 300 CallLog.append at the same time ==", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        barrier = mp.Barrier(4)
        start = time.perf_counter()
        workers = [mp.Process(target=append_worker, args=(tmp, w, 300, barrier)) for w in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=600)
        print(f"  4 x 300 appends took {time.perf_counter() - start:.1f} s (including process start)", flush=True)
        path = Path(tmp) / "calls.jsonl"
        raw_lines = path.read_text(encoding="utf-8").split("\n")
        blank = sum(1 for line in raw_lines[:-1] if not line)
        parsed = []
        malformed = 0
        for line in raw_lines[:-1]:
            try:
                parsed.append(json.loads(line))
            except ValueError:
                malformed += 1
        pairs = {(r["worker"], r["index"]) for r in parsed}
        print(f"  physical lines {len(raw_lines) - 1}, blank {blank}, malformed {malformed}, distinct records {len(pairs)}", flush=True)
        check("1200 lines, none blank or malformed", len(raw_lines) - 1 == 1200 and blank == 0 and malformed == 0)
        check("every (worker, index) record is present exactly once", len(pairs) == 1200 and len(parsed) == 1200)
        check("the file ends with a newline", raw_lines[-1] == "")
        teacher, *_ = fx.make_teacher(Path(tmp))
        check("the ledger reads back: 1200 x 10,000 tokens", teacher.call_log.tokens_quoted_for_billed_calls() == 12_000_000)
        check("all 4 processes exited 0", all(worker.exitcode == 0 for worker in workers))

    print("\n== Part 3: a stale lock is never broken automatically ==", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        teacher, provider, account, settings = fx.make_teacher(Path(tmp), lock_timeout_s=1.0)
        features, residuals, query = fx.arrays()
        teacher.fit(features, residuals)
        teacher.cache.lock_path.parent.mkdir(parents=True)
        stale = '{"pid": 999999, "acquired_utc": "2026-11-21T00:00:00+00:00"}'
        teacher.cache.lock_path.write_text(stale, encoding="ascii")
        provider.base_url()  # import the client first, so the wait below measures only the lock
        start = time.perf_counter()
        error = check.raises("a request while a stale lock exists", CacheLockedError,
                             lambda: teacher.predict_residual_distribution(query))
        waited = time.perf_counter() - start
        print(f"  waited {waited:.2f} s (lock_timeout_s = 1.0)", flush=True)
        check("it waited at least lock_timeout_s and gave up soon after", 1.0 <= waited < 10.0)
        check("the message names the lock file, the holder's pid and how to remove a stale lock",
              error is not None and str(teacher.cache.lock_path) in str(error) and "999999" in str(error)
              and "delete" in str(error))
        check("the stale lock is untouched", teacher.cache.lock_path.read_text(encoding="ascii") == stale)
        check("nothing was billed, quoted or logged",
              account.billed_predicts == 0 and provider.quote_reads == 0 and not settings.call_log.exists())
        teacher.cache.lock_path.unlink()  # the operator's decision, by hand
        teacher.predict_residual_distribution(query)
        check("after the operator removed it, the request runs", account.billed_predicts == 1 and fx.events(teacher)[-1] == "completed")

    check.finish()
