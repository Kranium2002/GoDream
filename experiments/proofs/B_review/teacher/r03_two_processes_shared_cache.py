"""Review r03: two processes that share tabpfn.cache_dir and tabpfn.call_log (the config has one of each).

Part 1: both processes ask for the SAME uncached request with tabpfn.project_token_budget = 10,000
(room for exactly one minimum-charge request). FixtureProvider (see _fixture.py) replaces only the
network; its predict() takes 0.5 s like a real round trip. Nothing locks the key or the budget.
Part 2: four processes append 300 records each through the real CallLog.append.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r03_two_processes_shared_cache.py
"""

import collections
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def same_request_worker(tmp: str, start_at: float) -> None:
    import _fixture as fx

    slow = lambda raw: (time.sleep(0.5), raw)[1]  # noqa: E731
    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp), response_hook=slow, budget=10_000)
    features, residuals, query = fx.arrays()
    teacher.fit(features, residuals)
    while time.time() < start_at:
        time.sleep(0.001)
    try:
        teacher.predict_residual_distribution(query)
        print(f"  pid {mp.current_process().pid}: predicted, billable predicts in this process = {account.billed_predicts}", flush=True)
    except Exception as exc:
        print(f"  pid {mp.current_process().pid}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)


def append_worker(path: str, worker: int, count: int) -> None:
    import _fixture as fx
    from learned_judgment.teacher_cache import CallLog

    log = CallLog(Path(path), clock=fx.make_clock())
    for index in range(count):
        log.append({"event": "dispatch", "billable_dispatch": True, "quote_tokens": 10_000,
                    "worker": worker, "index": index, "pad": "x" * 200})


if __name__ == "__main__":
    print("REVIEW REPRODUCTION - fixture, not evidence", flush=True)
    import _fixture as fx

    with tempfile.TemporaryDirectory() as tmp:
        print("== Part 1: same uncached request from two processes, project budget 10,000 ==")
        start_at = time.time() + 3.0
        workers = [mp.Process(target=same_request_worker, args=(tmp, start_at)) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        from learned_judgment.teacher_cache import CallLog

        log = CallLog(Path(tmp) / "calls.jsonl", clock=fx.make_clock())
        records = log.read()
        dispatches = [r for r in records if r.get("billable_dispatch") is True]
        print("  call-log events:", [r["event"] for r in records], flush=True)
        print("  dispatch records:", len(dispatches), "| distinct keys among them:", len({r["key"] for r in dispatches}))
        print("  tokens_quoted_for_billed_calls:", log.tokens_quoted_for_billed_calls(), "vs project_token_budget 10000")

    with tempfile.TemporaryDirectory() as tmp:
        print("\n== Part 2: 4 processes x 300 CallLog.append ==")
        path = Path(tmp) / "calls.jsonl"
        workers = [mp.Process(target=append_worker, args=(str(path), w, 300)) for w in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        raw_lines = path.read_text(encoding="utf-8").split("\n")
        blank = sum(1 for line in raw_lines[:-1] if not line)
        nul = path.read_bytes().count(b"\x00")
        print(f"  blank lines inside the log: {blank}; NUL bytes: {nul}")
        lines = [line for line in raw_lines if line]
        import json

        good, bad = 0, 0
        for line in lines:
            try:
                json.loads(line)
                good += 1
            except ValueError:
                bad += 1
        print(f"  expected 1200 records; physical lines {len(lines)}, parseable {good}, malformed {bad}")
        log = CallLog(path, clock=fx.make_clock())
        try:
            print("  tokens_quoted_for_billed_calls:", log.tokens_quoted_for_billed_calls(), "(expected 12000000)")
        except ValueError as exc:
            print("  tokens_quoted_for_billed_calls raised:", str(exc).replace(tmp, "<tmp>")[:160])
