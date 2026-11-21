"""Check 2: RequestCache single-file entries, the unvalidated store, CacheLock and CallLog, in a temporary directory.

FIXTURE arrays only; exercises learned_judgment.teacher_cache (disk only, no network). Nothing is
written outside a tempfile.TemporaryDirectory.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c2_cache_and_log.py
"""

import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.teacher_cache import (  # noqa: E402
    CacheCorruptError,
    CacheLock,
    CacheLockedError,
    CallLog,
    RequestCache,
)

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    label = label.encode("ascii", "backslashreplace").decode("ascii")
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


rng = np.random.default_rng(1)
special = np.array([0.0, -0.0, 5e-324, np.inf, -np.inf, np.nan, 1.7976931348623157e308])
nan_payload = np.frombuffer(np.uint64(0xFFF8000000000001).tobytes(), dtype=np.float64).copy()
REQUEST = {
    "x_train": rng.normal(size=(7, 3)),
    "y_train": np.concatenate([special, nan_payload]),
    "x_query": np.asfortranarray(rng.normal(size=(4, 3))),
}
RESPONSE = {"mean": rng.normal(size=4), "quantiles": np.sort(rng.normal(size=(4, 5)), axis=1)}
REQUEST_META = {"b": 0.1 + 0.2, "a": [1, 2.0, 1e-17], "nested": {"z": None, "y": True}, "text": "caf\u00e9 \u2028"}
RESPONSE_META = {"timestamp_utc": "2026-11-21T00:00:00+00:00", "quote": {"estimated_cost": 10000}}
KEY = "a" * 64
OTHER = "b" * 64
THIRD = "c" * 64

with tempfile.TemporaryDirectory() as tmp:
    cache = RequestCache(Path(tmp) / "cache")
    print("== RequestCache: one authoritative <key>.npz ==")
    check("get of an unknown key is None (directory does not exist yet)", cache.get(OTHER) is None)
    check("has_entries of a missing directory is False", cache.has_entries() is False)
    sidecar_error = cache.put(KEY, request_arrays=REQUEST, request_meta=REQUEST_META, response_arrays=RESPONSE, response_meta=RESPONSE_META)
    check("put returns no sidecar error", sidecar_error is None)
    entry = cache.get(KEY)
    check("entry returned", entry is not None)
    for name, array in REQUEST.items():
        check(f"request {name} bitwise equal (dtype {array.dtype}, shape {array.shape})",
              bitwise_equal(np.ascontiguousarray(array), np.ascontiguousarray(entry.request_arrays[name])))
    for name, array in RESPONSE.items():
        check(f"response {name} bitwise equal", bitwise_equal(array, entry.response_arrays[name]))
    check("request metadata round-trips (non-ASCII text included)", entry.request_meta == REQUEST_META)
    check("response metadata round-trips", entry.response_meta == RESPONSE_META)
    check("get of another unknown key is None", cache.get(OTHER) is None)
    files = sorted(p.name for p in (Path(tmp) / "cache").iterdir())
    print("  files:", [name[:8] + "..." + name[-5:] for name in files])
    check("exactly <key>.npz + its sidecar <key>.json, no temp files left", files == [f"{KEY}.json", f"{KEY}.npz"])
    check("has_entries is True", cache.has_entries() is True)

    npz_path, json_path = cache.paths(KEY)
    json_path.write_text("not even JSON", encoding="ascii")
    check("an edited sidecar is never read: the entry is still served", cache.get(KEY) is not None)
    json_path.unlink()
    check("a deleted sidecar is never needed: the entry is still served", cache.get(KEY) is not None)
    only_sidecar = cache.paths(THIRD)[1]
    only_sidecar.write_text("{}", encoding="ascii")
    check("a sidecar without its .npz is no entry: get is None", cache.get(THIRD) is None)
    only_sidecar.unlink()

    second_response = {"mean": RESPONSE["mean"] + 1.0, "quantiles": RESPONSE["quantiles"] + 1.0}
    cache.put(KEY, request_arrays=REQUEST, request_meta=REQUEST_META, response_arrays=second_response, response_meta=RESPONSE_META)
    again = cache.get(KEY)
    check("second put of the same key leaves a readable entry", again is not None)
    check("  ...holding the second response bitwise", bitwise_equal(second_response["mean"], again.response_arrays["mean"]))
    check("  ...no temp files left", sorted(p.name for p in (Path(tmp) / "cache").iterdir()) == files)

    try:
        cache.put(OTHER, request_arrays=REQUEST, request_meta={"bad": float("nan")}, response_arrays=RESPONSE, response_meta={})
        check("NaN in metadata is rejected", False)
    except ValueError as exc:
        check(f"NaN in metadata is rejected before writing: {exc}", cache.get(OTHER) is None and not cache.paths(OTHER)[0].exists())
    try:
        cache.put(OTHER, request_arrays={"x": np.array(["a"])}, request_meta={}, response_arrays={}, response_meta={})
        check("string array is rejected", False)
    except TypeError as exc:
        check(f"string array is rejected: {exc}", True)
    try:
        cache.get("not-a-key")
        check("malformed key is rejected", False)
    except ValueError as exc:
        check(f"malformed key is rejected: {exc}", True)

    pristine = npz_path.read_bytes()
    for label, position in (("near the end (zip directory)", len(pristine) - 30), ("inside the array data", len(pristine) // 3)):
        payload = bytearray(pristine)
        payload[position] ^= 0xFF
        npz_path.write_bytes(bytes(payload))
        try:
            cache.get(KEY)
            check(f"altered .npz {label} is detected", False)
        except CacheCorruptError as exc:
            check(f"altered .npz {label} is detected: {str(exc).replace(tmp, '<tmp>')[:150]}", True)
    npz_path.write_bytes(pristine)
    check("the restored entry reads again", cache.get(KEY) is not None)
    other_npz = cache.paths(OTHER)[0]
    other_npz.write_bytes(pristine)
    try:
        cache.get(OTHER)
        check("an entry copied under another key is detected", False)
    except CacheCorruptError as exc:
        check(f"an entry copied under another key is detected: {str(exc).replace(tmp, '<tmp>')[:120]}", True)
    other_npz.unlink()

    print("\n== unvalidated responses: stored exactly as received ==")
    raw = {
        "mean": np.array([1.0, np.nan, 3.0], dtype=np.float32),
        "median": [0.5, None, 1.5],
        "quantiles": [[1.0, 2.0], [3.0]],
        "note": "server text",
    }
    path = cache.put_unvalidated(THIRD, request_arrays=REQUEST, request_meta={"base_url": "x"}, raw_response=raw,
                                 response_meta={"quote": {"estimated_cost": 10000}})
    check("written to unvalidated/<key>.npz", path == cache.unvalidated_path(THIRD) and path.exists())
    stored = cache.get_unvalidated(THIRD)
    check("numeric item kept with its own dtype and NaN", stored.raw_response["mean"].dtype == np.float32
          and bitwise_equal(stored.raw_response["mean"], raw["mean"]))
    check("list with None kept as JSON", stored.raw_response["median"] == [0.5, None, 1.5])
    check("ragged list kept as JSON", stored.raw_response["quantiles"] == [[1.0, 2.0], [3.0]])
    check("text kept", stored.raw_response["note"] == "server text")
    check("request arrays kept bitwise",
          all(bitwise_equal(np.ascontiguousarray(REQUEST[n]), np.ascontiguousarray(stored.request_arrays[n])) for n in REQUEST))
    check("the unvalidated file is no validated entry", cache.get(THIRD) is None)
    cache.put_unvalidated(OTHER, request_arrays={}, request_meta={}, raw_response=[1, 2, 3], response_meta={})
    check("a non-mapping raw value is kept too", np.array_equal(cache.get_unvalidated(OTHER).raw_response, [1, 2, 3]))
    cache.remove_unvalidated(OTHER)
    check("remove_unvalidated deletes it", cache.get_unvalidated(OTHER) is None)
    moved = cache.abandon_unvalidated(THIRD, "20261121T120000000000Z")
    check("abandon moves it to unvalidated/abandoned/<key>.<stamp>.npz",
          moved is not None and moved.parent == cache.abandoned_dir and moved.exists() and not cache.unvalidated_path(THIRD).exists())
    check("abandon of a key without a file returns None", cache.abandon_unvalidated(THIRD, "20261121T120000000001Z") is None)

    print("\n== CacheLock ==")
    lock = CacheLock(cache.lock_path, timeout_s=0.3, poll_s=0.01)
    twin = CacheLock(cache.lock_path, timeout_s=0.3, poll_s=0.01)
    with lock:
        content = cache.lock_path.read_text(encoding="ascii")
        check("the lock file holds pid and UTC time", '"pid"' in content and '"acquired_utc"' in content)
        with twin:
            check("re-entrant for the holding thread, also through another CacheLock of the same file", twin.held_by_this_thread())
        check("still held after the inner release", cache.lock_path.exists())
        outcome: list[str] = []

        def other_thread() -> None:
            try:
                with CacheLock(cache.lock_path, timeout_s=0.3, poll_s=0.01):
                    outcome.append("acquired")
            except CacheLockedError:
                outcome.append("CacheLockedError")

        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
        check("another thread waits and gets CacheLockedError", outcome == ["CacheLockedError"])
    check("the outermost release removes the lock file", not cache.lock_path.exists())
    try:
        lock.release()
        check("release without holding raises", False)
    except RuntimeError as exc:
        check(f"release without holding raises: {str(exc).replace(tmp, '<tmp>')}", True)

    print("\n== CallLog ==")
    ticks = iter(datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=s) for s in range(100))
    log = CallLog(Path(tmp) / "logs" / "calls.jsonl", clock=lambda: next(ticks), lock=lock)
    check("read of a missing log is []", log.read() == [] and not log.exists())
    check("tokens of an empty log is 0", log.tokens_quoted_for_billed_calls() == 0)
    log.append({"event": "refused", "billable_dispatch": False, "quote_tokens": 10000})
    log.append({"event": "dispatch", "billable_dispatch": True, "quote_tokens": 10000})
    log.append({"event": "completed", "billable_dispatch": False, "quote_tokens": 10000})
    log.append({"event": "dispatch", "billable_dispatch": True, "quote_tokens": 12345})
    log.append({"event": "cache_hit", "billable_dispatch": False, "quote_tokens": None, "text": "caf\u00e9\u2028"})
    records = log.read()
    check("5 records read back in order", [r["event"] for r in records] == ["refused", "dispatch", "completed", "dispatch", "cache_hit"])
    check("each record carries the injected clock's UTC timestamp", records[0]["timestamp_utc"] == "2026-11-21T12:00:00+00:00")
    check("the file is ASCII and non-ASCII text round-trips",
          all(b < 128 for b in log.path.read_bytes()) and records[-1]["text"] == "caf\u00e9\u2028")
    total = log.tokens_quoted_for_billed_calls()
    check(f"tokens quoted for billed calls counts only dispatch records: {total}", total == 10000 + 12345)
    check("no lock file is left after appends and reads", not cache.lock_path.exists())
    for label, record in {
        "credential key at top level": {"event": "x", "access_token": "..."},
        "credential key nested": {"event": "x", "meta": {"Authorization": "..."}},
        "caller-set timestamp": {"event": "x", "timestamp_utc": "now"},
        "non-finite float": {"event": "x", "value": float("inf")},
    }.items():
        try:
            log.append(record)
            check(f"{label} is rejected", False)
        except ValueError as exc:
            check(f"{label} is rejected: {exc}", True)
    check("rejected records were not written", len(log.read()) == 5)
    naive = CallLog(Path(tmp) / "naive.jsonl", clock=lambda: datetime(2026, 11, 21, 12, 0), lock=lock)
    try:
        naive.append({"event": "x"})
        check("a naive (non-UTC) clock is rejected", False)
    except ValueError as exc:
        check(f"a naive (non-UTC) clock is rejected: {exc}", not (Path(tmp) / "naive.jsonl").exists())
    try:
        CallLog(Path(tmp) / "x.jsonl", clock=lambda: datetime.now(timezone.utc), lock=None)
        check("a call log without a lock is rejected", False)
    except TypeError as exc:
        check(f"a call log without a lock is rejected: {exc}", True)

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
