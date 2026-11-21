"""Check 2: RequestCache bitwise round trip and CallLog behaviour, in a temporary directory.

FIXTURE arrays only; exercises learned_judgment.teacher_cache (disk only, no network). Nothing is
written outside a tempfile.TemporaryDirectory.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c2_cache_and_log.py
"""

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

print("FIXTURE CHECK - utility test only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.teacher_cache import CacheCorruptError, CallLog, RequestCache  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
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
REQUEST_META = {"b": 0.1 + 0.2, "a": [1, 2.0, 1e-17], "nested": {"z": None, "y": True}}
RESPONSE_META = {"timestamp_utc": "2026-11-21T00:00:00+00:00", "quote": {"estimated_cost": 10000}}
KEY = "a" * 64
OTHER = "b" * 64

with tempfile.TemporaryDirectory() as tmp:
    cache = RequestCache(Path(tmp) / "cache")
    print("== RequestCache ==")
    check("get of an unknown key is None (directory does not exist yet)", cache.get(OTHER) is None)
    cache.put(KEY, request_arrays=REQUEST, request_meta=REQUEST_META, response_arrays=RESPONSE, response_meta=RESPONSE_META)
    entry = cache.get(KEY)
    check("entry returned", entry is not None)
    for name, array in REQUEST.items():
        check(f"request {name} bitwise equal (dtype {array.dtype}, shape {array.shape})", bitwise_equal(np.ascontiguousarray(array), entry.request_arrays[name]))
    for name, array in RESPONSE.items():
        check(f"response {name} bitwise equal", bitwise_equal(array, entry.response_arrays[name]))
    check("request metadata round-trips", entry.request_meta == REQUEST_META)
    check("response metadata round-trips", entry.response_meta == RESPONSE_META)
    check("get of another unknown key is None", cache.get(OTHER) is None)
    files = sorted(p.name for p in (Path(tmp) / "cache").iterdir())
    print("  files:", files)
    check("exactly <key>.npz + <key>.json, no temp files left", files == [f"{KEY}.json", f"{KEY}.npz"])

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
        check(f"NaN in metadata is rejected before writing: {exc}", cache.get(OTHER) is None)
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

    npz_path, _ = cache.paths(KEY)
    payload = bytearray(npz_path.read_bytes())
    payload[-30] ^= 0xFF
    npz_path.write_bytes(bytes(payload))
    try:
        cache.get(KEY)
        check("altered .npz is detected", False)
    except CacheCorruptError as exc:
        check(f"altered .npz is detected: {str(exc).replace(tmp, '<tmp>')}", True)

    print("\n== CallLog ==")
    ticks = iter(datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=s) for s in range(100))
    log = CallLog(Path(tmp) / "logs" / "calls.jsonl", clock=lambda: next(ticks))
    check("read of a missing log is []", log.read() == [])
    check("tokens of an empty log is 0", log.tokens_quoted_for_billed_calls() == 0)
    log.append({"event": "refused", "billable_dispatch": False, "quote_tokens": 10000})
    log.append({"event": "dispatch", "billable_dispatch": True, "quote_tokens": 10000})
    log.append({"event": "completed", "billable_dispatch": False, "quote_tokens": 10000})
    log.append({"event": "dispatch", "billable_dispatch": True, "quote_tokens": 12345})
    log.append({"event": "cache_hit", "billable_dispatch": False, "quote_tokens": None})
    records = log.read()
    check("5 records read back in order", [r["event"] for r in records] == ["refused", "dispatch", "completed", "dispatch", "cache_hit"])
    check("each record carries the injected clock's UTC timestamp", records[0]["timestamp_utc"] == "2026-11-21T12:00:00+00:00")
    total = log.tokens_quoted_for_billed_calls()
    check(f"tokens quoted for billed calls counts only dispatch records: {total}", total == 10000 + 12345)
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
    naive = CallLog(Path(tmp) / "naive.jsonl", clock=lambda: datetime(2026, 11, 21, 12, 0))
    try:
        naive.append({"event": "x"})
        check("a naive (non-UTC) clock is rejected", False)
    except ValueError as exc:
        check(f"a naive (non-UTC) clock is rejected: {exc}", not (Path(tmp) / "naive.jsonl").exists())

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
