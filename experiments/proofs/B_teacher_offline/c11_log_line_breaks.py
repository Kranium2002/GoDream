"""Check 11 (review T4): characters that str.splitlines() treats as line breaks never make the ledger unreadable.

Call-log lines are written with ensure_ascii (U+2028, U+2029, U+0085 ... become \\uXXXX escapes)
and the log is read by splitting on "\\n" only, so a raw U+2028 inside an older line is also read
as part of its record.
Old behaviour (before the fix): one provider error text with U+2028 split a record in two and
every later uncached request raised "is not a valid call-log line" (review r04 part 2).

FixtureProvider (see _fixture_provider.py); temporary directory only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c11_log_line_breaks.py
"""

import json
import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fixture_provider as fx  # noqa: E402
from learned_judgment.teacher_cache import parse_log_text  # noqa: E402

check = fx.Checks()
MESSAGE = "Fail to call predict: [HTTP 500] upstream said:\u2028retry later.\u2029next\u0085end \u00e9"


def run(teacher, seed: int):
    features, residuals, query = fx.arrays(seed=seed)
    teacher.fit(features, residuals)
    return teacher.predict_residual_distribution(query)


with tempfile.TemporaryDirectory() as tmp:
    def server_error(raw):
        raise RuntimeError(MESSAGE)

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=server_error)
    check.raises("a request whose provider error text holds U+2028 / U+2029 / U+0085", RuntimeError, lambda: run(teacher, 1))
    data = settings.call_log.read_bytes()
    check("the call log is pure ASCII", all(byte < 128 for byte in data))
    text = data.decode("ascii")
    check("records split on '\\n': 2, and str.splitlines() agrees", len(text.split("\n")) - 1 == 2 and len(text.splitlines()) == 2)
    records = teacher.call_log.read()
    check("read() returns dispatch, provider_error", [r["event"] for r in records] == ["dispatch", "provider_error"])
    check("the error text round-trips exactly", records[-1]["error"] == f"RuntimeError: {MESSAGE}")
    check("the ledger is readable: 10,000 tokens", teacher.call_log.tokens_quoted_for_billed_calls() == 10_000)
    provider.response_hook = None
    run(teacher, 2)
    check("the next uncached request (another key) is sent and completes", fx.events(teacher)[-1] == "completed")

    print("\n== a raw U+2028 inside an older line (written before the fix) ==")
    legacy = json.dumps({"event": "provider_error", "billable_dispatch": False, "error": "a\u2028b", "key": "legacy"},
                        ensure_ascii=False, separators=(",", ":"))
    with open(settings.call_log, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(legacy + "\n")
    records = teacher.call_log.read()
    check("it is read as one record with its text intact", records[-1]["error"] == "a\u2028b" and len(records) == 5)
    run(teacher, 3)
    check("later requests still work", fx.events(teacher)[-1] == "completed")

print("\n== parse_log_text rules ==")
check("'\\n'-only split keeps U+2028 inside a record", parse_log_text('{"a":"x\u2028y"}\n', "<text>") == [{"a": "x\u2028y"}])
check.raises("an unterminated last line", ValueError, lambda: parse_log_text('{"a":1}\n{"b":', "<text>"))
check.raises("a blank line", ValueError, lambda: parse_log_text('{"a":1}\n\n', "<text>"))

check.finish()
