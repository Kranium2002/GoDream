"""Review r04: project-budget accounting when the call log is missing, partial, or holds a Unicode line break.

FixtureProvider (see _fixture.py) replaces only the network; the real teacher gate and CallLog run.
Temporary directory only.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r04_call_log_budget.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import tempfile
from pathlib import Path

import _fixture as fx
from learned_judgment.teacher import BudgetError


def run(teacher, seed):
    features, residuals, query = fx.arrays(seed=seed)
    teacher.fit(features, residuals)
    teacher.predict_residual_distribution(query)


print("== 1. call log missing: the project budget silently restarts at zero ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp), budget=30_000)
    for seed in (1, 2, 3):
        run(teacher, seed)
    print("  after 3 requests: tokens quoted", log.tokens_quoted_for_billed_calls(), "| billed", account.billed_predicts,
          "| cache entries", len(list(settings.cache_dir.glob("*.json"))))
    try:
        run(teacher, 4)
    except BudgetError as exc:
        print("  4th request refused as expected:", str(exc).split(": ", 1)[1][:90])
    settings.call_log.unlink()  # e.g. results/ cleaned before a fresh evaluation run
    run(teacher, 4)
    print("  log removed -> 4th request sent; tokens quoted now", log.tokens_quoted_for_billed_calls(),
          "| billed in total", account.billed_predicts, "against project_token_budget 30000")

print("\n== 2. a provider error message containing U+2028 makes the log unreadable ==")
with tempfile.TemporaryDirectory() as tmp:
    def server_error(raw):
        raise RuntimeError("Fail to call predict: [HTTP 500] upstream said:\u2028retry later.")

    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp), response_hook=server_error)
    try:
        run(teacher, 1)
    except RuntimeError as exc:
        print("  provider error recorded:", repr(str(exc))[:70])
    text = settings.call_log.read_text(encoding="utf-8")
    print("  physical records split on '\\n':", len([l for l in text.split("\n") if l]),
          "| str.splitlines() pieces:", len(text.splitlines()))
    provider.response_hook = None
    try:
        run(teacher, 2)
    except ValueError as exc:
        print("  next uncached request:", type(exc).__name__, str(exc).replace(tmp, "<tmp>")[:110])

print("\n== 3. a partial last line (crash during append) ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, log, settings = fx.make_teacher(Path(tmp))
    run(teacher, 1)
    with open(settings.call_log, "a", encoding="utf-8", newline="\n") as stream:
        stream.write('{"billable_dispatch":true,"quote_tok')  # torn write
    run(teacher, 1)  # cache hit: appends onto the torn line
    try:
        run(teacher, 2)
    except ValueError as exc:
        print("  next uncached request:", type(exc).__name__, str(exc).replace(tmp, "<tmp>")[:110])
    print("  billed in total:", account.billed_predicts, "(fails closed: no extra billing)")
