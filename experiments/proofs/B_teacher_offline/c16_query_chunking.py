"""Check 16 (engine review "dream() sends n_dreams x C rows as one request", T9): query rows are
split into consecutive chunks of at most tabpfn.max_query_rows_per_request rows; each chunk is its
own request (key, gate, cache entry, log records) with the identical training context, and the
answers are concatenated in order.

Old behaviour (before the fix): every query matrix went out as one request, however large.

FixtureProvider (see _fixture_provider.py); the row cap is overridden to 7 for the check (it must
exceed Q = 5). Temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c16_query_chunking.py
"""

import dataclasses
import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _fixture_provider as fx  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import TeacherSettings  # noqa: E402

check = fx.Checks()
CAP = 7

print("== settings ==")
configured = TeacherSettings.from_config(load_config())
print(f"  config.yaml tabpfn.max_query_rows_per_request = {configured.max_query_rows_per_request}")
check.raises("a row cap equal to the number of quantile levels", ValueError,
             lambda: dataclasses.replace(configured, max_query_rows_per_request=len(configured.quantile_levels)))


def same_distribution(a, b) -> bool:
    return all(np.array_equal(getattr(a, name), getattr(b, name)) for name in ("mean", "median", "quantiles", "spread"))


def predict(tmp: Path, residuals, query, features, **overrides):
    teacher, provider, account, settings = fx.make_teacher(tmp, **overrides)
    teacher.fit(features, residuals)
    return teacher, account, teacher.predict_residual_distribution(query)


print("\n== stacked, 2 active outcomes, 10 query rows -> 20 stacked rows -> chunks 7, 7, 6 ==")
features, residuals, query = fx.arrays(m=10)
residuals[:, 2] = 0.5  # constant outcome: 2 active outcomes
with tempfile.TemporaryDirectory() as tmp:
    _, _, reference = predict(Path(tmp) / "one", residuals, query, features)
    teacher, account, chunked = predict(Path(tmp) / "chunked", residuals, query, features, max_query_rows_per_request=CAP)
    print("  sent rows per request:", [q.shape[0] for q in account.sent_queries])
    check("3 requests of 7, 7 and 6 rows", [q.shape[0] for q in account.sent_queries] == [7, 7, 6])
    check("3 keys listed, the last one is last_request_hash",
          len(teacher.last_request_hashes) == 3 and len(set(teacher.last_request_hashes)) == 3
          and teacher.last_request_hash == teacher.last_request_hashes[-1])
    check("the concatenated answer equals the unchunked one exactly", same_distribution(chunked, reference))
    first_x, first_y = account.sent_training[0]
    check("every chunk was sent the identical training context",
          all(np.array_equal(x, first_x) and np.array_equal(y, first_y) for x, y in account.sent_training))
    dispatches = [r for r in teacher.call_log.read() if r.get("billable_dispatch") is True]
    check("3 dispatch records with consecutive chunk bounds",
          [(r["chunk"]["index"], r["chunk"]["count"], r["chunk"]["first_row"], r["chunk"]["stop_row"]) for r in dispatches]
          == [(0, 3, 0, 7), (1, 3, 7, 14), (2, 3, 14, 20)])
    check("each chunk has its own cache entry", all(teacher.cache.get(key) is not None for key in teacher.last_request_hashes))
    check("tokens quoted: one quote per chunk (3 x 10,000)", teacher.call_log.tokens_quoted_for_billed_calls() == 30_000)
    again = teacher.predict_residual_distribution(query)
    check("a repeat is 3 cache hits and no billable call",
          fx.events(teacher)[-3:] == ["cache_hit"] * 3 and account.billed_predicts == 3 and same_distribution(again, reference))

print("\n== stacked, 3 active outcomes, 4 rows -> 12 stacked rows -> chunks 7 and 5 (5 = Q, padded to 6) ==")
features, residuals, query = fx.arrays(m=4)
with tempfile.TemporaryDirectory() as tmp:
    _, _, reference = predict(Path(tmp) / "one", residuals, query, features)
    teacher, account, chunked = predict(Path(tmp) / "chunked", residuals, query, features, max_query_rows_per_request=CAP)
    sent = [q.shape[0] for q in account.sent_queries]
    print("  sent rows per request:", sent)
    check("2 requests: 7 rows, then 5 rows padded to 6", sent == [7, 6])
    check("the concatenated answer equals the unchunked one exactly", same_distribution(chunked, reference))

print("\n== per_outcome, 3 outcomes, 10 rows -> per outcome chunks 7 and 3 ==")
features, residuals, query = fx.arrays(m=10)
with tempfile.TemporaryDirectory() as tmp:
    _, _, reference = predict(Path(tmp) / "one", residuals, query, features, multi_outcome_mode="per_outcome")
    teacher, account, chunked = predict(Path(tmp) / "chunked", residuals, query, features,
                                        multi_outcome_mode="per_outcome", max_query_rows_per_request=CAP)
    sent = [q.shape[0] for q in account.sent_queries]
    print("  sent rows per request:", sent)
    check("6 requests: 7 and 3 rows for each outcome", sent == [7, 3, 7, 3, 7, 3])
    check("6 keys listed", len(teacher.last_request_hashes) == 6)
    check("the concatenated answer equals the unchunked one exactly", same_distribution(chunked, reference))

check.finish()
