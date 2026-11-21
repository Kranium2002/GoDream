"""Check 12 (review T5): a request with as many query rows as quantile levels is padded, so the
quantile layout is unambiguous by shape; the median vs quantile(0.5) gap is recorded, not enforced.

Old behaviour (before the fix): with m == Q rows a transposed (m, Q) quantile array passed
validation and wrong quantiles were cached (review r06).

FixtureProvider (see _fixture_provider.py); temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c12_square_query.py
"""

import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _fixture_provider as fx  # noqa: E402
from learned_judgment.teacher import ProviderResponseError, parse_main_output, request_key, stack_training  # noqa: E402

check = fx.Checks()
features, residuals, _ = fx.arrays(n=8)
residuals[:, 1] = 0.25  # constant outcomes: one active outcome, so the stacked query has m rows
residuals[:, 2] = -1.0
_, _, query6 = fx.arrays(seed=9, m=6)
query = np.ascontiguousarray(query6[:5])  # m = 5 = Q

with tempfile.TemporaryDirectory() as tmp:
    settings = fx.settings_for(Path(tmp))
    Q = len(settings.quantile_levels)
    print(f"== pure rules (Q = {Q}) ==")
    check.raises("parse_main_output with n_rows == Q", ValueError,
                 lambda: parse_main_output({}, n_rows=Q, quantile_levels=settings.quantile_levels))
    provider = fx.FixtureProvider(settings, fx.Account())
    check.raises("provider.validate_request with Q query rows", ValueError,
                 lambda: provider.validate_request(features, residuals[:, 0].copy(), query,
                                                   categorical_indices=(1,), quantile_levels=settings.quantile_levels))
    check("no billable call was started", provider.prediction_calls_started == 0)

print("\n== the teacher pads a Q-row request by one duplicated row ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    teacher.fit(features, residuals)
    distribution = teacher.predict_residual_distribution(query)
    sent = account.sent_queries[0]
    check("6 rows were sent: the 5 query rows plus the last one again",
          sent.shape == (6, 6) and np.array_equal(sent[:5], query) and np.array_equal(sent[5], query[4]))
    check("5 answer rows are returned", distribution.median.shape == (5, 3) and distribution.quantiles.shape == (5, 3, Q))
    x_train, y_train, _, _ = stack_training(features, residuals, (0,))
    padded_key = request_key(settings=settings, x_train=x_train, y_train=y_train, x_query=sent,
                             categorical_indices=(1,), base_url=provider.base_url())
    check("the key covers the padded matrix", teacher.last_request_hash == padded_key)
    entry = teacher.cache.get(padded_key)
    check("the entry stores the 6 sent rows and 6 answer rows",
          entry.request_arrays["x_query"].shape == (6, 6) and entry.response_arrays["quantiles"].shape == (6, Q))
    check("the entry records padded_rows 1", entry.request_meta["padded_rows"] == 1)
    check("the response metadata records max|median - q0.5| (0.0 for the fixture)",
          entry.response_meta["median_vs_q50_max_abs_diff"] == 0.0)
    record = teacher.call_log.read()[-1]
    check("the completed record says padded_rows 1, n_query 6", record["padded_rows"] == 1 and record["n_query"] == 6)
    again = teacher.predict_residual_distribution(query)
    check("a repeat is a cache hit with the same 5 rows",
          fx.events(teacher)[-1] == "cache_hit" and np.array_equal(again.quantiles, distribution.quantiles))

    print("\n== the same 5 rows inside an unpadded 6-row request give the same answers ==")
    reference = teacher.predict_residual_distribution(query6)
    check("rows 0..4 equal", np.array_equal(reference.quantiles[:5], distribution.quantiles)
          and np.array_equal(reference.median[:5], distribution.median))

print("\n== a transposed answer is now rejected by shape (and kept unvalidated) ==")
with tempfile.TemporaryDirectory() as tmp:
    def transposed(raw):
        raw["quantiles"] = np.ascontiguousarray(raw["quantiles"].T)  # (n, Q) instead of (Q, n)
        return raw

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=transposed)
    teacher.fit(features, residuals)
    check.raises("a Q-row request answered in the transposed layout", ProviderResponseError,
                 lambda: teacher.predict_residual_distribution(query))
    key = teacher.call_log.read()[0]["key"]
    stored = teacher.cache.get_unvalidated(key)
    check("the raw (6, 5) answer is on disk", stored is not None and stored.raw_response["quantiles"].shape == (6, Q))
    check("nothing was cached as valid", teacher.cache.get(key) is None)

print("\n== per_outcome mode pads each Q-row request ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp), multi_outcome_mode="per_outcome")
    features2, residuals2, _ = fx.arrays(seed=3)
    teacher.fit(features2, residuals2)
    teacher.predict_residual_distribution(query)
    check("3 requests of 6 sent rows each", [q.shape[0] for q in account.sent_queries] == [6, 6, 6])

print("\n== a median that differs from quantile(0.5) is recorded, not rejected ==")
with tempfile.TemporaryDirectory() as tmp:
    def shifted_median(raw):
        raw["median"] = raw["median"] + 0.25
        return raw

    teacher, provider, account, settings = fx.make_teacher(Path(tmp), response_hook=shifted_median)
    teacher.fit(features, residuals)
    teacher.predict_residual_distribution(query)
    entry = teacher.cache.get(teacher.last_request_hash)
    gap = entry.response_meta["median_vs_q50_max_abs_diff"]
    print(f"  recorded gap: {gap!r}")
    check("the gap (0.25) is stored in the response metadata and the request completed",
          abs(gap - 0.25) < 1e-12 and fx.events(teacher)[-1] == "completed")

check.finish()
