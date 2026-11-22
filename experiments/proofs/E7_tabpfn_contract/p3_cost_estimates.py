"""E7 proof 3: quote costs with tabpfn_client.estimate_cost (NON-BILLABLE per installed source).

Source evidence that the quote is free and sends shapes only:
  cost.py:43-49     "Estimate one operation without uploading data or consuming quota. Only raw
                     row/column counts and the supplied configuration are sent."
  cost.py:74-96     request = EstimateCostRequest(train_rows, test_rows, raw_columns, model_version,
                     operation, n_estimators, thinking_effort) -- built from .shape only
  client.py:3188-3198 POST /tabpfn/estimate_cost, docstring "never upload or reserve quota"
Executed check: usage is read before and after all quotes; it must not change.
Inputs are np.empty arrays: only their shapes are read by estimate_cost (cost.py:17-29).

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\E7_tabpfn_contract\\p3_cost_estimates.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # D:\BerlinPrior, for prior_labs

import prior_labs  # noqa: E402
import tabpfn_client  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402


def usage():
    d = ServiceClient.get_api_usage(tabpfn_client.get_access_token())
    return {k: d[k] for k in ("current_usage", "daily_tokens_used", "monthly_tokens_used", "usage_limit")}


def quote(train_rows, test_rows, n_features, **kw):
    X_train = np.empty((train_rows, n_features))
    X_test = None if test_rows is None else np.empty((test_rows, n_features))
    t0 = time.perf_counter()
    try:
        r = tabpfn_client.estimate_cost(X_train, X_test, **kw)
        dt = time.perf_counter() - t0
        inp = r.inputs.model_dump(mode="json", exclude_none=True)
        return {"estimated_cost": r.estimated_cost, "pricing_version": r.pricing_version,
                "resolved_inputs": inp, "rtt_s": round(dt, 3)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", "rtt_s": round(time.perf_counter() - t0, 3)}


print("utc_now =", datetime.now(timezone.utc).isoformat(timespec="seconds"))
print("tabpfn_client", tabpfn_client.__version__)
prior_labs.connect()
u0 = usage()
print("usage BEFORE quotes =", json.dumps(u0))

print("\n=== required shapes: operation='predict', model/n_estimators resolved by server ===")
print("(train_rows, query_rows, n_features) -> result")
REQUIRED = [(5, 8, 6), (20, 64, 8), (40, 64, 8), (40, 256, 10), (40, 2000, 10), (200, 320, 10), (40, 10000, 10)]
for tr, te, nf in REQUIRED:
    print(f"({tr}, {te}, {nf}) ->", json.dumps(quote(tr, te, nf)))

print("\n=== dependence on n_estimators, shape (40, 64, 8), predict ===")
for n in (1, 2, 4, 8):
    print(f"n_estimators={n} ->", json.dumps(quote(40, 64, 8, n_estimators=n)))

print("\n=== dependence on model_version, shape (40, 64, 8), predict ===")
for mv in ("v3.5", "v3.5-fast", "v3", "v2.6", "v2.5"):
    print(f"model_version={mv} ->", json.dumps(quote(40, 64, 8, model_version=mv)))

print("\n=== operation='cache_predict' (assumes fit_mode='fit_with_cache' hit), shape (40, 64, 8) ===")
print("cache_predict ->", json.dumps(quote(40, 64, 8, operation="cache_predict")))

print("\n=== batching: one request of C candidates vs C requests of 1 ===")
one = quote(40, 1, 8)
print("(40, 1, 8) ->", json.dumps(one))
for c in (16, 64):
    b = quote(40, c, 8)
    print(f"(40, {c}, 8) ->", json.dumps(b))
    if "estimated_cost" in one and "estimated_cost" in b:
        print(f"   {c} x single-row requests = {c * one['estimated_cost']}  vs  1 batched request = {b['estimated_cost']}")

print("\n=== minimum-charge probes ===")
for tr, te, nf in [(2, 1, 1), (5, 1, 6), (40, None, 8)]:
    print(f"({tr}, {te}, {nf}) ->", json.dumps(quote(tr, te, nf)))

print("\n=== feature-count dependence at (40, 64, F) ===")
for nf in (1, 6, 10, 20):
    print(f"(40, 64, {nf}) ->", json.dumps(quote(40, 64, nf)))

u1 = usage()
print("\nusage AFTER quotes =", json.dumps(u1))
print("usage unchanged by quotes:", u0 == u1)
print("DONE (no fit/predict/upload called)")
