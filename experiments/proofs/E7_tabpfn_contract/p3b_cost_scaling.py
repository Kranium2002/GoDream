"""E7 proof 3b: where does a quota_v3 quote rise above the 10,000-token floor? (NON-BILLABLE)

Same free endpoint as p3 (POST /tabpfn/estimate_cost, shapes only). Usage read before/after.
Relevant for dream batches (many situations x candidates in one request).

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\E7_tabpfn_contract\\p3b_cost_scaling.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import prior_labs  # noqa: E402
import tabpfn_client  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402


def usage():
    d = ServiceClient.get_api_usage(tabpfn_client.get_access_token())
    return d["current_usage"]


def q(tr, te, nf, **kw):
    try:
        r = tabpfn_client.estimate_cost(np.empty((tr, nf)), np.empty((te, nf)), **kw)
        return r.estimated_cost, r.pricing_version, r.inputs.n_estimators
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:200]}"


prior_labs.connect()
u0 = usage()
print("current_usage BEFORE =", u0)
print("(train_rows, test_rows, n_features[, kw]) -> (estimated_cost, pricing_version, n_estimators)")
for args, kw in [
    ((40, 32_000, 10), {}),
    ((40, 100_000, 10), {}),
    ((40, 300_000, 10), {}),
    ((40, 1_000_000, 10), {}),
    ((1_000, 10_000, 10), {}),
    ((10_000, 10_000, 10), {}),
    ((10_000, 100_000, 10), {}),
    ((100_000, 100_000, 10), {}),
    ((100_000, 100_000, 10), {"n_estimators": 1}),
    ((100_000, 100_000, 50), {}),
    ((100_000, 100_000, 10), {"operation": "cache_predict"}),
]:
    print(args, kw or "", "->", q(*args, **kw))
u1 = usage()
print("current_usage AFTER =", u1, "| unchanged:", u0 == u1)
print("DONE (no fit/predict/upload called)")
