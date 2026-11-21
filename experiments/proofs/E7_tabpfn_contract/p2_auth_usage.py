"""E7 proof 2: authenticate via the project helper and read API usage (NON-BILLABLE).

Requests made (all read-only, per installed source):
  prior_labs.connect() -> tabpfn_client.init():
      GET  /protected/                    token check        (client.py:2947-2969)
      GET  /retrieve_greeting_messages/                       (client.py:3089-3100)
      GET  /tabpfn/get_settings           model limits       (client.py:547-568)
  tabpfn_client.get_api_usage():
      POST /get_api_usage/                usage read         (config.py:161-170, client.py:3201-3215)
  ServiceClient.get_api_usage(token):     same endpoint, raw dict
No fit / predict / upload endpoint is called. The token is never printed; unknown usage fields
are redacted (only allowlisted numeric/time fields are shown).

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\E7_tabpfn_contract\\p2_auth_usage.py
"""

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # D:\BerlinPrior, for prior_labs

warnings.simplefilter("always")
_caught = []
_orig_showwarning = warnings.showwarning
warnings.showwarning = lambda m, c, f, l, file=None, line=None: _caught.append(f"{c.__name__}: {m}")

import prior_labs  # noqa: E402
import tabpfn_client  # noqa: E402
from tabpfn_client.api_models import APIUsageData  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402

print("utc_now =", datetime.now(timezone.utc).isoformat(timespec="seconds"))
print("tabpfn_client", tabpfn_client.__version__, "| base_url =", ServiceClient.base_url)

t0 = time.perf_counter()
prior_labs.connect()
print(f"prior_labs.connect(): OK in {time.perf_counter() - t0:.2f}s "
      f"(Config.is_initialized={tabpfn_client.config.Config.is_initialized})")

ALLOW = set(APIUsageData.model_fields)  # numeric counters, limits and reset timestamps only

t0 = time.perf_counter()
msg = tabpfn_client.get_api_usage()
print(f"\ntabpfn_client.get_api_usage() -> {type(msg).__name__} in {time.perf_counter() - t0:.2f}s")
print("   ", msg)

t0 = time.perf_counter()
raw = ServiceClient.get_api_usage(tabpfn_client.get_access_token())
print(f"\nServiceClient.get_api_usage(token) -> {type(raw).__name__} in {time.perf_counter() - t0:.2f}s")
print("response keys =", sorted(raw))
for k in sorted(raw):
    v = raw[k]
    shown = v if k in ALLOW else f"<redacted {type(v).__name__}>"
    print(f"   {k} = {shown}")
try:
    parsed = APIUsageData.model_validate(raw)
    print("validates as api_models.APIUsageData: True")
    print("unlimited marker: usage_limit == -1 ->", parsed.usage_limit == -1)
except Exception as e:  # noqa: BLE001
    print("validates as api_models.APIUsageData: False ->", type(e).__name__, str(e)[:300])

s = ServiceClient._api_settings  # cached by init(); no extra request
print("\n=== server settings cached by init() (GET /tabpfn/get_settings; no account data) ===")
if s is None:
    print("settings: None")
else:
    print("default_model_version =", getattr(s.default_model_version, "value", s.default_model_version))
    print("thinking_default_model_version =", s.thinking_default_model_version)
    print("dataset_max_size_bytes =", s.dataset_max_size_bytes)
    print("inline_test_set_max_parquet_bytes =", s.inline_test_set_max_parquet_bytes)
    print("async_settings =", json.dumps(s.async_settings.model_dump(mode="json")))
    print("max_model_limit =", json.dumps(s.max_model_limit.model_dump(mode="json")))
    for ver, lim in sorted(s.model_limits.items(), key=lambda kv: str(kv[0])):
        print(f"model_limits[{getattr(ver, 'value', ver)}] =", json.dumps(lim.model_dump(mode="json")))
    print("thinking_max_timeout_secs =", s.thinking_max_timeout_secs)
    if s.thinking_model_limits:
        for ver, lim in sorted(s.thinking_model_limits.items(), key=lambda kv: str(kv[0])):
            d = lim.model_dump(mode="json")
            print(f"thinking_model_limits[{getattr(ver, 'value', ver)}] efforts =", d.get("efforts"),
                  "| max_timeout_secs =", d.get("max_timeout_secs"))

print("\nwarnings captured =", _caught or "none")
print("DONE (no fit/predict/upload called)")
