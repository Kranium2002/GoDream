"""E7 proof 1 (NO NETWORK): dump the tabpfn-client 0.10.1 contract from the installed package.

Any accidental HTTP request is forced to fail: TABPFN_CLIENT_API_URL points at a closed local
port before the package is imported (ServiceClient.base_url is bound at import time).
Only pure, client-side functions are executed. No token is read or printed.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\E7_tabpfn_contract\\p1_signatures_offline.py
"""

import os

os.environ["TABPFN_CLIENT_API_URL"] = "http://127.0.0.1:9"  # closed port: network is impossible
os.environ.pop("TABPFN_TOKEN", None)

import inspect
import io
import json
import warnings
from importlib.metadata import version
from typing import get_args

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import tabpfn_client
from tabpfn_client import (
    TabPFNClassifier,
    TabPFNRegressor,
    estimate_cost,
    get_api_usage,
    init,
)
from tabpfn_client import api_models as am
from tabpfn_client import client as cl
from tabpfn_client import constants, estimator as est
from tabpfn_client.client import ServiceClient, _serialize_to_parquet
from tabpfn_client.models import ClientOptions
from tabpfn_client.options import Options


def h(title):
    print(f"\n=== {title} ===")


h("versions")
print("tabpfn_client.__version__ =", tabpfn_client.__version__)
print("tabpfn-common-utils dist version =", version("tabpfn-common-utils"))
print("ServiceClient.base_url (forced dead) =", ServiceClient.base_url)
print("TORCH_AVAILABLE (estimator) =", est.TORCH_AVAILABLE)
try:
    import tabpfn  # noqa: F401

    print("tabpfn importable = True")
except ImportError:
    print("tabpfn importable = False")

h("signatures")
for name, obj in [
    ("TabPFNRegressor.__init__", TabPFNRegressor.__init__),
    ("TabPFNRegressor.fit", TabPFNRegressor.fit),
    ("TabPFNRegressor.predict", TabPFNRegressor.predict),
    ("TabPFNRegressor.submit_predict", TabPFNRegressor.submit_predict),
    ("TabPFNRegressor.save_model", TabPFNRegressor.save_model),
    ("TabPFNRegressor.load_model", TabPFNRegressor.load_model),
    ("TabPFNClassifier.__init__", TabPFNClassifier.__init__),
    ("TabPFNClassifier.fit", TabPFNClassifier.fit),
    ("TabPFNClassifier.predict", TabPFNClassifier.predict),
    ("TabPFNClassifier.predict_proba", TabPFNClassifier.predict_proba),
    ("tabpfn_client.estimate_cost", estimate_cost),
    ("tabpfn_client.get_api_usage", get_api_usage),
    ("tabpfn_client.init", init),
    ("ServiceClient.estimate_cost", ServiceClient.estimate_cost),
    ("ServiceClient.get_api_usage", ServiceClient.get_api_usage),
]:
    print(f"{name}{inspect.signature(obj)}")

h("legal output types (client-side Literal)")
print("RegressorOutputLiteral =", get_args(est.RegressorOutputLiteral))
print("ClassifierOutputLiteral =", get_args(est.ClassifierOutputLiteral))
print("RegressorOutputType enum =", [e.value for e in am.RegressorOutputType])
print("ClassifierOutputType enum (server schema) =", [e.value for e in am.ClassifierOutputType])
try:
    est._validate_output_type("bogus", est.RegressorOutputLiteral)
except ValueError as e:
    print("invalid output_type ->", type(e).__name__, ":", e)

h("model names / versions (client-side lists, no network)")
print("ModelVersion =", [v.value for v in am.ModelVersion])
print("_DEFAULT_MODEL_NAMES =", est._DEFAULT_MODEL_NAMES)
print("TabPFNRegressor.list_available_models() =", TabPFNRegressor.list_available_models())
print("TabPFNClassifier.list_available_models() =", TabPFNClassifier.list_available_models())
print("_AUTO_MODEL_PATH_ALIASES =", est._AUTO_MODEL_PATH_ALIASES)
r_v35 = TabPFNRegressor.create_default_for_version("v3.5")
print("create_default_for_version('v3.5').model_path =", r_v35.model_path)
print("QuotaOperation =", [o.value for o in am.QuotaOperation])
print("ThinkingEffort =", get_args(am.ThinkingEffort))

h("what is SENT: RegressorTabPFNConfig fields and default-constructed payload")
print("RegressorTabPFNConfig fields =", list(am.RegressorTabPFNConfig.model_fields))
print("ClassifierTabPFNConfig fields =", list(am.ClassifierTabPFNConfig.model_fields))
r = TabPFNRegressor()
cfg = r._get_tabpfn_config()
print("TabPFNRegressor() tabpfn_config sent (exclude_none) =",
      json.dumps(cfg.model_dump(mode="json", exclude_none=True)))
print("RegressorFitTaskConfig sent =",
      json.dumps(est._build_fit_task_config(cfg).model_dump(mode="json", exclude_none=True)))
print("tabpfn_systems default =", est._build_tabpfn_systems(False, False),
      "| paper_version=True ->", est._build_tabpfn_systems(True, False))
r2 = TabPFNRegressor(model_path="v3.5_default", n_estimators=8, random_state=42,
                     categorical_features_indices=[0], softmax_temperature=0.9)
print("example custom tabpfn_config sent =",
      json.dumps(r2._get_tabpfn_config().model_dump(mode="json", exclude_none=True)))
print("params NOT in tabpfn_config (client-side only / other request fields) =",
      sorted(set(r.get_params()) - set(am.RegressorTabPFNConfig.model_fields)))

h("predict params payload (what one predict request carries)")
print("RegressorPredictParams fields =", list(am.RegressorPredictParams.model_fields))
for kw in [
    {"output_type": "main", "quantiles": [0.1, 0.5, 0.9]},
    {"output_type": "quantiles", "quantiles": None},
    {"output_type": "median", "quantiles": None},
]:
    pp = r._get_predict_params(kw)
    task = am.RegressorConfig(tabpfn_config=cfg, predict_params=pp)
    print(f"{kw} -> task_config =", json.dumps(task.model_dump(mode="json", exclude_none=True)))

h("_finish_predict shaping (pure; synthetic PredictionResult, no server)")
from tabpfn_client.models import PredictionResult  # noqa: E402

q_arr = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])  # server sends (n_quantiles, n_rows)
out = r._finish_predict([PredictionResult(y_pred=q_arr)], "quantiles")
print("quantiles: server (2,3) array ->", type(out).__name__, "len", len(out), "elem shape", out[0].shape)
main_like = cl.ServiceClient._prediction_result(
    {"mean": [1.0, 2.0, 3.0], "median": [1.0, 2.0, 3.0], "mode": [1.0, 2.0, 3.0],
     "quantiles": [[0.5, 1.5, 2.5], [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]},
    am.RegressorMetadata(test_set_num_rows=3, test_set_num_cols=2, package_version="x",
                         tabpfn_config=am.RegressorTabPFNConfig()),
    None, None)
out = r._finish_predict([main_like], "main")
print("main (synthetic dict as server would send) ->", type(out).__name__,
      {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in out.items()})
print("metadata dict keys kept on estimator._last_meta =", sorted(main_like.metadata))
print("_FULL_OUTPUT_ROW_AXIS =", est._FULL_OUTPUT_ROW_AXIS)

h("targets: 1-D only, NaN rejected (pure _validate_targets)")
for label, y in [
    ("y 1-D float", np.array([0.1, 0.2, 0.3])),
    ("y (n,1)", np.array([[0.1], [0.2], [0.3]])),
    ("y (n,2) multi-target", np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])),
    ("y with NaN", np.array([0.1, np.nan, 0.3])),
]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r._validate_targets(y)
            print(f"{label}: accepted", f"(warning: {w[0].category.__name__}: {w[0].message})" if w else "")
        except Exception as e:  # noqa: BLE001
            print(f"{label}: {type(e).__name__}: {e}")
print("MIN_TRAIN_ROWS =", constants.MIN_TRAIN_ROWS)

h("categoricals / text: client-side transform and the parquet actually uploaded")
X = pd.DataFrame({
    "side_cat": pd.Categorical(["left", "right", "left", "none"], categories=["left", "none", "right"]),
    "side_str": ["left, hard", "right", "left", "none"],
    "side_int": np.array([0, 2, 0, 1], dtype=np.int64),
    "speed": np.array([0.3, 0.6, np.nan, 0.9], dtype=np.float64),
})
Xc = est._clean_text_features(X)
print("input dtypes =", dict(X.dtypes.astype(str)))
print("after _clean_text_features dtypes =", dict(Xc.dtypes.astype(str)))
print("side_str after clean =", Xc["side_str"].tolist(), "(commas stripped, whitespace collapsed)")
print("side_cat after clean =", Xc["side_cat"].tolist())
raw, crc = _serialize_to_parquet(Xc)
tbl = pq.read_table(io.BytesIO(raw))
print("uploaded parquet schema:")
for f in tbl.schema:
    print("   ", f.name, ":", f.type)
print("uploaded parquet bytes =", len(raw), "| crc32c(base64, used as dedup hash) =", crc)
rt = pd.read_parquet(io.BytesIO(raw))
print("round-trip pandas dtypes =", dict(rt.dtypes.astype(str)), "| speed NaN kept:", bool(rt["speed"].isna().iloc[2]))
Xn = np.array([[0.3, 1.0], [0.6, np.nan]])
print("numeric numpy passes through unchanged:", est._clean_text_features(Xn) is Xn)
raw2, crc2 = _serialize_to_parquet(pd.DataFrame(Xn))
print("numpy -> DataFrame column names in parquet =", pq.read_table(io.BytesIO(raw2)).schema.names)

h("identical inputs -> identical upload bytes/hash (basis of server-side upload dedup)")
_, crc_a = _serialize_to_parquet(Xc)
_, crc_b = _serialize_to_parquet(Xc.copy())
print("crc equal for identical frame:", crc_a == crc_b)

h("client limits / timeouts / retry constants (source constants)")
o = Options()
print("Options defaults: TABPFN_CLIENT_TIMEOUT =", o.TABPFN_CLIENT_TIMEOUT,
      "| UPLOAD_TIMEOUT =", o.TABPFN_CLIENT_UPLOAD_TIMEOUT,
      "| DEDUP_DATASETS =", o.TABPFN_CLIENT_DEDUP_DATASETS,
      "| FORCE_REUPLOAD =", o.TABPFN_CLIENT_FORCE_REUPLOAD,
      "| ASYNC_USE_ABOVE_TRAINSET_SIZE =", o.TABPFN_CLIENT_ASYNC_USE_ABOVE_TRAINSET_SIZE,
      "| ASYNC_POLL_TIMEOUT =", o.TABPFN_CLIENT_ASYNC_POLL_TIMEOUT)
print("ClientOptions().timeout =", ClientOptions().timeout)
print("FALLBACK_PREDICT_ROW_PAIRS_BUDGET =", est.FALLBACK_PREDICT_ROW_PAIRS_BUDGET)
print("_MAX_PREDICT_JOBS_IN_FLIGHT =", cl._MAX_PREDICT_JOBS_IN_FLIGHT,
      "| _MAX_POLL_TIMEOUT_SECS =", cl._MAX_POLL_TIMEOUT_SECS,
      "| _MIN_RETRY_INTERVAL_SECS =", cl._MIN_RETRY_INTERVAL_SECS)
print("_REFIT_ERROR_CODES =", sorted(cl._REFIT_ERROR_CODES))
print("CACHE_DIR (token cache location; contents never read) =", constants.CACHE_DIR,
      "| exists:", constants.CACHE_DIR.exists())

h("retry decorators on billable calls (backoff config read from wrapped functions)")
for fn_name in ["_fit", "_submit_fit_job", "_predict", "_submit_predict_job"]:
    src = inspect.getsource(getattr(ServiceClient, fn_name))
    head = src.split("def ")[0]
    exc = head[head.find("(", head.find("on_exception")) + 1: head.find("max_tries")]
    exc = " ".join(exc.split())
    tries = head[head.find("max_tries"):].split(",")[0]
    print(f"ServiceClient.{fn_name}: retries on {exc} {tries}")

h("estimate_cost request model (what the quote sends)")
print("EstimateCostRequest fields =", list(am.EstimateCostRequest.model_fields))
print("EstimateCostResponse fields =", list(am.EstimateCostResponse.model_fields))
print("APIUsageData fields =", list(am.APIUsageData.model_fields))
print("RegressorMetadata fields =", list(am.RegressorMetadata.model_fields))
print("ModelLimit fields =", list(am.ModelLimit.model_fields))
print("\nDONE (no network used)")
