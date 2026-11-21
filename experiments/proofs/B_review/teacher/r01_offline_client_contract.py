"""Review r01: offline check of the provider call sequence against the INSTALLED tabpfn-client.

No network: the provider is never connected; only the estimator constructor and its private
config builder run. Prints signatures and the exact tabpfn_config / predict params the pinned
client would send for the settings in config.yaml.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r01_offline_client_contract.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import tabpfn_client  # noqa: E402
from tabpfn_client import TabPFNRegressor, estimate_cost  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402
from tabpfn_client.utils import model_version_from_path  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import TabPFNProvider, TeacherSettings, _request_description  # noqa: E402

print("installed tabpfn_client", tabpfn_client.__version__)
init_params = inspect.signature(TabPFNRegressor.__init__).parameters
used = ["model_path", "n_estimators", "random_state", "categorical_features_indices",
        "ignore_pretraining_limits", "api_mode", "paper_version", "text_handling", "thinking_mode"]
print("constructor accepts every kwarg the provider passes:", all(name in init_params for name in used))
print("fit signature:", inspect.signature(TabPFNRegressor.fit))
print("predict signature:", inspect.signature(TabPFNRegressor.predict))
print("estimate_cost signature:", inspect.signature(estimate_cost))
print("ServiceClient.get_api_usage signature:", inspect.signature(ServiceClient.get_api_usage))
print("model_version_from_path signature:", inspect.signature(model_version_from_path))
print("has get_timings / get_access_token:", hasattr(TabPFNRegressor, "get_timings"), hasattr(tabpfn_client, "get_access_token"))

settings = TeacherSettings.from_config(load_config())
provider = TabPFNProvider(settings)
for indices in ([], [1], [1, 5]):
    sent = provider.outgoing_config(indices)
    print(f"outgoing tabpfn_config for categorical {indices}: {sent}")
regressor, sent = provider._build_regressor([1])
params = regressor._get_predict_params({"output_type": "main", "quantiles": list(settings.quantile_levels)})
print("predict_params the client would send:", params.model_dump(mode="json", exclude_none=True))
print("keyed description:", _request_description(settings, [1], ["a", "b"]))
print("model version used for the quote:", provider._model_version())
print("ServiceClient.base_url (not keyed):", ServiceClient.base_url)
