"""p01: where does the ~1e-15 logit gap between the torch model and the numpy network come from?

FIXTURE CHECK - mechanics only. An untrained model (seed 3) on the fixture dream's encoded
situations, and random 0/1 spike patterns. Compares torch's Linear (MKL) with numpy's matmul
(OpenBLAS) for the first-layer current (D = 4 inputs) and for the readout current (64 spikes).

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\p01_matmul_rounding_probe.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fixture_snn_spec import FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, build_model, export_network  # noqa: E402

config = load_config()
settings = SNNSettings.from_config(config)
spec = FixtureSnnSpec()
inputs = InputEncoder.from_variables(spec.situation).encode(fixture_dreams(spec, config).situations)
model = build_model(inputs.shape[1], {"h": 35}, settings, seed=3)
network = export_network(model, "h")
with torch.no_grad():
    torch_current = model.trunk_linears[0](torch.as_tensor(inputs)).numpy()
numpy_current = inputs @ network.hidden_weights[0].T + network.hidden_biases[0]
print(
    f"first-layer current (D={inputs.shape[1]}): max |torch - numpy| = {np.abs(torch_current - numpy_current).max():.3e}, "
    f"bitwise equal {np.array_equal(torch_current, numpy_current)}"
)
spikes = (np.random.default_rng(0).uniform(size=(1000, settings.hidden_sizes[-1])) > 0.7).astype(np.float64)
with torch.no_grad():
    torch_readout = model.head_linears["h"](torch.as_tensor(spikes)).numpy()
numpy_readout = spikes @ network.readout_weight.T + network.readout_bias
print(
    f"readout current ({settings.hidden_sizes[-1]} spikes -> 35): max |torch - numpy| = "
    f"{np.abs(torch_readout - numpy_readout).max():.3e}, bitwise equal {np.array_equal(torch_readout, numpy_readout)}"
)
print("torch BLAS:", [line.strip() for line in torch.__config__.show().splitlines() if "Math Kernel" in line])
print("numpy BLAS:", np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"])
