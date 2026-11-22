"""Probe: what the shared spiking model's state dict holds, and whether a numpy round trip of
it rebuilds the model exactly (the shared student saves weights as an .npz, never a pickle).

FIXTURE CHECK - mechanics only. Random encoded inputs (fixture), a model from
learned_judgment.students.snn_train.build_model with config.yaml students.snn settings and
learn_beta / learn_threshold switched on once (so both buffer and parameter variants are seen).
Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\p00_state_dict_probe.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

import torch  # noqa: E402

settings = SNNSettings.from_config(load_config())
inputs = np.random.default_rng(3).normal(size=(40, 9))
for label, variant in (
    ("config settings", settings),
    ("learn_beta + learn_threshold", dataclasses.replace(settings, learn_beta=True, learn_threshold=True)),
):
    print(f"\n== {label} ==")
    model = snn_train.build_model(9, {"head_a": 5, "head_b": 7}, variant, seed=11)
    with torch.no_grad():
        model(torch.as_tensor(inputs), "head_a")  # a forward fills the non-persistent mem buffers
    state = model.state_dict()
    for key, value in state.items():
        kind = "parameter" if key in dict(model.named_parameters()) else "buffer"
        print(f"  {key:40s} {kind:9s} dtype={value.dtype} shape={tuple(value.shape)}")
    print(f"  'mem' in any key: {any(key.endswith('.mem') for key in state)}")
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in state.items()}
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        restored = {key: archive[key] for key in archive.files}
    print(f"  npz round trip keeps keys: {sorted(restored) == sorted(arrays)}; dtypes kept: "
          f"{all(restored[k].dtype == arrays[k].dtype for k in arrays)}")
    rebuilt = snn_train.build_model(9, {"head_a": 5, "head_b": 7}, variant, seed=999)
    result = rebuilt.load_state_dict({key: torch.as_tensor(value) for key, value in restored.items()}, strict=True)
    print(f"  load_state_dict(strict=True): {result}")
    rebuilt.eval()
    with torch.no_grad():
        for head in ("head_a", "head_b"):
            a = model(torch.as_tensor(inputs), head).numpy()
            b = rebuilt(torch.as_tensor(inputs), head).numpy()
            print(f"  head {head}: rebuilt logits identical: {np.array_equal(a, b)}")
    model.train()
    with torch.no_grad():
        same_mode = np.array_equal(model(torch.as_tensor(inputs), "head_a").numpy(), rebuilt(torch.as_tensor(inputs), "head_a").numpy())
    print(f"  train() vs eval() mode gives identical logits: {same_mode}")
    exported = snn_train.export_network(rebuilt, "head_b")
    print(f"  export of the rebuilt model: layer sizes {exported.layer_sizes}, digest == original export: "
          f"{exported.digest() == snn_train.export_network(model, 'head_b').digest()}")
