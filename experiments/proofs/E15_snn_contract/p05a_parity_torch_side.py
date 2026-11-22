"""p05a: train a FIXTURE spiking student, export weights (.npz float32 and float64), and save torch outputs
(logits and every hidden spike) on 1000 FIXTURE inputs for the numpy parity check in p05b (which never imports torch).
Model: hidden (64,64), T=10, CE, 600 steps, seed 0, 1 thread.
"""
import copy

import numpy as np
import torch

from snn_common import SpikingStudent, Standardizer, export_npz, make_fixture, train, N_IN, N_CLASSES, evaluate

torch.set_num_threads(1)
X, y, q, _ = make_fixture(2000, seed=1)
st = Standardizer(X)
torch.manual_seed(0)
m = SpikingStudent(N_IN, N_CLASSES, hidden=(64, 64), T=10)
h = train(m, st(X), y, q, kind="ce", steps=600, batch=64, lr=2e-3, seed=0)
print("trained: loss first", round(h["loss_first"], 4), "last-epoch mean", round(h["loss_last_epoch_mean"], 4), "seconds", round(h["train_seconds"], 1))
torch.save(m.state_dict(), "p05_model_f32_state_dict.pt")
export_npz(m, "p05_params_f32.npz", st.mean, st.std)

X_ev, y_ev, _, _ = make_fixture(1000, seed=4242)
xs = st(X_ev)  # float64 standardized
m.eval()
with torch.no_grad():
    lg32, rec32 = m(torch.as_tensor(xs, dtype=torch.float32), record=True)
    single32 = torch.cat([m(torch.as_tensor(xs[i:i + 1], dtype=torch.float32)) for i in range(len(xs))])
print("float32 torch: batched vs single-row logits max|diff| =", float((lg32 - single32).abs().max()),
      " argmax agree =", float((lg32.argmax(1) == single32.argmax(1)).float().mean()))

m64 = SpikingStudent(N_IN, N_CLASSES, hidden=(64, 64), T=10)
m64.load_state_dict(m.state_dict())
m64 = m64.double().eval()
print("m64 beta dtype/value:", m64.lifs[0].beta.dtype, repr(float(m64.lifs[0].beta)))
export_npz(m64, "p05_params_f64.npz", st.mean, st.std)
with torch.no_grad():
    lg64, rec64 = m64(torch.as_tensor(xs, dtype=torch.float64), record=True)
print("torch float32 vs float64 argmax agreement:", float((lg32.argmax(1) == lg64.argmax(1)).float().mean()),
      " spike mismatches f32 vs f64:", int(sum(int((a.double() != b).sum()) for a, b in zip(rec32, rec64))))
print("hold-out (fixture) agreement f32:", float((lg32.argmax(1).numpy() == y_ev).mean()))
np.savez("p05_torch_outputs.npz", xs=xs, y=y_ev,
         logits32=lg32.numpy(), logits64=lg64.numpy(), single32=single32.numpy(),
         **{f"spk32_{i}": r.numpy().astype(np.uint8) for i, r in enumerate(rec32)},
         **{f"spk64_{i}": r.numpy().astype(np.uint8) for i, r in enumerate(rec64)})
print("saved p05_torch_outputs.npz, p05_params_f32.npz, p05_params_f64.npz, p05_model_f32_state_dict.pt")
