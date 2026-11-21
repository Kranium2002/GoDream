"""p04: bitwise determinism of training on the FIXTURE task.
Usage: python p04_determinism.py <num_threads> <deterministic 0|1>
Trains hidden (64,64), T=10, n=1000, CE, 300 steps, batch 64, seed 0, then prints sha256 over the
concatenated state_dict bytes (keys + raw tensor bytes, in state_dict order) and of the hold-out logits.
"""
import hashlib
import sys

import numpy as np
import torch

from snn_common import SpikingStudent, Standardizer, make_fixture, state_sha256, train, N_IN, N_CLASSES

nt, det = int(sys.argv[1]), bool(int(sys.argv[2]))
torch.set_num_threads(nt)
torch.use_deterministic_algorithms(det)
X, y, q, _ = make_fixture(1000, seed=1)
st = Standardizer(X)
torch.manual_seed(0)
m = SpikingStudent(N_IN, N_CLASSES, hidden=(64, 64), T=10)
h = train(m, st(X), y, q, kind="ce", steps=300, batch=64, lr=2e-3, seed=0)
X_te, _, _, _ = make_fixture(500, seed=999)
with torch.no_grad():
    lg = m(torch.as_tensor(st(X_te), dtype=torch.float32)).numpy()
print(f"threads={torch.get_num_threads()} deterministic={torch.are_deterministic_algorithms_enabled()} "
      f"final_loss={h['loss_last_epoch_mean']:.6f} state_sha256={state_sha256(m)} "
      f"holdout_logits_sha256={hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest()}")
