"""p06c: clean (sequential, otherwise idle machine) training-step wall time on the FIXTURE task.
Per config: 20 warm-up steps then 200 timed Adam steps (batch 64, CE, n_train=2000). Reports ms/step.
Usage: python p06c_train_step_time.py <num_threads> <configs: all|one>"""
import sys, time
import numpy as np, torch
from snn_common import SpikingStudent, Standardizer, make_fixture, train, N_IN, N_CLASSES
nt = int(sys.argv[1]); which = sys.argv[2]
torch.set_num_threads(nt)
X, y, q, _ = make_fixture(2000, seed=1); st = Standardizer(X)
cfgs = [(h, T) for h in [(32,), (64,), (64, 64)] for T in [5, 10, 25]] if which == "all" else [((64, 64), 10)]
for hidden, T in cfgs:
    torch.manual_seed(0)
    m = SpikingStudent(N_IN, N_CLASSES, hidden=hidden, T=T)
    train(m, st(X), y, q, kind="ce", steps=20, batch=64, seed=0)
    h = train(m, st(X), y, q, kind="ce", steps=200, batch=64, seed=1)
    print(f"threads={torch.get_num_threads():2d} hidden={str(hidden):9s} T={T:2d}: {1000*h['train_seconds']/h['steps']:6.1f} ms/step (batch 64)", flush=True)
