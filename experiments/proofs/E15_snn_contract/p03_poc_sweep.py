"""p03: proof-of-concept training sweep on the FIXTURE task (mechanics only, not project results).

Usage: python p03_poc_sweep.py <worker_id> <n_workers> <out_jsonl>
Grid: n_train {200,1000,2000} x hidden {(32,),(64,),(64,64)} x T {5,10,25} x loss {ce, kl} = 54 runs,
split round-robin across workers. Each process uses torch.set_num_threads(1).
Fixed: steps=1500 Adam steps, batch=64 (min(64,n)), lr=2e-3, beta=0.9, threshold=1.0, readout_beta=0.9,
spike_grad=None (snnTorch default atan(alpha=2)), seed=0. Train data: first n rows of fixture seed 1;
hold-out: 2000 rows of fixture seed 999.
"""
import itertools
import json
import os
import sys
import time

import torch

from snn_common import SpikingStudent, Standardizer, evaluate, make_fixture, train, N_IN, N_CLASSES

torch.set_num_threads(1)
wid, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
X_all, y_all, q_all, _ = make_fixture(2000, seed=1)
X_te, y_te, q_te, _ = make_fixture(2000, seed=999)
grid = list(itertools.product([200, 1000, 2000], [(32,), (64,), (64, 64)], [5, 10, 25], ["ce", "kl"]))
with open(out, "w", encoding="utf-8") as f:
    for k, (n, hidden, T, kind) in enumerate(grid):
        if k % nw != wid:
            continue
        X, y, q = X_all[:n], y_all[:n], q_all[:n]
        st = Standardizer(X)
        torch.manual_seed(0)
        m = SpikingStudent(N_IN, N_CLASSES, hidden=hidden, T=T)
        ev0 = evaluate(m, st(X_te), y_te, q_te)
        h = train(m, st(X), y, q, kind=kind, steps=1500, batch=64, lr=2e-3, seed=0)
        ev_tr = evaluate(m, st(X), y, q)
        ev_te = evaluate(m, st(X_te), y_te, q_te)
        rec = dict(n_train=n, hidden=list(hidden), T=T, loss=kind, steps=h["steps"], epochs=h["epochs"], batch=h["batch"],
                   threads=torch.get_num_threads(), train_seconds=round(h["train_seconds"], 2),
                   loss_first=round(h["loss_first"], 4), loss_last_epoch_mean=round(h["loss_last_epoch_mean"], 4),
                   init_holdout_agree=round(ev0["agree"], 4), init_fire=[round(v, 3) for v in ev0["fire_rate_per_layer"]],
                   train_agree=round(ev_tr["agree"], 4), holdout_agree=round(ev_te["agree"], 4),
                   holdout_kl=round(ev_te["kl_to_soft"], 4),
                   fire=[round(v, 3) for v in ev_te["fire_rate_per_layer"]],
                   dead_frac=[round(v, 3) for v in ev_te["dead_frac_per_layer"]],
                   sat_frac=[round(v, 3) for v in ev_te["saturated_frac_per_layer"]])
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps(rec), flush=True)
