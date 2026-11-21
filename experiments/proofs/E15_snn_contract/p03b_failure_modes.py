"""p03b: failure modes on the FIXTURE task (mechanics only, not project results).

One factor at a time from a base config: n_train=2000, hidden=(64,), T=10, CE loss, beta=0.9, threshold=1.0,
input scale 1 (standardized), default nn.Linear init, default surrogate (atan alpha=2), 600 Adam steps,
batch 64, lr 2e-3, seed 0, 1 thread. For each variant: firing statistics at init, first-layer gradient
norm on one batch at init, loss first/last, hold-out agreement with the fixture label, final firing stats.
Usage: python p03b_failure_modes.py <worker_id> <n_workers> <out_jsonl>
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from snntorch import surrogate

from snn_common import SpikingStudent, Standardizer, evaluate, make_fixture, train, N_IN, N_CLASSES

torch.set_num_threads(1)
wid, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
X, y, q, _ = make_fixture(2000, seed=1)
X_te, y_te, q_te, _ = make_fixture(2000, seed=999)
st = Standardizer(X)

base = dict(hidden=(64,), T=10, beta=0.9, threshold=1.0, scale=1.0, w1_scale=1.0, sg="atan2", learn_beta=False)
variants = []
for s in [0.1, 0.3, 3.0, 10.0]:
    variants.append(("input_scale", dict(base, scale=s)))
for th in [0.1, 5.0, 20.0]:
    variants.append(("threshold", dict(base, threshold=th)))
for b in [0.5, 0.99]:
    variants.append(("beta", dict(base, beta=b)))
variants.append(("learn_beta", dict(base, learn_beta=True)))
variants.append(("learn_beta_beta0.5", dict(base, beta=0.5, learn_beta=True)))
for T in [1, 2, 5]:
    variants.append(("T", dict(base, T=T)))
for ws in [0.1, 10.0]:
    variants.append(("w1_init_scale", dict(base, w1_scale=ws)))
for sg in ["fs25", "fs5"]:
    variants.append(("surrogate", dict(base, sg=sg)))
variants.append(("surrogate_fs25_threshold5", dict(base, sg="fs25", threshold=5.0)))
variants.append(("two_layer", dict(base, hidden=(64, 64))))
variants.append(("two_layer_threshold5", dict(base, hidden=(64, 64), threshold=5.0)))
variants.append(("base", dict(base)))
SG = {"atan2": None, "fs25": surrogate.fast_sigmoid(slope=25), "fs5": surrogate.fast_sigmoid(slope=5)}

with open(out, "w", encoding="utf-8") as f:
    for k, (name, c) in enumerate(variants):
        if k % nw != wid:
            continue
        torch.manual_seed(0)
        m = SpikingStudent(N_IN, N_CLASSES, hidden=c["hidden"], T=c["T"], beta=c["beta"], threshold=c["threshold"],
                           spike_grad=SG[c["sg"]], learn_beta=c["learn_beta"])
        with torch.no_grad():
            m.fcs[0].weight.mul_(c["w1_scale"])
            m.fcs[0].bias.mul_(c["w1_scale"])
        Xtr = st(X) * c["scale"]
        Xte = st(X_te) * c["scale"]
        ev0 = evaluate(m, Xte, y_te)
        # first-layer grad norm on one batch at init
        xb = torch.as_tensor(Xtr[:64], dtype=torch.float32)
        loss0 = F.cross_entropy(m(xb), torch.as_tensor(y[:64]))
        m.zero_grad()
        loss0.backward()
        g1 = float(m.fcs[0].weight.grad.norm())
        gout = float(m.fc_out.weight.grad.norm())
        m.zero_grad()
        h = train(m, Xtr, y, q, kind="ce", steps=600, batch=64, lr=2e-3, seed=0)
        ev = evaluate(m, Xte, y_te)
        betas = [float(l.beta.detach().clamp(0, 1).mean()) for l in m.lifs] if c["learn_beta"] else None
        rec = dict(variant=name, cfg={k2: (list(v) if isinstance(v, tuple) else v) for k2, v in c.items()},
                   init_fire=[round(v, 4) for v in ev0["fire_rate_per_layer"]],
                   init_dead_frac=[round(v, 3) for v in ev0["dead_frac_per_layer"]],
                   init_sat_frac=[round(v, 3) for v in ev0["saturated_frac_per_layer"]],
                   init_grad_norm_W1=float(f"{g1:.3e}"), init_grad_norm_Wout=float(f"{gout:.3e}"),
                   loss_first=round(h["loss_first"], 4), loss_last_epoch_mean=round(h["loss_last_epoch_mean"], 4),
                   holdout_agree=round(ev["agree"], 4), fire=[round(v, 4) for v in ev["fire_rate_per_layer"]],
                   dead_frac=[round(v, 3) for v in ev["dead_frac_per_layer"]],
                   sat_frac=[round(v, 3) for v in ev["saturated_frac_per_layer"]],
                   learned_beta_effective=betas, train_seconds=round(h["train_seconds"], 1))
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps(rec), flush=True)
