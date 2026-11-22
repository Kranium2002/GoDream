"""E15 shared code: FIXTURE task, spiking student model (snnTorch), training loop, .npz export.

EVERYTHING HERE IS A MECHANICS FIXTURE. The fixture task below is an analytic toy function chosen
by this research agent. Numbers produced with it are API/mechanics evidence only. They are NOT
evidence about the Go2 robot, the MuJoCo simulator, TabPFN, or the final project student.

Fixture task (documented, deterministic given the seed):
  * inputs x in R^8, x ~ N(0, I_8)  (then standardized with train mean/std, as real features would be)
  * C = 64 classes on an 8x8 action grid: class k = i*8 + j  <->  a_k = (u_i, v_j),
    u_i = linspace(-1, 1, 8)[i], v_j = linspace(-1, 1, 8)[j]
  * two fixed direction vectors w_u, w_v ~ N(0, 0.3^2 I_8) drawn from np.random.default_rng(12345)
  * m_u(x) = tanh(x . w_u),  m_v(x) = tanh(x . w_v)
  * score s(x, a) = -((u - m_u)^2 + (v - m_v)^2) / (2 * 0.25^2) + 0.5 * x[2] * u * v
  * hard label  y(x) = argmax_k s(x, a_k)
  * soft target q(x) = softmax_k(s(x, a_k) / TAU), TAU = 0.5
"""
from __future__ import annotations

import hashlib
import io
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

GRID_N = 8
N_IN = 8
N_CLASSES = GRID_N * GRID_N
TAU = 0.5
SIGMA = 0.25
_U = np.linspace(-1.0, 1.0, GRID_N)
GRID = np.array([(u, v) for u in _U for v in _U])  # (64, 2), class k = i*8 + j
_rng_w = np.random.default_rng(12345)
W_U = _rng_w.normal(0.0, 0.3, N_IN)
W_V = _rng_w.normal(0.0, 0.3, N_IN)


def fixture_scores(x: np.ndarray) -> np.ndarray:
    """FIXTURE analytic scores, shape (n, 64)."""
    mu = np.tanh(x @ W_U)[:, None]
    mv = np.tanh(x @ W_V)[:, None]
    u = GRID[None, :, 0]
    v = GRID[None, :, 1]
    return -((u - mu) ** 2 + (v - mv) ** 2) / (2 * SIGMA**2) + 0.5 * x[:, 2:3] * u * v


def make_fixture(n: int, seed: int):
    """Return x (n,8) float64, hard labels (n,), soft targets (n,64), continuous 2-D argmax action (n,2)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, N_IN))
    s = fixture_scores(x)
    y = s.argmax(1)
    z = s / TAU
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(1, keepdims=True)
    return x, y, q, GRID[y]


class Standardizer:
    def __init__(self, x_train: np.ndarray):
        self.mean = x_train.mean(0)
        self.std = x_train.std(0) + 1e-8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class SpikingStudent(nn.Module):
    """Direct (constant-current) encoding -> [Linear -> Leaky(subtract)] x L -> Linear -> Leaky(reset none).

    logits = mean over T steps of the readout membrane. Hidden states are passed explicitly and
    initialised to zero on every forward call, so no state leaks between calls/batches.
    """

    def __init__(self, n_in, n_out, hidden=(64,), T=10, beta=0.9, threshold=1.0, readout_beta=0.9,
                 spike_grad=None, learn_beta=False, learn_threshold=False):
        super().__init__()
        self.T = int(T)
        sizes = [n_in, *hidden]
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.lifs = nn.ModuleList(
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, learn_beta=learn_beta,
                      learn_threshold=learn_threshold, reset_mechanism="subtract")
            for _ in hidden)
        self.fc_out = nn.Linear(sizes[-1], n_out)
        self.lif_out = snn.Leaky(beta=readout_beta, reset_mechanism="none")

    def forward(self, x, record: bool = False):
        B = x.shape[0]
        mems = [torch.zeros(B, fc.out_features, dtype=x.dtype) for fc in self.fcs]
        mem_out = torch.zeros(B, self.fc_out.out_features, dtype=x.dtype)
        cur0 = self.fcs[0](x)  # constant input current: computed once, reused for all T steps
        acc = 0
        rec = [[] for _ in self.fcs] if record else None
        for _t in range(self.T):
            cur = cur0
            spk = None
            for i, lif in enumerate(self.lifs):
                if i > 0:
                    cur = self.fcs[i](spk)
                spk, mems[i] = lif(cur, mems[i])
                spk = spk.to(x.dtype)  # snnTorch surrogate forward returns float32 even in .double() mode (p02b)
                if record:
                    rec[i].append(spk)
            _, mem_out = self.lif_out(self.fc_out(spk), mem_out)
            acc = acc + mem_out
        logits = acc / self.T
        if record:
            return logits, [torch.stack(r) for r in rec]  # each [T, B, H]
        return logits


def loss_fn(kind, logits, y_hard, q_soft):
    if kind == "ce":
        return F.cross_entropy(logits, y_hard)
    if kind == "kl":
        return F.kl_div(F.log_softmax(logits, -1), q_soft, reduction="batchmean")
    raise ValueError(kind)


def train(model, Xs, y, q, *, kind="ce", steps=1500, batch=64, lr=2e-3, seed=0, log_every=None):
    """Epoch-wise permutation minibatches until `steps` optimizer steps are done. Returns history dict."""
    Xt = torch.as_tensor(Xs, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    qt = torch.as_tensor(q, dtype=torch.float32)
    n = Xt.shape[0]
    b = min(batch, n)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss_first": None, "loss_last_epoch_mean": None, "losses": []}
    step = 0
    epochs = 0
    t0 = time.perf_counter()
    while step < steps:
        perm = torch.randperm(n, generator=g)
        ep_losses = []
        for s in range(0, n, b):
            idx = perm[s:s + b]
            logits = model(Xt[idx])
            loss = loss_fn(kind, logits, yt[idx], qt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lv = float(loss.detach())
            if hist["loss_first"] is None:
                hist["loss_first"] = lv
            ep_losses.append(lv)
            step += 1
            if log_every and step % log_every == 0:
                hist["losses"].append((step, lv))
            if step >= steps:
                break
        epochs += 1
        hist["loss_last_epoch_mean"] = float(np.mean(ep_losses))
    hist["train_seconds"] = time.perf_counter() - t0
    hist["epochs"] = epochs
    hist["steps"] = step
    hist["batch"] = b
    return hist


@torch.no_grad()
def evaluate(model, Xs, y, q=None):
    model.eval()
    logits, rec = model(torch.as_tensor(Xs, dtype=next(model.parameters()).dtype), record=True)
    pred = logits.argmax(1).numpy()
    out = {"agree": float((pred == y).mean())}
    out["fire_rate_per_layer"] = [float(r.mean()) for r in rec]
    # fraction of neurons that never fire on any input/step, and that fire on every step for every input
    out["dead_frac_per_layer"] = [float((r.sum((0, 1)) == 0).float().mean()) for r in rec]
    out["saturated_frac_per_layer"] = [float((r.mean((0, 1)) == 1).float().mean()) for r in rec]
    if q is not None:
        lp = F.log_softmax(logits.float(), -1)
        out["kl_to_soft"] = float(F.kl_div(lp, torch.as_tensor(q, dtype=torch.float32), reduction="batchmean"))
    model.train()
    return out


def export_npz(model: SpikingStudent, path, x_mean, x_std):
    """Export everything the numpy forward needs. dtype follows the model (call .double() first for float64)."""
    d = {}
    for i, (fc, lif) in enumerate(zip(model.fcs, model.lifs)):
        d[f"W{i}"] = fc.weight.detach().numpy().copy()
        d[f"b{i}"] = fc.bias.detach().numpy().copy()
        d[f"beta{i}"] = lif.beta.detach().numpy().copy()
        d[f"thr{i}"] = lif.threshold.detach().numpy().copy()
    d["Wo"] = model.fc_out.weight.detach().numpy().copy()
    d["bo"] = model.fc_out.bias.detach().numpy().copy()
    d["beta_o"] = model.lif_out.beta.detach().numpy().copy()
    d["n_hidden_layers"] = np.array(len(model.lifs))
    d["T"] = np.array(model.T)
    dt = d["Wo"].dtype
    d["x_mean"] = np.asarray(x_mean, dtype=dt)
    d["x_std"] = np.asarray(x_std, dtype=dt)
    np.savez(path, **d)
    return d


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
