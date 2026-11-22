"""Pure-numpy forward pass for the E15 spiking student. MUST NOT import torch.

Reproduces snntorch 1.0.0 Leaky dynamics (reset_delay=True default) as read from
snntorch/_neurons/leaky.py:209-243 and neurons.py:73-108:
    reset_t = H(U_{t-1} - thr)                     (computed from the PREVIOUS membrane, detached)
    U_t     = (clip(beta,0,1) * U_{t-1} + I_t) - reset_t * thr   ("subtract")
    S_t     = H(U_t - thr)  with H(z) = 1 if z > 0 else 0   (strict >)
Readout Leaky(reset_mechanism="none"):  M_t = clip(beta_o,0,1) * M_{t-1} + (S^L_t @ Wo.T + bo)
logits = (sum_t M_t) / T
"""
import numpy as np


def load_params(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def snn_forward_np(xs, P, count=False, record=False):
    """xs: (B, n_in) ALREADY-STANDARDIZED features in the params' dtype. P: dict from load_params.

    Returns logits (B, C); if count=True also returns per-layer spike counts array of shape
    (n_hidden_layers, T, B) (spikes emitted by each hidden layer at each step, per input).
    If record=True returns (logits, [spikes of layer i as uint8 array (T, B, H_i)]).
    """
    dt = P["Wo"].dtype
    xs = np.asarray(xs, dtype=dt)
    L = int(P["n_hidden_layers"])
    T = int(P["T"])
    B = xs.shape[0]
    betas = [np.clip(P[f"beta{i}"], 0, 1) for i in range(L)]
    thrs = [P[f"thr{i}"] for i in range(L)]
    beta_o = np.clip(P["beta_o"], 0, 1)
    cur0 = xs @ P["W0"].T + P["b0"]  # constant input current, computed once
    mems = [np.zeros((B, P[f"W{i}"].shape[0]), dt) for i in range(L)]
    mem_o = np.zeros((B, P["Wo"].shape[0]), dt)
    acc = np.zeros_like(mem_o)
    counts = np.zeros((L, T, B), dtype=np.int64) if count else None
    rec = [np.zeros((T, B, P[f"W{i}"].shape[0]), np.uint8) for i in range(L)] if record else None
    for t in range(T):
        cur = cur0
        spk = None
        for i in range(L):
            if i > 0:
                cur = spk @ P[f"W{i}"].T + P[f"b{i}"]
            reset = (mems[i] > thrs[i]).astype(dt)
            mems[i] = (betas[i] * mems[i] + cur) - reset * thrs[i]
            spk = (mems[i] > thrs[i]).astype(dt)
            if count:
                counts[i, t] = spk.sum(1).astype(np.int64)
            if record:
                rec[i][t] = spk
        mem_o = beta_o * mem_o + (spk @ P["Wo"].T + P["bo"])
        acc = acc + mem_o
    logits = acc / dt.type(T)
    if record:
        return logits, rec
    return (logits, counts) if count else logits


def standardize(x, P):
    dt = P["Wo"].dtype
    return ((np.asarray(x, dtype=np.float64) - P["x_mean"].astype(np.float64)) / P["x_std"].astype(np.float64)).astype(dt)


def decide(x_raw_row, P):
    """One decision: raw feature vector (n_in,) -> class index."""
    return int(snn_forward_np(standardize(x_raw_row[None, :], P), P).argmax(1)[0])


def op_counts(P, spike_counts_LT):
    """Operation counts for ONE decision given spike counts of shape (L, T) for that input.

    synops: sum over spiking layers l of spikes_l * fan_out_l  (fan_out = next layer width, or C for last).
    first_layer_macs_once: n_in * h1 (constant current computed once);  _per_step: n_in * h1 * T.
    neuron_updates: (sum hidden widths + C readout) * T.
    dense_macs_reference: sum of fan_in * fan_out over all Linear layers (one pass of a non-spiking MLP).
    """
    L = int(P["n_hidden_layers"])
    T = int(P["T"])
    widths = [P[f"W{i}"].shape[0] for i in range(L)]
    n_in = P["W0"].shape[1]
    C = P["Wo"].shape[0]
    fan_out = widths[1:] + [C]
    spikes_per_layer = [int(spike_counts_LT[i].sum()) for i in range(L)]
    synops_per_layer = [spikes_per_layer[i] * fan_out[i] for i in range(L)]
    dense = n_in * widths[0] + sum(widths[i] * fan_out[i] for i in range(L))
    return {
        "spikes_per_layer_per_step": [[int(v) for v in spike_counts_LT[i]] for i in range(L)],
        "spikes_per_layer": spikes_per_layer,
        "total_spikes": int(sum(spikes_per_layer)),
        "synops_per_layer": synops_per_layer,
        "synops_total": int(sum(synops_per_layer)),
        "first_layer_macs_once": int(n_in * widths[0]),
        "first_layer_macs_if_recomputed_each_step": int(n_in * widths[0] * T),
        "neuron_state_updates": int((sum(widths) + C) * T),
        "mean_firing_rate_per_layer": [spikes_per_layer[i] / (widths[i] * T) for i in range(L)],
        "dense_mlp_macs_reference": int(dense),
    }
