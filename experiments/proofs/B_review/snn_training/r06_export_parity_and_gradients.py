"""R06: export/parity with learned beta/threshold pushed out of range, two hidden layers; gradients.

  (a) build_model with learn_beta and learn_threshold, 1 and 2 hidden layers; set the learned
      parameters to values training could reach (beta 1.7 / -0.4, threshold -0.5 / 0.0 / 2.5)
      and check that export_network + check_parity hold (spike counts and decisions);
  (b) compare the torch model step by step against a hand-written float64 reference of
      snnTorch's Leaky (membrane, spikes) to confirm the order leak -> input -> reset(previous U);
  (c) surrogate wrapper: gradients of a KL loss w.r.t. every parameter with the wrapper vs the raw
      snnTorch surrogate (threshold 1.0, exact in float32) -> identical?
  (d) spike-rate penalty: the gradient of the penalty term alone reaches the trunk weights;
  (e) readout: the readout neuron's spikes and reset have no effect (logits = mean of a pure
      leaky integration of W_o S + b_o).
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

import _review_common  # noqa: E402,F401
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

config = load_config()
base = SNNSettings.from_config(config)
rng = np.random.default_rng(11)
x = rng.normal(size=(300, 6)) * 1.5

print("== (a) learned parameters pushed out of range ==")
for hidden in ((32,), (32, 24)):
    settings = dataclasses.replace(base, hidden_sizes=hidden, learn_beta=True, learn_threshold=True, threshold=0.3)
    for betas, thresholds in (((1.7, -0.4), (-0.5, 2.5)), ((-0.4, 1.7), (0.0, -0.5)), ((0.95, 0.5), (2.5, 0.0))):
        model = snn_train.build_model(6, {"h": 7}, settings, seed=4)
        with torch.no_grad():
            for neuron, b, t in zip(model.trunk_neurons, betas, thresholds):
                neuron.beta.fill_(b)
                neuron.threshold.fill_(t)
        network = snn_train.export_network(model, "h")
        try:
            report = snn_train.check_parity(model, "h", network, x)
            print(f"  hidden {hidden} betas {betas[:len(hidden)]} thr {thresholds[:len(hidden)]}: exported betas {network.betas} "
                  f"thr {network.thresholds}; mismatches {report['spike_count_mismatches_per_layer']} "
                  f"decisions differing {report['decisions_differing']} max|dlogit| {report['max_abs_logit_difference']:.1e}")
        except Exception as error:  # noqa: BLE001
            print(f"  hidden {hidden} betas {betas} thr {thresholds}: {type(error).__name__}: {str(error)[:200]}")

print("== (b) torch model vs hand float64 reference of Leaky ==")
settings = dataclasses.replace(base, hidden_sizes=(16, 8), threshold=0.3, beta=0.8, readout_beta=0.6, time_steps=7)
model = snn_train.build_model(6, {"h": 5}, settings, seed=2)
with torch.no_grad():
    logits, spikes = model(torch.as_tensor(x), "h", record=True)
W = [l.weight.detach().numpy() for l in model.trunk_linears]
b = [l.bias.detach().numpy() for l in model.trunk_linears]
Wo, bo = model.head_linears["h"].weight.detach().numpy(), model.head_linears["h"].bias.detach().numpy()
U = [np.zeros((300, 16)), np.zeros((300, 8))]
M = np.zeros((300, 5))
acc = np.zeros((300, 5))
ref_spikes = [[], []]
for t in range(7):
    inp = x @ W[0].T + b[0]
    for layer in range(2):
        if layer:
            inp = s @ W[1].T + b[1]
        reset = (U[layer] - 0.3 > 0).astype(float)          # from the PREVIOUS membrane
        U[layer] = 0.8 * U[layer] + inp - reset * 0.3        # leak, input, subtractive reset
        s = (U[layer] - 0.3 > 0).astype(float)               # strict '>'
        ref_spikes[layer].append(s)
    M = 0.6 * M + (s @ Wo.T + bo)                            # readout: no reset, spikes unused
    acc = acc + M
ref_logits = acc / 7
print(f"  spikes equal per layer: {[bool(np.array_equal(np.stack(ref_spikes[l]), spikes[l].numpy())) for l in range(2)]}; "
      f"max|logit - ref| = {float(np.max(np.abs(logits.numpy() - ref_logits))):.1e}")

print("== (c) surrogate wrapper vs raw snnTorch surrogate: gradients ==")
# Without the wrapper the float64 model cannot run (float32 spikes into a float64 Linear), so the
# comparison is made on the spike function itself: upstream gradient g (float64) through the
# wrapper, versus snnTorch's raw function fed float32(g) (what autograd hands it either way),
# versus the analytic float64 surrogate derivative.
for kind, alpha in (("atan", 2.0), ("fast_sigmoid", 25.0)):
    u = torch.linspace(-3.0, 3.0, 2001, dtype=torch.float64, requires_grad=True)
    g = torch.as_tensor(rng.normal(size=2001), dtype=torch.float64)
    wrapper = snn_train._SpikeInMembraneDtype(kind, alpha)
    out = wrapper(u)
    out.backward(g)
    grad_wrapped = u.grad.clone()
    u.grad = None
    raw_out = wrapper._function(u)
    raw_out.backward(g.float())
    grad_raw = u.grad.clone()
    ud = u.detach()
    analytic = (alpha / 2 / (1 + (np.pi / 2 * alpha * ud) ** 2) if kind == "atan" else 1 / (alpha * ud.abs() + 1) ** 2) * g
    print(f"  {kind}: forward dtype {out.dtype} values equal raw {bool(torch.equal(out, raw_out.double()))}; "
          f"grad wrapper == raw snnTorch: {bool(torch.equal(grad_wrapped, grad_raw))}; "
          f"max rel diff to analytic float64 = {float(((grad_wrapped - analytic).abs() / analytic.abs().clamp_min(1e-300)).max()):.1e}")

print("== (d) spike-rate penalty gradient reaches the trunk ==")
model = snn_train.build_model(6, {"h": 7}, dataclasses.replace(base, hidden_sizes=(32, 16)), seed=8)
_, spk = model(torch.as_tensor(x), "h", record=True)
penalty = sum(s.sum() for s in spk) / float(sum(s.numel() for s in spk))
penalty.backward()
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"  {name}: grad None")
    else:
        print(f"  {name}: |grad| sum = {float(p.grad.abs().sum()):.3e}")

print("== (e) readout neuron: threshold has no effect ==")
model = snn_train.build_model(6, {"h": 7}, base, seed=8)
with torch.no_grad():
    before = model(torch.as_tensor(x), "h")
    model.head_neurons["h"].threshold.fill_(-5.0)
    after = model(torch.as_tensor(x), "h")
print(f"  logits identical after changing the readout threshold to -5: {bool(torch.equal(before, after))}")
