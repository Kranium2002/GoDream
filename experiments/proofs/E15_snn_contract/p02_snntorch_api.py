"""p02: snnTorch 1.0.0 API + dynamics, verified by execution (no training; tiny hand-made tensors = FIXTURES)."""
import copy
import gc
import inspect
import io
import pickle
import warnings

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate, utils

warnings.simplefilter("always")
torch.set_printoptions(precision=6)
print("snntorch", snn.__version__, "torch", torch.__version__)

print("\n=== 1. signatures ===")
print("snn.Leaky", inspect.signature(snn.Leaky.__init__))
print("snn.Leaky.forward", inspect.signature(snn.Leaky.forward))
print("surrogate.fast_sigmoid", inspect.signature(surrogate.fast_sigmoid))
print("surrogate.atan", inspect.signature(surrogate.atan))
print("surrogate.sigmoid", inspect.signature(surrogate.sigmoid))
print("surrogate.triangular", inspect.signature(surrogate.triangular))
for name in ["ce_rate_loss", "ce_count_loss", "ce_max_membrane_loss", "mse_membrane_loss", "mse_count_loss"]:
    print(f"SF.{name}", inspect.signature(getattr(SF, name).__init__))
print("utils.reset", inspect.signature(utils.reset))
lif = snn.Leaky(beta=0.9)
print("default spike_grad:", lif.spike_grad.__qualname__, "closure alpha =",
      [c.cell_contents for c in lif.spike_grad.__closure__])
print("buffers:", {k: (tuple(v.shape), v.dtype, v.tolist() if v.numel() < 3 else '...') for k, v in lif.named_buffers()})
print("parameters (learn_beta=False):", [k for k, _ in lif.named_parameters()])
print("state_dict keys:", list(lif.state_dict().keys()))

print("\n=== 2. step dynamics per reset mechanism (beta=0.5, threshold=1.0, constant input 0.6, 6 steps) ===")
def run(mech, cur_seq, beta=0.5, thr=1.0):
    l = snn.Leaky(beta=beta, threshold=thr, reset_mechanism=mech)
    mem = torch.zeros(1)
    rows = []
    for c in cur_seq:
        spk, mem = l(torch.tensor([c]), mem)
        rows.append((round(float(mem), 6), int(spk)))
    return rows
cur_seq = [0.6] * 6
for mech in ["subtract", "zero", "none"]:
    print(f"  {mech:8s} (mem_t, spk_t):", run(mech, cur_seq))
# hand-computed reference for 'subtract': U_t = 0.5*U_{t-1} + 0.6 - H(U_{t-1}-1)*1 ; S_t = H(U_t - 1)
U, ref = 0.0, []
for _ in range(6):
    reset = 1.0 if U > 1.0 else 0.0
    U = 0.5 * U + 0.6 - reset * 1.0
    ref.append((round(U, 6), int(U > 1.0)))
print("  hand-ref subtract (prev-step reset):", ref)
U, ref = 0.0, []
for _ in range(6):
    reset = 1.0 if U > 1.0 else 0.0
    U = 0.5 * (1 - reset) * U + 0.6
    ref.append((round(U, 6), int(U > 1.0)))
print("  hand-ref zero     (prev-step reset):", ref)

print("\n=== 3. firing condition strict '>' : mem exactly == threshold ===")
l = snn.Leaky(beta=0.0, threshold=1.0)
spk, mem = l(torch.tensor([1.0]), torch.zeros(1))
print("  input=1.0 beta=0 -> mem", float(mem), "spk", float(spk))
spk, mem = l(torch.tensor([1.0 + 1e-6]), torch.zeros(1))
print("  input=1.000001 -> mem", float(mem), "spk", float(spk))

print("\n=== 4. return values for init_hidden / output ===")
for ih, out in [(False, False), (False, True), (True, False), (True, True)]:
    l = snn.Leaky(beta=0.9, init_hidden=ih, output=out)
    r = l(torch.ones(2, 3))
    kind = type(r).__name__ + (f" of len {len(r)}" if isinstance(r, tuple) else f" shape {tuple(r.shape)}")
    print(f"  init_hidden={ih} output={out}: forward(input) returns {kind}")
l = snn.Leaky(beta=0.9, init_hidden=True)
try:
    l(torch.ones(1), torch.zeros(1))
except TypeError as e:
    print("  init_hidden=True and mem passed ->", type(e).__name__, e)

print("\n=== 5. STATE PITFALL: init_hidden=False but mem not passed -> internal buffer carries over ===")
l = snn.Leaky(beta=0.9)
s1, m1 = l(torch.full((1,), 0.5))
s2, m2 = l(torch.full((1,), 0.5))
print("  call1 mem", float(m1), " call2 mem", float(m2), "(second call continued from first: 0.9*0.5+0.5=0.95)")
l.reset_mem()
s3, m3 = l(torch.full((1,), 0.5))
print("  after reset_mem(): mem", float(m3))

print("\n=== 6. graph-retention error if stale state is not reset between training batches ===")
lin = nn.Linear(2, 3)
l = snn.Leaky(beta=0.9)
x = torch.randn(4, 2)
for it in range(2):
    try:
        spk, mem = l(lin(x))  # no mem passed: uses buffer from previous iteration (still attached to old graph)
        mem.sum().backward()
        print(f"  iter {it}: backward OK")
    except RuntimeError as e:
        print(f"  iter {it}: RuntimeError: {str(e)[:110]}")
print("  fix A: pass mem explicitly each forward (mem = torch.zeros(B, H)); fix B: l.reset_mem() / utils.reset(net) before each batch")

print("\n=== 7. utils.reset(net) only inspects net._modules (top level) but resets ALL Leaky instances globally ===")
class Flat(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(2, 2); self.lif = snn.Leaky(beta=0.9)
class Nested(nn.Module):
    def __init__(self):
        super().__init__(); self.layers = nn.ModuleList([nn.Linear(2, 2), snn.Leaky(beta=0.9)])
f, nmod, other = Flat(), Nested(), snn.Leaky(beta=0.9)
for m in [f.lif, nmod.layers[1], other]:
    m(torch.ones(1, 2))
utils.reset(nmod)
print("  after utils.reset(Nested): nested lif mem =", nmod.layers[1].mem.tolist(), "(NOT reset: Leaky not a direct child)")
utils.reset(f)
print("  after utils.reset(Flat): flat lif mem =", f.lif.mem.tolist(), "; unrelated standalone Leaky mem =", other.mem.tolist(), "(also zeroed: global class list)")

print("\n=== 8. SpikingNeuron.instances global list grows with every neuron ever built ===")
n0 = len(snn.SpikingNeuron.instances)
for _ in range(100):
    _m = snn.Leaky(beta=0.9)
del _m; gc.collect()
print("  instances before", n0, "after building+deleting 100 Leaky:", len(snn.SpikingNeuron.instances))
snn.SpikingNeuron.init()
print("  after SpikingNeuron.init():", len(snn.SpikingNeuron.instances))

print("\n=== 9. learn_beta / learn_threshold semantics ===")
l = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
print("  parameters:", [(k, tuple(p.shape)) for k, p in l.named_parameters()])
with torch.no_grad():
    l.beta.fill_(1.5)
inp = torch.tensor([0.3], requires_grad=False)
mem0 = torch.tensor([0.4])
spk, mem = l(inp, mem0)
mem.sum().backward()
print("  beta param stored=1.5 -> effective clamp(0,1)=1: mem =", float(mem), "(=1*0.4+0.3); d mem/d beta =", float(l.beta.grad), "(clamp kills grad outside [0,1])")
l2 = snn.Leaky(beta=torch.full((4,), 0.9), learn_beta=True)
print("  per-neuron beta: pass beta tensor of shape (H,) ->", tuple(l2.beta.shape))
l3 = snn.Leaky(beta=0.9, threshold=-0.5, learn_threshold=True)
spk, mem = l3(torch.zeros(1), torch.zeros(1))
print("  threshold is NOT clamped: thr=-0.5, zero input/state -> spk", float(spk), " mem", float(mem), "(reset at t0 from mem0=0 > -0.5 fired too)")

print("\n=== 10. surrogate gradients (value of dS/dU at U-thr = 0, 0.5, -2) ===")
for name, fn in [("atan() default alpha=2", surrogate.atan()), ("fast_sigmoid(slope=25)", surrogate.fast_sigmoid()),
                 ("fast_sigmoid(slope=5)", surrogate.fast_sigmoid(slope=5))]:
    z = torch.tensor([0.0, 0.5, -2.0], requires_grad=True)
    s = fn(z)
    s.sum().backward()
    print(f"  {name}: forward {s.tolist()} dtype {s.dtype}; grad {[round(g, 6) for g in z.grad.tolist()]}")

print("\n=== 11. float64 via .double() ===")
class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(3, 4); self.lif = snn.Leaky(beta=0.9)
    def forward(self, x):
        mem = torch.zeros(x.shape[0], 4, dtype=x.dtype)
        tot = 0
        for _ in range(3):
            spk, mem = self.lif(self.fc(x), mem)
            tot = tot + mem
        return tot, spk
t = Tiny().double()
print("  beta buffer after .double():", t.lif.beta.dtype, repr(float(t.lif.beta)), "<- float32(0.9) widened, NOT 0.9")
o, spk = t(torch.randn(2, 3, dtype=torch.float64) * 3)
o.sum().backward()
print("  forward OK: mem dtype", o.dtype, "spk dtype", spk.dtype, "fc.weight.grad dtype", t.fc.weight.grad.dtype)
t2 = Tiny()
t2.lif = snn.Leaky(beta=torch.tensor(0.9, dtype=torch.float64))
t2 = t2.double()
print("  Leaky(beta=torch.tensor(0.9, float64)).double() beta:", repr(float(t2.lif.beta)))

print("\n=== 12. snntorch.functional losses with [T, B, C] tensors (T=4, B=3, C=5) ===")
torch.manual_seed(0)
spk_out = (torch.rand(4, 3, 5) > 0.5).float().requires_grad_()
mem_out = torch.randn(4, 3, 5, requires_grad=True)
tgt = torch.tensor([0, 2, 4])
for name, inp, fn in [("ce_rate_loss()", spk_out, SF.ce_rate_loss()), ("ce_count_loss()", spk_out, SF.ce_count_loss()),
                      ("ce_max_membrane_loss()", mem_out, SF.ce_max_membrane_loss()),
                      ("mse_membrane_loss()", mem_out, SF.mse_membrane_loss()),
                      ("mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)", spk_out, SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2))]:
    v = fn(inp, tgt)
    v.backward()
    print(f"  SF.{name}: value {v.detach().tolist()} shape {tuple(v.shape)} dtype {v.dtype}; backward OK")
md = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
v = SF.ce_rate_loss()(md, tgt)
print("  SF.ce_rate_loss on float64 input -> loss dtype", v.dtype, "(accumulator hard-coded dtype=torch.float, loss.py:12,123)")
print("  accuracy helper: SF.acc.accuracy_rate(spk_out, tgt) =", float(SF.acc.accuracy_rate(spk_out.detach(), tgt)))

print("\n=== 13. serialization pitfalls ===")
m = Tiny()
o, _ = m(torch.randn(2, 3))
try:
    copy.deepcopy(m)
    print("  deepcopy after a grad-enabled forward: OK")
except RuntimeError as e:
    print("  deepcopy after a grad-enabled forward -> RuntimeError:", str(e)[:120])
buf = io.BytesIO()
try:
    torch.save(m, buf)
    print("  torch.save(whole module): OK")
except Exception as e:
    print("  torch.save(whole module) ->", type(e).__name__, str(e)[:120])
buf = io.BytesIO()
torch.save(m.state_dict(), buf)
print("  torch.save(state_dict) OK; keys:", list(m.state_dict().keys()))
print("  mem buffer persistent? ", "lif.mem" in m.state_dict())
