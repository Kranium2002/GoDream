"""p08: shared multi-skill spiking network MECHANICS on three FIXTURE "skills" (analytic toy tasks; NOT project skills).

Input = [one-hot(skill, N_SKILLS_MAX) | per-skill-standardized features zero-padded to N_FEAT_MAX]
-> shared trunk Linear->Leaky(subtract) -> per-skill head Linear->Leaky(reset none), logits = mean head membrane.
Loss only on the active skill's head (single-skill minibatches). Skills added sequentially; for each new skill
we train (i) without rehearsal and (ii) with rehearsal (alternate new-skill and stored earlier-skill minibatches,
targets = stored soft targets). Then: state_dict save/load, .npz export, numpy per-head inference parity.
Also demonstrates the optimizer / registration pitfalls. KL to soft targets, Adam lr 2e-3, batch 64, 1 thread.
"""
import io
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

torch.set_num_threads(1)
N_SKILLS_MAX, N_FEAT_MAX, HIDDEN, T = 4, 12, 64, 10


# ---------------- FIXTURE skills ----------------
def fixture_skill(name, n, seed):
    rng_w = np.random.default_rng({"A": 1, "B": 2, "C": 3}[name])
    rng = np.random.default_rng(seed)
    if name == "A":  # 8 features, 8x8 grid = 64 classes
        nf, gu, gv = 8, np.linspace(-1, 1, 8), np.linspace(-1, 1, 8)
    elif name == "B":  # 5 features, 1-D grid of 30 classes
        nf, gu, gv = 5, np.linspace(-1, 1, 30), np.array([0.0])
    else:  # 10 features, 6x8 grid = 48 classes
        nf, gu, gv = 10, np.linspace(-1, 1, 6), np.linspace(-1, 1, 8)
    wu, wv = rng_w.normal(0, 0.3, nf), rng_w.normal(0, 0.3, nf)
    grid = np.array([(u, v) for u in gu for v in gv])
    x = rng.normal(0, 1, (n, nf))
    s = -((grid[None, :, 0] - np.tanh(x @ wu)[:, None]) ** 2 + (grid[None, :, 1] - np.tanh(x @ wv)[:, None] * (len(gv) > 1)) ** 2) / (2 * 0.25**2)
    z = s / 0.5
    q = np.exp(z - z.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    return x, s.argmax(1), q, len(grid)


class MultiSkillSNN(nn.Module):
    def __init__(self, n_skills_max=N_SKILLS_MAX, n_feat_max=N_FEAT_MAX, hidden=HIDDEN, T=T, beta=0.9, threshold=1.0):
        super().__init__()
        self.n_skills_max, self.n_feat_max, self.T = n_skills_max, n_feat_max, T
        self.fc1 = nn.Linear(n_skills_max + n_feat_max, hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.heads = nn.ModuleDict()       # registered -> in .parameters() and state_dict
        self.head_lifs = nn.ModuleDict()
        self.skills = []                   # ordered names; skill id = index (store alongside checkpoints)

    def add_skill(self, name, n_classes, readout_beta=0.9):
        assert len(self.skills) < self.n_skills_max, "one-hot width is fixed at construction"
        self.skills.append(name)
        self.heads[name] = nn.Linear(self.fc1.out_features, n_classes)
        self.head_lifs[name] = snn.Leaky(beta=readout_beta, reset_mechanism="none")
        return self.heads[name]

    def encode(self, name, xs, dtype=np.float32):
        B, nf = xs.shape
        out = np.zeros((B, self.n_skills_max + self.n_feat_max), dtype)
        out[:, self.skills.index(name)] = 1.0
        out[:, self.n_skills_max:self.n_skills_max + nf] = xs
        return torch.as_tensor(out)

    def forward(self, x, name):
        B = x.shape[0]
        mem1 = torch.zeros(B, self.fc1.out_features, dtype=x.dtype)
        head, hl = self.heads[name], self.head_lifs[name]
        memo = torch.zeros(B, head.out_features, dtype=x.dtype)
        cur1 = self.fc1(x)
        acc = 0
        for _ in range(self.T):
            spk, mem1 = self.lif1(cur1, mem1)
            _, memo = hl(head(spk.to(x.dtype)), memo)
            acc = acc + memo
        return acc / self.T


def kl(logits, q):
    return F.kl_div(F.log_softmax(logits, -1), q, reduction="batchmean")


data = {}
for nm, seed in [("A", 10), ("B", 20), ("C", 30)]:
    x, y, q, C = fixture_skill(nm, 2000, seed)
    xt, yt, qt, _ = fixture_skill(nm, 1000, seed + 1)
    mu, sd = x.mean(0), x.std(0) + 1e-8
    data[nm] = dict(xs=(x - mu) / sd, y=y, q=q.astype(np.float32), xs_te=(xt - mu) / sd, y_te=yt, C=C, mu=mu, sd=sd)


@torch.no_grad()
def agree(m, nm):
    d = data[nm]
    return float((m(m.encode(nm, d["xs_te"]), nm).argmax(1).numpy() == d["y_te"]).mean())


def batches(nm, g, b=64):
    d = data[nm]
    while True:
        for idx in torch.randperm(len(d["xs"]), generator=g).split(b):
            idx = idx.numpy()
            yield nm, d["xs"][idx], torch.as_tensor(d["q"][idx])


def run(m, opt, new, old, steps, rehearse, seed):
    g = torch.Generator().manual_seed(seed)
    it_new = batches(new, g)
    it_old = {o: batches(o, g) for o in old}
    for s in range(steps):
        streams = [it_new] + ([it_old[old[s % len(old)]]] if (rehearse and old) else [])
        for st in streams:
            nm, xb, qb = next(st)
            loss = kl(m(m.encode(nm, xb), nm), qb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


print("=== pitfall 1: optimizer built before add_skill does not contain the new head ===")
torch.manual_seed(0)
m = MultiSkillSNN()
m.add_skill("A", data["A"]["C"])
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
hB = m.add_skill("B", data["B"]["C"])
w_before = hB.weight.detach().clone()
loss = kl(m(m.encode("B", data["B"]["xs"][:64]), "B"), torch.as_tensor(data["B"]["q"][:64]))
opt.zero_grad(); loss.backward(); opt.step()
print("  head B weight changed after opt.step()?", not torch.equal(w_before, hB.weight), "(grad present:", hB.weight.grad is not None, ")")
opt.add_param_group({"params": list(hB.parameters()) + list(m.head_lifs["B"].parameters())})
opt.zero_grad(); kl(m(m.encode("B", data["B"]["xs"][:64]), "B"), torch.as_tensor(data["B"]["q"][:64])).backward(); opt.step()
print("  after opt.add_param_group(new head params): changed?", not torch.equal(w_before, hB.weight),
      "| Adam state kept for trunk:", len(opt.state[m.fc1.weight]) > 0)

print("=== pitfall 2: inactive head drifts under Adam if grads are zero tensors instead of None ===")
for set_none in [True, False]:
    torch.manual_seed(0)
    m2 = MultiSkillSNN(); m2.add_skill("A", data["A"]["C"]); m2.add_skill("B", data["B"]["C"])
    o2 = torch.optim.Adam(m2.parameters(), lr=2e-3)
    for _ in range(3):  # give head A some Adam momentum
        o2.zero_grad(set_to_none=set_none); kl(m2(m2.encode("A", data["A"]["xs"][:64]), "A"), torch.as_tensor(data["A"]["q"][:64])).backward(); o2.step()
    wA = m2.heads["A"].weight.detach().clone()
    for _ in range(3):  # now train only B
        o2.zero_grad(set_to_none=set_none); kl(m2(m2.encode("B", data["B"]["xs"][:64]), "B"), torch.as_tensor(data["B"]["q"][:64])).backward(); o2.step()
    print(f"  zero_grad(set_to_none={set_none}): head A max|change| while training only B = {float((m2.heads['A'].weight - wA).abs().max()):.3e}")

print("=== pitfall 3: registration ===")
class Bad(nn.Module):
    def __init__(self):
        super().__init__(); self.heads = {"A": nn.Linear(4, 3)}  # plain dict: NOT registered
print("  plain dict heads -> n params", sum(p.numel() for p in Bad().parameters()), "state_dict keys", list(Bad().state_dict().keys()))
try:
    nn.ModuleDict({"dodge.v2": nn.Linear(2, 2)})
except KeyError as e:
    print("  ModuleDict key with '.' ->", "KeyError", e)

print("=== sequential skills A -> B -> C, without vs with rehearsal (FIXTURE mechanics) ===")
torch.manual_seed(0)
base = MultiSkillSNN()
base.add_skill("A", data["A"]["C"])
opt = torch.optim.Adam(base.parameters(), lr=2e-3)
run(base, opt, "A", [], 600, False, seed=0)
print(f"  after A: agree A={agree(base, 'A'):.3f}")
ck = {"skills": [("A", data["A"]["C"])], "state": {k: v.clone() for k, v in base.state_dict().items()}}
results = {}
for rehearse in [False, True]:
    m = MultiSkillSNN()
    for nm, C in ck["skills"]:
        m.add_skill(nm, C)
    m.load_state_dict(ck["state"])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)  # fresh optimizer after (re)building heads
    for new, old in [("B", ["A"]), ("C", ["A", "B"])]:
        h = m.add_skill(new, data[new]["C"])
        opt.add_param_group({"params": list(h.parameters()) + list(m.head_lifs[new].parameters())})
        run(m, opt, new, old, 600, rehearse, seed=1)
        print(f"  rehearsal={rehearse} after {new}: " + ", ".join(f"{s}={agree(m, s):.3f}" for s in m.skills))
    results[rehearse] = m

print("=== save / load (state_dict + skill spec) ===")
m = results[True]
buf = io.BytesIO()
torch.save({"skills": [(s, m.heads[s].out_features) for s in m.skills], "state_dict": m.state_dict(),
            "n_skills_max": m.n_skills_max, "n_feat_max": m.n_feat_max}, buf)
buf.seek(0)
blob = torch.load(buf, weights_only=True)
fresh = MultiSkillSNN()
try:
    fresh.load_state_dict(blob["state_dict"])
except RuntimeError as e:
    print("  load into model without heads ->", str(e).splitlines()[0][:100], "...")
for s, C in blob["skills"]:
    fresh.add_skill(s, C)
fresh.load_state_dict(blob["state_dict"])
print("  after add_skill for each stored head: load OK; agreement", {s: round(agree(fresh, s), 3) for s in fresh.skills})

print("=== .npz export + numpy per-head inference (float64) ===")
m64 = fresh.double()
npz = {"W1": m64.fc1.weight.detach().numpy(), "b1": m64.fc1.bias.detach().numpy(), "beta1": m64.lif1.beta.detach().numpy(),
       "thr1": m64.lif1.threshold.detach().numpy(), "T": np.array(m64.T), "skills": np.array(m64.skills),
       "n_skills_max": np.array(m64.n_skills_max), "n_feat_max": np.array(m64.n_feat_max)}
for s in m64.skills:
    npz[f"head_{s}_W"] = m64.heads[s].weight.detach().numpy()
    npz[f"head_{s}_b"] = m64.heads[s].bias.detach().numpy()
    npz[f"head_{s}_beta"] = m64.head_lifs[s].beta.detach().numpy()
    npz[f"x_mean_{s}"], npz[f"x_std_{s}"] = data[s]["mu"], data[s]["sd"]
np.savez("p08_multiskill_f64.npz", **npz)
P = dict(np.load("p08_multiskill_f64.npz"))


def np_forward_head(xs, skill, P):
    skills = [str(s) for s in P["skills"]]
    B = xs.shape[0]
    x = np.zeros((B, int(P["n_skills_max"]) + int(P["n_feat_max"])))
    x[:, skills.index(skill)] = 1.0
    x[:, int(P["n_skills_max"]):int(P["n_skills_max"]) + xs.shape[1]] = xs
    cur1 = x @ P["W1"].T + P["b1"]
    b1, th = np.clip(P["beta1"], 0, 1), P["thr1"]
    W, b, bo = P[f"head_{skill}_W"], P[f"head_{skill}_b"], np.clip(P[f"head_{skill}_beta"], 0, 1)
    mem1, memo = np.zeros((B, W.shape[1])), np.zeros((B, W.shape[0]))
    acc = np.zeros_like(memo)
    for _ in range(int(P["T"])):
        reset = (mem1 > th).astype(np.float64)
        mem1 = (b1 * mem1 + cur1) - reset * th
        spk = (mem1 > th).astype(np.float64)
        memo = bo * memo + (spk @ W.T + b)
        acc = acc + memo
    return acc / np.float64(P["T"])


for s in m64.skills:
    xs = data[s]["xs_te"]
    with torch.no_grad():
        enc = m64.encode(s, xs, np.float64)
        tl = m64(enc, s).numpy()
    nl = np_forward_head(xs, s, P)
    print(f"  skill {s}: max|torch-numpy| {np.abs(tl - nl).max():.2e}, argmax agreement {(tl.argmax(1) == nl.argmax(1)).mean():.4f}, C={tl.shape[1]}")
