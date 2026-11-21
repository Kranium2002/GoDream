"""p03c: regression readout variant on FIXTURES (runs-only check + multimodality mechanics).

(a) Readout membrane (n_out=2, mean over T) regressed with MSE onto the continuous 2-D grid action of the
    fixture argmax label. Confirms the loop runs and the loss decreases.
(b) Bimodal fixture: target action = (+0.8, 0) or (-0.8, 0) chosen by a coin that is NOT in the input
    (e.g. two equally good dodge sides). MSE regression -> predictions collapse toward the conditional mean (0, 0);
    a classifier over the 64-grid with soft targets puts its mass on the two modes instead.
hidden (64,), T=10, 800 steps, batch 64, lr 2e-3, seed 0, 1 thread.
"""
import numpy as np
import torch
import torch.nn.functional as F

from snn_common import GRID, SpikingStudent, Standardizer, make_fixture, N_IN, N_CLASSES

torch.set_num_threads(1)


def fit(m, X, Y, loss, steps=800, batch=64, lr=2e-3, seed=0):
    Xt, Yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    first = None
    for s in range(steps):
        idx = torch.randint(0, len(Xt), (batch,), generator=g)
        l = loss(m(Xt[idx]), Yt[idx])
        opt.zero_grad(); l.backward(); opt.step()
        first = float(l) if first is None else first
    return first, float(l)


X, y, q, a = make_fixture(2000, seed=1)
st = Standardizer(X)
X_te, y_te, _, a_te = make_fixture(1000, seed=999)
torch.manual_seed(0)
m = SpikingStudent(N_IN, 2, hidden=(64,), T=10)
f0, f1 = fit(m, st(X), a.astype(np.float32), F.mse_loss)
with torch.no_grad():
    pred = m(torch.as_tensor(st(X_te), dtype=torch.float32)).numpy()
print(f"(a) MSE regression onto 2-D argmax action: loss {f0:.4f} -> {f1:.4f}; hold-out RMSE per dim {np.sqrt(((pred - a_te) ** 2).mean(0)).round(4)}")

# (b) bimodal
rng = np.random.default_rng(7)
coin = rng.integers(0, 2, len(X))
Yb = np.stack([np.where(coin == 1, 0.8, -0.8), np.zeros(len(X))], 1).astype(np.float32)
torch.manual_seed(0)
mr = SpikingStudent(N_IN, 2, hidden=(64,), T=10)
fit(mr, st(X), Yb, F.mse_loss)
with torch.no_grad():
    pr = mr(torch.as_tensor(st(X_te), dtype=torch.float32)).numpy()
print(f"(b) bimodal targets (+-0.8, 0): MSE-regression mean prediction {pr.mean(0).round(3)}, mean |u_pred| {np.abs(pr[:, 0]).mean():.3f} "
      f"(modes at |u|=0.8); fraction of predictions within 0.2 of a mode: {(np.abs(np.abs(pr[:, 0]) - 0.8) < 0.2).mean():.3f}")
# classifier over the grid with soft targets: half mass on each mode's nearest grid cells
k_pos = int(np.argmin(((GRID - [0.8, 0]) ** 2).sum(1)))
k_neg = int(np.argmin(((GRID - [-0.8, 0]) ** 2).sum(1)))
Q = np.zeros((len(X), N_CLASSES), np.float32)
Q[np.arange(len(X)), np.where(coin == 1, k_pos, k_neg)] = 1.0
torch.manual_seed(0)
mc = SpikingStudent(N_IN, N_CLASSES, hidden=(64,), T=10)
fit(mc, st(X), Q, lambda lg, qq: F.kl_div(F.log_softmax(lg, -1), qq, reduction="batchmean"))
with torch.no_grad():
    p = torch.softmax(mc(torch.as_tensor(st(X_te), dtype=torch.float32)), -1).numpy()
am = p.argmax(1)
print(f"(b) grid classifier: mean prob on mode cells {GRID[k_pos].round(3)}/{GRID[k_neg].round(3)} = {p[:, k_pos].mean():.3f}/{p[:, k_neg].mean():.3f}; "
      f"argmax lands on a mode cell for {np.isin(am, [k_pos, k_neg]).mean():.3f} of inputs; argmax |u| mean {np.abs(GRID[am, 0]).mean():.3f}")
