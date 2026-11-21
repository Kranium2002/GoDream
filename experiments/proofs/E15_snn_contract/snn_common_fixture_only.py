"""numpy-only copy of the E15 FIXTURE generator (identical formulas to snn_common.make_fixture) so numpy-only
proofs can generate the same fixture inputs without importing torch. FIXTURE ONLY - not project data."""
import numpy as np

GRID_N, N_IN, TAU, SIGMA = 8, 8, 0.5, 0.25
_U = np.linspace(-1.0, 1.0, GRID_N)
GRID = np.array([(u, v) for u in _U for v in _U])
_rng_w = np.random.default_rng(12345)
W_U = _rng_w.normal(0.0, 0.3, N_IN)
W_V = _rng_w.normal(0.0, 0.3, N_IN)


def make_fixture(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, N_IN))
    mu = np.tanh(x @ W_U)[:, None]
    mv = np.tanh(x @ W_V)[:, None]
    s = -((GRID[None, :, 0] - mu) ** 2 + (GRID[None, :, 1] - mv) ** 2) / (2 * SIGMA**2) + 0.5 * x[:, 2:3] * GRID[None, :, 0] * GRID[None, :, 1]
    y = s.argmax(1)
    z = s / TAU
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(1, keepdims=True)
    return x, y, q, GRID[y]
