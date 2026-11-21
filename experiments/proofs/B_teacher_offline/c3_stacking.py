"""Check 3: long-format build and un-stack are exact inverses; de-standardization is exact to float precision.

FIXTURE arrays only; exercises the pure functions constant_outcomes, stack_training, stack_query,
unstack_rows, destandardize and fill_outcomes of learned_judgment.teacher. No network.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c3_stacking.py
"""

import sys
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.teacher import (  # noqa: E402
    constant_outcomes,
    destandardize,
    fill_outcomes,
    stack_query,
    stack_training,
    unstack_rows,
)

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


EPS = np.finfo(np.float64).eps


def max_eps_units(restored: np.ndarray, original: np.ndarray, standardized: np.ndarray, mean, std) -> float:
    """max |restored - original| in units of eps * (|original| + |mean| + std * |z|), the scale of
    the round trip z = (r - mean) / std -> z * std + mean."""
    scale = np.abs(original) + np.abs(mean) + np.asarray(std) * np.abs(standardized)
    return float(np.max(np.abs(restored - original) / (EPS * scale)))


rng = np.random.default_rng(3)
N, M, F = 5, 3, 4
features = rng.normal(size=(N, F))
features[:, 1] = [0.0, 1.0, 1.0, 0.0, 1.0]
residuals = np.column_stack([rng.normal(2.0, 3.0, N), np.full(N, -0.75), rng.normal(-1.0, 0.01, N)])
query = rng.normal(size=(M, F))

print("== constant outcomes ==")
constant = constant_outcomes(residuals)
print("  constant mask:", constant.tolist())
check("outcome 1 (all -0.75) is constant, outcomes 0 and 2 are not", constant.tolist() == [False, True, False])
active = tuple(int(k) for k in np.flatnonzero(~constant))
check("active outcomes are (0, 2)", active == (0, 2))

print("\n== two active outcomes: long format with an outcome column ==")
x_long, y_long, mean, std = stack_training(features, residuals, active)
print(f"  x_long {x_long.shape}, y_long {y_long.shape}, mean {mean.tolist()}, std {std.tolist()}")
check("x_long shape (A*n, F+1)", x_long.shape == (2 * N, F + 1))
check("y_long shape (A*n,)", y_long.shape == (2 * N,))
blocks = unstack_rows(x_long, N, len(active))  # (n, A, F+1)
check("every block's feature columns equal the training features exactly", all(np.array_equal(blocks[:, j, :F], features) for j in range(2)))
check("outcome column holds the outcome index k of each block", all(np.all(blocks[:, j, F] == float(k)) for j, k in enumerate(active)))
check("rows are outcome-major (first n rows are outcome 0)", np.all(x_long[:N, F] == 0.0) and np.all(x_long[N:, F] == 2.0))
standardized = unstack_rows(y_long, N, len(active))  # (n, A)
check("standardized targets have mean 0 and std 1 per outcome (float precision)",
      np.allclose(standardized.mean(axis=0), 0.0, atol=1e-15) and np.allclose(standardized.std(axis=0), 1.0, rtol=1e-14))
restored = destandardize(standardized, mean, std)
units = max_eps_units(restored, residuals[:, list(active)], standardized, mean, std)
print(f"  de-standardized vs original residuals: max |diff| {np.max(np.abs(restored - residuals[:, list(active)])):.3e}, "
      f"{units:.2f} eps units of scale")
check("de-standardization restores the residuals within 4 eps units of scale", units <= 4.0)

q_long = stack_query(query, active)
check("query expansion shape (A*m, F+1)", q_long.shape == (2 * M, F + 1))
q_blocks = unstack_rows(q_long, M, 2)
check("query blocks equal the query exactly, outcome column = k",
      all(np.array_equal(q_blocks[:, j, :F], query) and np.all(q_blocks[:, j, F] == float(k)) for j, k in enumerate(active)))

print("\n== un-stacking predicted quantiles (stand-in arrays shaped like a response) ==")
levels = 5
pred_q = np.sort(rng.normal(size=(2 * M, levels)), axis=1)  # (A*m, Q) standardized units
per_outcome = unstack_rows(pred_q, M, 2)  # (m, A, Q)
check("un-stacked quantiles (m, A, Q)", per_outcome.shape == (M, 2, levels))
check("row (i, outcome j) of the result is long row j*m + i", all(np.array_equal(per_outcome[i, j], pred_q[j * M + i]) for i in range(M) for j in range(2)))
real_q = destandardize(per_outcome, mean, std)
check("de-standardization keeps quantiles non-decreasing", np.all(np.diff(real_q, axis=2) >= 0.0))
check("de-standardized quantile = value * std_k + mean_k exactly",
      all(np.array_equal(real_q[:, j, :], per_outcome[:, j, :] * std[j] + mean[j]) for j in range(2)))

print("\n== filling the full outcome order (constant outcome) ==")
full_q = fill_outcomes(real_q, active, residuals[0])
check("full quantiles (m, K, Q)", full_q.shape == (M, 3, levels))
check("constant outcome 1 is exactly -0.75 at every quantile", np.all(full_q[:, 1, :] == -0.75))
check("active outcomes are placed unchanged", np.array_equal(full_q[:, [0, 2], :], real_q))
full_mean = fill_outcomes(np.zeros((M, 0)), (), residuals[0])
check("no active outcome: every column is its constant", full_mean.shape == (M, 3) and np.array_equal(full_mean, np.tile(residuals[0], (M, 1))))

print("\n== exactly one active outcome: no outcome column ==")
one_x, one_y, one_mean, one_std = stack_training(features, residuals, (2,))
check("x equals the features exactly (n, F), no extra column", one_x.shape == (N, F) and np.array_equal(one_x, features))
check("query is unchanged (m, F)", np.array_equal(stack_query(query, (2,)), query))
one_restored = destandardize(unstack_rows(one_y, N, 1), one_mean, one_std)[:, 0]
one_units = max_eps_units(one_restored, residuals[:, 2], one_y, one_mean[0], one_std[0])
print(f"  single-outcome restore: {one_units:.2f} eps units of scale")
check("single-outcome de-standardization within 4 eps units of scale", one_units <= 4.0)

print("\n== rejected inputs ==")
for label, call in {
    "a constant outcome among the active ones": lambda: stack_training(features, residuals, (0, 1)),
    "unsorted active indices": lambda: stack_training(features, residuals, (2, 0)),
    "active index out of range": lambda: stack_training(features, residuals, (0, 3)),
    "wrong row count for unstack": lambda: unstack_rows(np.zeros(7), 3, 2),
}.items():
    try:
        call()
        check(f"{label} raises", False)
    except ValueError as exc:
        check(f"{label} raises ValueError: {exc}", True)

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
