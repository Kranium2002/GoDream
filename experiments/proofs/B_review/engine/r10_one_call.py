"""R10: dream() makes exactly one residual-model call for n situations x C candidates, and its
size; nothing in the engine reads engine.candidate_batch_size."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, ReviewSpec  # noqa: E402

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.dream import dream  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.types import ResidualDistribution  # noqa: E402


class CountingModel:
    """FIXTURE: zero residual, counts calls and query rows (no provider)."""

    def __init__(self):
        self.calls, self.rows = 0, []
        self.last_request_hash = None

    def fit(self, f, r):
        return self

    def predict_residuals(self, q):
        d = self.predict_residual_distribution(q)
        return d.median, d.spread

    def predict_residual_distribution(self, q):
        self.calls += 1
        self.rows.append(q.shape)
        return ResidualDistribution.zeros(q.shape[0], 2, np.array(LEVELS), SPREAD)


spec = ReviewSpec()
layout = build_layout(spec)
config = load_config()
n = int(require(config, "engine.dreams_per_skill_initial"))
for points in ([3, 3], [5, 5]):
    m = CountingModel()
    dream(spec, m, layout, n_dreams=n, rng=stream_rng(1, "rv/dreams"), points_per_dim=points, quantile_levels=LEVELS,
          spread_levels=SPREAD, method="teacher")
    c = int(np.prod(points))
    print(f"dream(n={n}, grid {points}): model calls = {m.calls}, query matrix = {m.rows[0]}; "
          f"stacked request rows for K={spec.n_outcome} outcomes = {m.rows[0][0] * spec.n_outcome}")
print("engine.candidate_batch_size in config =", config["engine"]["candidate_batch_size"],
      "| tabpfn.max_tokens_per_request =", require(config, "tabpfn.max_tokens_per_request"))
