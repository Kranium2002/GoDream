"""Review r05: stacked-mode standardization / grouping / outcome order, end to end through the real teacher.

LookupRegressor (fixture) answers each query row with the y of the identical training row (so the
expected teacher output is the original residual exactly) and puts quantile q at y + (q - 0.5) in
the units it was trained in. Cases: three active outcomes; a constant outcome in the middle; one
active outcome; all constant; binary-valued residuals; per_outcome mode for comparison.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r05_stacked_roundtrip.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import tempfile
from pathlib import Path

import numpy as np

import _fixture as fx
from learned_judgment.teacher import _tabpfn_config


class LookupRegressor(fx.FakeRegressor):
    def fit(self, x, y):
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self.sent_train = (self._x.copy(), self._y.copy())
        return self

    def predict(self, x, output_type, quantiles):
        self.account.billed_predicts += 1
        index = [int(np.flatnonzero(np.all(self._x == row, axis=1))[0]) for row in x]
        y = self._y[index]
        return {"mean": y, "median": y, "mode": y,
                "quantiles": np.stack([y + (level - 0.5) for level in quantiles])}  # (Q, n)


class LookupProvider(fx.FixtureProvider):
    def _build_regressor(self, categorical_indices):
        self.last_regressor = LookupRegressor(self.account)
        return self.last_regressor, _tabpfn_config(self.settings, categorical_indices)


def run(label, residuals, mode="stacked"):
    features, _, _ = fx.arrays(seed=11, n=6)
    with tempfile.TemporaryDirectory() as tmp:
        teacher, provider, account, log, settings = fx.make_teacher(Path(tmp), mode=mode)
        provider = LookupProvider(settings, account)
        teacher.provider = provider
        teacher.fit(features, residuals)
        dist = teacher.predict_residual_distribution(features)  # query = training rows
        levels = np.asarray(settings.quantile_levels)
        std = residuals.std(axis=0)
        unit = std if mode == "stacked" else np.ones_like(std)
        constant = np.all(residuals == residuals[0], axis=0)
        expected_q = residuals[:, :, None] + np.where(constant, 0.0, unit)[None, :, None] * (levels - 0.5)[None, None, :]
        print(f"== {label} ({mode}) ==")
        print("  requests:", account.billed_predicts, "| active outcomes:", list(np.flatnonzero(~constant)))
        if account.billed_predicts:
            x_sent, y_sent = provider.last_regressor.sent_train
            print("  last request shape:", x_sent.shape, "| y mean/std sent:", np.round(y_sent.mean(), 12), np.round(y_sent.std(), 12))
        print("  max |median - residual|:", float(np.max(np.abs(dist.median - residuals))))
        print("  max |mean - residual|:  ", float(np.max(np.abs(dist.mean - residuals))))
        print("  max |quantiles - expected (residual + (q-0.5)*std_k)|:", float(np.max(np.abs(dist.quantiles - expected_q))))
        print("  spread per outcome (row 0):", np.round(dist.spread[0], 6).tolist(), "| expected 0.4*unit:",
              np.round(np.where(constant, 0.0, 0.4 * unit), 6).tolist())


rng = np.random.default_rng(3)
base = rng.normal(size=(6, 3)) * np.array([0.01, 5.0, 300.0]) + np.array([0.0, -2.0, 1e3])
run("three active outcomes on very different scales", base)
middle = base.copy()
middle[:, 1] = 0.25
run("constant outcome in the middle", middle)
one = base.copy()
one[:, 0] = 0.0
one[:, 2] = -1.5
run("one active outcome", one)
run("all outcomes constant", np.tile(np.array([0.0, 1.0, -2.0]), (6, 1)))
binary = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=float)
run("binary-valued residuals", binary)
run("three active outcomes", base, mode="per_outcome")
