"""Review reproduction: paired_difference_ci with rare binary per-episode losses (collision / fall
counts), where the bootstrap distribution is discrete. Does the BCa interval it returns as
'method_used = BCa' contain its own point estimate?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from learned_judgment.eval.bootstrap import paired_difference_ci  # noqa: E402

print("REVIEW REPRODUCTION - fixture, not evidence")

cases = {}
# 100 situations x 3 repeats, method a never collides; method b collides once / twice / in one situation every repeat
for label, cells in {
    "one collision (1 of 300 episodes)": [(5, 0)],
    "two collisions, same situation": [(5, 0), (5, 1)],
    "three collisions, one situation": [(5, 0), (5, 1), (5, 2)],
    "two collisions, two situations": [(5, 0), (9, 1)],
}.items():
    a = np.zeros((100, 3))
    b = np.zeros((100, 3))
    for s, r in cells:
        b[s, r] = 1.0
    cases[label] = (a, b)
# both methods collide sometimes, mostly agreeing
rng = np.random.default_rng(0)
a = (rng.uniform(size=(100, 3)) < 0.02).astype(float)
b = a.copy()
b[3, 0] = 1.0
cases["2% collisions, b adds one"] = (a, b)

for label, (a, b) in cases.items():
    res = paired_difference_ci(a, b, n_resamples=9999, confidence_level=0.95, method="BCa", seed=20261121)
    inside = res["low"] <= res["estimate"] <= res["high"]
    print(f"{label:36s} estimate={res['estimate']:.5f} CI=({res['low']:.5f}, {res['high']:.5f}) "
          f"method_used={res['method_used']} contains_estimate={inside} warnings={len(res['warnings'])}")
