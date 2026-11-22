"""R04: does the number of rows n change the draws of earlier rows?
(a) SkillSpec.sample_situations (default, used by dream() and eval.splits) draws column by
    column, so row i's 2nd..S-th columns depend on n.
(b) exploration_actions draws all n explore flags, then all n picks, so row i's pick depends on n.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, ReviewSpec  # noqa: E402

from learned_judgment.engine.collect import exploration_actions  # noqa: E402
from learned_judgment.engine.dream import dream  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402

spec = ReviewSpec()
layout = build_layout(spec)

s5 = spec.sample_situations(stream_rng(7, "rv/context/situations"), 5)
s40 = spec.sample_situations(stream_rng(7, "rv/context/situations"), 40)
print("A. sample_situations(n=5) == sample_situations(n=40)[:5] ?", np.array_equal(s5, s40[:5]))
print("   column 0 equal:", np.array_equal(s5[:, 0], s40[:5, 0]), "| column 1 (categorical) equal:", np.array_equal(s5[:, 1], s40[:5, 1]))
print("   n=5  surface:", s5[:, 1].tolist())
print("   n=40 surface:", s40[:5, 1].tolist())

d100 = dream(spec, None, layout, n_dreams=100, rng=stream_rng(7, "rv/dreams/situations"), points_per_dim=[3, 3],
             quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)
d1000 = dream(spec, None, layout, n_dreams=1000, rng=stream_rng(7, "rv/dreams/situations"), points_per_dim=[3, 3],
              quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)
print("B. 100-dream situations are the first 100 of the 1000-dream set ?", np.array_equal(d100.situations, d1000.situations[:100]))

ctx = s40
a20 = exploration_actions(spec, ctx[:20], rng=stream_rng(7, "rv/context/actions"), points_per_dim=[3, 3],
                          exploration_fraction=1.0, quantile_levels=LEVELS, spread_levels=SPREAD, layout=layout)
a40 = exploration_actions(spec, ctx, rng=stream_rng(7, "rv/context/actions"), points_per_dim=[3, 3],
                          exploration_fraction=1.0, quantile_levels=LEVELS, spread_levels=SPREAD, layout=layout)
same = np.all(a20 == a40[:20], axis=1)
print("C. exploration_actions(n=20) rows equal to the first 20 rows of n=40:", int(same.sum()), "/ 20")
