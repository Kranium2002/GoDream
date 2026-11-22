"""R03: are the shared student's "held-out" rows really unseen when dream situations repeat?

add_skill splits dream ROWS (stream "<skill>/snn/validation_split"), never situations. With a
categorical-only situation (finitely many distinct situations) or a dream set that contains repeated
situations, held-out rows are copies of training rows: same situation, same dreamed scores. The record
still calls them held_out_in_dream and nothing reports the overlap.

Two FIXTURE skills: fx_cat (situation = two categorical variables, 2 x 3 = 6 distinct situations) and
fx_beta whose 300-row dream is two copies of the same 150 rows (e.g. two dream batches drawn from one
stream). Tiny training (60 steps) - only the bookkeeping matters here.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\snn_shared_measure\\r03_duplicate_situations_heldout.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import POINTS, FixtureSharedSpec, concat, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.dream import dream  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.skills.base import ActionDim, Variable  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(fixture_config(config)), max_steps=60, evaluate_every_steps=20)
cat_spec = FixtureSharedSpec(
    "fx_cat",
    (
        Variable("fx_c_floor", "categorical", "label", "fixture floor", levels=("tile", "carpet")),
        Variable("fx_c_load", "categorical", "label", "fixture load", levels=("none", "light", "heavy")),
    ),
    (ActionDim("fx_c_speed", "float", "m/s", "fixture speed", low=0.0, high=0.9, role="command", command_field="fixture_vx"),),
    mixing_seed=7,
)
cat_dreams = dream(
    cat_spec, None, build_layout(cat_spec), n_dreams=300, rng=stream_rng(5, "fx_cat/dreams/situations"), points_per_dim=(9,),
    quantile_levels=require(config, "engine.quantile_levels"), spread_levels=tuple(require(config, "engine.spread_levels")),
    method=PHYSICS_ONLY,
)
beta_spec = fixture_spec("fx_beta")
half = fixture_dreams(beta_spec, config, 150)
beta_dreams = concat(half, half)

student = SharedSNNStudent(skills=["fx_cat", "fx_beta"], feature_width=5, settings=settings, rehearsal_fraction=0.5)
for spec, dreams, points in ((cat_spec, cat_dreams, (9,)), (beta_spec, beta_dreams, POINTS["fx_beta"])):
    record = student.add_skill(spec, dreams, points_per_dim=points, base_seed=4, rehearsal=True)
    held_out = student.held_out_rows(spec.name)
    train_rows = np.setdiff1d(np.arange(dreams.n_rows), held_out)
    seen = {situation.tobytes() for situation in dreams.situations[train_rows]}
    duplicated = sum(situation.tobytes() in seen for situation in dreams.situations[held_out])
    distinct = len({situation.tobytes() for situation in dreams.situations})
    numbers = record["held_out_in_dream"][spec.name]
    print(
        f"{spec.name}: {dreams.n_rows} dream rows, {distinct} distinct situations; held-out rows {held_out.size}, of which "
        f"{duplicated} have a situation that is also a TRAINING row ({duplicated / held_out.size:.0%}); "
        f"record 'held_out_in_dream' agreement {numbers['agreement_with_dream_choice']:.3f}"
    )
    print(f"    record keys about the held-out rows: {sorted(numbers)} (no overlap / duplicate count)")
