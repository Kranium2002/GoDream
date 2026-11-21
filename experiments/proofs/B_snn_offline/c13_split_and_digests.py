"""Check 13 (decisions M2/M4): rows are split by distinct situation; training situations are recorded.

FIXTURE CHECK - mechanics only. Old behaviour (review findings "held-out rows can overlap training
rows", reproductions experiments/proofs/B_review/snn_shared_measure/r02 and r03): the dream ROWS
were permuted, so a repeated situation could be a training row and an early-stopping row at once,
and the student stored only one sha256 over all its rows, so a later held-out set that is a
subset, superset or filtered copy of the training dream could not be detected. Now the rows are
split by DISTINCT situation (identical situations stay on one side), and meta['student'] stores
the sha256 of every situation the optimiser or early stopping saw; situation_digests() returns
the digest set of any situations so that later code can refuse overlapping held-out dreams.
Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c13_split_and_digests.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.dream import DreamSet  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import (  # noqa: E402
    SNNStudent,
    situation_digests,
    situation_row_digests,
    split_rows_by_situation,
)
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

BASE_SEED = 29  # fixture run seed

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=150, evaluate_every_steps=50)
spec = FixtureSnnSpec()
base = fixture_dreams(spec, config, 300)


def stacked(dreams: DreamSet, copies: int) -> DreamSet:
    """FIXTURE: the dream's rows repeated ``copies`` times (every situation appears ``copies`` times).
    Every per-row array field (an array whose first axis has n_rows entries) is repeated."""
    per_row = {
        item.name: np.concatenate([getattr(dreams, item.name)] * copies)
        for item in dataclasses.fields(dreams)
        if isinstance(getattr(dreams, item.name), np.ndarray) and getattr(dreams, item.name).shape[:1] == (dreams.n_rows,)
    }
    return dataclasses.replace(dreams, **per_row)


print("\n== situation digests ==")
digests = situation_row_digests(base.situations)
checks.expect(len(digests) == base.n_rows and len(set(digests)) == base.n_rows, f"{base.n_rows} distinct fixture situations -> {base.n_rows} distinct digests")
checks.expect(situation_digests(base.situations) == set(digests), "situation_digests returns the set of the row digests")
checks.expect(situation_row_digests(base.situations[:5].copy()) == digests[:5], "the digest depends on the values only (copies give the same digests)")
negative_zero = np.array([[-0.0, 3.0, 1.0]])
checks.expect(situation_row_digests(negative_zero) == situation_row_digests(np.array([[0.0, 3.0, 1.0]])), "-0.0 and 0.0 give the same digest")
nudged = base.situations[:1].copy()
nudged[0, 0] = np.nextafter(nudged[0, 0], 2.0)
checks.expect(situation_row_digests(nudged) != digests[:1], "one ulp of difference gives another digest")
checks.raises("digests of NaN situations", ValueError, lambda: situation_digests(np.array([[np.nan, 1.0, 0.0]])), "non-finite")

print("\n== split_rows_by_situation ==")
fraction = settings.validation_fraction
train, held = split_rows_by_situation(base.situations, fraction=fraction, rng=stream_rng(BASE_SEED, "fixture_snn/snn/validation_split"))
order = stream_rng(BASE_SEED, "fixture_snn/snn/validation_split").permutation(base.n_rows)
n_held = max(1, int(round(fraction * base.n_rows)))
checks.expect(
    bool(np.array_equal(held, np.sort(order[:n_held])) and np.array_equal(train, np.sort(order[n_held:]))),
    "with distinct situations the split equals the previous row permutation (same rows on each side)",
)
doubled = stacked(base, 2)
train2, held2 = split_rows_by_situation(doubled.situations, fraction=fraction, rng=stream_rng(BASE_SEED, "fixture_snn/snn/validation_split"))
digests2 = situation_row_digests(doubled.situations)
overlap = {digests2[i] for i in train2} & {digests2[i] for i in held2}
checks.expect(not overlap, f"dream with every situation twice: no situation on both sides ({held2.size} held rows, {train2.size} training rows)")
checks.expect(held2.size == 2 * n_held, f"{n_held} distinct situations held -> {2 * n_held} rows")
old_order = stream_rng(BASE_SEED, "fixture_snn/snn/validation_split").permutation(doubled.n_rows)
old_held = old_order[: max(1, int(round(fraction * doubled.n_rows)))]
old_overlap = {digests2[i] for i in old_order[len(old_held):]} & {digests2[i] for i in old_held}
print(f"  (old row split on the same dream: {len(old_overlap)} of {len(old_held)} early-stopping rows have a training copy)")
checks.expect(len(old_overlap) > 0, "the old row permutation puts copies of one situation on both sides of this dream")
checks.raises("one distinct situation", ValueError, lambda: split_rows_by_situation(np.ones((10, 3)), fraction=fraction, rng=np.random.default_rng(0)), "leaves no training rows")

print("\n== training on a dream whose situations repeat ==")
student = SNNStudent.train_in_dream(spec, stacked(base, 3), settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
seen = student.meta["student"]["situations_seen"]
dream_meta = student.meta["dream"]
print(f"  dream: {dream_meta['n_rows_used']} rows, {dream_meta['n_distinct_situations']} distinct situations, "
      f"{dream_meta['n_rows_train']} training rows, {dream_meta['n_rows_early_stopping']} early-stopping rows; split: {dream_meta['split']}")
optimiser, early = set(seen["optimiser_rows"]), set(seen["early_stopping_rows"])
checks.expect(not optimiser & early, "no situation is both an optimiser row and an early-stopping row")
checks.expect(optimiser | early == set(digests), "every situation of the dream is recorded, as an optimiser or an early-stopping situation")
checks.expect(
    dream_meta["n_distinct_situations"] == base.n_rows and dream_meta["n_rows_early_stopping"] == 3 * len(early),
    "meta counts 300 distinct situations; each early-stopping situation contributes its 3 copies",
)
checks.expect(seen["optimiser_rows"] == sorted(optimiser) and "sha256" in seen["definition"], "digests are stored sorted, with their definition")
checks.expect(student.training_situation_digests == frozenset(digests), "student.training_situation_digests = every situation training saw")

print("\n== later code can find overlapping held-out dreams ==")
fresh = fixture_dreams(spec, config, 200, stream="fixture_snn/check/heldout")
checks.expect(not situation_digests(fresh.situations) & student.training_situation_digests, "fresh dreams from another stream: no overlap")
subset = base.situations[10:60]
checks.expect(bool(np.all(student.deployed().seen_in_training(subset))), "a subset of the training dream: every row is flagged as seen")
mixed = np.vstack([fresh.situations[:20], base.situations[:5]])
flags = student.deployed().seen_in_training(mixed)
checks.expect(bool(np.array_equal(flags, np.r_[np.zeros(20, bool), np.ones(5, bool)])), "a superset: exactly the 5 training rows are flagged")
checks.finish()
