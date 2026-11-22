"""Check 10 (decision S5): the soft targets' score scale is global_median or per_row.

FIXTURE CHECK - mechanics only. Old behaviour (review finding "one global median spread flattens
small-scale rows", reproduction experiments/proofs/B_review/snn_training/r05 part d): the
temperature was always in units of ONE median spread over all training rows, so on rows whose
scores vary on a much smaller scale than the median row the targets were practically uniform.
Now students.snn.target_spread selects "global_median" (as before) or "per_row" (each row divided
by its own score spread; a row with zero spread gets the uniform target), soft_targets implements
both, and training records the mode. The mixed-scale score matrix below is a hand-made fixture
for the pure target function; the trained students use the fixture dream. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c10_target_spread.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402
from learned_judgment.students.targets import TARGET_SPREADS, flat_rows, row_spreads, score_spread, soft_targets  # noqa: E402

BASE_SEED = 17  # fixture run seed
C, N_LARGE, N_SMALL = 35, 110, 90  # fixture: 55% large-scale rows, 45% small-scale rows

checks = Checks()
config = load_config()
temperature = float(require(config, "students.snn.target_temperature"))
print(f"\ntemperature {temperature} (config); config target_spread {require(config, 'students.snn.target_spread')!r}; modes {TARGET_SPREADS}")
checks.expect(SNNSettings.from_config(config).target_spread == require(config, "students.snn.target_spread"), "SNNSettings.target_spread comes from config.yaml")

print("\n== a mixed-scale score matrix (hand-made fixture) ==")
rng = np.random.default_rng(1)
motion = np.stack([rng.permutation(C) / (C - 1.0) for _ in range(N_LARGE + N_SMALL)])  # 0..1 per candidate
collision = rng.random((N_LARGE, C)) < 0.4
scores = -0.05 * motion
scores[:N_LARGE] -= 10.0 * collision  # a large collision penalty on some candidates of the large-scale rows
small = slice(N_LARGE, None)
spread = score_spread(scores)
print(f"  global spread (median row std) {spread:.4f}; small-scale row std ~ {np.median(row_spreads(scores[small])):.4f}")
global_targets = soft_targets(scores, temperature=temperature, target_spread="global_median", spread=spread)
row_targets = soft_targets(scores, temperature=temperature, target_spread="per_row", spread=None)
global_max = float(global_targets[small].max(axis=1).mean())
row_max = float(row_targets[small].max(axis=1).mean())
print(f"  small-scale rows: mean max target probability global_median {global_max:.4f}, per_row {row_max:.4f} (uniform {1 / C:.4f})")
checks.expect(global_max < 1.1 / C, "global_median (the old behaviour): small-scale rows get practically uniform targets")
checks.expect(row_max > 3.0 / C, "per_row: small-scale rows get targets concentrated on their best candidates")
ranks_kept = all(
    np.array_equal(np.argsort(-row_targets[i], kind="stable"), np.argsort(-scores[i], kind="stable")) for i in range(scores.shape[0])
)
checks.expect(ranks_kept, "per_row: every row's targets rank the candidates exactly as its scores do")
for name, targets in (("global_median", global_targets), ("per_row", row_targets)):
    error = float(np.max(np.abs(targets.sum(axis=1) - 1.0)))
    checks.expect(error <= 1e-12 and bool(np.all(targets >= 0.0)), f"{name}: rows are distributions (max |sum - 1| = {error:.1e})")
    checks.expect(bool(np.array_equal(np.argmax(targets, axis=1), np.argmax(scores, axis=1))), f"{name}: argmax of every row = argmax of its scores")
alone = soft_targets(scores[N_LARGE : N_LARGE + 1], temperature=temperature, target_spread="per_row", spread=None)
checks.expect(bool(np.array_equal(alone[0], row_targets[N_LARGE])), "per_row: a row's targets do not depend on the other rows")
scaled = soft_targets(7.0 * scores + 3.0, temperature=temperature, target_spread="per_row", spread=None)
checks.expect(float(np.max(np.abs(scaled - row_targets))) < 1e-12, "per_row: invariant to a positive affine change of the scores (rounding level)")

print("\n== flat rows and argument errors ==")
mixed = np.vstack([np.zeros((1, 4)), np.array([[0.0, 1.0, 2.0, 3.0]])])
flat = soft_targets(mixed, temperature=temperature, target_spread="per_row", spread=None)
checks.expect(bool(np.array_equal(flat[0], np.full(4, 0.25))), "per_row: a row whose candidates all score the same gets the uniform target")
tenths = np.full((1, 3), 0.1)
checks.expect(
    bool(flat_rows(tenths)[0]) and float(row_spreads(tenths)[0]) != 0.0
    and bool(np.array_equal(soft_targets(tenths, temperature=temperature, target_spread="per_row", spread=None)[0], np.full(3, 1 / 3))),
    f"per_row: a flat row whose computed std is {float(row_spreads(tenths)[0]):.1e} (rounding of the mean) is still flat -> uniform",
)
checks.raises("per_row with a spread", ValueError, lambda: soft_targets(scores, temperature=1.0, target_spread="per_row", spread=1.0), "spread must be None")
checks.raises("global_median without a spread", ValueError, lambda: soft_targets(scores, temperature=1.0, target_spread="global_median", spread=None), "needs spread")
checks.raises("an unknown mode", ValueError, lambda: soft_targets(scores, temperature=1.0, target_spread="median", spread=1.0), "target_spread must be one of")
checks.raises(
    "per_row on a non-flat row whose std underflows to 0", ValueError,
    lambda: soft_targets(mixed * 1e-200, temperature=1.0, target_spread="per_row", spread=None), "not a finite number > 0",
)
checks.raises(
    "per_row with temperature x spread underflowing to 0", ValueError,
    lambda: soft_targets(mixed * 1e-160, temperature=1e-200, target_spread="per_row", spread=None), "not a finite number > 0",
)
checks.raises("SNNSettings with target_spread 'median'", ValueError, lambda: dataclasses.replace(SNNSettings.from_config(config), target_spread="median"), "students.snn.target_spread")

print("\n== training uses the configured mode ==")
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 500)
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=150, evaluate_every_steps=50)
fixture_targets = soft_targets(dreams.scores, temperature=temperature, target_spread="per_row", spread=None)
checks.expect(bool(np.array_equal(np.argmax(fixture_targets, axis=1), dreams.chosen)), "per_row on the fixture dream: argmax of every row = the dream's choice")
students = {}
for mode in TARGET_SPREADS:
    students[mode] = SNNStudent.train_in_dream(
        spec, dreams, settings=dataclasses.replace(settings, target_spread=mode), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED
    )
    record = students[mode].meta["dream"]
    print(f"  {mode}: digest {students[mode].digest[:16]}..., meta dream.target_spread {record['target_spread']!r}, "
          f"score_spread {record['score_spread']}, training_row_spread {record['training_row_spread']}")
    checks.expect(record["target_spread"] == mode and students[mode].meta["settings"]["target_spread"] == mode, f"{mode}: the mode is recorded in meta")
checks.expect(students["per_row"].meta["dream"]["score_spread"] is None, "per_row: no global score_spread is recorded")
checks.expect(students["global_median"].digest != students["per_row"].digest, "the two modes train different networks from the same seed (the setting reaches training)")
flat_dreams = dataclasses.replace(dreams, scores=np.zeros_like(dreams.scores))
checks.raises(
    "per_row training on a dream whose rows are all flat", ValueError,
    lambda: SNNStudent.train_in_dream(spec, flat_dreams, settings=dataclasses.replace(settings, target_spread="per_row"), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED),
    "score every candidate identically",
)
checks.finish()
