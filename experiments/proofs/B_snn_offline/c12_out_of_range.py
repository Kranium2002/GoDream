"""Check 12 (decision S7): situations outside the declared ranges are refused or clipped and counted.

FIXTURE CHECK - mechanics only. Old behaviour (review finding "out-of-range situations are decided
on silently", reproduction experiments/proofs/B_review/snn_training/r01): the student refused
non-finite values only; a value outside its declared range was extrapolated by the encoder
(1.7e6 for one fixture feature) and decided on with nothing to mark it. Now
students.snn.out_of_range is "error" (refuse, naming the feature, value, row and range) or "clip"
(clip into the range and count the clipped values in the student's out_of_range log); training
refuses dream situations outside the declared ranges. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c12_out_of_range.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import OutOfRangeError, SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import OUT_OF_RANGE_POLICIES, SNNSettings  # noqa: E402

BASE_SEED = 23  # fixture run seed

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=150, evaluate_every_steps=50)
print(f"\nconfig students.snn.out_of_range = {require(config, 'students.snn.out_of_range')!r}; policies {OUT_OF_RANGE_POLICIES}")
checks.expect(SNNSettings.from_config(config).out_of_range == require(config, "students.snn.out_of_range"), "SNNSettings.out_of_range comes from config.yaml")
checks.raises("SNNSettings with out_of_range 'ignore'", ValueError, lambda: dataclasses.replace(settings, out_of_range="ignore"), "students.snn.out_of_range")

spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 400)
strict = SNNStudent.train_in_dream(spec, dreams, settings=dataclasses.replace(settings, out_of_range="error"), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
lenient = SNNStudent.train_in_dream(spec, dreams, settings=dataclasses.replace(settings, out_of_range="clip"), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
checks.expect(strict.digest == lenient.digest, "the policy does not change training: same network from the same seed")
checks.expect(
    strict.meta["student"]["out_of_range"] == "error" and lenient.meta["student"]["out_of_range"] == "clip",
    "the policy is stored in the student record (it travels with a saved student)",
)

situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 20)
outside = situations.copy()
outside[3, 1] = 9.0  # fx_speed declared in [2, 6]
outside[7, 0] = -1.5  # fx_lateral declared in [-1, 1]
outside[7, 1] = 1.0
edges = situations.copy()
edges[0, 0], edges[1, 0], edges[2, 1], edges[4, 1] = -1.0, 1.0, 2.0, 6.0

print("\n== policy 'error' ==")
for label, function in (
    ("decide_index", lambda: strict.decide_index(outside)),
    ("decide", lambda: strict.decide(spec, outside)),
    ("decide_index_one", lambda: strict.decide_index_one(outside[3])),
    ("decide_one", lambda: strict.decide_one(spec, outside[3])),
    ("operation_counts", lambda: strict.operation_counts(outside)),
):
    checks.raises(f"{label} with fx_speed = 9.0", OutOfRangeError, function, "'fx_speed' = 9.0")
caught = checks.raises("the message gives the range and the count", OutOfRangeError, lambda: strict.decide_index(outside), "[2.0, 6.0] (3 value(s) out of range in total)")
checks.expect(isinstance(caught, ValueError), "OutOfRangeError is a ValueError")
checks.expect(strict.decide_index(edges).shape == (20,), "values exactly at the declared bounds are accepted")
checks.expect(strict.out_of_range_log.as_dict()["values_clipped"] == 0, "nothing is ever clipped under 'error'")

print("\n== policy 'clip' ==")
clipped_by_hand = np.clip(outside, [-1.0, 2.0, -np.inf], [1.0, 6.0, np.inf])
checked, mask = lenient.deployed().check_situations(outside)
checks.expect(bool(np.array_equal(checked, clipped_by_hand)), "check_situations clips into [low, high]")
checks.expect(
    mask.sum() == 3 and bool(mask[3, 1] and mask[7, 0] and mask[7, 1]),
    "check_situations returns the (n, S) mask of clipped values (3 values in 2 situations)",
)
before = lenient.out_of_range_log.as_dict()
index = lenient.decide_index(outside)
checks.expect(bool(np.array_equal(index, lenient.decide_index(clipped_by_hand))), "decide_index on out-of-range situations = decide_index on the clipped situations")
log = lenient.out_of_range_log.as_dict()
print(f"  out_of_range log: {log}")
checks.expect(
    log["decisions"] - before["decisions"] == 40 and log["decisions_with_clipping"] - before["decisions_with_clipping"] == 2
    and log["values_clipped"] - before["values_clipped"] == 3,
    "the log counts 40 decisions (two calls of 20), 2 with clipping, 3 clipped values",
)
checks.expect(log["per_feature"] == {"fx_lateral": 1, "fx_speed": 2, "fx_surface": 0}, "the log counts clipped values per feature")
one = lenient.decide_index_one(outside[3])
checks.expect(one == int(index[3]), "decide_index_one clips the same way as the batched path")
action = lenient.decide_one(spec, outside[7])
checks.expect(bool(np.array_equal(action, lenient.decide(spec, clipped_by_hand)[7])), "decide_one on a clipped situation = decide on the clipped situation")
checks.expect(lenient.out_of_range_log.as_dict()["values_clipped"] == 3 + 1 + 2, "single decisions are counted too (1 + 2 more clipped values)")

print("\n== refused under both policies ==")
for student in (strict, lenient):
    policy = student.meta["student"]["out_of_range"]
    bad_level = situations.copy()
    bad_level[5, 2] = 2.0  # fx_surface has 2 levels
    checks.raises(f"{policy}: an invalid categorical level index", ValueError, lambda s=student, b=bad_level: s.decide_index(b), "'fx_surface' = 2.0 at row 5 is not a level index")
    nan = situations.copy()
    nan[6, 0] = np.nan
    checks.raises(f"{policy}: NaN", ValueError, lambda s=student, b=nan: s.decide_index(b), "non-finite values (first at row 6, feature 'fx_lateral')")

print("\n== training refuses dream situations outside the declared ranges ==")
wide = dataclasses.replace(dreams, situations=np.where(np.arange(dreams.n_rows)[:, None] == 10, dreams.situations * [1.0, 2.0, 1.0], dreams.situations))
print(f"  dream row 10 fx_speed set to {wide.situations[10, 1]:.3f}")
checks.raises(
    "a dream with fx_speed outside [2, 6]",
    ValueError,
    lambda: SNNStudent.train_in_dream(spec, wide, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED),
    "dream row 10: situation feature 'fx_speed'",
)
checks.finish()
