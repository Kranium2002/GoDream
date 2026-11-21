"""Check 4 (lead decision M2): dreams that share any situation with training are not accepted as held out.

FIXTURE CHECK - mechanics only. A per-skill student is trained by the real engine step
(engine.train_in_dream) inside a physics-only dream of the FIXTURE spec fixture_specs.FixtureMeasureSpec
(config.yaml with skills.fixture_measure.candidate_points_per_dim added and students.snn.max_steps
reduced; both printed). Then dreams built from the training dream's rows are passed as "held out":
an exact copy, a subset, only the optimiser's rows, only the early-stopping rows, a superset (the
training dream plus fresh rows) and fresh rows plus one training row. student_record must refuse
each and state how many rows overlap; agreement_with_dream must refuse them too unless
allow_seen=True, and then label the record "includes situations seen in training". Fresh dreams
from another named stream are accepted. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c4_overlap.py
"""

from _checks import BANNER, Checks

print(BANNER)

import copy  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import FixtureMeasureSpec, concat_dreams, dream_rows, make_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.students.measure import SEEN_IN_TRAINING_LABEL, HeldOutOverlapError, agreement_with_dream, student_record  # noqa: E402
from learned_judgment.students.snn import situation_row_digests  # noqa: E402

SKILL = "fixture_measure"
POINTS_PER_DIM = (7, 5)  # fixture grid
BASE_SEED = 23  # fixture run seed
N_TRAIN, N_FRESH, N_SITUATIONS = 1000, 500, 200  # fixture sizes
MAX_STEPS = 400  # reduced from config students.snn.max_steps for a short mechanics check

checks = Checks()
file_config = load_config()
config = copy.deepcopy(file_config)
config["skills"][SKILL] = {"candidate_points_per_dim": list(POINTS_PER_DIM)}
config["students"]["snn"]["max_steps"] = MAX_STEPS
print(
    f"\nconfig: config.yaml plus skills.{SKILL}.candidate_points_per_dim={list(POINTS_PER_DIM)}; "
    f"students.snn.max_steps reduced to {MAX_STEPS} (config.yaml value {file_config['students']['snn']['max_steps']})"
)
spec = FixtureMeasureSpec()
train = make_dreams(spec, None, PHYSICS_ONLY, stream_rng(BASE_SEED, f"{SKILL}/dreams/situations"), N_TRAIN, POINTS_PER_DIM, config)
fresh = make_dreams(spec, None, PHYSICS_ONLY, stream_rng(BASE_SEED, f"{SKILL}/heldout_dreams/situations"), N_FRESH, POINTS_PER_DIM, config)
situations = spec.sample_situations(stream_rng(BASE_SEED, f"{SKILL}/measure/situations"), N_SITUATIONS)
student = train_in_dream(spec, train, config, base_seed=BASE_SEED)
deployed = student.deployed()
seen = student.meta["student"]["situations_seen"]
digests = situation_row_digests(train.situations)
optimiser_rows = [row for row, digest in enumerate(digests) if digest in set(seen["optimiser_rows"])]
early_rows = [row for row, digest in enumerate(digests) if digest in set(seen["early_stopping_rows"])]
print(
    f"  trained: {len(optimiser_rows)} optimiser rows and {len(early_rows)} early-stopping rows of the {N_TRAIN}-row training dream; "
    f"{len(deployed.training_situation_digests)} training situation digests"
)

cases = {
    # name: (dreams passed as held out, rows that overlap, total rows)
    "exact copy of the training dream": (train, N_TRAIN, N_TRAIN),
    "subset: the first 150 training rows": (dream_rows(train, np.arange(150)), 150, 150),
    "only the optimiser's rows": (dream_rows(train, optimiser_rows), len(optimiser_rows), len(optimiser_rows)),
    "only the early-stopping rows": (dream_rows(train, early_rows), len(early_rows), len(early_rows)),
    "superset: training dream + fresh rows": (concat_dreams(train, fresh), N_TRAIN, N_TRAIN + N_FRESH),
    "fresh rows + one training row": (concat_dreams(fresh, dream_rows(train, [7])), 1, N_FRESH + 1),
}

print("\n== student_record refuses every overlap and states its size ==")
for name, (dreams, overlapping, total) in cases.items():
    checks.raises(
        f"student_record, {name}",
        HeldOutOverlapError,
        lambda dreams=dreams: student_record(student, spec, dreams, situations, config, seed=BASE_SEED),
        f"{overlapping} of {total} dream rows",
    )

print("\n== agreement_with_dream refuses them unless allow_seen=True, and then labels the record ==")
for name, (dreams, overlapping, total) in cases.items():
    checks.raises(
        f"agreement_with_dream, {name}",
        HeldOutOverlapError,
        lambda dreams=dreams: agreement_with_dream(deployed, dreams),
        f"{overlapping} of {total} dream rows",
    )
subset = cases["subset: the first 150 training rows"][0]
seen_record = agreement_with_dream(deployed, subset, allow_seen=True)
fresh_record = agreement_with_dream(deployed, fresh)
print(
    f"  MECHANICS numbers: agreement with the dream's choice on 150 training rows {seen_record['student']['agreement_with_dream_choice']:.3f} "
    f"(labelled {seen_record['label'][:60]!r}...) vs on {N_FRESH} fresh rows {fresh_record['student']['agreement_with_dream_choice']:.3f}"
)
checks.expect(
    seen_record["label"].startswith(SEEN_IN_TRAINING_LABEL) and seen_record["held_out"] is False
    and seen_record["includes_situations_seen_in_training"] is True
    and seen_record["overlap_with_training"]["rows_seen_in_training"] == 150
    and seen_record["overlap_with_training"]["allow_seen"] is True,
    f"allow_seen=True on the subset: label starts with {SEEN_IN_TRAINING_LABEL!r}, held_out False, 150 rows seen",
)
checks.expect(
    fresh_record["held_out"] is True and fresh_record["overlap_with_training"]["rows_seen_in_training"] == 0
    and fresh_record["label"].startswith("held out"),
    "fresh dreams from another stream: no overlap, labelled held out",
)
fresh_allowed = agreement_with_dream(deployed, fresh, allow_seen=True)
checks.expect(fresh_allowed["held_out"] is True and fresh_allowed["label"] == fresh_record["label"], "allow_seen=True with no overlap: still labelled held out")
checks.raises("allow_seen given as a string", TypeError, lambda: agreement_with_dream(deployed, fresh, allow_seen="yes"), "allow_seen must be True or False")

print("\n== fresh dreams are accepted by student_record ==")
record = student_record(student, spec, fresh, situations, config, seed=BASE_SEED)
overlap = record["dream_provenance"]["heldout_overlap_with_training"]
checks.expect(
    overlap["rows_seen_in_training"] == 0 and overlap["n_rows"] == N_FRESH and overlap["n_training_situations"] == len(deployed.training_situation_digests)
    and record["agreement_with_heldout_dream"]["held_out"] is True,
    f"fresh dreams: the record states 0 of {N_FRESH} rows seen in training and the number of training situations",
)
checks.finish()
