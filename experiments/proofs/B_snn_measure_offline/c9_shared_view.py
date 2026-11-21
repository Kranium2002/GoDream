"""Check 9: the measurement code on a real view of the shared student, and engine.dream.select_rows.

FIXTURE CHECK - mechanics only. Both round-three lanes reported this combination as NOT RUN: the
measurement lane used a hand-padded view, the shared lane ran it once without saving it. Here a
SharedSNNStudent is created from the check config (config.yaml in memory with the fixture
shared_feature_width and reduced max_steps and latency_decisions, printed below), one fixture skill
is added, and measure.student_record is run on ``shared.deployed(skill)``:

  * on the skill's retention rows (never trained on, never used for early stopping) it returns a
    record for that view's network and states that no row was seen in training;
  * on the skill's whole stored dream it refuses, naming how many rows the student saw.

The retention rows are selected with engine.dream.select_rows, whose contract is checked too.
Every number printed is a MECHANICS number on a fixture dream, not evidence about the robot, the
simulator, TabPFN or a model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c9_shared_view.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "B_snn_shared_offline"))

from fixture_shared_specs import FIXTURE_FEATURE_WIDTH, POINTS, Checks, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.dream import keep_confident, select_rows  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import measure  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn import DeployedStudent, situation_digests  # noqa: E402

MAX_STEPS = 300  # reduced from config students.snn.max_steps for a short mechanics check
LATENCY_DECISIONS = 100  # reduced from config evaluation.latency_decisions for a short mechanics check
BASE_SEED = 9  # fixture run seed
SKILL = "fx_alpha"

checks = Checks()
config = load_config()
check_config = fixture_config(config)
check_config["students"]["snn"]["max_steps"] = MAX_STEPS
check_config["evaluation"]["latency_decisions"] = LATENCY_DECISIONS
print(
    f"\ncheck config: config.yaml with students.snn.shared_feature_width={FIXTURE_FEATURE_WIDTH} (fixture), "
    f"students.snn.max_steps={MAX_STEPS} (config.yaml {config['students']['snn']['max_steps']}) and "
    f"evaluation.latency_decisions={LATENCY_DECISIONS} (config.yaml {config['evaluation']['latency_decisions']})"
)

spec = fixture_spec(SKILL)
dreams = fixture_dreams(spec, config)

print("\n== engine.dream.select_rows ==")
rows = np.array([5, 2, 9])
picked = select_rows(dreams, rows)
checks.expect(
    picked.n_rows == 3 and np.array_equal(picked.situations, dreams.situations[rows])
    and np.array_equal(picked.scores, dreams.scores[rows]) and np.array_equal(picked.chosen, dreams.chosen[rows]),
    "indices select those rows in the given order",
)
mask = np.zeros(dreams.n_rows, dtype=bool)
mask[[2, 5, 9]] = True
masked = select_rows(dreams, mask)
checks.expect(np.array_equal(masked.situations, dreams.situations[[2, 5, 9]]), "a boolean mask selects the marked rows in row order")
checks.expect(picked.method == dreams.method and picked.skill == dreams.skill and picked.n_candidates == dreams.n_candidates,
              "the header (skill, method, candidate count) is kept")
checks.expect(select_rows(dreams, np.empty(0, dtype=np.int64)).n_rows == 0, "selecting no rows gives an empty DreamSet")
# keep_confident is now written with select_rows; it must keep exactly the rows at or below the
# threshold. Thresholds: the smallest, the median and the largest spread of the fixture dream.
for label, threshold in (
    ("smallest", float(dreams.chosen_spread.min())),
    ("median", float(np.median(dreams.chosen_spread))),
    ("largest", float(dreams.chosen_spread.max())),
):
    kept = keep_confident(dreams, threshold)
    expected = dreams.chosen_spread <= threshold
    checks.expect(
        kept.n_rows == int(expected.sum()) and np.array_equal(kept.situations, dreams.situations[expected])
        and np.array_equal(kept.chosen_spread, dreams.chosen_spread[expected]),
        f"keep_confident at the {label} spread ({threshold:.6g}) keeps exactly the rows at or below it ({kept.n_rows} of {dreams.n_rows})",
    )
checks.raises("a repeated row index is refused", ValueError, lambda: select_rows(dreams, [1, 1]), "must not repeat")
checks.raises("a row index outside the set is refused", ValueError, lambda: select_rows(dreams, [dreams.n_rows]), "must lie in")
checks.raises("a mask of the wrong length is refused", ValueError, lambda: select_rows(dreams, mask[:-1]), "must have length")
checks.raises("non-integer row indices are refused", ValueError, lambda: select_rows(dreams, [0.5]), "integer indices")

print("\n== a shared student holding one skill ==")
student = SharedSNNStudent.create(check_config)
student.add_skill(spec, dreams, points_per_dim=POINTS[SKILL], base_seed=BASE_SEED, rehearsal=True)
view = student.deployed(SKILL)
retention = select_rows(student.stored_dreams(SKILL), student.retention_rows(SKILL))
print(
    f"  slots {student.slots}; input width {student.input_width}; retention rows {retention.n_rows} of "
    f"{student.stored_dreams(SKILL).n_rows}; view network digest {view.digest[:12]}"
)
checks.expect(isinstance(view, DeployedStudent), "shared.deployed(skill) is a DeployedStudent")
checks.expect(
    situation_digests(retention.situations) == set(student.retention_situation_digests(SKILL)),
    "the selected rows are exactly the skill's retention situations",
)
checks.expect(not view.seen_in_training(retention.situations).any(), "the view saw none of the retention situations in training")

print("\n== measure.student_record on the shared view ==")
situations = spec.sample_situations(stream_rng(BASE_SEED, f"{SKILL}/check/measure_situations"), 200)
record = measure.student_record(view, spec, retention, situations, check_config, seed=BASE_SEED)
text = json.dumps(record, allow_nan=False)
print(f"  record keys: {sorted(record)}")
print(f"  record size {len(text)} bytes; JSON without NaN: yes")
checks.expect(view.digest in text, "the record names the view's network digest")
checks.expect(student.network_view(SKILL).digest() == view.digest, "that digest is the shared student's network view of the skill")
checks.expect(
    f'"rows_seen_in_training": 0' in text,
    "the record states that 0 held-out rows were seen in training",
)
first_layer = student.input_width * student.settings.hidden_sizes[0]
checks.expect(
    str(first_layer) in text,
    f"the operation counts use the padded input: first layer {student.input_width} x {student.settings.hidden_sizes[0]} = {first_layer} MACs",
)

checks.raises(
    "the skill's whole stored dream is refused as held-out data",
    measure.HeldOutOverlapError,
    lambda: measure.student_record(view, spec, student.stored_dreams(SKILL), situations, check_config, seed=BASE_SEED),
    "dream rows",
)

checks.finish()
