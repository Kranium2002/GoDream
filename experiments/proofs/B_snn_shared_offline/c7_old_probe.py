"""Check 7 worker: the scenarios of checks (a)-(e) run through the OLD API of the shared student.

FIXTURE CHECK - mechanics only. Started by c7_old_behaviour.py inside its temporary overlay B, where
the student package (students/shared_snn.py and the modules it ran with) is the version of the
commit that last changed the committed shared_snn.py. It uses only names and signatures that code
had, and prints one line per decision: "OLD <item>: DEFECT <what the old code did>" or
"OLD <item>: NO DEFECT <...>". It is not meant to run against the current code (the API changed).
Writes only inside a temporary directory that it removes.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_shared_specs import FIXTURE_FEATURE_WIDTH, POINTS, fixture_dreams, fixture_spec, repeated_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

MAX_STEPS = 100  # reduced: the old code only has to run, not to learn well
BASE_SEED = 7  # fixture run seed


def report(item: str, defect: bool, text: str) -> None:
    print(f"OLD {item}: {'DEFECT' if defect else 'NO DEFECT'} {text}")


def row_set(array: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in np.ascontiguousarray(array, dtype=np.float64)}


config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=MAX_STEPS, evaluate_every_steps=50)
fraction = require(config, "students.snn.rehearsal_fraction")
alpha, beta = fixture_spec("fx_alpha"), fixture_spec("fx_beta")
dreams = {"fx_alpha": fixture_dreams(alpha, config), "fx_beta": fixture_dreams(beta, config)}
validation_calls: list[dict[str, np.ndarray]] = []
real_fit = snn_train.fit


def recording_fit(model, tasks, **kwargs):  # noqa: ANN001, ANN003, ANN201
    validation_calls.append({head: np.array(pair[0], copy=True) for head, pair in kwargs["validation"].items()})
    return real_fit(model, tasks, **kwargs)


snn_train.fit = recording_fit


def old_student(skills: list[str]) -> SharedSNNStudent:
    return SharedSNNStudent(skills=skills, feature_width=FIXTURE_FEATURE_WIDTH, settings=settings, rehearsal_fraction=fraction)


# H1 / H2: reserved slots; one skill held and a second added later.
one = old_student(["fx_alpha"])
one.add_skill(alpha, dreams["fx_alpha"], points_per_dim=POINTS["fx_alpha"], base_seed=BASE_SEED, rehearsal=True)
try:
    one.add_skill(beta, dreams["fx_beta"], points_per_dim=POINTS["fx_beta"], base_seed=BASE_SEED, rehearsal=True)
    report("H1", False, "fx_beta was added to a student created for ['fx_alpha']")
except ValueError as error:
    report(
        "H1",
        True,
        f"no reserved slots: input width {one.input_width} = feature width {FIXTURE_FEATURE_WIDTH} + one input per skill NAMED "
        f"at creation; a student holding fx_alpha cannot take a second skill later: {str(error)[:140]}",
    )

# H3: the rows of the retention matrix are the early-stopping rows fit used.
two = old_student(["fx_alpha", "fx_beta"])
before = len(validation_calls)
two.add_skill(alpha, dreams["fx_alpha"], points_per_dim=POINTS["fx_alpha"], base_seed=BASE_SEED, rehearsal=True)
two.add_skill(beta, dreams["fx_beta"], points_per_dim=POINTS["fx_beta"], base_seed=BASE_SEED, rehearsal=True)
held_rows = two.held_out_rows("fx_alpha")
encoded_held = two._encoder.encode("fx_alpha", InputEncoder.from_variables(alpha.situation), dreams["fx_alpha"].situations[held_rows])
same = [row_set(call["fx_alpha"]) == row_set(encoded_held) for call in validation_calls[before:] if "fx_alpha" in call]
report(
    "H3",
    len(same) == 2 and all(same),
    f"the held-out rows behind fx_alpha's retention numbers ({held_rows.size} rows) are exactly the early-stopping rows "
    f"fit received at {sum(same)} of {len(same)} adds (its own add and the rehearsal add of fx_beta)",
)
repeated = old_student(["fx_beta"])
doubled = repeated_dreams(dreams["fx_beta"], 2)
repeated.add_skill(beta, doubled, points_per_dim=POINTS["fx_beta"], base_seed=BASE_SEED, rehearsal=True)
n = dreams["fx_beta"].n_rows
held_doubled = repeated.held_out_rows("fx_beta")
train_doubled = set(np.setdiff1d(np.arange(2 * n), held_doubled).tolist())
duplicated = sum(1 for row in held_doubled if (row + n) % (2 * n) in train_doubled)
report(
    "H3b",
    duplicated > 0,
    f"rows split by row: on a dream with every situation twice, {duplicated} of {held_doubled.size} held-out rows have a copy "
    "among the training rows",
)

# H5: no deployed view, no earlier stage.
report("H5a", not hasattr(two, "deployed"), "SharedSNNStudent has no deployed(skill, stage): the per-skill measurement code has no object to run on")
stage0 = two.records[0]["network_digests"]["fx_alpha"]
try:
    two.network_view("fx_alpha", 0)
    report("H5b", False, "network_view accepted a stage")
except TypeError:
    scratch = Path(tempfile.mkdtemp(prefix="b_snn_shared_c7_old_"))
    try:
        two.save(scratch / "student")
        files = sorted(path.relative_to(scratch / "student").as_posix() for path in (scratch / "student").rglob("*.npz"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report(
        "H5b",
        stage0 != two.network_view("fx_alpha").digest(),
        f"network_view takes no stage; fx_alpha's view before fx_beta was added (digest {stage0[:12]}...) survives only as a "
        f"digest in the records; save wrote {files}",
    )

# H4: out-of-range situations and changed grids.
situations = dreams["fx_alpha"].situations[:20].copy()
far = situations.copy()
far[3, 1] = 9.0
try:
    actions = two.decide(alpha, far)
    report("H4a", True, f"decide on fx_a_speed = 9.0 (declared [0.5, 2.0]) returned {actions.shape} actions with no error and no count")
except ValueError as error:
    report("H4a", False, f"refused: {error}")
try:
    narrow = two.decide(fixture_spec("fx_alpha", alpha_vy_bounds=(-0.3, 0.3)), situations)
    changed = int(np.sum(np.any(narrow != two.decide(alpha, situations), axis=1)))
    report("H4b", True, f"decide with side-speed bounds [-0.3, 0.3] instead of [-0.6, 0.6] returned actions with no error ({changed} of 20 commands changed)")
except ValueError as error:
    report("H4b", False, f"refused: {error}")
try:
    flipped = two.decide(fixture_spec("fx_alpha", reverse_grid=True), situations)
    changed = int(np.sum(np.any(flipped != two.decide(alpha, situations), axis=1)))
    report("H4c", True, f"decide with the candidate order reversed returned actions with no error ({changed} of 20 commands changed)")
except ValueError as error:
    report("H4c", False, f"refused: {error}")
snn_train.fit = real_fit
