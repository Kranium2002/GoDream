"""Check 5: guard rails of the shared spiking student.

FIXTURE CHECK - mechanics only. Refused with clear errors: config.yaml as is (shared_feature_width is
null), a config that deselects the shared student, invalid constructor values, a skill not in
shared_skills, an encoded width above shared_feature_width, adding the same skill twice, a dream of
another skill or another grid, too few rows, invalid rehearsal / seed arguments, deciding for a skill
not held, non-finite or mis-shaped situations, a spec whose encoding changed, saving an empty
student, and tampered files on load. Also: every refusal before training happens without importing
torch; a TrainingError and a failure injected after training (the check script wraps
snn_train.fit; a FIXTURE failure, not production code) leave the student exactly as it was, and the
next add then matches a student that never failed. Students here train a few steps only (printed).
Files go to a temporary directory that is removed at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c5_guard_rails.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_shared_specs import POINTS, Checks, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import ConfigError, load_config, require  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402

SHORT_STEPS = 30  # a few steps: these students only exist to exercise the guard rails
BASE_SEED = 5  # fixture run seed

checks = Checks()
config = load_config()
check_config = fixture_config(config)
settings = dataclasses.replace(SNNSettings.from_config(check_config), max_steps=SHORT_STEPS, evaluate_every_steps=10)
skills = require(check_config, "students.snn.shared_skills")
width = require(check_config, "students.snn.shared_feature_width")
fraction = require(check_config, "students.snn.rehearsal_fraction")
print(f"\nsettings: students.snn with max_steps={SHORT_STEPS} evaluate_every_steps=10; shared_skills={skills} width={width} (fixture)")

specs = {name: fixture_spec(name) for name in ("fx_alpha", "fx_beta", "fx_wide", "fx_omega")}
dreams = {name: fixture_dreams(spec, config) for name, spec in specs.items()}
alpha, beta = specs["fx_alpha"], specs["fx_beta"]


def student(settings_: SNNSettings = settings, **overrides: object) -> SharedSNNStudent:
    arguments = {"skills": skills, "feature_width": width, "settings": settings_, "rehearsal_fraction": fraction}
    arguments.update(overrides)
    return SharedSNNStudent(**arguments)


def add(target: SharedSNNStudent, name: str, **overrides: object) -> dict:
    arguments = {"points_per_dim": POINTS[name], "base_seed": BASE_SEED, "rehearsal": True}
    arguments.update(overrides)
    data = arguments.pop("dreams", dreams[name])
    return target.add_skill(specs[name] if "spec" not in arguments else arguments.pop("spec"), data, **arguments)


print("\n== construction refusals ==")
checks.raises("create with config.yaml as is", ConfigError, lambda: SharedSNNStudent.create(config), "students.snn.shared_feature_width")
no_shared = copy.deepcopy(check_config)
no_shared["students"]["shared_snn"] = False
checks.raises("create with students.shared_snn false", ValueError, lambda: SharedSNNStudent.create(no_shared), "students.shared_snn")
other_kind = copy.deepcopy(check_config)
other_kind["students"]["kind"] = "mlp"
checks.raises("create with students.kind 'mlp'", ValueError, lambda: SharedSNNStudent.create(other_kind), "students.kind")
no_fraction = copy.deepcopy(check_config)
del no_fraction["students"]["snn"]["rehearsal_fraction"]
checks.raises("create without students.snn.rehearsal_fraction", ConfigError, lambda: SharedSNNStudent.create(no_fraction), "students.snn.rehearsal_fraction")
checks.raises("rehearsal_fraction 0.0", ValueError, lambda: student(rehearsal_fraction=0.0), "students.snn.rehearsal_fraction must lie in (0, 1)")
checks.raises("rehearsal_fraction 1.0", ValueError, lambda: student(rehearsal_fraction=1.0), "students.snn.rehearsal_fraction must lie in (0, 1)")
checks.raises("a dotted skill name", ValueError, lambda: student(skills=["fx_alpha", "dodge.v2"]), "shared_skills[1] must be an identifier")
checks.raises("duplicate skill names", ValueError, lambda: student(skills=["fx_alpha", "fx_alpha"]), "unique names")
checks.raises("feature_width 6.0", ValueError, lambda: student(feature_width=6.0), "shared_feature_width must be an integer")
checks.raises("feature_width 0", ValueError, lambda: student(feature_width=0), "feature_width must be >= 1")
checks.raises("settings given as a dict", TypeError, lambda: student(settings_=settings.describe()), "settings must be SNNSettings")

print("\n== add_skill and decide refusals on an empty student (before any training) ==")
empty = student()
situations = dreams["fx_alpha"].situations[:20].copy()
checks.raises("a skill not in shared_skills (fx_omega)", ValueError, lambda: add(empty, "fx_omega"), "is not one of the shared network's skills")
checks.raises("an encoded width above shared_feature_width (fx_wide: 7 > 6)", ValueError, lambda: add(empty, "fx_wide"), "encodes its situation into 7 columns but students.snn.shared_feature_width is 6")
checks.raises("rehearsal='yes'", ValueError, lambda: add(empty, "fx_alpha", rehearsal="yes"), "rehearsal must be True or False")
checks.raises("base_seed -1", ValueError, lambda: add(empty, "fx_alpha", base_seed=-1), "base_seed must be a non-negative integer")
checks.raises("fx_beta's dreams for fx_alpha", ValueError, lambda: add(empty, "fx_alpha", dreams=dreams["fx_beta"]), "DreamSet.skill 'fx_beta'")
checks.raises("points_per_dim (5, 7) instead of (7, 5)", ValueError, lambda: add(empty, "fx_alpha", points_per_dim=(5, 7)), "are not the grid")
checks.raises("a single dream row", ValueError, lambda: add(empty, "fx_alpha", dreams=fixture_dreams(alpha, config, 1)), "leave no training rows")
checks.raises("decide_index on an empty student", ValueError, lambda: empty.decide_index("fx_alpha", situations), "has not been added yet")
scratch = Path(tempfile.mkdtemp(prefix="b_snn_shared_c5_"))
checks.redactions[str(scratch)] = "<tmp>"
checks.raises("save an empty student", ValueError, lambda: empty.save(scratch / "empty"), "holds no skill")
checks.expect(
    empty.skills_held == () and "torch" not in sys.modules and "snntorch" not in sys.modules,
    "every refusal above happened before training: nothing held, torch and snntorch never imported",
)

print(f"\n== a TrainingError leaves the student unchanged (learning_rate 1e-300, {SHORT_STEPS} steps) ==")
frozen = student(settings_=dataclasses.replace(settings, learning_rate=1e-300))
try:
    add(frozen, "fx_alpha")
    checks.expect(False, "learning_rate 1e-300 did not raise TrainingError")
except TrainingError as error:
    checks.expect(error.report is not None and error.report.best_step == 0, f"TrainingError with report (best_step 0): {str(error)[:120]}")
checks.expect(frozen.skills_held == () and frozen.records == [] and frozen.retention_matrix()["skills"] == [], "after the TrainingError the student is still empty")

print(f"\n== guard rails on a student holding fx_alpha ({SHORT_STEPS} steps) ==")
held = student()
add(held, "fx_alpha")
checks.raises("adding fx_alpha a second time", ValueError, lambda: add(held, "fx_alpha"), "skill 'fx_alpha' was already added (position 0")
nan_situations = situations.copy()
nan_situations[4, 1] = np.nan
inf_situations = situations.copy()
inf_situations[2, 0] = np.inf
checks.raises("decide_index for fx_beta (shared, not added)", ValueError, lambda: held.decide_index("fx_beta", situations), "has not been added yet")
checks.raises("decide_index for fx_omega (not shared)", ValueError, lambda: held.decide_index("fx_omega", situations), "is not one of the shared skills")
checks.raises("decide_index with NaN", ValueError, lambda: held.decide_index("fx_alpha", nan_situations), "non-finite values (first at row 4, feature 'fx_a_speed')")
checks.raises("decide with inf", ValueError, lambda: held.decide(alpha, inf_situations), "non-finite values (first at row 2, feature 'fx_a_offset')")
checks.raises("operation_counts with NaN", ValueError, lambda: held.operation_counts("fx_alpha", nan_situations), "non-finite")
checks.raises("decide_index with 2 columns", ValueError, lambda: held.decide_index("fx_alpha", situations[:, :2]), "shape (n, 3)")
checks.raises("decide with fx_beta's spec", ValueError, lambda: held.decide(beta, situations), "has not been added yet")
checks.raises(
    "decide with an fx_alpha spec whose speed range changed",
    ValueError,
    lambda: held.decide(fixture_spec("fx_alpha", alpha_speed_high=3.0), situations),
    "encodes its situation differently",
)

print("\n== a failure after training leaves the student as it was ==")
before_digests = held.network_digests()
before_decisions = held.decide_index("fx_alpha", situations)
before_records = held.records
real_fit = snn_train.fit


def failing_fit(*args: object, **kwargs: object) -> object:
    real_fit(*args, **kwargs)
    raise RuntimeError("FIXTURE failure injected after training")


snn_train.fit = failing_fit
try:
    checks.raises("add fx_beta with a failure injected after fit", RuntimeError, lambda: add(held, "fx_beta"), "FIXTURE failure injected")
finally:
    snn_train.fit = real_fit
checks.expect(
    held.skills_held == ("fx_alpha",) and held.network_digests() == before_digests and held.records == before_records
    and bool(np.array_equal(held.decide_index("fx_alpha", situations), before_decisions)),
    "after the failure: same skills held, network digests, records and decisions",
)
record_after_failure = add(held, "fx_beta")
never_failed = student()
add(never_failed, "fx_alpha")
record_never_failed = add(never_failed, "fx_beta")
checks.expect(
    record_after_failure["network_digests"] == record_never_failed["network_digests"],
    "the next add (torch model rebuilt from the committed state) matches a student that never failed",
)

print("\n== tampered files are refused on load ==")
try:
    directory = scratch / "student"
    held.save(directory)
    checks.expect(SharedSNNStudent.load(directory).network_digests() == held.network_digests(), "the untouched copy loads")

    def tampered(label: str, change) -> Path:  # noqa: ANN001
        target = scratch / label
        shutil.copytree(directory, target)
        change(target)
        return target

    def flip_byte(path: Path) -> None:
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))

    state = tampered("state", lambda d: flip_byte(d / "model_state.npz"))
    checks.raises("a changed model_state.npz", ValueError, lambda: SharedSNNStudent.load(state), "does not match the sha256 in the manifest")
    network = tampered("network", lambda d: shutil.copyfile(d / "networks" / "fx_beta.npz", d / "networks" / "fx_alpha.npz"))
    checks.raises("fx_beta's network arrays in place of fx_alpha's", ValueError, lambda: SharedSNNStudent.load(network), "does not match the digest")
    stored = tampered("dreams", lambda d: flip_byte(d / "dreams" / "fx_alpha.npz"))
    checks.raises("a changed stored dream set", ValueError, lambda: SharedSNNStudent.load(stored), "does not match the sha256")

    def other_format(target: Path) -> None:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["format"] = "something-else"
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    wrong = tampered("format", other_format)
    checks.raises("a manifest of another format", ValueError, lambda: SharedSNNStudent.load(wrong), "is not a learned-judgment/shared-spiking-student/v1 manifest")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
