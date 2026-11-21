"""Check (e): guard rails of the shared spiking student.

FIXTURE CHECK - mechanics only. Refused with clear errors:
  construction   config.yaml as is (shared_feature_width is null), a config that deselects the shared
                 student, missing shared_skill_slots / shared_retention_fraction / rehearsal_fraction,
                 invalid slot counts, widths, fractions and settings;
  adding         NO FREE SLOT, a skill HELD TWICE (also when no slot is free), an ENCODED WIDTH ABOVE
                 the shared feature width, a name that is not an identifier, a dream of another skill or
                 grid, too few distinct situations for the three-way split, dream situations outside the
                 declared ranges, invalid rehearsal / seed arguments;
  deciding       a skill not held, non-finite or mis-shaped situations, a spec whose situation encoding
                 changed, a spec whose GRID CHANGED (GridMismatchError: other action bounds, and the same
                 action records with the candidate order reversed), a SITUATION OUTSIDE THE DECLARED
                 RANGES (OutOfRangeError under out_of_range "error"; clipped and counted under "clip");
  persistence    saving an empty student, tampered files, a retired v1 manifest.
Also: every refusal before training happens without importing torch; a TrainingError and a failure
injected after training (the check wraps snn_train.fit; a FIXTURE failure, not production code)
leave the student exactly as it was, and the next add matches a student that never failed.
Students here train a few steps only (printed). Files go to a temporary directory removed at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c5_guard_rails.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

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
from learned_judgment.students.snn import GridMismatchError, OutOfRangeError  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402

SHORT_STEPS = 30  # a few steps: these students only exist to exercise the guard rails
BASE_SEED = 5  # fixture run seed
ONE_SLOT = 1  # fixture slot count of the student that runs out of slots

checks = Checks()
config = load_config()
check_config = fixture_config(config)
settings = dataclasses.replace(SNNSettings.from_config(check_config), max_steps=SHORT_STEPS, evaluate_every_steps=10)
n_slots = require(check_config, "students.snn.shared_skill_slots")
width = require(check_config, "students.snn.shared_feature_width")
fraction = require(check_config, "students.snn.rehearsal_fraction")
retention = require(check_config, "students.snn.shared_retention_fraction")
print(
    f"\nsettings: students.snn with max_steps={SHORT_STEPS} evaluate_every_steps=10 (config {config['students']['snn']['max_steps']} / "
    f"{config['students']['snn']['evaluate_every_steps']}); n_slots={n_slots} (config) width={width} (fixture) "
    f"rehearsal_fraction={fraction} retention_fraction={retention} (config) out_of_range={settings.out_of_range!r} (config)"
)

specs = {name: fixture_spec(name) for name in ("fx_alpha", "fx_beta", "fx_wide")}
dreams = {name: fixture_dreams(spec, config) for name, spec in specs.items()}
alpha, beta = specs["fx_alpha"], specs["fx_beta"]


def student(settings_: SNNSettings = settings, **overrides: object) -> SharedSNNStudent:
    arguments = {
        "n_slots": n_slots,
        "feature_width": width,
        "settings": settings_,
        "rehearsal_fraction": fraction,
        "retention_fraction": retention,
    }
    arguments.update(overrides)
    return SharedSNNStudent(**arguments)


def add(target: SharedSNNStudent, name: str, **overrides: object) -> dict:
    arguments = {"points_per_dim": POINTS[name], "base_seed": BASE_SEED, "rehearsal": True}
    arguments.update(overrides)
    data = arguments.pop("dreams", dreams[name])
    spec = arguments.pop("spec", specs[name])
    return target.add_skill(spec, data, **arguments)


def without(path: str) -> dict:
    changed = copy.deepcopy(check_config)
    node = changed
    *parents, last = path.split(".")
    for part in parents:
        node = node[part]
    del node[last]
    return changed


print("\n== construction refusals ==")
checks.raises("create with config.yaml as is", ConfigError, lambda: SharedSNNStudent.create(config), "students.snn.shared_feature_width")
no_shared = copy.deepcopy(check_config)
no_shared["students"]["shared_snn"] = False
checks.raises("create with students.shared_snn false", ValueError, lambda: SharedSNNStudent.create(no_shared), "students.shared_snn")
other_kind = copy.deepcopy(check_config)
other_kind["students"]["kind"] = "mlp"
checks.raises("create with students.kind 'mlp'", ValueError, lambda: SharedSNNStudent.create(other_kind), "students.kind")
for key in ("students.snn.shared_skill_slots", "students.snn.shared_retention_fraction", "students.snn.rehearsal_fraction"):
    checks.raises(f"create without {key}", ConfigError, lambda key=key: SharedSNNStudent.create(without(key)), key)
old_layout = without("students.snn.shared_skill_slots")
old_layout["students"]["snn"]["shared_skills"] = ["fx_alpha", "fx_beta"]
checks.raises("create from an old config that lists shared_skills instead of slots", ConfigError, lambda: SharedSNNStudent.create(old_layout), "students.snn.shared_skill_slots")
checks.raises("rehearsal_fraction 0.0", ValueError, lambda: student(rehearsal_fraction=0.0), "students.snn.rehearsal_fraction must lie in (0, 1)")
checks.raises("rehearsal_fraction 1.0", ValueError, lambda: student(rehearsal_fraction=1.0), "students.snn.rehearsal_fraction must lie in (0, 1)")
checks.raises("retention_fraction 0.0", ValueError, lambda: student(retention_fraction=0.0), "students.snn.shared_retention_fraction must lie in (0, 1)")
checks.raises("retention_fraction 1.0", ValueError, lambda: student(retention_fraction=1.0), "students.snn.shared_retention_fraction must lie in (0, 1)")
checks.raises("retention_fraction nan", ValueError, lambda: student(retention_fraction=float("nan")), "students.snn.shared_retention_fraction must lie in (0, 1)")
checks.raises("n_slots 0", ValueError, lambda: student(n_slots=0), "students.snn.shared_skill_slots must be an integer >= 1")
checks.raises("n_slots 2.5", ValueError, lambda: student(n_slots=2.5), "students.snn.shared_skill_slots must be an integer >= 1")
checks.raises("n_slots True", ValueError, lambda: student(n_slots=True), "students.snn.shared_skill_slots must be an integer >= 1")
checks.raises("feature_width 6.0", ValueError, lambda: student(feature_width=6.0), "students.snn.shared_feature_width must be an integer >= 1")
checks.raises("feature_width 0", ValueError, lambda: student(feature_width=0), "students.snn.shared_feature_width must be an integer >= 1")
checks.raises("settings given as a dict", TypeError, lambda: student(settings_=settings.describe()), "settings must be SNNSettings")

print("\n== add_skill and decide refusals on an empty student (before any training) ==")
empty = student()
situations = dreams["fx_alpha"].situations[:20].copy()
dotted = fixture_spec("fx_alpha")
dotted.name = "fx.alpha"  # FIXTURE: a spec whose name is not an identifier
checks.raises("a skill name that is not an identifier ('fx.alpha')", ValueError, lambda: add(empty, "fx_alpha", spec=dotted), "must be a Python identifier")
checks.raises(
    "ENCODED WIDTH ABOVE the shared width (fx_wide encodes 7 > 6)", ValueError, lambda: add(empty, "fx_wide"),
    "encodes its situation into 7 columns but students.snn.shared_feature_width is 6",
)
checks.raises("rehearsal='yes'", ValueError, lambda: add(empty, "fx_alpha", rehearsal="yes"), "rehearsal must be True or False")
checks.raises("base_seed -1", ValueError, lambda: add(empty, "fx_alpha", base_seed=-1), "base_seed must be a non-negative integer")
checks.raises("fx_beta's dreams for fx_alpha", ValueError, lambda: add(empty, "fx_alpha", dreams=dreams["fx_beta"]), "DreamSet.skill 'fx_beta'")
checks.raises("points_per_dim (5, 7) instead of (7, 5)", ValueError, lambda: add(empty, "fx_alpha", points_per_dim=(5, 7)), "are not the grid")
checks.raises("a single dream row", ValueError, lambda: add(empty, "fx_alpha", dreams=fixture_dreams(alpha, config, 1)), "cannot hold out retention situations")
checks.raises("two dream rows", ValueError, lambda: add(empty, "fx_alpha", dreams=fixture_dreams(alpha, config, 2)), "cannot hold out early-stopping situations")
outside = dreams["fx_alpha"].situations.copy()
outside[10, 1] = 7.5  # FIXTURE: fx_a_speed outside its declared range [0.5, 2.0]
checks.raises(
    "a dream situation outside the declared ranges", ValueError,
    lambda: add(empty, "fx_alpha", dreams=dataclasses.replace(dreams["fx_alpha"], situations=outside)),
    "dream row 10: situation feature 'fx_a_speed' = 7.5 lies outside its declared range [0.5, 2.0]",
)
checks.raises("decide_index on an empty student", ValueError, lambda: empty.decide_index("fx_alpha", situations), "has not been added to this shared student")
checks.raises("deployed() on an empty student", ValueError, lambda: empty.deployed("fx_alpha"), "has not been added to this shared student")
checks.raises("network_digests() on an empty student", ValueError, lambda: empty.network_digests(), "holds no skill yet; it has no stage")
scratch = Path(tempfile.mkdtemp(prefix="b_snn_shared_c5_"))
checks.redactions[str(scratch)] = "<tmp>"
checks.raises("save an empty student", ValueError, lambda: empty.save(scratch / "empty"), "holds no skill yet")
checks.expect(
    empty.skills_held == () and empty.free_slots == n_slots and empty.retention_matrix()["skills"] == []
    and "torch" not in sys.modules and "snntorch" not in sys.modules,
    "every refusal above happened before training: nothing held, every slot free, torch and snntorch never imported",
)

print(f"\n== a TrainingError leaves the student unchanged (learning_rate 1e-300, {SHORT_STEPS} steps) ==")
frozen = student(settings_=dataclasses.replace(settings, learning_rate=1e-300))
try:
    add(frozen, "fx_alpha")
    checks.expect(False, "learning_rate 1e-300 did not raise TrainingError")
except TrainingError as error:
    checks.expect(error.report is not None and error.report.best_step == 0, f"TrainingError with report (best_step 0): {str(error)[:120]}")
checks.expect(
    frozen.skills_held == () and frozen.records == [] and frozen.n_stages == 0 and frozen.free_slots == n_slots,
    "after the TrainingError the student is still empty and every slot is free",
)

print(f"\n== NO FREE SLOT and HELD TWICE (a fixture student with {ONE_SLOT} slot, {SHORT_STEPS} steps) ==")
one = student(n_slots=ONE_SLOT)
add(one, "fx_alpha")
one_digests, one_records = one.network_digests(), one.records
checks.expect(one.input_width == width + ONE_SLOT and one.free_slots == 0, f"one slot: input width {one.input_width}, no free slot after fx_alpha")
checks.raises("adding fx_beta with no free slot", ValueError, lambda: add(one, "fx_beta"), "no free skill slot: all 1 slots (students.snn.shared_skill_slots) are held by {'fx_alpha': 0}")
checks.raises("adding fx_alpha again when no slot is free (held twice wins)", ValueError, lambda: add(one, "fx_alpha"), "skill 'fx_alpha' already holds slot 0")
checks.expect(
    one.skills_held == ("fx_alpha",) and one.n_stages == 1 and one.network_digests() == one_digests and one.records == one_records,
    "after both refusals the student is unchanged",
)

print(f"\n== guard rails on a student holding fx_alpha ({SHORT_STEPS} steps) ==")
held = student()
add(held, "fx_alpha")
checks.raises("adding fx_alpha a second time (HELD TWICE)", ValueError, lambda: add(held, "fx_alpha"), "skill 'fx_alpha' already holds slot 0 (added at stage 0")
nan_situations = situations.copy()
nan_situations[4, 1] = np.nan
inf_situations = situations.copy()
inf_situations[2, 0] = np.inf
checks.raises("decide_index for fx_beta (not held)", ValueError, lambda: held.decide_index("fx_beta", situations), "has not been added to this shared student")
checks.raises("decide_index with NaN", ValueError, lambda: held.decide_index("fx_alpha", nan_situations), "non-finite values (first at row 4, feature 'fx_a_speed')")
checks.raises("decide with inf", ValueError, lambda: held.decide(alpha, inf_situations), "non-finite values (first at row 2, feature 'fx_a_offset')")
checks.raises("operation_counts with NaN", ValueError, lambda: held.operation_counts("fx_alpha", nan_situations), "non-finite")
checks.raises("decide_index with 2 columns", ValueError, lambda: held.decide_index("fx_alpha", situations[:, :2]), "shape (n, 3)")
checks.raises("decide with fx_beta's spec", ValueError, lambda: held.decide(beta, situations), "has not been added to this shared student")
checks.raises(
    "decide with an fx_alpha spec whose speed range changed", ValueError,
    lambda: held.decide(fixture_spec("fx_alpha", alpha_speed_high=3.0), situations), "encodes its situation differently",
)
narrow = fixture_spec("fx_alpha", alpha_vy_bounds=(-0.3, 0.3))
reversed_grid = fixture_spec("fx_alpha", reverse_grid=True)
checks.raises("GRID CHANGED: decide with side-speed bounds [-0.3, 0.3]", GridMismatchError, lambda: held.decide(narrow, situations), "action dimension 0 ('fx_a_vy') has low -0.3")
checks.raises("GRID CHANGED: decide_one with side-speed bounds [-0.3, 0.3]", GridMismatchError, lambda: held.decide_one(narrow, situations[0]), "action dimension 0 ('fx_a_vy')")
checks.raises("GRID CHANGED: check_spec with side-speed bounds [-0.3, 0.3]", GridMismatchError, lambda: held.check_spec(narrow), "the student was trained with -0.6")
checks.raises(
    "GRID CHANGED: same action records, candidate order reversed (only the grid digest sees it)", GridMismatchError,
    lambda: held.decide(reversed_grid, situations), "candidate_actions over the",
)
checks.raises("GRID CHANGED: the stage-0 view refuses too", GridMismatchError, lambda: held.deployed("fx_alpha", stage=0).decide(reversed_grid, situations), "differs from the grid the student was trained on")
far = situations.copy()
far[3, 1] = 9.0  # FIXTURE: fx_a_speed outside its declared range [0.5, 2.0]
needle = "situation feature 'fx_a_speed' = 9.0 at row 3 lies outside its declared range [0.5, 2.0]"
checks.raises("OUT OF RANGE: decide_index (out_of_range 'error')", OutOfRangeError, lambda: held.decide_index("fx_alpha", far), needle)
checks.raises("OUT OF RANGE: decide", OutOfRangeError, lambda: held.decide(alpha, far), needle)
checks.raises("OUT OF RANGE: operation_counts", OutOfRangeError, lambda: held.operation_counts("fx_alpha", far), needle)
checks.raises("OUT OF RANGE: decide_index_one", OutOfRangeError, lambda: held.decide_index_one("fx_alpha", far[3]), "situation feature 'fx_a_speed' = 9.0 at row 0")
checks.raises("OUT OF RANGE: decide_one", OutOfRangeError, lambda: held.decide_one(alpha, far[3]), "situation feature 'fx_a_speed' = 9.0 at row 0")
checks.expect(held.deployed("fx_alpha").out_of_range == settings.out_of_range == "error", "the deployed view carries students.snn.out_of_range ('error')")

print(f"\n== out_of_range 'clip' (a fixture student, {SHORT_STEPS} steps) ==")
clipping = student(settings_=dataclasses.replace(settings, out_of_range="clip"))
add(clipping, "fx_alpha")
clipped = far.copy()
clipped[3, 1] = 2.0
decisions = clipping.decide_index("fx_alpha", far)
log = clipping.deployed("fx_alpha").out_of_range_log.as_dict()
print(f"  out_of_range log: {log}")
checks.expect(
    bool(np.array_equal(decisions, clipping.decide_index("fx_alpha", clipped)))
    and log["policy"] == "clip" and log["decisions"] == far.shape[0] and log["decisions_with_clipping"] == 1
    and log["values_clipped"] == 1 and log["per_feature"]["fx_a_speed"] == 1,
    "clip: the out-of-range value is clipped into [0.5, 2.0], decided like the clipped situation, and counted",
)

print("\n== a failure after training leaves the student as it was ==")
before_digests = held.network_digests()
before_decisions = held.decide_index("fx_alpha", situations)
before_records = held.records
before_slots = held.slots
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
    held.skills_held == ("fx_alpha",) and held.slots == before_slots and held.free_slots == n_slots - 1 and held.n_stages == 1
    and held.network_digests() == before_digests and held.records == before_records
    and bool(np.array_equal(held.decide_index("fx_alpha", situations), before_decisions)),
    "after the failure: same skills, slots, stages, network digests, records and decisions",
)
record_after_failure = add(held, "fx_beta")
never_failed = student()
add(never_failed, "fx_alpha")
record_never_failed = add(never_failed, "fx_beta")
checks.expect(
    record_after_failure["network_digests"] == record_never_failed["network_digests"] and record_after_failure["slot"] == 1,
    "the next add (torch model rebuilt from the committed state) takes slot 1 and matches a student that never failed",
)

print("\n== tampered files are refused on load ==")
try:
    directory = scratch / "student"
    held.save(directory)
    checks.expect(
        SharedSNNStudent.load(directory).network_digests(0) == held.network_digests(0)
        and SharedSNNStudent.load(directory).network_digests() == held.network_digests(),
        "the untouched copy loads with both stages",
    )

    def tampered(label: str, change) -> Path:  # noqa: ANN001
        target = scratch / label
        shutil.copytree(directory, target)
        change(target)
        return target

    def flip_byte(path: Path) -> None:
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))

    def edit_manifest(change):  # noqa: ANN001, ANN202
        def apply(target: Path) -> None:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            change(manifest)
            (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        return apply

    def swap_stage_view(target: Path) -> None:
        for extension in ("npz", "json"):
            shutil.copyfile(target / "networks" / "stage_1" / f"fx_alpha.{extension}", target / "networks" / "stage_0" / f"fx_alpha.{extension}")

    def move_retention_row(manifest: dict) -> None:
        rows = manifest["held"][0]["rows"]
        moved = rows["retention"].pop(0)
        rows["training"] = sorted([*rows["training"], moved])

    state = tampered("state", lambda d: flip_byte(d / "model_state.npz"))
    checks.raises("a changed model_state.npz", ValueError, lambda: SharedSNNStudent.load(state), "does not match the sha256 in the manifest")
    network = tampered("network", lambda d: shutil.copyfile(d / "networks" / "stage_1" / "fx_alpha.npz", d / "networks" / "stage_0" / "fx_alpha.npz"))
    checks.raises("stage 1's arrays of fx_alpha in place of stage 0's", ValueError, lambda: SharedSNNStudent.load(network), "does not match the digest")
    swapped = tampered("swapped", swap_stage_view)
    checks.raises("stage 1's whole view of fx_alpha in place of stage 0's", ValueError, lambda: SharedSNNStudent.load(swapped), "is not the view of 'fx_alpha' at stage 0")
    stored = tampered("dreams", lambda d: flip_byte(d / "dreams" / "fx_alpha.npz"))
    checks.raises("a changed stored dream set", ValueError, lambda: SharedSNNStudent.load(stored), "does not match the sha256")
    rows = tampered("rows", edit_manifest(move_retention_row))
    checks.raises("a retention row moved into the training rows", ValueError, lambda: SharedSNNStudent.load(rows), "do not match the recorded situation digests")
    slot = tampered("slot", edit_manifest(lambda m: m["held"][1].__setitem__("slot", 5)))
    checks.raises("a manifest that moves fx_beta to slot 5", ValueError, lambda: SharedSNNStudent.load(slot), "records slot 5; the next free slot was 1")
    path = tampered("path", edit_manifest(lambda m: m["stages"][0]["views"]["fx_alpha"].__setitem__("file", "networks/stage_1/fx_alpha")))
    checks.raises("a stage entry pointing at another stage's file", ValueError, lambda: SharedSNNStudent.load(path), "names 'networks/stage_1/fx_alpha' for 'fx_alpha'")
    wrong = tampered("format", edit_manifest(lambda m: m.__setitem__("format", "something-else")))
    checks.raises("a manifest of another format", ValueError, lambda: SharedSNNStudent.load(wrong), "is not a learned-judgment/shared-spiking-student/v2 manifest")
    old = tampered("v1", edit_manifest(lambda m: m.__setitem__("format", "learned-judgment/shared-spiking-student/v1")))
    checks.raises(
        "a retired v1 manifest", ValueError, lambda: SharedSNNStudent.load(old),
        "is an old learned-judgment/shared-spiking-student/v1 manifest, which is no longer read",
    )
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
