"""Check 7 (decision S1): a student never decides under a spec whose candidate grid changed.

FIXTURE CHECK - mechanics only. Students are trained for a few steps (printed below) inside fixture
dreams of fixture_snn_spec.py. Old behaviour (review finding "student decides under a spec whose
candidate grid changed", reproduction experiments/proofs/B_review/snn_training/r02): decide()
compared only the spec name, situation encoding and action NAMES, so changed side-speed bounds
silently changed every command and swapped categorical levels flipped every side; meta held no
action bounds, levels or grid digest. Now meta['student'] stores a full record of every action
dimension and a digest of the candidate grid over the first distinct training situations (at most
16, stored with it); deciding, also after save/load, recomputes both and raises GridMismatchError
on any difference. Files go to a temporary directory that is removed at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c7_grid_mismatch.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import (  # noqa: E402
    FIXTURE_POINTS_PER_DIM,
    FIXTURE_SIDE_POINTS_PER_DIM,
    Checks,
    FixtureReversedGridSpec,
    FixtureSideSpec,
    FixtureSnnSpec,
    fixture_dreams,
)
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import (  # noqa: E402
    ACTION_RECORD_KEYS,
    GRID_PROBE_MAX_ROWS,
    GridMismatchError,
    SNNStudent,
    action_record,
    candidate_grid_sha256,
    situation_row_digests,
)
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

MAX_STEPS = 150  # reduced from config students.snn.max_steps: these students only exercise the guard
BASE_SEED = 3  # fixture run seed
N_DREAMS = 400

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=MAX_STEPS, evaluate_every_steps=50)
print(f"\nsettings: config.yaml students.snn with max_steps={MAX_STEPS} evaluate_every_steps=50")

spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, N_DREAMS)
student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
record = student.meta["student"]
situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 50)

print("\n== the stored record ==")
print(f"  action_records: {record['action_records']}")
checks.expect(
    record["action_records"] == [action_record(dim) for dim in spec.actions]
    and all(list(item) == list(ACTION_RECORD_KEYS) for item in record["action_records"]),
    f"meta['student']['action_records'] holds every field {list(ACTION_RECORD_KEYS)} of every action dimension",
)
probe = np.asarray(record["candidate_grid"]["probe_situations"])
train_digests = set(record["situations_seen"]["optimiser_rows"])
probe_digests = situation_row_digests(probe)
checks.expect(
    probe.shape == (GRID_PROBE_MAX_ROWS, spec.n_situation) and len(set(probe_digests)) == probe.shape[0]
    and set(probe_digests) <= train_digests,
    f"grid probe = {probe.shape[0]} distinct training situations (at most {GRID_PROBE_MAX_ROWS})",
)
first_training = [d for d in dict.fromkeys(situation_row_digests(dreams.situations)) if d in train_digests][:GRID_PROBE_MAX_ROWS]
checks.expect(probe_digests == first_training, "the probe is the FIRST distinct training situations, in row order")
checks.expect(
    record["candidate_grid"]["sha256"] == candidate_grid_sha256(spec, probe, FIXTURE_POINTS_PER_DIM),
    f"stored grid digest {record['candidate_grid']['sha256'][:16]}... = candidate_grid_sha256(spec, probe, points_per_dim)",
)

print("\n== the unchanged spec decides ==")
reference = student.decide(spec, situations)
checks.expect(reference.shape == (50, 2), "decide with the training spec -> (50, 2) actions")
student.check_spec(FixtureSnnSpec())
checks.expect(True, "check_spec accepts a fresh instance of the same spec")

print("\n== changed specs are refused (GridMismatchError, a ValueError) ==")
narrow = FixtureSnnSpec(side_bounds=(-0.3, 0.3))
caught = checks.raises("side-speed bounds [-0.3, 0.3] instead of [-0.6, 0.6] (decide)", GridMismatchError, lambda: student.decide(narrow, situations), "has low -0.3")
checks.expect(isinstance(caught, ValueError), "GridMismatchError is a ValueError")
checks.raises("same change through decide_one", GridMismatchError, lambda: student.decide_one(narrow, situations[0]), "has low -0.3")
checks.raises("same change through check_spec", GridMismatchError, lambda: student.check_spec(narrow), "trained with -0.6")
checks.raises(
    "another command field ('fixture_vx')",
    GridMismatchError,
    lambda: student.decide(FixtureSnnSpec(command_field="fixture_vx"), situations),
    "has command_field 'fixture_vx'",
)
checks.raises(
    "same action dimensions, candidate_actions reversed (only the grid digest can see it)",
    GridMismatchError,
    lambda: student.decide(FixtureReversedGridSpec(), situations),
    f"over the {GRID_PROBE_MAX_ROWS} stored probe situations",
)
reversed_spec = FixtureReversedGridSpec()
checks.expect(
    [action_record(dim) for dim in reversed_spec.actions] == record["action_records"],
    "(the reversed-grid spec has identical action records: the digest is what refuses it)",
)
renamed = FixtureSnnSpec()
renamed.actions = (dataclasses.replace(renamed.actions[0], name="fx_side"), renamed.actions[1])
checks.raises("an action renamed", GridMismatchError, lambda: student.decide(renamed, situations), "has name 'fx_side'")
dropped = FixtureSnnSpec()
dropped.actions = dropped.actions[:1]
checks.raises("an action dimension removed", GridMismatchError, lambda: student.decide(dropped, situations), "1 action dimensions instead of the trained 2")
checks.raises(
    "another skill name (ValueError, not a grid change)",
    ValueError,
    lambda: student.decide(FixtureSnnSpec(name="fixture_other"), situations),
    "trained for skill 'fixture_snn', not 'fixture_other'",
)
indices = student.decide_index(situations)
checks.expect(
    bool(np.array_equal(reference, np.stack([spec.candidate_actions(s, FIXTURE_POINTS_PER_DIM)[i] for s, i in zip(situations, indices)]))),
    "decide_index (no spec, grid indices only) gives the indices decide(spec) mapped through the grid",
)

print("\n== categorical command dimension ==")
side = FixtureSideSpec()
side_dreams = fixture_dreams(side, config, N_DREAMS, points_per_dim=FIXTURE_SIDE_POINTS_PER_DIM)
side_student = SNNStudent.train_in_dream(
    side, side_dreams, settings=settings, points_per_dim=FIXTURE_SIDE_POINTS_PER_DIM, base_seed=BASE_SEED
)
side_actions = side_student.decide(side, situations)
checks.expect(side_actions.shape == (50, 2), f"the categorical-side student decides ({side_dreams.n_candidates} candidates)")
swapped = FixtureSideSpec(levels=("right", "stay", "left"), command_values=(-0.5, 0.0, 0.5))
checks.expect(
    candidate_grid_sha256(swapped, side_student.deployed().grid_probe, FIXTURE_SIDE_POINTS_PER_DIM)
    == side_student.meta["student"]["candidate_grid"]["sha256"],
    "(swapping levels leaves the numeric grid, and so its digest, unchanged: the record must catch it)",
)
checks.raises("levels swapped (left <-> right)", GridMismatchError, lambda: side_student.decide(swapped, situations), "has levels ['right', 'stay', 'left']")
checks.raises(
    "command values changed only",
    GridMismatchError,
    lambda: side_student.decide(FixtureSideSpec(command_values=(0.4, 0.0, -0.4)), situations),
    "has command_values [0.4, 0.0, -0.4]",
)

print("\n== load-then-decide ==")
scratch = Path(tempfile.mkdtemp(prefix="b_snn_offline_c7_"))
checks.redactions[str(scratch)] = "<tmp>"
try:
    student.save(scratch / "student")
    loaded = SNNStudent.load(scratch / "student")
    checks.expect(bool(np.array_equal(loaded.decide(spec, situations), reference)), "loaded student decides as before with the training spec")
    checks.raises("loaded student, side-speed bounds changed", GridMismatchError, lambda: loaded.decide(narrow, situations), "has low -0.3")
    checks.raises("loaded student, reversed grid", GridMismatchError, lambda: loaded.decide_one(FixtureReversedGridSpec(), situations[0]), "stored probe situations")

    meta_path = scratch / "student.json"
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    stored["student"]["candidate_grid"]["sha256"] = "0" * 64
    meta_path.write_text(json.dumps(stored), encoding="utf-8")
    tampered = SNNStudent.load(scratch / "student")
    checks.raises("a file whose stored grid digest was changed", GridMismatchError, lambda: tampered.decide(spec, situations), "differs from the grid")

    old_style = json.loads(json.dumps(stored))
    for key in ("action_records", "candidate_grid", "situations_seen", "out_of_range", "format"):
        del old_style["student"][key]
    meta_path.write_text(json.dumps(old_style), encoding="utf-8")
    checks.raises(
        "a file in the old layout (no action records, no grid digest)",
        ValueError,
        lambda: SNNStudent.load(scratch / "student"),
        "cannot be checked against a spec and is refused",
    )
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
