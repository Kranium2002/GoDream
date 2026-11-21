"""Check 11 (review finding E4): a run never appends to an existing attempt file, and each run
has a manifest written before its first episode and completed after its last.

Reviewer reproduction r05: a crash at episode 7 of 12 left 6 rows; rerunning the same
command appended (18 rows), and a "shift" evaluation with the same run_id appended again
(30 rows). THROWAWAY harness runs (fixture_spec.FileMechanicsSpec, whose episode returns a
constant and whose "world" is None) exercise the file handling only, in a temporary directory.
On the old code the rerun appended instead of raising FileExistsError, so this check fails there.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import Checks, FileMechanicsSpec, fixed_clock, throwaway_compare_run, throwaway_world_factory  # noqa: E402
from learned_judgment.config import config_digest, load_config  # noqa: E402
from learned_judgment.engine.rows import read_rows  # noqa: E402
from learned_judgment.eval.harness import compare_methods, evaluate_actions, manifest_path, read_manifest  # noqa: E402
from learned_judgment.eval.splits import noise_seeds, split_situations  # noqa: E402

checks = Checks()
expect = checks.expect
DIGEST = config_digest(load_config())

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    print("== crash, then the same command again ==")
    crashing = FileMechanicsSpec(crash_at=7)
    try:
        throwaway_compare_run(root, run_id="dev_run", methods=("physics_only",), spec=crashing)
        expect(False, "the throwaway run crashes at episode 7")
    except RuntimeError as error:
        expect(True, f"the throwaway run crashed at episode 7: {error}")
    attempt = root / "dev_run__physics_only.jsonl"
    manifest_file = manifest_path(attempt)
    rows_after_crash = len(read_rows(attempt))
    crashed = read_manifest(manifest_file)
    expect(rows_after_crash == 6, f"rows on disk after the crash: {rows_after_crash}")
    expect(crashed["completed"] is False and crashed["finished_utc"] is None and len(crashed["episodes"]) == 12,
           "the crashed run's manifest exists, lists 12 planned episodes and says completed: false")
    asked = []
    try:
        throwaway_compare_run(root, run_id="dev_run", methods=("physics_only",))
        expect(False, "a rerun into the same attempt file is refused")
    except FileExistsError as error:
        expect(True, f"a rerun into the same attempt file is refused: {str(error)[:140]}")
    expect(len(read_rows(attempt)) == 6, "the attempt file still holds the 6 crashed rows (nothing appended)")
    try:
        throwaway_compare_run(root, run_id="dev_run", methods=("physics_only",), split="shift")
        expect(False, "another split with the same run_id is refused")
    except FileExistsError as error:
        expect(True, "another split with the same run_id is refused (FileExistsError)")

    spec = FileMechanicsSpec()
    situations = split_situations(spec, base_seed=7, split="dev", n=4, world_preset="default")
    seeds = noise_seeds(spec, base_seed=7, split="dev", world_preset="default", n_situations=4, n_repeats=3)

    def counting_policy(s: np.ndarray) -> np.ndarray:
        asked.append(s.shape[0])
        return np.zeros((s.shape[0], 2))

    only_manifest = root / "only_manifest"
    only_manifest.mkdir()
    (only_manifest / "x__physics_only.manifest.json").write_text("{}", encoding="utf-8")
    try:
        compare_methods(spec, {"physics_only": counting_policy}, world_factory=throwaway_world_factory,
                        situations=situations, noise_seeds=seeds, run_id="x", world_preset="default", split="dev",
                        base_seed=7, config_digest=DIGEST, out_dir=only_manifest, clock=fixed_clock)
        expect(False, "an existing manifest alone also refuses the run")
    except FileExistsError:
        expect(True, "an existing manifest alone also refuses the run")
    expect(asked == [], "no policy was asked for actions before the refusal (a billed teacher would not be called)")

    print("\n== a complete run: manifest before the first episode, completed after the last ==")
    fresh = root / "fresh"
    seen_at_first_episode = []

    def observing_factory(seed: int) -> None:  # THROWAWAY: returns None, records the manifest state
        target = fresh / "dev_run__physics_only.manifest.json"
        if not seen_at_first_episode:
            seen_at_first_episode.append(json.loads(target.read_text(encoding="utf-8")) if target.exists() else None)
        return None

    result = compare_methods(spec, {"physics_only": counting_policy}, world_factory=observing_factory,
                             situations=situations, noise_seeds=seeds, run_id="dev_run", world_preset="default",
                             split="dev", base_seed=7, config_digest=DIGEST, out_dir=fresh, clock=fixed_clock)["physics_only"]
    first = seen_at_first_episode[0]
    expect(first is not None and first["completed"] is False, "the manifest existed (completed: false) before the first episode")
    done = read_manifest(result.manifest_path)
    expect(done["completed"] is True and done["finished_utc"] == fixed_clock(), "after the last episode: completed true, finished_utc set")
    expected_fields = {"run_id": "dev_run", "row_run_id": "dev_run/physics_only", "method": "physics_only", "skill": "fixture",
                       "split": "dev", "world_preset": "default", "base_seed": 7, "n_situations": 4, "n_repeats": 3,
                       "config_digest": DIGEST, "started_utc": fixed_clock(), "attempt_file": "dev_run__physics_only.jsonl"}
    expect(all(done[key] == value for key, value in expected_fields.items()), f"manifest fields {expected_fields}")
    plan = [[i, r, int(seeds[i, r])] for r in range(3) for i in range(4)]
    rows = read_rows(Path(result.rows_path))
    expect(done["episodes"] == plan and [row.seed for row in rows] == [seed for _, _, seed in plan],
           "episodes list (situation index, repeat, seed) in run order == the row order of the attempt file")
    expect(result.losses.shape == (4, 3) and result.measured.shape == (4, 3, 2), "returned shapes (n, R) and (n, R, K)")

    print("\n== evaluate_actions ==")
    single = root / "single" / "rows.jsonl"
    measured = evaluate_actions(spec, throwaway_world_factory, situations, np.zeros((4, 2)), noise_seeds=seeds[:, 2],
                                run_id="single", method="physics_only", world_preset="default", split="dev", base_seed=7,
                                repeat=2, config_digest=DIGEST, out_path=single, clock=fixed_clock)
    single_manifest = read_manifest(manifest_path(single))
    expect(measured.shape == (4, 2) and single_manifest["completed"] is True and single_manifest["n_repeats"] == 1
           and single_manifest["episodes"] == [[i, 2, int(seeds[i, 2])] for i in range(4)],
           "evaluate_actions writes a completed manifest recording repeat 2 for every episode")
    try:
        evaluate_actions(spec, throwaway_world_factory, situations, np.zeros((4, 2)), noise_seeds=seeds[:, 2],
                         run_id="single", method="physics_only", world_preset="default", split="dev", base_seed=7,
                         repeat=2, config_digest=DIGEST, out_path=single, clock=fixed_clock)
        expect(False, "evaluate_actions refuses an existing attempt file")
    except FileExistsError:
        expect(True, "evaluate_actions refuses an existing attempt file")
    expect(len(read_rows(single)) == 4, "the refused rerun appended nothing")

checks.finish()
