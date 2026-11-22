"""Check 11 (review finding E4, engine review G-seed): a run never appends to an existing attempt
file, each run has a manifest written before its first episode and completed after its last,
and the harness uses two levels of seed.

Reviewer reproduction r05: a crash at episode 7 of 12 left 6 rows; rerunning the same
command appended (18 rows), and a "shift" evaluation with the same run_id appended again
(30 rows). On the old code the rerun appended instead of raising FileExistsError.

Two levels of seed (DECISIONS.md 2026-11-21): compare_methods must build ONE fresh world per
method from world_factory(world_seed), run every episode of that method in it, and hand each
episode its own attempt seed through spec.episode(world, situation, action, attempt_seed); the
manifest records world_seed and the attempt seeds. On the old code the harness built a new
world per episode from the episode seed and never passed an attempt seed, so these checks
fail there.

THROWAWAY harness runs (fixture_spec.FileMechanicsSpec, whose episode returns a constant, with
ThrowawayWorldToken "worlds" that are identity markers and not worlds) exercise the file and
seed handling only, in a temporary directory.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import (  # noqa: E402
    FIXTURE_COMMAND_SCHEMA,
    FIXTURE_WORLD_SEED,
    Checks,
    FileMechanicsSpec,
    ThrowawayWorldToken,
    fixed_clock,
    throwaway_compare_run,
    throwaway_world_factory,
)
from learned_judgment.config import config_digest, load_config  # noqa: E402
from learned_judgment.engine.rows import read_rows  # noqa: E402
from learned_judgment.eval.harness import compare_methods, evaluate_actions, manifest_path, read_manifest  # noqa: E402
from learned_judgment.eval.splits import noise_seeds, split_situations  # noqa: E402

checks = Checks()
expect = checks.expect
DIGEST = config_digest(load_config())


def section(title: str) -> None:
    print(f"\n== {title} ==")


def attempt(label: str, function):  # noqa: ANN001, ANN201
    """Run one group of checks; an unexpected exception fails the group instead of the script."""
    try:
        function()
    except Exception as error:  # noqa: BLE001
        expect(False, f"{label}: unexpected {type(error).__name__}: {error}")


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    spec_for_seeds = FileMechanicsSpec()
    situations = split_situations(spec_for_seeds, base_seed=7, split="dev", n=4, world_preset="default")
    seeds = noise_seeds(spec_for_seeds, base_seed=7, split="dev", world_preset="default", n_situations=4, n_repeats=3)

    def crash_and_rerun() -> None:
        section("crash, then the same command again")
        crashing = FileMechanicsSpec(crash_at=7)
        try:
            throwaway_compare_run(root, run_id="dev_run", methods=("physics_only",), spec=crashing)
            expect(False, "the throwaway run crashes at episode 7")
        except RuntimeError as error:
            expect(True, f"the throwaway run crashed at episode 7: {error}")
        attempt_file = root / "dev_run__physics_only.jsonl"
        rows_after_crash = len(read_rows(attempt_file))
        crashed = read_manifest(manifest_path(attempt_file))
        expect(rows_after_crash == 6, f"rows on disk after the crash: {rows_after_crash}")
        expect(crashed["completed"] is False and crashed["finished_utc"] is None and len(crashed["episodes"]) == 12,
               "the crashed run's manifest exists, lists 12 planned episodes and says completed: false")
        try:
            throwaway_compare_run(root, run_id="dev_run", methods=("physics_only",))
            expect(False, "a rerun into the same attempt file is refused")
        except FileExistsError as error:
            expect(True, f"a rerun into the same attempt file is refused: {str(error)[:140]}")
        expect(len(read_rows(attempt_file)) == 6, "the attempt file still holds the 6 crashed rows (nothing appended)")
        try:
            throwaway_compare_run(root, run_id="dev_run", methods=("physics_only",), split="shift")
            expect(False, "another split with the same run_id is refused")
        except FileExistsError:
            expect(True, "another split with the same run_id is refused (FileExistsError)")

        asked: list[int] = []

        def counting_policy(s: np.ndarray) -> np.ndarray:
            asked.append(s.shape[0])
            return np.zeros((s.shape[0], 2))

        only_manifest = root / "only_manifest"
        only_manifest.mkdir()
        (only_manifest / "x__physics_only.manifest.json").write_text("{}", encoding="utf-8")
        built: list[int] = []
        try:
            compare_methods(FileMechanicsSpec(), {"physics_only": counting_policy},
                            world_factory=lambda seed: built.append(seed), world_seed=FIXTURE_WORLD_SEED,
                            situations=situations, attempt_seeds=seeds, command_schema=FIXTURE_COMMAND_SCHEMA,
                            run_id="x", world_preset="default", split="dev", base_seed=7, config_digest=DIGEST,
                            out_dir=only_manifest, clock=fixed_clock)
            expect(False, "an existing manifest alone also refuses the run")
        except FileExistsError:
            expect(True, "an existing manifest alone also refuses the run")
        expect(asked == [] and built == [],
               "no policy was asked for actions and no world was built before the refusal (a billed teacher would not be called)")

    def complete_run() -> None:
        section("a complete run: manifest before the first episode, one fresh world per method, attempt seeds")
        fresh = root / "fresh"
        seen_manifests: dict[str, dict | None] = {}
        factory_calls: list[int] = []

        def observing_factory(seed: int) -> ThrowawayWorldToken:  # THROWAWAY: identity marker, not a world
            factory_calls.append(seed)
            return throwaway_world_factory(seed)

        def observe(world, situation, action, attempt_seed) -> None:  # noqa: ANN001
            # Record each method's manifest as it is when its first episode runs.
            for method in ("physics_only", "constant_bias"):
                target = fresh / f"dev_run__{method}.manifest.json"
                if method not in seen_manifests and target.exists() and not (fresh / f"dev_run__{method}.jsonl").exists():
                    seen_manifests[method] = json.loads(target.read_text(encoding="utf-8"))

        spec = FileMechanicsSpec(on_episode=observe)
        policies = {"physics_only": lambda s: np.zeros((s.shape[0], 2)), "constant_bias": lambda s: np.zeros((s.shape[0], 2))}
        results = compare_methods(spec, policies, world_factory=observing_factory, world_seed=FIXTURE_WORLD_SEED,
                                  situations=situations, attempt_seeds=seeds, command_schema=FIXTURE_COMMAND_SCHEMA,
                                  run_id="dev_run", world_preset="default", split="dev", base_seed=7,
                                  config_digest=DIGEST, out_dir=fresh, clock=fixed_clock)
        plan = [[i, r, int(seeds[i, r])] for r in range(3) for i in range(4)]
        n_episodes = len(plan)
        expect(factory_calls == [FIXTURE_WORLD_SEED, FIXTURE_WORLD_SEED],
               f"world_factory called once per method, each time with the world seed: {factory_calls}")
        worlds = [world for world, _ in spec.seen]
        first, second = worlds[:n_episodes], worlds[n_episodes:]
        expect(len(spec.seen) == 2 * n_episodes and all(w is first[0] for w in first) and all(w is second[0] for w in second),
               "every episode of a method ran in that method's one world")
        expect(first[0] is not second[0] and first[0].seed == second[0].seed == FIXTURE_WORLD_SEED,
               "the second method got a fresh world built from the same world seed")
        passed_seeds = [seed for _, seed in spec.seen]
        expect(passed_seeds[:n_episodes] == passed_seeds[n_episodes:] == [seed for _, _, seed in plan],
               "spec.episode received the attempt seeds in plan order, identical for both methods")
        for method, result in results.items():
            done = read_manifest(result.manifest_path)
            at_start = seen_manifests.get(method)
            expect(at_start is not None and at_start["completed"] is False,
                   f"{method}: the manifest existed (completed: false) when its first episode ran")
            expect(done["completed"] is True and done["finished_utc"] == fixed_clock(),
                   f"{method}: after the last episode: completed true, finished_utc set")
            expected_fields = {"run_id": "dev_run", "row_run_id": f"dev_run/{method}", "method": method, "skill": "fixture",
                               "split": "dev", "world_preset": "default", "base_seed": 7, "world_seed": FIXTURE_WORLD_SEED,
                               "n_situations": 4, "n_repeats": 3, "config_digest": DIGEST, "started_utc": fixed_clock(),
                               "attempt_file": f"dev_run__{method}.jsonl"}
            expect(all(done.get(key) == value for key, value in expected_fields.items()),
                   f"{method}: manifest fields incl. world_seed {expected_fields}")
            rows = read_rows(Path(result.rows_path))
            expect(done["episodes"] == plan and [row.seed for row in rows] == [seed for _, _, seed in plan],
                   f"{method}: episodes list (situation index, repeat, attempt seed) == the row order and row seeds of the attempt file")
            expect(result.losses.shape == (4, 3) and result.measured.shape == (4, 3, 2), f"{method}: returned shapes (n, R) and (n, R, K)")

    def single_world() -> None:
        section("evaluate_actions: one given world, attempt seeds, world seed in the manifest")
        spec = FileMechanicsSpec()
        token = throwaway_world_factory(23)
        single = root / "single" / "rows.jsonl"
        measured = evaluate_actions(spec, token, situations, np.zeros((4, 2)), attempt_seeds=seeds[:, 2],
                                    command_schema=FIXTURE_COMMAND_SCHEMA, world_seed=23, run_id="single",
                                    method="physics_only", world_preset="default", split="dev", base_seed=7,
                                    repeat=2, config_digest=DIGEST, out_path=single, clock=fixed_clock)
        single_manifest = read_manifest(manifest_path(single))
        expect(measured.shape == (4, 2) and single_manifest["completed"] is True and single_manifest["n_repeats"] == 1
               and single_manifest["world_seed"] == 23
               and single_manifest["episodes"] == [[i, 2, int(seeds[i, 2])] for i in range(4)],
               "evaluate_actions writes a completed manifest recording world seed 23 and repeat 2 for every episode")
        expect([world for world, _ in spec.seen] == [token] * 4 and [seed for _, seed in spec.seen] == seeds[:, 2].tolist(),
               "every episode ran in the given world with its own attempt seed")
        try:
            evaluate_actions(spec, token, situations, np.zeros((4, 2)), attempt_seeds=seeds[:, 2],
                             command_schema=FIXTURE_COMMAND_SCHEMA, world_seed=23, run_id="single",
                             method="physics_only", world_preset="default", split="dev", base_seed=7,
                             repeat=2, config_digest=DIGEST, out_path=single, clock=fixed_clock)
            expect(False, "evaluate_actions refuses an existing attempt file")
        except FileExistsError:
            expect(True, "evaluate_actions refuses an existing attempt file")
        expect(len(read_rows(single)) == 4 and len(spec.seen) == 4, "the refused rerun appended nothing and ran no episode")

        retired = root / "single" / "old.manifest.json"
        record = json.loads(manifest_path(single).read_text(encoding="utf-8"))
        record["format"] = "learned_judgment.run_manifest/v1"
        retired.write_text(json.dumps(record), encoding="utf-8")
        try:
            read_manifest(retired)
            expect(False, "a v1 manifest (one world per episode) is refused")
        except ValueError as error:
            expect("v1" in str(error) and "world" in str(error), f"a v1 manifest is refused with its reason: {str(error)[:150]}")

    attempt("crash and rerun", crash_and_rerun)
    attempt("complete run", complete_run)
    attempt("evaluate_actions", single_world)

checks.finish()
