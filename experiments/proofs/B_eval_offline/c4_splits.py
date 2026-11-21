"""Check 4: fixed seeded splits -- disjoint, reproducible across processes, test split sealed.

The cross-process part re-runs this script as a child with a different PYTHONHASHSEED in
"--digest" mode and compares sha256 digests of every split and of the noise seeds.
The test-access log is written only inside a temporary directory. Fixture spec only.
"""

import sys

from fixture_spec import FIXTURE_BANNER

if "--digest" not in sys.argv:
    print(FIXTURE_BANNER)

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import Checks, FixtureSpec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.eval.splits import (  # noqa: E402
    SealedSplitError,
    check_disjoint,
    noise_seed,
    noise_seeds,
    split_situations,
    test_situations,
)

config = load_config()
spec = FixtureSpec()
BASE_SEED = int(require(config, "runtime.seed"))
N_DEV = int(require(config, "evaluation.dev_situations_per_skill"))
N_TEST = int(require(config, "evaluation.test_situations_per_skill"))
N_REPEATS = int(require(config, "evaluation.minimum_seeds"))
SHIFT_PRESET = "heavy_payload"  # a preset name from config.yaml world.presets


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def draw_all(log_path: Path) -> dict[str, np.ndarray]:
    return {
        "dev": split_situations(spec, base_seed=BASE_SEED, split="dev", n=N_DEV, world_preset="default"),
        "context": split_situations(spec, base_seed=BASE_SEED, split="context", n=N_DEV, world_preset="default"),
        "shift": split_situations(spec, base_seed=BASE_SEED, split="shift", n=N_DEV, world_preset=SHIFT_PRESET),
        "test": test_situations(
            spec, base_seed=BASE_SEED, n=N_TEST, world_preset="default",
            final_evaluation=True, access_log_path=log_path, clock=utc_now,
        ),
        "dev_noise": noise_seeds(spec, base_seed=BASE_SEED, split="dev", world_preset="default", n_situations=N_DEV, n_repeats=N_REPEATS),
    }


def digests(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for name, array in arrays.items()}


if "--digest" in sys.argv:
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(digests(draw_all(Path(directory) / "test_access.jsonl")), sort_keys=True))
    raise SystemExit(0)

checks = Checks()
expect = checks.expect
with tempfile.TemporaryDirectory() as directory:
    log_path = Path(directory) / "test_access.jsonl"

    print("== sealed test split ==")
    try:
        split_situations(spec, base_seed=BASE_SEED, split="test", n=N_TEST, world_preset="default")
        expect(False, "split_situations refuses split='test'")
    except SealedSplitError as error:
        expect(True, f"split_situations refuses split='test': {error}")
    try:
        test_situations(spec, base_seed=BASE_SEED, n=N_TEST, world_preset="default",
                        final_evaluation=False, access_log_path=log_path, clock=utc_now)
        expect(False, "test_situations refuses final_evaluation=False")
    except SealedSplitError as error:
        expect(True, f"test_situations refuses final_evaluation=False: {error}")
    expect(not log_path.exists(), "a refused access writes no log line")
    try:
        test_situations(spec, base_seed=BASE_SEED, n=N_TEST, world_preset="default",
                        final_evaluation=True, access_log_path=log_path, clock=lambda: "2026-11-21T12:00:00")
        expect(False, "test_situations refuses a clock without a UTC offset")
    except ValueError as error:
        expect(True, f"test_situations refuses a clock without a UTC offset: {error}")

    print("\n== draw all splits ==")
    arrays = draw_all(log_path)
    for name in ("dev", "context", "shift", "test"):
        expect(arrays[name].shape[1] == spec.n_situation, f"{name}: shape {arrays[name].shape}")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    expect(len(lines) == 1 and record["skill"] == spec.name and record["n"] == N_TEST,
           f"one access-log line written: {lines[-1]}")

    print("\n== disjoint ==")
    situations = {name: arrays[name] for name in ("dev", "context", "shift", "test")}
    check_disjoint(situations)
    expect(True, "check_disjoint: the four splits share no identical situation row")
    shared = 0
    names = list(situations)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared += int(np.sum(np.all(situations[first][:, None, :] == situations[second][None, :, :], axis=2)))
    expect(shared == 0, f"brute-force pairwise comparison: {shared} shared rows")
    try:
        check_disjoint({"dev": arrays["dev"], "overlap": arrays["dev"][:3]})
        expect(False, "check_disjoint detects an overlap")
    except ValueError as error:
        expect(True, f"check_disjoint detects an overlap: {error}")

    print("\n== noise seeds ==")
    seeds = arrays["dev_noise"]
    expect(seeds.shape == (N_DEV, N_REPEATS) and seeds.dtype == np.int64, f"noise seeds shape {seeds.shape} dtype {seeds.dtype}")
    expect(int(seeds[7, 1]) == noise_seed(spec, base_seed=BASE_SEED, split="dev", world_preset="default", situation_index=7, repeat=1),
           "noise_seeds[i, r] == noise_seed(i, r) (no method argument: shared by all methods)")
    expect(np.unique(seeds).size == seeds.size, f"{seeds.size} distinct per-episode seeds")

    print("\n== same process: repeated draw ==")
    again = draw_all(Path(directory) / "second_log.jsonl")
    expect(all(np.array_equal(arrays[name], again[name]) for name in arrays), "repeated draw is bitwise identical")

local = digests(arrays)
print("\n== across processes ==")
for hash_seed in ("1", "2"):
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--digest"],
        capture_output=True, text=True, env=environment, cwd=str(Path(__file__).resolve().parents[3]),
    )
    remote = json.loads(child.stdout.strip().splitlines()[-1]) if child.returncode == 0 else {}
    expect(child.returncode == 0 and remote == local,
           f"child process PYTHONHASHSEED={hash_seed} exit={child.returncode}: identical sha256 for {sorted(remote)}")
for name, digest in sorted(local.items()):
    print(f"     sha256 {name}: {digest}")

checks.finish()
