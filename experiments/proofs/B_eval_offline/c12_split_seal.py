"""Check 12 (review finding E6): the test split is reachable through no public function except
test_situations, and every access is appended to config.yaml's evaluation.test_access_log.

Reviewer reproduction r04: test noise seeds came from noise_seeds(split="test") without the
flag, and test_situations wrote its audit line to a caller-chosen file. The check passes a COPY
of config.yaml whose test_access_log points into a temporary directory (config.yaml itself is
unchanged); it also confirms that results/test_access_log.jsonl is not touched. On the old code
noise_seeds(split="test") returned seeds and access_log_path was accepted, so this check fails there.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import copy  # noqa: E402
import inspect  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import REPO_ROOT, Checks, FixtureSpec, fixed_clock  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.eval import splits  # noqa: E402
from learned_judgment.eval.splits import SealedSplitError, noise_seed, noise_seeds, split_situations, test_situations  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()  # read only
spec = FixtureSpec()
BASE = int(require(config, "runtime.seed"))
project_log = REPO_ROOT / str(require(config, "evaluation.test_access_log"))
project_before = project_log.read_bytes() if project_log.exists() else None

print("== the open-split functions refuse the test split and its aliases ==")
aliases = ["test", "TEST", " test", "Test\n", np.str_("test")]
for alias in aliases:
    for name, call in (
        ("split_situations", lambda a=alias: split_situations(spec, base_seed=BASE, split=a, n=5, world_preset="default")),
        ("noise_seed", lambda a=alias: noise_seed(spec, base_seed=BASE, split=a, world_preset="default", situation_index=0, repeat=0)),
        ("noise_seeds", lambda a=alias: noise_seeds(spec, base_seed=BASE, split=a, world_preset="default", n_situations=2, n_repeats=1)),
    ):
        try:
            call()
            expect(False, f"{name} refuses split {alias!r}")
        except SealedSplitError:
            expect(True, f"{name} refuses split {alias!r} (SealedSplitError)")
public = [name for name in splits.__all__ if callable(getattr(splits, name)) and not isinstance(getattr(splits, name), type)]
print(f"   public functions of eval.splits: {public}")
expect(set(public) == {"check_disjoint", "noise_seed", "noise_seeds", "split_situations", "test_situations"},
       "no other public function in eval.splits could build a test stream name")

print("\n== test_situations takes the config and always logs to its test_access_log ==")
parameters = inspect.signature(test_situations).parameters
expect("access_log_path" not in parameters and "config" in parameters, f"test_situations parameters: {list(parameters)}")
with tempfile.TemporaryDirectory() as directory:
    local = copy.deepcopy(config)
    local["evaluation"]["test_access_log"] = str(Path(directory) / "nested" / "test_access.jsonl")
    log = Path(local["evaluation"]["test_access_log"])
    try:
        test_situations(spec, local, base_seed=BASE, n=5, n_repeats=2, world_preset="default", final_evaluation=True,
                        clock=fixed_clock, access_log_path=Path(directory) / "elsewhere.jsonl")
        expect(False, "a caller-chosen log path is not accepted")
    except TypeError as error:
        expect(True, f"a caller-chosen log path is not accepted: {error}")
    first = test_situations(spec, local, base_seed=BASE, n=5, n_repeats=2, world_preset="default", final_evaluation=True,
                            clock=fixed_clock)
    second = test_situations(spec, local, base_seed=BASE, n=5, n_repeats=2, world_preset="default", final_evaluation=True,
                             clock=fixed_clock)
    lines = log.read_text(encoding="utf-8").splitlines()
    expect(len(lines) == 2, f"two final accesses -> two lines in the config's log: {lines[-1]}")
    expect(first.noise_seeds.shape == (5, 2) and np.array_equal(first.situations, second.situations),
           "the test noise seeds come only with the logged access (TestSplit.noise_seeds)")
    expect(not (Path(directory) / "elsewhere.jsonl").exists(), "nothing was written to the caller-chosen path")
    rebuilt = np.asarray(spec.sample_situations(stream_rng(BASE, f"{spec.name}/test/default/situations", 0), 5))
    print("   NOTE (open issue, not fixed here): typing the stream name into engine.streams.stream_rng still "
          f"reproduces the test situations: {np.array_equal(rebuilt, first.situations)} -- only a secret salt "
          "outside the repository would prevent that")
project_after = project_log.read_bytes() if project_log.exists() else None
expect(project_before == project_after, f"the project log {project_log} was not touched by this check")

checks.finish()
