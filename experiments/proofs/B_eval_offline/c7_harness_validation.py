"""Check 7: harness input validation and the latency-record shape. NO episode runs here.

evaluate_actions / compare_methods need Person A's Go2World and stay NOT RUN. This check
only shows that invalid inputs are refused BEFORE any world is built: the world factory
passed below raises if it is ever called, so it never creates a world of any kind.
measure_latency is called on a no-op function only to check the keys of its record; the
timings of a no-op are printed nowhere and mean nothing.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import tempfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import Checks, FixtureSpec, fixture_rows  # noqa: E402
from learned_judgment.eval.harness import compare_methods, evaluate_actions, measure_latency  # noqa: E402

checks = Checks()
expect = checks.expect
spec = FixtureSpec()
built = []


def refusing_factory(seed: int):
    built.append(seed)
    raise RuntimeError("check 7 never builds a world")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


situations, actions, _, _ = fixture_rows(spec, 3, seed=31)
with tempfile.TemporaryDirectory() as directory:
    out = Path(directory) / "rows.jsonl"
    cases = (
        ("action outside the declared bounds", dict(actions=actions + np.array([[1.0, 0.0]]), noise_seeds=np.arange(3))),
        ("noise seeds of the wrong shape", dict(actions=actions, noise_seeds=np.arange(2))),
        ("float noise seeds", dict(actions=actions, noise_seeds=np.arange(3, dtype=float))),
        ("actions of the wrong width", dict(actions=actions[:, :1], noise_seeds=np.arange(3))),
    )
    for label, arguments in cases:
        try:
            evaluate_actions(spec, refusing_factory, situations, arguments["actions"], noise_seeds=arguments["noise_seeds"],
                             run_id="check7", world_preset="default", out_path=out, clock=utc_now)
            expect(False, f"evaluate_actions refuses {label}")
        except ValueError as error:
            expect(True, f"evaluate_actions refuses {label}: {error}")
    try:
        compare_methods(
            spec,
            {"ok": lambda s: actions, "bad": lambda s: actions + np.array([[1.0, 0.0]])},
            world_factory=refusing_factory, situations=situations, noise_seeds=np.arange(6).reshape(3, 2),
            run_id="check7", world_preset="default", out_dir=directory, clock=utc_now,
        )
        expect(False, "compare_methods validates every method's actions before the first episode")
    except ValueError as error:
        expect(True, f"compare_methods validates every method's actions before the first episode: {error}")
    expect(built == [], "the world factory was never called")
    expect(not out.exists() and not any(Path(directory).iterdir()), "no attempt file was written")

record = measure_latency(lambda x: x, [1, 2, 3], 50)
expect(set(record) == {"median_s", "p95_s", "n_decisions", "n_inputs", "timer", "platform", "processor", "cpu_count"},
       f"latency record keys {sorted(record)}")
expect(record["p95_s"] >= record["median_s"] >= 0.0 and record["n_decisions"] == 50, "p95 >= median >= 0, n_decisions recorded")

checks.finish()
