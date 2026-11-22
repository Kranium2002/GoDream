"""Review reproduction: compare_methods / evaluate_actions file handling on a crash and a rerun.

THROWAWAY STAND-IN: `ThrowawaySpec.episode` returns CONSTANTS and `throwaway_world` is a bare
object. They exist only in this review script to exercise the harness's argument and file
handling. They are not a world, not a simulator, not an episode and not evidence of anything.
All files go to a temporary directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "proofs" / "B_eval_offline"))

from fixture_spec import FixtureSpec  # noqa: E402

from learned_judgment.engine.rows import read_rows  # noqa: E402
from learned_judgment.eval.harness import compare_methods  # noqa: E402
from learned_judgment.eval.splits import noise_seeds, split_situations  # noqa: E402

print("REVIEW REPRODUCTION - fixture, not evidence")


class ThrowawaySpec(FixtureSpec):
    """THROWAWAY: constant 'episode' to exercise file handling only; crashes on call `crash_at`."""

    crash_at: int | None = None
    calls = 0

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, float]:
        type(self).calls += 1
        if self.crash_at is not None and type(self).calls == self.crash_at:
            raise RuntimeError("simulated crash in the middle of an evaluation (throwaway)")
        return {"separation": 1.0, "contact": 0.0}


def throwaway_world(seed: int) -> object:  # THROWAWAY: not a world
    return object()


spec = ThrowawaySpec()
situations = split_situations(spec, base_seed=7, split="dev", n=4, world_preset="default")
seeds = noise_seeds(spec, base_seed=7, split="dev", world_preset="default", n_situations=4, n_repeats=3)
policies = {"physics_only": lambda s: np.tile([0.0, 0.0], (s.shape[0], 1))}
clock = lambda: "2026-11-21T00:00:00+00:00"  # noqa: E731

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    # run 1 crashes at episode 7 of 12 (4 situations x 3 repeats)
    ThrowawaySpec.crash_at, ThrowawaySpec.calls = 7, 0
    try:
        compare_methods(spec, policies, world_factory=throwaway_world, situations=situations, noise_seeds=seeds,
                        run_id="dev_run", world_preset="default", out_dir=out, clock=clock)
    except RuntimeError as error:
        print("run 1 raised:", error)
    path = out / "dev_run__physics_only.jsonl"
    print("rows on disk after the crash:", len(read_rows(path, spec=spec)))

    # run 2: the same command is re-run after the crash (same run_id, same out_dir)
    ThrowawaySpec.crash_at, ThrowawaySpec.calls = None, 0
    result = compare_methods(spec, policies, world_factory=throwaway_world, situations=situations, noise_seeds=seeds,
                             run_id="dev_run", world_preset="default", out_dir=out, clock=clock)
    rows = read_rows(path, spec=spec)
    print("returned losses shape:", result["physics_only"].losses.shape, "(12 episodes)")
    print("rows on disk after the rerun:", len(rows), "-> the saved artifact now holds", len(rows) - 12, "extra rows")
    keys = [(r.seed, tuple(sorted(r.situation.items()))) for r in rows]
    print("distinct (seed, situation) pairs on disk:", len(set(keys)))

    # run 3: a different split evaluated with the same run_id appends into the same file
    shift = split_situations(spec, base_seed=7, split="shift", n=4, world_preset="default")
    shift_seeds = noise_seeds(spec, base_seed=7, split="shift", world_preset="default", n_situations=4, n_repeats=3)
    compare_methods(spec, policies, world_factory=throwaway_world, situations=shift, noise_seeds=shift_seeds,
                    run_id="dev_run", world_preset="default", out_dir=out, clock=clock)
    rows = read_rows(path, spec=spec)
    print("after a 'shift' evaluation with the same run_id:", len(rows), "rows in", path.name,
          "| fields that name the split or repeat in a row:", [k for k in ("split", "repeat", "situation_index") if hasattr(rows[0], k)])
