"""Helpers shared by the B_sidestep_dev checks: import path, pass/fail harness, the real world.

Importing this module puts the repository root on ``sys.path`` (the package is not installed in
the venv). Every check runs from the repository root, because Go2World reads its assets by paths
relative to it.
"""

from __future__ import annotations

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[3]
CHECK_DIR = Path(__file__).resolve().parent
OUT_DIR = CHECK_DIR / "out"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from learned_judgment.config import require  # noqa: E402

DEV_RUN_LABEL = (
    "DEV RUN on the real Go2 simulator (Person A's Go2World, preset default, no ball) - development check of the "
    "sidestep_reach dev skill; not an evaluation result, not in any report or table"
)
PLUMBING_LABEL = (
    "PLUMBING CHECK - fixture values, pure functions only; no simulator, no model. Its numbers go into no report."
)
WORLD_SEED_STREAM = "sidestep_reach/dev/world"  # named stream of the world seed (engine.streams.stream_seed)


class Checks:
    """Pass/fail harness; ``finish`` exits 0 only when every check held."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        condition = bool(condition)
        if condition:
            self.passed += 1
        else:
            self.failed.append(label)
        print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""), flush=True)
        return condition

    def finish(self) -> None:
        print(f"\n{self.passed} passed, {len(self.failed)} failed")
        for label in self.failed:
            print(f"  failed: {label}")
        sys.exit(1 if self.failed else 0)


def world_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``config`` with ``world.save_traces`` set from ``dev_checks.sidestep_reach.save_traces``."""
    run = copy.deepcopy(dict(config))
    save = require(config, "dev_checks.sidestep_reach.save_traces")
    if not isinstance(save, bool):
        raise ValueError(f"dev_checks.sidestep_reach.save_traces must be true or false; got {save!r}")
    run["world"]["save_traces"] = save
    return run


def determinism_case(config: Mapping[str, Any], spec: Any) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    """The c2 episode: dev situation 0, its physics-only action, and attempt seeds.

    Returns (situation, action, seeds): seeds[0] is the dev noise seed of situation 0, repeat 0
    (the seed every method meets on that situation); seeds[1:] are repeats 1..k with k =
    ``dev_checks.sidestep_reach.determinism_other_attempt_seeds``. All come from named streams.
    """
    from learned_judgment.engine.decide import PHYSICS_ONLY, decide
    from learned_judgment.engine.features import build_layout
    from learned_judgment.eval.splits import noise_seed, split_situations

    base_seed = int(require(config, "runtime.seed"))
    preset = str(require(config, "dev_checks.sidestep_reach.world_preset"))
    others = int(require(config, "dev_checks.sidestep_reach.determinism_other_attempt_seeds"))
    situation = split_situations(spec, base_seed=base_seed, split="dev", n=1, world_preset=preset)[0]
    decision = decide(
        spec,
        situation,
        model=None,
        layout=build_layout(spec),
        points_per_dim=spec.candidate_points_per_dim,
        quantile_levels=require(config, "engine.quantile_levels"),
        spread_levels=tuple(require(config, "engine.spread_levels")),
        method=PHYSICS_ONLY,
    )
    seeds = [
        noise_seed(spec, base_seed=base_seed, split="dev", world_preset=preset, situation_index=0, repeat=repeat)
        for repeat in range(others + 1)
    ]
    return spec.decode_situation(situation), spec.decode_action(decision.action), seeds


def utc_now() -> str:
    """Current time, ISO 8601 with a zero UTC offset (the engine's clock convention)."""
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    """Compact UTC time for directory names, e.g. 20261121T183000Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def versions() -> dict[str, str]:
    """Versions of the interpreter and the libraries the checks depend on."""
    import mujoco
    import numpy
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "mujoco": mujoco.__version__,
    }
