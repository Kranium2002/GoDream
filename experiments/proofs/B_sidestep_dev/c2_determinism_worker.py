"""Worker of check 2: one sidestep_reach episode in a fresh process, printed exactly.

DEV RUN on the real Go2 simulator. Builds Go2World(config, world seed, preset), recomputes the
c2 case (dev situation 0, its physics-only action, the dev noise seed of repeat 0) from the named
streams, runs one episode and prints one line ``RESULT <json>`` holding the inputs and every
outcome as float.hex, so the parent can compare bit for bit. Not an evaluation result.

Run by c2_determinism.py, from the repository root.
"""

from __future__ import annotations

import json

from _common import WORLD_SEED_STREAM, determinism_case, world_config

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from sidestep_reach_spec import ROOT, for_world  # noqa: E402

config = load_config()
world_seed = stream_seed(int(require(config, "runtime.seed")), WORLD_SEED_STREAM)
world = Go2World(world_config(config), world_seed, str(require(config, f"{ROOT}.world_preset")))
spec = for_world(config, world)
situation, action, seeds = determinism_case(config, spec)
outcome = spec.episode(world, situation, action, seeds[0])
print("RESULT " + json.dumps({
    "world_seed": world_seed,
    "attempt_seed": seeds[0],
    "situation": {key: (value.hex() if isinstance(value, float) else value) for key, value in situation.items()},
    "action": {key: value.hex() for key, value in action.items()},
    "outcome": {key: float(value).hex() for key, value in outcome.items()},
}, sort_keys=True))
