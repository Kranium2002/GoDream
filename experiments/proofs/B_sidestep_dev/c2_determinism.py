"""Check 2: determinism of a sidestep_reach episode on the real Go2 simulator.

DEV RUN on the real Go2 simulator (Person A's Go2World, preset default, no ball). Not an
evaluation result. One case: dev situation 0, its physics-only action, the dev noise seed of
repeat 0 (all from named streams; see _common.determinism_case).

    1. same world, same inputs, twice in one process           -> identical outcome, bit for bit
    2. a fresh world with the same world seed, same inputs     -> identical outcome
    3. two fresh processes (c2_determinism_worker.py)          -> identical inputs and outcome
    4. other attempt seeds (repeats 1..k), same world and inputs -> may differ; reported, not
       required: the difference is the per-attempt noise the oracle would see.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_sidestep_dev\\c2_determinism.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

from _common import CHECK_DIR, DEV_RUN_LABEL, ROOT_DIR, WORLD_SEED_STREAM, Checks, determinism_case, world_config

print(DEV_RUN_LABEL, flush=True)

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from sidestep_reach_spec import ROOT, for_world  # noqa: E402

checks = Checks()
config = load_config()
preset = str(require(config, f"{ROOT}.world_preset"))
world_seed = stream_seed(int(require(config, "runtime.seed")), WORLD_SEED_STREAM)
print(f"config digest {config_digest(config)[:12]} | world seed {world_seed} (stream '{WORLD_SEED_STREAM}') | preset {preset}")

world = Go2World(world_config(config), world_seed, preset)
spec = for_world(config, world)
situation, action, seeds = determinism_case(config, spec)
print(f"situation {situation}")
print(f"action    {action} (physics-only choice)")
print(f"attempt seeds: {seeds[0]} (dev noise, situation 0, repeat 0); others {seeds[1:]} (repeats 1..{len(seeds) - 1})")


def as_hex(outcome: dict) -> dict:
    return {key: float(value).hex() for key, value in outcome.items()}


walls = []
started = time.perf_counter()
first = spec.episode(world, situation, action, seeds[0])
walls.append(time.perf_counter() - started)
started = time.perf_counter()
second = spec.episode(world, situation, action, seeds[0])
walls.append(time.perf_counter() - started)
print(f"\noutcome: {dict(first)}")
checks.check("same world, same inputs, twice in one process: identical outcome bit for bit", as_hex(first) == as_hex(second),
             f"{as_hex(first)}")

fresh = Go2World(world_config(config), world_seed, preset)
started = time.perf_counter()
third = spec.episode(fresh, situation, action, seeds[0])
walls.append(time.perf_counter() - started)
checks.check("a fresh world with the same world seed: identical outcome bit for bit", as_hex(third) == as_hex(first), "")

worker = CHECK_DIR / "c2_determinism_worker.py"
expected_inputs = {
    "world_seed": world_seed,
    "attempt_seed": seeds[0],
    "situation": {key: (value.hex() if isinstance(value, float) else value) for key, value in situation.items()},
    "action": {key: value.hex() for key, value in action.items()},
}
for run in (1, 2):
    started = time.perf_counter()
    done = subprocess.run([sys.executable, str(worker)], cwd=ROOT_DIR, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode != 0 or len(lines) != 1:
        checks.check(f"process {run}: worker ran", False, f"exit {done.returncode}; stderr tail {done.stderr[-500:]!r}")
        continue
    result = json.loads(lines[0][len("RESULT "):])
    inputs = {key: result[key] for key in expected_inputs}
    checks.check(
        f"process {run} (fresh interpreter, {time.perf_counter() - started:.1f} s): same inputs and identical outcome bit for bit",
        inputs == expected_inputs and result["outcome"] == as_hex(first),
        f"exit {done.returncode}",
    )

print("\n== other attempt seeds (allowed to differ; this is the noise between attempts) ==")
differences = {name: [] for name in spec.outcome_names}
for seed in seeds[1:]:
    started = time.perf_counter()
    other = spec.episode(world, situation, action, seed)
    walls.append(time.perf_counter() - started)
    for name in spec.outcome_names:
        differences[name].append(abs(float(other[name]) - float(first[name])))
    print(f"  attempt seed {seed}: {dict(other)}")
differs = any(value > 0.0 for values in differences.values() for value in values)
print(f"outcome differs from attempt seed {seeds[0]}: {'yes' if differs else 'no'}; largest |difference| per outcome: "
      + ", ".join(f"{name} {max(values):.3e}" for name, values in differences.items()))
print("(Go2World draws no noise yet: its docstring says the simulator and the policy are deterministic.)")
print(f"\nwall time per episode in this process: mean {sum(walls) / len(walls):.2f} s over {len(walls)} episodes")

checks.finish()
