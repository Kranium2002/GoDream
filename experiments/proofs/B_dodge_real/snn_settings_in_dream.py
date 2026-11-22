"""Compare SNN sizes and step limits inside a saved DODGE dream (no simulator episode, no TabPFN request).

DEV RUN. The DODGE chain's first run (experiment B-dodge-chain) left the SNN unconverged at its
step limit and worse than the teacher. This script takes that run's saved dream, keeps a share of
its situations out of every training, trains one spiking student per variant listed under
dev_checks.snn_settings_in_dream.variants on the remaining dreams, and measures each on the kept-out
dreams: how often it picks the dream's best candidate and how much dreamed objective it gives up
(dream regret), next to the physics-only choice's regret on the same dreams.

It measures decision quality INSIDE the dream only. Whether a variant decides better on the
simulator is measured by running the chain with it; nothing here is an evaluation result.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_real\\snn_settings_in_dream.py <chain output directory>
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print("DEV RUN inside a saved dream - no simulator episode and no TabPFN request; not an evaluation result", flush=True)

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.dream import load_dreams, select_rows  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from learned_judgment.skills.dodge import for_world  # noqa: E402
from learned_judgment.students import measure  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
chain_dir = Path(sys.argv[1]).resolve()
manifest = json.loads((chain_dir / "manifest.json").read_text(encoding="utf-8"))
config = load_config()
ROOT = "dev_checks.snn_settings_in_dream"
heldout_fraction = float(require(config, f"{ROOT}.heldout_fraction"))
variants = list(require(config, f"{ROOT}.variants"))
if not 0.0 < heldout_fraction < 1.0:
    raise ValueError(f"'{ROOT}.heldout_fraction' must lie in (0, 1); got {heldout_fraction}")
base_seed = int(require(config, "runtime.seed"))
points = tuple(require(config, "skills.dodge.candidate_points_per_dim"))

dreams, dream_meta = load_dreams(chain_dir / "dreams.npz")
world = Go2World(config, int(manifest["world_seed"]), str(manifest["world_preset"]))
spec = for_world(config, world)
order = stream_rng(base_seed, f"{spec.name}/dev/snn_settings/heldout").permutation(dreams.n_rows)
n_heldout = int(round(heldout_fraction * dreams.n_rows))
heldout = select_rows(dreams, np.sort(order[:n_heldout]))
training = select_rows(dreams, np.sort(order[n_heldout:]))
print(f"config digest {config_digest(config)[:12]} | dream of run {dream_meta.get('run')} ({dreams.method}, {dreams.n_rows} rows, "
      f"{dreams.n_candidates} candidates) | training rows {training.n_rows}, kept-out rows {heldout.n_rows} "
      f"(stream '{spec.name}/dev/snn_settings/heldout')", flush=True)

base = SNNSettings.from_config(config)
results = []
for variant in variants:
    name = str(variant["name"])
    settings = dataclasses.replace(base, hidden_sizes=tuple(int(h) for h in variant["hidden_sizes"]), max_steps=int(variant["max_steps"]))
    began = time.perf_counter()
    try:
        student = SNNStudent.train_in_dream(spec, training, settings=settings, points_per_dim=points, base_seed=base_seed)
    except TrainingError as error:
        print(f"  {name}: TRAINING REFUSED after {time.perf_counter() - began:.1f} s: {error}", flush=True)
        results.append({"name": name, "variant": variant, "training_error": str(error)})
        continue
    wall = time.perf_counter() - began
    report = dict(student.meta.get("training", {}))
    agreement = measure.agreement_with_dream(student.deployed(), heldout)
    scalars = {key: value for key, value in agreement.items() if isinstance(value, (int, float, str, bool)) or value is None}
    nested = {key: {k: v for k, v in value.items() if isinstance(v, (int, float, str, bool)) or v is None}
              for key, value in agreement.items() if isinstance(value, dict)}
    record = {"name": name, "variant": variant, "train_wall_s": wall,
              "training": {key: report.get(key) for key in ("steps_run", "best_step", "stopped_early", "initial_validation_loss",
                                                            "best_validation_loss", "firing_rate_per_layer")},
              "heldout": scalars, "heldout_detail": nested, "digest": student.digest}
    results.append(record)
    print(f"\n  {name}: hidden {list(settings.hidden_sizes)}, max_steps {settings.max_steps}; trained in {wall:.1f} s; {record['training']}")
    print(f"    kept-out dreams: {scalars}")
    for key, value in nested.items():
        print(f"    {key}: {value}")
    sys.stdout.flush()

out_path = chain_dir / "snn_settings_in_dream.json"
out_path.write_text(json.dumps({"label": "DEV RUN inside a saved dream; not an evaluation result", "config_digest": config_digest(config),
                                "chain": manifest.get("started_utc"), "training_rows": int(training.n_rows),
                                "heldout_rows": int(heldout.n_rows), "results": results}, indent=2, default=str) + "\n", encoding="utf-8")
print(f"\nwritten: {out_path.relative_to(ROOT_DIR).as_posix()}")
