"""What should the SNN be given as input? Compared inside a saved DODGE dream (no simulator, no TabPFN request).

DEV RUN, design probe. The first DODGE chain (experiment B-dodge-chain) and the settings comparison
(snn_settings_in_dream.py) showed that a spiking student given only the situation decides worse,
on kept-out dreams, than the plain physics choice. The teacher itself does not start from nothing:
it corrects the physics guess. This probe gives the student the same footing, in three variants:

  situation                      the encoded situation only (what the student gets today);
  situation_and_physics_action   plus the action the physics formula alone would choose
                                 (encoded like a situation feature; A values);
  situation_and_physics_scores   plus the physics-only objective of every candidate, standardised
                                 within the row (C values).

Both extra inputs come from the SkillSpec's formula alone, so they are available at decision time
without any request, and neither contains a prediction of the teacher. Each variant is trained on the
same dreams with the same targets and seed (through students.snn_train directly; this is a probe,
not the production student) and measured on the same kept-out dreams: mean dream regret (dreamed
objective given up against the dream's best candidate) next to the physics choice's regret, how
often it picks the dream's best candidate, and in how many rows its regret is lower or higher than
the physics choice's.

Decision quality INSIDE the dream only; nothing here is an evaluation result.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_real\\snn_inputs_in_dream.py <chain output directory>
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

print("DEV RUN inside a saved dream - design probe; no simulator episode and no TabPFN request; not an evaluation result", flush=True)

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.dream import load_dreams, select_rows  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng, stream_seed  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from learned_judgment.skills.dodge import for_world  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402
from learned_judgment.students.targets import dream_regret, score_spread, soft_targets  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
chain_dir = Path(sys.argv[1]).resolve()
manifest = json.loads((chain_dir / "manifest.json").read_text(encoding="utf-8"))
config = load_config()
ROOT = "dev_checks.snn_inputs_in_dream"
heldout_fraction = float(require(config, f"{ROOT}.heldout_fraction"))
validation_fraction = float(require(config, f"{ROOT}.validation_fraction"))
max_steps = int(require(config, f"{ROOT}.max_steps"))
input_kinds = list(require(config, f"{ROOT}.inputs"))
shapes = [tuple(int(h) for h in shape) for shape in require(config, f"{ROOT}.hidden_sizes")]
base_seed = int(require(config, "runtime.seed"))
points = tuple(require(config, "skills.dodge.candidate_points_per_dim"))
quantile_levels = tuple(require(config, "engine.quantile_levels"))
spread_levels = tuple(require(config, "engine.spread_levels"))

dreams, dream_meta = load_dreams(chain_dir / "dreams.npz")
world = Go2World(config, int(manifest["world_seed"]), str(manifest["world_preset"]))
spec = for_world(config, world)

# The same kept-out rows as snn_settings_in_dream.py (same stream), then an early-stopping share of the rest.
order = stream_rng(base_seed, f"{spec.name}/dev/snn_settings/heldout").permutation(dreams.n_rows)
n_heldout = int(round(heldout_fraction * dreams.n_rows))
held_rows, rest = np.sort(order[:n_heldout]), order[n_heldout:]
n_validation = int(round(validation_fraction * rest.size))
validation_rows, train_rows = np.sort(rest[:n_validation]), np.sort(rest[n_validation:])

# Physics-only decisions for every dreamed situation: the formula's objective per candidate and its choice.
physics = decide_batch(spec, dreams.situations, model=None, method=PHYSICS_ONLY, layout=build_layout(spec),
                       points_per_dim=points, quantile_levels=quantile_levels, spread_levels=spread_levels)
physics_scores = np.stack([decision.scores for decision in physics])
physics_index = np.array([decision.chosen_index for decision in physics])
physics_actions = np.stack([decision.action for decision in physics])
if not np.array_equal(physics_index, dreams.physics_choice):
    raise RuntimeError("the physics choices recomputed here differ from the ones stored in the dream")
row_std = physics_scores.std(axis=1, keepdims=True)
standardised = np.where(row_std > 0, (physics_scores - physics_scores.mean(axis=1, keepdims=True)) / np.where(row_std > 0, row_std, 1.0), 0.0)

situation_block = InputEncoder.from_variables(spec.situation).encode(dreams.situations)
action_block = InputEncoder.from_variables(spec.actions).encode(physics_actions)
INPUTS = {
    "situation": situation_block,
    "situation_and_physics_action": np.concatenate([situation_block, action_block], axis=1),
    "situation_and_physics_scores": np.concatenate([situation_block, standardised], axis=1),
}
unknown = [kind for kind in input_kinds if kind not in INPUTS]
if unknown:
    raise ValueError(f"'{ROOT}.inputs' names {unknown}; known: {sorted(INPUTS)}")

base = SNNSettings.from_config(config)
targets = soft_targets(dreams.scores, temperature=base.target_temperature, target_spread=base.target_spread,
                       spread=score_spread(dreams.scores[train_rows]) if base.target_spread == "global_median" else None)
held = select_rows(dreams, held_rows)
physics_regret = dream_regret(held.scores, held.physics_choice)
scale = score_spread(held.scores)
print(f"config digest {config_digest(config)[:12]} | dream of run {dream_meta.get('run')} ({dreams.method}, {dreams.n_rows} rows, "
      f"{dreams.n_candidates} candidates) | training {train_rows.size}, early stopping {validation_rows.size}, kept out {held_rows.size}")
print(f"kept-out dreams: the physics choice gives up {physics_regret.mean():.4f} of dreamed objective on average "
      f"({physics_regret.mean() / scale:.3f} of the typical score spread {scale:.4f}); it is the dream's best candidate in "
      f"{float(np.mean(held.physics_choice == held.chosen)):.3f} of rows", flush=True)

seed = stream_seed(base_seed, f"{spec.name}/dev/snn_inputs/train")
results = []
print(f"\n{'input':30s} {'hidden':>10s} {'width':>5s} {'steps':>5s} {'val loss':>9s} {'regret':>8s} {'vs phys':>8s} {'best':>5s} {'lower':>6s} {'equal':>6s} {'higher':>6s} {'s':>5s}")
for kind in input_kinds:
    inputs = INPUTS[kind]
    for shape in shapes:
        settings = dataclasses.replace(base, hidden_sizes=shape, max_steps=max_steps)
        began = time.perf_counter()
        model = snn_train.build_model(inputs.shape[1], {spec.name: dreams.n_candidates}, settings, seed=seed)
        try:
            report = snn_train.fit(model, {spec.name: (inputs[train_rows], targets[train_rows])},
                                   validation={spec.name: (inputs[validation_rows], targets[validation_rows])},
                                   sampling={spec.name: 1.0}, settings=settings, seed=seed)
        except TrainingError as error:
            print(f"{kind:30s} {str(list(shape)):>10s} TRAINING REFUSED: {error}", flush=True)
            results.append({"input": kind, "hidden_sizes": list(shape), "training_error": str(error)})
            continue
        wall = time.perf_counter() - began
        chosen = snn_train.export_network(model, spec.name).decide(inputs[held_rows]).astype(np.int64)
        regret = dream_regret(held.scores, chosen)
        record = {
            "input": kind, "hidden_sizes": list(shape), "input_width": int(inputs.shape[1]), "steps_run": report.steps_run,
            "best_step": report.best_step, "best_validation_loss": report.best_validation_loss,
            "mean_regret": float(regret.mean()), "mean_regret_minus_physics": float((regret - physics_regret).mean()),
            "picks_dream_best": float(np.mean(chosen == held.chosen)), "regret_lower_than_physics": float(np.mean(regret < physics_regret)),
            "regret_equal_to_physics": float(np.mean(regret == physics_regret)), "regret_higher_than_physics": float(np.mean(regret > physics_regret)),
            "train_wall_s": wall,
        }
        results.append(record)
        print(f"{kind:30s} {str(list(shape)):>10s} {record['input_width']:5d} {record['steps_run']:5d} {record['best_validation_loss']:9.4f} "
              f"{record['mean_regret']:8.4f} {record['mean_regret_minus_physics']:+8.4f} {record['picks_dream_best']:5.2f} "
              f"{record['regret_lower_than_physics']:6.2f} {record['regret_equal_to_physics']:6.2f} {record['regret_higher_than_physics']:6.2f} {wall:5.0f}",
              flush=True)

print("\nregret: mean dreamed objective given up on the kept-out dreams (0 = always the dream's best candidate); vs phys: that mean "
      "minus the physics choice's (negative = better than the formula alone); best: share of rows where the dream's best candidate "
      "is picked; lower / equal / higher: share of rows with regret below / equal to / above the physics choice's.")
out_path = chain_dir / "snn_inputs_in_dream.json"
out_path.write_text(json.dumps({"label": "DEV RUN inside a saved dream, design probe; not an evaluation result", "config_digest": config_digest(config),
                                "training_rows": int(train_rows.size), "validation_rows": int(validation_rows.size), "heldout_rows": int(held_rows.size),
                                "physics_choice_mean_regret": float(physics_regret.mean()), "score_spread": float(scale),
                                "results": results}, indent=2) + "\n", encoding="utf-8")
print(f"written: {out_path.relative_to(ROOT_DIR).as_posix()}")
