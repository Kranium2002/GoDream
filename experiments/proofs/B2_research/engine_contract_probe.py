"""B2 research probe: what the generic engine accepts for a vision DODGE (read-only, offline).

RESEARCH PROBE, NOT EVIDENCE. Part 1 reads the real DODGE spec (Person A's throw model, no world,
no episode). Part 2 builds an in-memory FIXTURE SkillSpec (1,152 float pixel features, 4 discrete
actions) only to exercise pure engine functions (validate_spec, build_layout, decide_batch with
physics only, dream with physics only, rows, soft targets). Its physics is a placeholder; nothing
here is a Go2, TabPFN or SNN result. Part 3 reads TabPFN's free get_settings limits (no predict).
Nothing is written to disk.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B2_research\\engine_contract_probe.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.dream import _FLOAT_ARRAYS, _INT_ARRAYS, dream  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.rows import build_real_row  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.skills import available_skills, load_skill  # noqa: E402
from learned_judgment.skills.base import (  # noqa: E402
    ActionDim,
    CommandSchema,
    Outcome,
    SkillSpec,
    SkillSpecError,
    Variable,
    validate_spec,
)
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.targets import soft_targets  # noqa: E402

config = load_config()
schema = CommandSchema.from_config(config)
settings = dict(
    hidden_value_names=tuple(require(config, "world.hidden_effects")),
    quantile_levels=tuple(require(config, "engine.quantile_levels")),
    probe_seed=int(require(config, "runtime.seed")),
    probe_situations=int(require(config, "engine.spec_validation.probe_situations")),
    probe_points_per_dim=int(require(config, "engine.spec_validation.probe_points_per_dim")),
)
levels = tuple(require(config, "engine.quantile_levels"))
spread_levels = tuple(require(config, "engine.spread_levels"))

print("== 0. registry and CLI ==")
print("available_skills():", available_skills())
for argv in (["--help"], ["run", "--help"], ["status"]):
    done = subprocess.run([sys.executable, "-m", "learned_judgment.pipeline", *argv], cwd=ROOT, capture_output=True, text=True)
    print(f"python -m learned_judgment.pipeline {' '.join(argv)} -> exit {done.returncode}")
    print("  " + "\n  ".join((done.stdout or done.stderr).strip().splitlines()[:60]))

print("\n== 1. real DODGE spec (Person A's throw model; no world, no episode) ==")
try:
    load_skill("dodge", config)
except Exception as error:  # noqa: BLE001
    print(f"load_skill('dodge', config) without a world -> {type(error).__name__}: {str(error)[:160]}")
from learned_judgment.env import sample_dodge_situations  # noqa: E402
from learned_judgment.skills.dodge import throw_sampler_from  # noqa: E402

dodge = load_skill("dodge", config, throw_sampler=throw_sampler_from(sample_dodge_situations, config),
                   declared_payload_kg=0.0, declared_surface="default")
layout = build_layout(dodge)
print(f"S={dodge.n_situation} A={dodge.n_action} K={dodge.n_outcome} outcome_type={dodge.outcome_type}")
print("situation:", dodge.situation_names)
print("actions:", dodge.action_names, "outcomes:", dodge.outcome_names, "primary:", dodge.primary_outcome)
print(f"layout F={layout.n_features} blocks={layout.describe()['blocks']} categorical={layout.categorical}")
print("layout names:", layout.names)
points = tuple(require(config, "skills.dodge.candidate_points_per_dim"))
situations = dodge.sample_situations(stream_rng(0, "b2/probe/dodge"), 3)
decisions = decide_batch(dodge, situations, model=None, layout=layout, points_per_dim=points,
                         quantile_levels=levels, spread_levels=spread_levels, method=PHYSICS_ONLY)
print(f"decide_batch physics_only: candidates {decisions[0].candidates.shape}, scores {decisions[0].scores.shape}, "
      f"chosen {decisions[0].chosen_index}, physics_index {decisions[0].physics_index}")
try:
    dodge.encode_situation({**dodge.decode_situation(situations[0]), "scenario_truth": 1.0})
except ValueError as error:
    print(f"encode_situation with an extra non-feature key -> ValueError: {error}")

print("\n== 2. FIXTURE vision spec (pure-function probe; placeholder physics; not evidence) ==")
FRAMES, SIDE = 8, 12
PIXELS = tuple(f"px_t{t}_r{r}_c{c}" for t in range(FRAMES) for r in range(SIDE) for c in range(SIDE))
MOVES = ("NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT")
COMMAND = {"NO_OP": (0.0, 0.0), "DODGE_LEFT": (0.0, 0.8), "DODGE_RIGHT": (0.0, -0.8), "RETREAT": (-0.5, 0.0)}


class VisionProbeSpec(SkillSpec):
    skill_name = "vision_probe"

    def __init__(self, with_commands: bool) -> None:
        self.name = self.skill_name
        self.description = "fixture: 8 x 12 x 12 grayscale frames, 4 discrete moves"
        self.situation = tuple(Variable(n, "float", "intensity", "fixture pixel", low=0.0, high=1.0) for n in PIXELS)
        self.actions = (ActionDim("move", "categorical", "m/s", "fixture move", levels=MOVES, role="command",
                                  command_field="vel_y", command_values=tuple(COMMAND[m][1] for m in MOVES)),)
        self.outcomes = (Outcome("min_separation_m", "regression", "m", "fixture", low=0.0),
                         Outcome("collision", "binary", "flag", "fixture"))
        self.primary_outcome = "min_separation_m"
        self.hidden_effects = ("fixture",)
        self._with_commands = with_commands

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n = situations.shape[0]
        return np.column_stack([0.1 + 0.1 * actions[:, 0], np.full(n, 0.5)])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        return -10.0 * predicted.mean_of("collision") + predicted.quantile_of("min_separation_m", predicted.quantile_levels[0])

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int) -> Mapping[str, float]:
        raise NotImplementedError("fixture: never run")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        return measured[:, 1].astype(np.float64)

    def success(self, measured: np.ndarray) -> np.ndarray:
        return measured[:, 1] == 0.0


class VisionProbeWithCommands(VisionProbeSpec):
    def reset_situation(self, situation: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def commands(self, situation: Mapping[str, Any], action: Mapping[str, Any]) -> list[tuple[dict[str, float], float]]:
        vx, vy = COMMAND[action["move"]]
        return [({"vel_x": vx, "vel_y": vy, "yaw_rate": 0.0}, 1.0)]


for spec in (VisionProbeSpec(False), VisionProbeWithCommands(True)):
    began = time.perf_counter()
    validate_spec(spec, schema, **settings)
    print(f"validate_spec({type(spec).__name__}) passed in {time.perf_counter() - began:.2f} s "
          f"(S={spec.n_situation}, A={spec.n_action}, K={spec.n_outcome})")
spec = VisionProbeWithCommands(True)
grid = spec.candidate_actions(np.zeros(spec.n_situation), [1])
print("candidate_actions(points_per_dim=[1]) ->", grid.tolist(), "(categorical: every level, points ignored)")
print("candidate_actions(points_per_dim=[7]) shape ->", spec.candidate_actions(np.zeros(spec.n_situation), [7]).shape)
vlayout = build_layout(spec)
print(f"vision layout F={vlayout.n_features} blocks={vlayout.describe()['blocks']} categorical={vlayout.categorical}")
rng = stream_rng(0, "b2/probe/vision")
began = time.perf_counter()
vd = decide_batch(spec, spec.sample_situations(rng, 100), model=None, layout=vlayout, points_per_dim=[1],
                  quantile_levels=levels, spread_levels=spread_levels, method=PHYSICS_ONLY)
print(f"decide_batch physics_only x100: {time.perf_counter() - began:.3f} s; candidates {vd[0].candidates.shape}")
query = build_features(vlayout, np.repeat(vd[0].situation[None, :], 4, axis=0), vd[0].candidates,
                       physics=vd[0].predicted.physics)
print("query features for one situation:", query.shape)
began = time.perf_counter()
ds = dream(spec, None, vlayout, n_dreams=2000, rng=rng, points_per_dim=[1], quantile_levels=levels,
           spread_levels=spread_levels, method=PHYSICS_ONLY)
buffer = io.BytesIO()
np.savez(buffer, **{name: getattr(ds, name) for name in (*_FLOAT_ARRAYS, *_INT_ARRAYS)})
print(f"dream physics_only n=2000 in {time.perf_counter() - began:.2f} s: situations {ds.situations.shape}, "
      f"candidates {ds.candidates.shape}, scores {ds.scores.shape}, chosen {ds.chosen.shape}; npz {len(buffer.getvalue()) / 1e6:.1f} MB")
print("soft_targets shape:", soft_targets(ds.scores, temperature=0.5, target_spread="per_row", spread=None).shape)
print("InputEncoder width:", InputEncoder.from_variables(spec.situation).width)
row = build_real_row(spec, run_id="probe", seed=0, world_preset="default", situation=ds.situations[0],
                     action=ds.candidates[0, 0], measured_outcome={"min_separation_m": 0.3, "collision": 0.0},
                     timestamp_utc="2026-11-21T00:00:00+00:00")
print(f"one real AttemptRow JSON line: {len(row.to_json_line())} bytes (in memory, not written)")


class BadRetreat(VisionProbeWithCommands):
    def commands(self, situation: Mapping[str, Any], action: Mapping[str, Any]) -> list[tuple[dict[str, float], float]]:
        vx, vy = COMMAND[action["move"]]
        return [({"vel_x": vx * 3.0, "vel_y": vy, "yaw_rate": 0.0}, 1.0)]


try:
    validate_spec(BadRetreat(True), schema, **settings)
except SkillSpecError as error:
    print(f"commands() sending vel_x=-1.5 -> SkillSpecError: {str(error)[:200]}")

print("\n== 3. TabPFN free get_settings (no predict) ==")
try:
    import prior_labs

    prior_labs.connect()
    from tabpfn_client.client import ServiceClient

    api = ServiceClient.get_settings()
    if api is None:
        print("get_settings returned None")
    else:
        print("default_model_version:", api.default_model_version)
        wanted = ("train_set_max_rows", "train_set_max_cells", "train_set_max_upload_cells", "test_set_max_rows",
                  "max_cols", "test_set_max_cells", "test_set_max_rows_w_full_regression_output")
        print("max_model_limit:", {k: getattr(api.max_model_limit, k, None) for k in wanted})
        for version, limit in api.model_limits.items():
            print(f"model_limits[{getattr(version, 'value', version)}]:", {k: getattr(limit, k, None) for k in wanted})
except Exception as error:  # noqa: BLE001
    print(f"get_settings not read: {type(error).__name__}: {str(error)[:200]}")
print("\nexit 0")
