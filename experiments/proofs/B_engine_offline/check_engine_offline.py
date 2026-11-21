"""Offline checks of the engine lane's pure functions (rows, collect.exploration_actions,
dream, adapt) on a hand-written FIXTURE skill.

FIXTURE CHECK ONLY. The FixtureToySpec below is a toy SkillSpec written for this script;
it is not a robot skill, there is no world, no episode and no provider. Section 4b runs the
production ConstantBiasResidual (learned_judgment.eval.baselines, B2) on the fixture rows
only to exercise fit_residual_model and the model-backed dream() plumbing.
Nothing printed here is evidence about the Go2, the simulator, TabPFN or any model, and
nothing here may enter a results table. Files are written only inside a temporary
directory that is deleted at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_engine_offline\\check_engine_offline.py
Exit code 0 means every check below held; 1 means at least one failed.
"""

from __future__ import annotations

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root

from learned_judgment.config import load_config, require
from learned_judgment.engine.adapt import (
    recovery_attempts,
    select_context,
    world_id_column,
    world_id_layout,
    world_id_variable,
)
from learned_judgment.engine.collect import exploration_actions
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch
from learned_judgment.engine.dream import (
    DreamSet,
    dream,
    dream_rows,
    dreams_from_decisions,
    keep_confident,
    load_dreams,
    save_dreams,
)
from learned_judgment.engine.features import build_features, build_layout
from learned_judgment.engine.rows import (
    AttemptRow,
    append_rows,
    build_dream_row,
    build_real_row,
    read_rows,
    rows_to_arrays,
)
from learned_judgment.engine.learn import fit_residual_model
from learned_judgment.engine.streams import stream_rng
from learned_judgment.skills.base import ActionDim, Outcome, SkillSpec, Variable

try:  # production B2 baseline from the baselines lane (written in parallel with this lane)
    from learned_judgment.eval.baselines import ConstantBiasResidual

    BASELINE_IMPORT_ERROR = ""
except Exception as import_error:  # noqa: BLE001
    ConstantBiasResidual = None
    BASELINE_IMPORT_ERROR = repr(import_error)

# ---------------------------------------------------------------------------------------
# Fixture skill (toy formulas; exists only in this script)
# ---------------------------------------------------------------------------------------


class FixtureToySpec(SkillSpec):
    """FIXTURE: toy SkillSpec for pure-function checks. Not a robot skill."""

    def __init__(self) -> None:
        self.name = "fixture_toy"
        self.description = "fixture for pure-function engine checks; not a robot skill"
        self.situation = (
            Variable("fx_distance", "float", "m", "toy distance", low=0.0, high=2.0),
            Variable("fx_surface", "categorical", "label", "toy surface", levels=("a", "b")),
        )
        self.actions = (
            ActionDim("fx_speed", "float", "m/s", "toy speed", low=0.0, high=1.0, role="command", command_field="fx_cmd"),
            ActionDim("fx_delay", "float", "s", "toy delay", low=0.0, high=0.5, role="timing"),
        )
        self.outcomes = (
            Outcome("fx_error", "regression", "m", "toy signed error"),
            Outcome("fx_fail", "binary", "", "toy failure flag"),
        )
        self.primary_outcome = "fx_error"
        self.hidden_effects = ("fixture: no hidden effects",)

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        error = situations[:, 0] - actions[:, 0] * (1.0 - actions[:, 1]) - 0.1 * situations[:, 1]
        fail = 0.5 * actions[:, 0] + 0.2 * situations[:, 1]
        return np.column_stack([error, fail])

    def objective(self, predicted):  # noqa: ANN001
        return -np.abs(predicted.median_of("fx_error")) - predicted.mean_of("fx_fail") - predicted.spread_of("fx_error")

    def episode(self, world, situation, action, attempt_seed):  # noqa: ANN001
        raise RuntimeError("the fixture spec has no episode; collect() with a world is NOT RUN in this script")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        return np.abs(measured[:, 0])

    def success(self, measured: np.ndarray) -> np.ndarray:
        return np.abs(measured[:, 0]) <= 0.1


class FixtureUnevenGridSpec(FixtureToySpec):
    """FIXTURE: same toy skill, but surface 'b' gets a smaller candidate grid."""

    def candidate_actions(self, situation: np.ndarray, points_per_dim):  # noqa: ANN001
        if situation[1] == 1.0:
            return super().candidate_actions(situation, [2] * self.n_action)
        return super().candidate_actions(situation, points_per_dim)


# ---------------------------------------------------------------------------------------
# Tiny harness
# ---------------------------------------------------------------------------------------

RESULTS: list[tuple[str, bool]] = []
SCRATCH = {"dir": ""}  # the temporary directory, replaced by <tmp> in printed messages


def check(label: str, condition: bool) -> None:
    RESULTS.append((label, bool(condition)))
    if SCRATCH["dir"]:
        label = label.replace(SCRATCH["dir"], "<tmp>")
    print(f"  [{'ok' if condition else 'FAILED'}] {label}")


def raises(label: str, error: type[BaseException], function, needle: str = "") -> None:  # noqa: ANN001
    try:
        function()
    except error as caught:
        check(f"{label} -> {error.__name__}: {str(caught)[:110]}", needle in str(caught))
        return
    except Exception as other:  # noqa: BLE001
        check(f"{label} raised {type(other).__name__} instead of {error.__name__}: {other}", False)
        return
    check(f"{label} did not raise {error.__name__}", False)


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def float_bits(row: AttemptRow) -> list[int]:
    values = []
    for mapping in (row.situation, row.action, row.physics_guess, row.measured_outcome or {}, row.residual or {}):
        values += [bits(v) for k, v in sorted(mapping.items()) if isinstance(v, float)]
    return values


def section(title: str):  # noqa: ANN201
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------------------
# Shared fixture inputs (config values read from the real config.yaml where they exist)
# ---------------------------------------------------------------------------------------

config = load_config()
LEVELS = tuple(require(config, "engine.quantile_levels"))
SPREAD = tuple(require(config, "engine.spread_levels"))
PRESETS = tuple(require(config, "world.presets").keys())
HIDDEN = tuple(require(config, "world.hidden_effects").keys())
POINTS = [3, 3]  # fixture grid resolution (the real value is skills.<skill>.candidate_points_per_dim)
SEED = 7  # fixture base seed for the named streams of this script
TS = "2026-11-21T12:00:00+00:00"

spec = FixtureToySpec()
layout = build_layout(spec)
print("numpy", np.__version__, "| python", sys.version.split()[0])
print("config quantile_levels", LEVELS, "spread_levels", SPREAD, "presets", PRESETS)


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="b_engine_offline_") as scratch:
        tmp = Path(scratch)
        SCRATCH["dir"] = str(tmp)

        # ---------------------------------------------------------------- check 1
        section("1. build_real_row: residual sign/order, strict keys, JSONL bitwise round trip, dream rows")
        situation = np.array([1.2, 1.0])
        action = np.array([0.5, 0.25])
        physics = spec.physics_guess(situation[None, :], action[None, :])[0]
        measured = {"fx_fail": 1.0, "fx_error": 0.9}  # keys deliberately not in outcome order
        row = build_real_row(
            spec, run_id="fx-run", seed=SEED, world_preset=PRESETS[0], situation=situation,
            action=action, measured_outcome=measured, timestamp_utc=TS,
        )
        print("  physics", physics.tolist(), "| measured", measured, "| residual", row.residual)
        check("physics_guess/residual keys are in the spec outcome order", tuple(row.physics_guess) == spec.outcome_names
              and tuple(row.residual) == spec.outcome_names)
        check("residual == measured - physics, per outcome (exact)",
              row.residual["fx_error"] == 0.9 - physics[0] and row.residual["fx_fail"] == 1.0 - physics[1])
        check("residual sign: measured > physics gives a positive residual",
              (0.9 > physics[0]) == (row.residual["fx_error"] > 0.0) and row.residual["fx_fail"] > 0.0)
        check("situation stored with the categorical as its level string", row.situation == {"fx_distance": 1.2, "fx_surface": "b"})
        raises("extra key in measured outcome", ValueError,
               lambda: build_real_row(spec, run_id="r", seed=0, world_preset="default", situation=situation, action=action,
                                      measured_outcome={**measured, "hidden_payload_kg": 2.0}, timestamp_utc=TS),
               "extra=['hidden_payload_kg']")
        raises("missing key in measured outcome", ValueError,
               lambda: build_real_row(spec, run_id="r", seed=0, world_preset="default", situation=situation, action=action,
                                      measured_outcome={"fx_error": 0.1}, timestamp_utc=TS),
               "missing=['fx_fail']")
        raises("binary outcome not 0/1", ValueError,
               lambda: build_real_row(spec, run_id="r", seed=0, world_preset="default", situation=situation, action=action,
                                      measured_outcome={"fx_error": 0.1, "fx_fail": 0.5}, timestamp_utc=TS),
               "must be 0 or 1")
        raises("action outside bounds", ValueError,
               lambda: build_real_row(spec, run_id="r", seed=0, world_preset="default", situation=situation,
                                      action=np.array([1.5, 0.0]), measured_outcome=measured, timestamp_utc=TS),
               "outside the declared bounds")
        raises("timestamp without UTC offset", ValueError,
               lambda: build_real_row(spec, run_id="r", seed=0, world_preset="default", situation=situation, action=action,
                                      measured_outcome=measured, timestamp_utc="2026-11-21T12:00:00"),
               "zero UTC offset")
        raises("residual inconsistent with measured - physics", ValueError,
               lambda: dataclasses.replace(row, residual={"fx_error": 0.0, "fx_fail": 0.0}),
               "residual != measured_outcome - physics_guess")

        # Awkward floats for the bitwise round trip (all inside the fixture bounds).
        awkward = [(5e-324, 1.0 / 3.0), (0.1 + 0.2, 2.0 ** -40), (1.9999999999999998, 0.7)]
        rows = []
        for index, (distance, speed) in enumerate(awkward):
            rows.append(build_real_row(
                spec, run_id="fx-run", seed=SEED + 1 + index, world_preset=PRESETS[index % len(PRESETS)],
                situation=np.array([distance, float(index % 2)]), action=np.array([speed, 0.5]),
                measured_outcome={"fx_error": -1e-300 * (index + 1), "fx_fail": float(index % 2)}, timestamp_utc=TS,
            ))
        attempts = tmp / "attempts" / "fixture.jsonl"
        append_rows(attempts, [row], spec=spec)
        append_rows(attempts, rows, spec=spec)
        back = read_rows(attempts, spec=spec)
        check("JSONL round trip returns equal rows", back == [row, *rows])
        check("JSONL round trip is bitwise for every float", all(float_bits(a) == float_bits(b) for a, b in zip(back, [row, *rows])))
        check("re-serialized rows reproduce the file bytes", b"".join(r.to_json_line() for r in back) == attempts.read_bytes())
        first_line = attempts.read_bytes().split(b"\n")[0].decode()
        print("  first line:", first_line[:160], "...")

        bad = tmp / "bad.jsonl"
        bad.write_bytes(attempts.read_bytes().replace(b'"fx_error":0.9', b'"fx_error":NaN', 1))
        raises("read_rows rejects NaN", ValueError, lambda: read_rows(bad), "non-finite JSON constant NaN")
        bad.write_bytes(attempts.read_bytes()[:-5])
        raises("read_rows rejects an incomplete last line", ValueError, lambda: read_rows(bad), "incomplete")
        raises("append_rows refuses to append after an incomplete line", ValueError, lambda: append_rows(bad, [row], spec=spec), "incomplete line")
        other = FixtureToySpec()
        other.name = "fixture_other"
        raises("read_rows(spec=...) rejects a row of another skill", ValueError, lambda: read_rows(attempts, spec=other), "belongs to skill")

        decision = decide_batch(spec, situation[None, :], model=None, layout=layout, points_per_dim=POINTS,
                                quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)[0]
        dream_row = build_dream_row(spec, run_id="fx-dream", seed=SEED, world_preset=PRESETS[0], decision=decision, timestamp_utc=TS)
        check("dream row: source 'dream', measured_outcome and residual are None",
              dream_row.source == "dream" and dream_row.measured_outcome is None and dream_row.residual is None)
        check("dream row: action is the decision's chosen action", spec.encode_action(dream_row.action).tolist() == decision.action.tolist())
        check("dream row: physics_guess is the physics at the chosen candidate",
              [dream_row.physics_guess[n] for n in spec.outcome_names] == decision.predicted.physics[decision.chosen_index].tolist())
        check("dream row: teacher summary indices", dream_row.teacher["chosen_index"] == decision.chosen_index
              and dream_row.teacher["physics_index"] == decision.physics_index and dream_row.teacher["n_candidates"] == 9)
        raises("a dream row with a measured outcome is rejected", ValueError,
               lambda: dataclasses.replace(dream_row, measured_outcome=dict(dream_row.physics_guess)), "not a measurement")
        raises("a real row without a measured outcome is rejected", ValueError,
               lambda: dataclasses.replace(row, measured_outcome=None), "needs measured_outcome")
        dream_file = tmp / "dream_rows.jsonl"
        append_rows(dream_file, [dream_row], spec=spec)
        check("dream row JSONL round trip", read_rows(dream_file, spec=spec) == [dream_row])

        # ---------------------------------------------------------------- check 2
        section("2. rows_to_arrays: shapes, equality with inputs, refuses dream rows")
        rng = stream_rng(SEED, "fixture/rows/situations")
        situations = spec.sample_situations(rng, 6)
        actions = np.column_stack([np.linspace(0.0, 1.0, 6), np.linspace(0.0, 0.5, 6)])
        measured_arr = np.column_stack([np.linspace(-0.3, 0.4, 6), np.array([0, 1, 0, 1, 1, 0], dtype=np.float64)])
        real_rows = [
            build_real_row(spec, run_id="fx-run", seed=100 + i, world_preset=PRESETS[i % 2], situation=situations[i],
                           action=actions[i], measured_outcome=dict(zip(spec.outcome_names, measured_arr[i])), timestamp_utc=TS)
            for i in range(6)
        ]
        arrays = rows_to_arrays(spec, real_rows, on_physics_mismatch="error")
        check("shapes (6,2) (6,2) (6,2) (6,2) (6,2)", [a.shape for a in (arrays.situations, arrays.actions, arrays.physics,
              arrays.measured, arrays.residuals)] == [(6, 2)] * 5)
        check("situations/actions/measured equal the inputs exactly",
              np.array_equal(arrays.situations, situations) and np.array_equal(arrays.actions, actions)
              and np.array_equal(arrays.measured, measured_arr))
        check("physics equals spec.physics_guess of the batch", np.array_equal(arrays.physics, spec.physics_guess(situations, actions)))
        check("residuals == measured - physics exactly", np.array_equal(arrays.residuals, arrays.measured - arrays.physics))
        check("world_presets in row order", arrays.world_presets == [PRESETS[i % 2] for i in range(6)])
        path2 = tmp / "attempts" / "fixture2.jsonl"
        append_rows(path2, real_rows, spec=spec)
        arrays_back = rows_to_arrays(spec, read_rows(path2), on_physics_mismatch="error")
        check("arrays identical after a JSONL round trip", all(np.array_equal(getattr(arrays, f), getattr(arrays_back, f))
              for f in ("situations", "actions", "physics", "measured", "residuals")))
        check("zero rows give (0, width) arrays", rows_to_arrays(spec, [], on_physics_mismatch="error").residuals.shape == (0, 2))
        raises("rows_to_arrays refuses dream rows", ValueError,
               lambda: rows_to_arrays(spec, [*real_rows, dream_row], on_physics_mismatch="error"), "only real rows")

        # ---------------------------------------------------------------- check 3
        section("3. exploration_actions: reproducible, in bounds, fraction 0.0 = physics-only")
        context = spec.sample_situations(stream_rng(SEED, "fixture/context/situations"), 40)

        def explore(fraction: float, index: int = 0) -> np.ndarray:
            return exploration_actions(spec, context, rng=stream_rng(SEED, "fixture/context/actions", index),
                                       points_per_dim=POINTS, exploration_fraction=fraction,
                                       quantile_levels=LEVELS, spread_levels=SPREAD, layout=layout)

        half_a, half_b = explore(0.5), explore(0.5)
        check("same stream -> identical actions", np.array_equal(half_a, half_b))
        check("actions shape (40, 2)", half_a.shape == (40, 2))
        check("every action inside the spec bounds", bool(np.all(spec.action_in_bounds(half_a))))
        grids = [spec.candidate_actions(s, POINTS) for s in context]
        check("every action is one of its situation's candidates",
              all(any(np.array_equal(a, c) for c in grid) for a, grid in zip(half_a, grids)))
        physics_only = np.stack([d.physics_action for d in decide_batch(spec, context, model=None, layout=layout,
                                 points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)])
        check("exploration_fraction 0.0 equals the physics-only choice", np.array_equal(explore(0.0), physics_only))
        differs = int(np.sum(np.any(half_a != physics_only, axis=1)))
        print(f"  rows differing from physics-only at fraction 0.5: {differs}/40 (fixture observation, not a metric)")
        check("exploration_fraction 1.0 stays on the candidate grids",
              all(any(np.array_equal(a, c) for c in grid) for a, grid in zip(explore(1.0), grids)))
        check("another stream index gives a different draw", not np.array_equal(explore(0.5, 1), half_a))
        raises("exploration_fraction outside [0, 1]", ValueError, lambda: explore(1.5), "must lie in [0, 1]")

        # ---------------------------------------------------------------- check 4
        section("4. dream(): physics-only path and DreamSet construction from decide_batch decisions")
        dreams = dream(spec, None, layout, n_dreams=50, rng=stream_rng(SEED, "fixture/dreams/situations"),
                       points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)
        check("DreamSet shapes: situations (50,2), candidates (50,9,2), scores (50,9)",
              dreams.situations.shape == (50, 2) and dreams.candidates.shape == (50, 9, 2) and dreams.scores.shape == (50, 9))
        check("situations come from spec.sample_situations on the same stream",
              np.array_equal(dreams.situations, spec.sample_situations(stream_rng(SEED, "fixture/dreams/situations"), 50)))
        check("physics only: chosen == physics_choice and chosen_spread == 0",
              np.array_equal(dreams.chosen, dreams.physics_choice) and bool(np.all(dreams.chosen_spread == 0.0)))
        check("physics only: no request hash", dreams.request_hashes == () and dreams.request_groups == ()
              and bool(np.all(dreams.request_group == -1)))
        same = dreams_from_decisions(spec, decide_batch(spec, dreams.situations, model=None, layout=layout, points_per_dim=POINTS,
                                     quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY), spread_levels=SPREAD)
        check("dreams_from_decisions(decide_batch(...)) equals dream() on the same situations",
              all(np.array_equal(getattr(same, f), getattr(dreams, f)) for f in ("candidates", "scores", "chosen", "chosen_physics")))
        check("chosen_physics equals spec.physics_guess at the chosen actions",
              np.array_equal(dreams.chosen_physics, spec.physics_guess(dreams.situations, dreams.chosen_actions)))
        uneven = FixtureUnevenGridSpec()
        raises("dream() raises when candidate counts differ", ValueError,
               lambda: dream(uneven, None, build_layout(uneven), n_dreams=20, rng=stream_rng(SEED, "fixture/dreams/uneven"),
                             points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY),
               "candidate counts differ")
        rows_d = dream_rows(spec, dreams, run_id="fx-dream", seed=SEED, world_preset=PRESETS[0], timestamp_utc=TS)
        check("dream_rows: 50 rows, all source 'dream' with no measurement",
              len(rows_d) == 50 and all(r.source == "dream" and r.measured_outcome is None and r.residual is None for r in rows_d))
        check("dream_rows actions are the chosen actions",
              np.array_equal(np.stack([spec.encode_action(r.action) for r in rows_d]), dreams.chosen_actions))
        dream_file2 = tmp / "dreams.jsonl"
        append_rows(dream_file2, rows_d, spec=spec)
        check("dream rows JSONL round trip", read_rows(dream_file2, spec=spec) == rows_d)
        # ---------------------------------------------------------------- check 4b
        section("4b. fit_residual_model + model-backed dream() with the production ConstantBiasResidual "
                "(plumbing on fixture rows; not a performance result)")
        if ConstantBiasResidual is None:
            print(f"  NOT RUN: learned_judgment.eval.baselines.ConstantBiasResidual did not import: {BASELINE_IMPORT_ERROR}")
        else:
            model = ConstantBiasResidual(layout, LEVELS, SPREAD, seed=SEED)
            fitted = fit_residual_model(spec, model, real_rows, layout, on_physics_mismatch="error")
            probe = build_features(layout, arrays.situations, arrays.actions, physics=arrays.physics)
            fitted_dist = fitted.predict_residual_distribution(probe)
            check("fit_residual_model returns the fitted model, trained on all 6 rows", fitted is model and model.n_train == 6)
            check("targets were the stored residuals: constant mean == residuals.mean(axis=0) exactly",
                  np.array_equal(fitted_dist.mean[0], arrays.residuals.mean(axis=0)))
            raises("fit_residual_model refuses zero rows", ValueError,
                   lambda: fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=SEED), [], layout,
                                              on_physics_mismatch="error"), "no attempts")
            raises("fit_residual_model refuses dream rows", ValueError,
                   lambda: fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=SEED), [*real_rows, dream_row], layout,
                                              on_physics_mismatch="error"),
                   "only real rows")
            raises("fit_residual_model refuses a layout of another skill", ValueError,
                   lambda: fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=SEED), real_rows, build_layout(other),
                                              on_physics_mismatch="error"),
                   "layout skill")
            direct_layout = build_layout(spec, include_physics=False)
            direct = fit_residual_model(spec, ConstantBiasResidual(direct_layout, LEVELS, SPREAD, seed=SEED), real_rows, direct_layout,
                                        on_physics_mismatch="error")
            check("a layout without a physics block fits (direct baseline layout)", direct.n_train == 6)

            dreams_m = dream(spec, fitted, layout, n_dreams=30, rng=stream_rng(SEED, "fixture/dreams/model"), points_per_dim=POINTS,
                             quantile_levels=LEVELS, spread_levels=SPREAD, method="constant_bias")
            spread_primary = fitted_dist.spread[0, spec.outcome_index(spec.primary_outcome)]
            check("model-backed DreamSet shapes (30, 9, 2) and method label",
                  dreams_m.candidates.shape == (30, 9, 2) and dreams_m.method == "constant_bias")
            check("chosen_spread is the model's primary-outcome spread", bool(np.all(dreams_m.chosen_spread == spread_primary)))
            check("chosen_median == clip(chosen_physics + model median residual, outcome bounds)",
                  np.array_equal(dreams_m.chosen_median, np.clip(dreams_m.chosen_physics + fitted_dist.median[0],
                                                                 spec.outcome_low, spec.outcome_high)))
            check("a model without last_request_hash records no request", dreams_m.request_hashes == ()
                  and dreams_m.request_groups == () and bool(np.all(dreams_m.request_group == -1)))
            check("keep_confident at exactly the constant spread keeps every row; just below keeps none",
                  keep_confident(dreams_m, float(spread_primary)).n_rows == 30
                  and keep_confident(dreams_m, float(np.nextafter(spread_primary, 0.0))).n_rows == 0)
            print(f"  fixture observation: constant spread of {spec.primary_outcome} = {spread_primary!r}; "
                  f"chosen != physics in {int(np.sum(dreams_m.chosen != dreams_m.physics_choice))}/30 dreams (not a metric)")
            raises("model given with the physics-only label", ValueError,
                   lambda: dream(spec, fitted, layout, n_dreams=3, rng=stream_rng(SEED, "fixture/dreams/model"), points_per_dim=POINTS,
                                 quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY), "disagree")

            world_var = world_id_variable(PRESETS)
            layout_w = world_id_layout(spec, PRESETS, hidden_value_names=HIDDEN, include_physics=True)
            fitted_w = fit_residual_model(spec, ConstantBiasResidual(layout_w, LEVELS, SPREAD, seed=SEED), real_rows, layout_w,
                                          on_physics_mismatch="error", extra=world_id_column(real_rows, world_var))
            dreams_w = dream(spec, fitted_w, layout_w, n_dreams=10, rng=stream_rng(SEED, "fixture/dreams/world"), points_per_dim=POINTS,
                             quantile_levels=LEVELS, spread_levels=SPREAD, method="constant_bias",
                             extra=world_id_column([PRESETS[1]] * 10, world_var))
            check("world-id layout: fit with an extra column and dream with a declared world id", dreams_w.n_rows == 10
                  and fitted_w.n_train == 6)
            raises("world-id layout without the extra block", ValueError,
                   lambda: fit_residual_model(spec, ConstantBiasResidual(layout_w, LEVELS, SPREAD, seed=SEED), real_rows, layout_w,
                                              on_physics_mismatch="error"),
                   "needs a extra block")

        # ---------------------------------------------------------------- check 5
        section("5. DreamSet save/load, keep_confident, select_context, world id, recovery_attempts")
        store = tmp / "dreams" / "fixture.npz"
        meta = {"config_digest": "fixture", "seed": SEED, "n_dreams": 50}
        save_dreams(store, dreams, meta)
        loaded, meta_back = load_dreams(store)
        fields = ("situations", "candidates", "scores", "chosen", "physics_choice", "chosen_spread", "request_group",
                  "chosen_physics", "chosen_mean", "chosen_median", "chosen_outcome_spread")
        check("save/load arrays are bitwise identical (bytes and dtype)",
              all(getattr(loaded, f).tobytes() == getattr(dreams, f).tobytes() and getattr(loaded, f).dtype == getattr(dreams, f).dtype
                  for f in fields))
        check("save/load names, provenance and meta", meta_back == meta and loaded.outcome_names == dreams.outcome_names
              and loaded.spread_levels == dreams.spread_levels and loaded.method == dreams.method
              and loaded.request_hashes == dreams.request_hashes and loaded.request_groups == dreams.request_groups)
        check("only the .npz and .json are left (no temp files)", sorted(p.name for p in store.parent.iterdir()) == ["fixture.json", "fixture.npz"])
        store.write_bytes(store.read_bytes() + b"x")
        raises("load_dreams detects arrays that do not match the recorded sha256", ValueError, lambda: load_dreams(store), "sha256")
        raises("save_dreams rejects non-finite meta", ValueError, lambda: save_dreams(tmp / "d2.npz", dreams, {"x": float("nan")}), "meta")

        # Hand-written fixture spreads to exercise the filter (the physics-only set has spread 0 everywhere).
        spreads = np.zeros((50, 2))
        spreads[:, 0] = np.linspace(0.0, 0.49, 50)  # primary outcome spread 0.00, 0.01, ..., 0.49
        hand = dataclasses.replace(dreams, chosen_outcome_spread=spreads, chosen_spread=spreads[:, 0].copy())
        check("DreamSet is a DreamSet after replace", isinstance(hand, DreamSet))
        threshold = float(spreads[20, 0])
        kept = keep_confident(hand, threshold)
        check(f"keep_confident({threshold!r}) keeps exactly rows with spread <= threshold (boundary row kept)",
              kept.n_rows == int(np.sum(spreads[:, 0] <= threshold)) == 21 and bool(np.all(kept.chosen_spread <= threshold))
              and np.array_equal(kept.situations, hand.situations[:21]))
        check("keep_confident keeps C and every per-row array aligned", kept.candidates.shape == (21, 9, 2)
              and np.array_equal(kept.chosen_actions, hand.chosen_actions[:21]))
        check("keep_confident(0.0) keeps only the single zero-spread row", keep_confident(hand, 0.0).n_rows == 1)
        raises("keep_confident rejects a negative threshold", ValueError, lambda: keep_confident(hand, -0.1), "must be >= 0")
        empty = keep_confident(dataclasses.replace(hand, chosen_outcome_spread=spreads + 1.0, chosen_spread=spreads[:, 0] + 1.0), 0.5)
        save_dreams(tmp / "empty.npz", empty, {})
        check("an empty filtered set saves and loads", load_dreams(tmp / "empty.npz")[0].candidates.shape == (0, 9, 2))

        mixed = [real_rows[0], dream_row, real_rows[1], real_rows[2], dream_row, real_rows[3], real_rows[4], real_rows[5]]
        check("select_context 'all' -> every real row in order", select_context(mixed, policy="all", recent_window=None) == real_rows)
        check("select_context 'recent' 4 -> last 4 real rows", select_context(mixed, policy="recent", recent_window=4) == real_rows[2:])
        check("select_context 'recent' 10 -> all 6 real rows", select_context(mixed, policy="recent", recent_window=10) == real_rows)
        raises("'recent' without a window", ValueError, lambda: select_context(mixed, policy="recent", recent_window=None), "recent_window")
        raises("'all' with a window", ValueError, lambda: select_context(mixed, policy="all", recent_window=3), "only by policy 'recent'")
        raises("unknown policy", ValueError, lambda: select_context(mixed, policy="world_id", recent_window=None), "unknown context policy")

        variable = world_id_variable(PRESETS)
        column = world_id_column(real_rows, variable)
        check("world_id levels are the declared presets in config order", variable.levels == PRESETS and variable.name == "world_id")
        check("world_id_column -> (6, 1) level indices", column.shape == (6, 1) and column[:, 0].tolist() == [0.0, 1.0] * 3)
        check("world_id_column accepts preset names", world_id_column(list(PRESETS[::-1]), variable)[:, 0].tolist() == [4.0, 3.0, 2.0, 1.0, 0.0])
        raises("undeclared preset", ValueError, lambda: world_id_column(["moon"], variable), "not one of")
        layout_w = world_id_layout(spec, PRESETS, hidden_value_names=HIDDEN, include_physics=True)
        features = build_features(layout_w, arrays.situations, arrays.actions, physics=arrays.physics, extra=column)
        check("layout with world_id: features (6, 7), world_id column categorical", features.shape == (6, 7)
              and layout_w.categorical_indices == (1, 6))

        curve = {0: 2.0, 5: 1.3, 10: 1.19, 15: 1.0, 20: 1.1}
        check("recovery_attempts: threshold 1.0*(1+0.2) -> first 10, sustained 10",
              recovery_attempts(curve, 1.0, 0.2) == {"first": 10, "sustained": 10, "threshold": 1.2, "pre_shift_error_is_zero": False})
        check("recovery_attempts: exact boundary counts (1.25 <= 1.0*(1+0.25))", recovery_attempts({0: 3.0, 5: 1.25}, 1.0, 0.25)["first"] == 5)
        check("recovery_attempts: not reached -> None, None",
              [recovery_attempts({0: 3.0, 5: 2.0}, 1.0, 0.2)[key] for key in ("first", "sustained")] == [None, None])
        check("recovery_attempts: smallest count, not first key", recovery_attempts({20: 1.0, 10: 1.1, 0: 5.0}, 1.0, 0.2)["first"] == 10)
        raises("recovery_attempts: negative count", ValueError, lambda: recovery_attempts({-5: 1.0}, 1.0, 0.2), "int >= 0")
        raises("recovery_attempts: empty", ValueError, lambda: recovery_attempts({}, 1.0, 0.2), "empty")


try:
    run()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    RESULTS.append(("script completed without an unexpected exception", False))

failed = [label for label, ok in RESULTS if not ok]
print(f"\nchecks: {len(RESULTS)} run, {len(RESULTS) - len(failed)} held, {len(failed)} failed")
print("NOT RUN: collect() (needs Person A's Go2World; never called with a stand-in world); dream() with the "
      "TabPFN teacher (no real TabPFN prediction has been made)")
if ConstantBiasResidual is None:
    print("NOT RUN: fit_residual_model() and model-backed dream() (ConstantBiasResidual did not import)")
sys.exit(1 if failed else 0)
