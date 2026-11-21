"""Regression checks for the engine-lane review findings (engine-data-integrity review, 2026-11-21).

FIXTURE CHECK ONLY. FixtureSpec below is a toy SkillSpec written for this script; it is not a
robot skill. ConstantEpisodeSpec is a THROWAWAY: its episode() is NOT an episode -- it records
the world token and attempt seed it was handed and returns a constant mapping -- so that the
file, seed and validation mechanics of collect() can be exercised. ThrowawayWorldToken is NOT
a world (no reset, no step, no physics), only an identity marker. TwoRequestModel is a
duck-typed fixture residual model that names fixture request keys the way the teacher does
when it makes several requests for one predict call; it is not TabPFN and makes no request.
Nothing printed here is evidence about the Go2, the simulator, TabPFN or any model, and nothing
here may enter a results table. Files are written only inside a temporary directory that is
deleted at the end.

Each section below is one review item. It ran against the code before the fix (a snapshot of
the package) and failed there; it passes on the fixed code. Sections are independent: a
failure in one is reported and the next one still runs.

    G1       collect() enforces the verified command schema before the world is stepped
    G-seed   collect() runs every attempt in ONE world with its own attempt seed; validates
             every input before the first episode
    G2       rows_to_arrays / fit_residual_model re-check the stored physics guess
    G3       every provider request hash of every decision is recorded (DreamSet, dream rows)
    G5       exploration_actions: row i does not depend on n
    G6       append_rows re-validates every row immediately before writing
    G7       (run_id, seed) names one real attempt
    G11      world-id layout through build_layout with required hidden value names
    G13      recovery_attempts reports first / sustained / threshold / zero pre-shift error
    G-index  a situation index maps to the same situation at every n (splits, dreams)

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_engine_offline\\check_review_fixes.py
Exit code 0 means every check held; 1 means at least one failed.
"""

from __future__ import annotations

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root

import learned_judgment.engine.adapt as adapt
import learned_judgment.engine.collect as collect_mod
import learned_judgment.engine.dream as dream_mod
import learned_judgment.engine.learn as learn
import learned_judgment.engine.rows as rows_mod
from learned_judgment.config import load_config, require
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch
from learned_judgment.engine.features import build_layout
from learned_judgment.engine.streams import stream_rng
from learned_judgment.engine.types import ResidualDistribution
from learned_judgment.eval.splits import noise_seeds, split_situations
from learned_judgment.skills.base import (
    ActionDim,
    CommandField,
    CommandSchema,
    Outcome,
    SkillSpec,
    SkillSpecError,
    Variable,
)

try:  # production B2 baseline, used only to exercise fit_residual_model plumbing
    from learned_judgment.eval.baselines import ConstantBiasResidual
except Exception as import_error:  # noqa: BLE001
    ConstantBiasResidual = None
    BASELINE_IMPORT_ERROR = repr(import_error)

# ---------------------------------------------------------------------------------------
# Fixtures (exist only in this script)
# ---------------------------------------------------------------------------------------

# Fixture command schema: NOT the Go2 walking policy's schema.
FIXTURE_SCHEMA = CommandSchema(
    fields={"fx_cmd": CommandField(name="fx_cmd", unit="m/s", low=-1.0, high=1.0)},
    update_period_s=0.02,
    source="fixture (B_engine_offline check_review_fixes)",
)


class FixtureSpec(SkillSpec):
    """FIXTURE: toy SkillSpec (2 features, one categorical; 2 actions; 2 outcomes, one binary).
    ``gain`` is a toy physics constant, so a "recalibrated" physics can be simulated."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain
        self.name = "fixture_review"
        self.description = "fixture for engine review regression checks; not a robot skill"
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
        situations = np.asarray(situations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        error = situations[:, 0] - self.gain * actions[:, 0] * (1.0 - actions[:, 1]) - 0.1 * situations[:, 1]
        fail = 0.5 * actions[:, 0] + 0.2 * situations[:, 1]
        return np.column_stack([error, fail])

    def objective(self, predicted):  # noqa: ANN001, ANN201
        return -np.abs(predicted.median_of("fx_error")) - predicted.mean_of("fx_fail") - predicted.spread_of("fx_error")

    def episode(self, world, situation, action, attempt_seed):  # noqa: ANN001, ANN201
        raise RuntimeError("FixtureSpec has no episode")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(measured)[:, 0])

    def success(self, measured: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(measured)[:, 0]) <= 0.1


class OutOfBoundsPhysicsSpec(FixtureSpec):
    """FIXTURE: a 'changed' physics whose failure probability leaves [0, 1]."""

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        physics = super().physics_guess(situations, actions)
        physics[:, 1] = 1.5
        return physics


class ThrowawayWorldToken:
    """THROWAWAY identity marker, NOT a world: no reset, no step, no physics."""

    def __repr__(self) -> str:
        return "ThrowawayWorldToken()"


class ConstantEpisodeSpec(FixtureSpec):
    """THROWAWAY: episode() is NOT an episode. It records what collect() handed it and returns a
    constant mapping, so collect()'s file / seed / validation mechanics can be exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, dict, dict, object]] = []

    def episode(self, world, situation, action, attempt_seed):  # noqa: ANN001, ANN201
        self.calls.append((world, dict(situation), dict(action), attempt_seed))
        return {"fx_error": 0.25, "fx_fail": 0.0}


class TwoRequestModel:
    """FIXTURE duck-typed residual model: answers one predict call and names the given fixture
    request keys, as the teacher does when it splits one predict call into several requests.
    Its spread grows with the first feature. It is not TabPFN and makes no request."""

    def __init__(self, keys: tuple[str, ...], levels: tuple[float, ...], spread_levels: tuple[float, float]) -> None:
        self.keys = keys
        self.levels = np.asarray(levels, dtype=np.float64)
        self.spread_levels = spread_levels
        self.last_request_hash = None
        self.last_request_hashes: tuple[str, ...] = ()

    def fit(self, features, residuals):  # noqa: ANN001, ANN201
        return self

    def predict_residuals(self, query_features):  # noqa: ANN001, ANN201
        distribution = self.predict_residual_distribution(query_features)
        return distribution.median, distribution.spread

    def predict_residual_distribution(self, query_features):  # noqa: ANN001, ANN201
        n = query_features.shape[0]
        width = 0.05 + 0.1 * query_features[:, 0]
        offsets = np.linspace(-1.0, 1.0, self.levels.size)
        quantiles = np.zeros((n, 2, self.levels.size))
        quantiles[:, 0, :] = width[:, None] * offsets[None, :]
        self.last_request_hashes = tuple(self.keys)
        self.last_request_hash = self.keys[-1]
        return ResidualDistribution.from_quantiles(np.zeros((n, 2)), np.zeros((n, 2)), self.levels, quantiles,
                                                   self.spread_levels)


# ---------------------------------------------------------------------------------------
# Tiny harness
# ---------------------------------------------------------------------------------------

RESULTS: list[tuple[str, bool]] = []
SCRATCH = {"dir": ""}


def check(label: str, condition: bool) -> None:
    RESULTS.append((label, bool(condition)))
    if SCRATCH["dir"]:
        label = label.replace(SCRATCH["dir"], "<tmp>")
    print(f"  [{'ok' if condition else 'FAILED'}] {label}")


def raises(label: str, error: type[BaseException], function, needle: str = "") -> None:  # noqa: ANN001
    try:
        function()
    except error as caught:
        check(f"{label} -> {type(caught).__name__}: {str(caught)[:150]}", needle in str(caught))
        return
    except Exception as other:  # noqa: BLE001
        check(f"{label} raised {type(other).__name__} instead of {error.__name__}: {str(other)[:150]}", False)
        return
    check(f"{label} did not raise {error.__name__}", False)


def section(title: str, body) -> None:  # noqa: ANN001
    print(f"\n== {title} ==")
    try:
        body()
    except Exception as error:  # noqa: BLE001
        lines = traceback.format_exception_only(type(error), error)
        check(f"section '{title.split(' ')[0]}' ran without an unexpected exception: {''.join(lines).strip()[:200]}", False)


config = load_config()
LEVELS = tuple(require(config, "engine.quantile_levels"))
SPREAD = tuple(require(config, "engine.spread_levels"))
PRESETS = tuple(require(config, "world.presets").keys())
HIDDEN = tuple(require(config, "world.hidden_effects").keys())
POINTS = [3, 3]  # fixture grid resolution
SEED = 11  # fixture base seed of this script's named streams
TS = "2026-11-21T12:00:00+00:00"


def clock() -> str:
    return TS


spec = FixtureSpec()
layout = build_layout(spec)
print("numpy", np.__version__, "| python", sys.version.split()[0])


def real_row(fixture: FixtureSpec, index: int, *, run_id: str = "fx-run", seed: int | None = None):  # noqa: ANN201
    situation = np.array([0.2 + 0.15 * (index % 10), float(index % 2)])  # inside [0, 2] for every index
    action = np.array([0.5 + 0.1 * (index % 3), 0.1 * (index % 4)])
    return rows_mod.build_real_row(
        fixture, run_id=run_id, seed=100 + index if seed is None else seed, world_preset=PRESETS[0],
        situation=situation, action=action, measured_outcome={"fx_error": 0.05 * index - 0.1, "fx_fail": float(index % 2)},
        timestamp_utc=TS,
    )


def run(tmp: Path) -> None:
    # ------------------------------------------------------------------------------ G1
    def g1() -> None:
        world = ThrowawayWorldToken()
        throwaway = ConstantEpisodeSpec()
        situations = np.array([[0.5, 0.0], [1.0, 1.0]])
        actions = np.array([[0.5, 0.1], [1.0, 0.0]])
        schemas = (
            ("a schema without the spec's command field", CommandSchema(
                fields={"other": CommandField("other", "m/s", -1.0, 1.0)}, update_period_s=0.02, source="fixture")),
            ("a schema narrower than the spec's action range", CommandSchema(
                fields={"fx_cmd": CommandField("fx_cmd", "m/s", -0.5, 0.5)}, update_period_s=0.02, source="fixture")),
            ("a schema with another unit", CommandSchema(
                fields={"fx_cmd": CommandField("fx_cmd", "rad/s", -1.0, 1.0)}, update_period_s=0.02, source="fixture")),
        )
        for label, schema in schemas:
            target = tmp / "g1" / f"{label.split()[2]}.jsonl"
            raises(f"collect refuses {label}", SkillSpecError,
                   lambda schema=schema, target=target: collect_mod.collect(
                       throwaway, world, situations, actions, attempt_seeds=np.array([1, 2]), command_schema=schema,
                       run_id="fx-g1", world_preset=PRESETS[0], out_path=target, clock=clock),
                   "verified")
        check("no episode was reached and no attempt file was written",
              throwaway.calls == [] and not (tmp / "g1").exists())

    # ------------------------------------------------------------------------------ G-seed
    def g_seed() -> None:
        world = ThrowawayWorldToken()
        throwaway = ConstantEpisodeSpec()
        situations = np.array([[0.5, 0.0], [1.0, 1.0], [1.5, 0.0]])
        actions = np.array([[0.5, 0.1], [1.0, 0.0], [0.0, 0.5]])
        out = tmp / "gseed" / "attempts.jsonl"
        rows = collect_mod.collect(throwaway, world, situations, actions, attempt_seeds=np.array([31, 7, 19]),
                                   command_schema=FIXTURE_SCHEMA, run_id="fx-collect", world_preset=PRESETS[0],
                                   out_path=out, clock=clock)
        check("every attempt ran in the ONE world the caller built", [call[0] for call in throwaway.calls] == [world] * 3
              and all(call[0] is world for call in throwaway.calls))
        check("attempt i got its own attempt seed, as a Python int",
              [call[3] for call in throwaway.calls] == [31, 7, 19] and all(type(call[3]) is int for call in throwaway.calls))
        check("situation / action mappings handed to the episode are the decoded inputs",
              [call[1] for call in throwaway.calls] == [spec.decode_situation(s) for s in situations]
              and [call[2] for call in throwaway.calls] == [spec.decode_action(a) for a in actions])
        check("each row's seed is its attempt seed", [row.seed for row in rows] == [31, 7, 19])
        check("the file holds exactly the returned rows", rows_mod.read_rows(out, spec=throwaway) == rows)

        before = out.read_bytes()
        good = dict(attempt_seeds=np.array([41, 42, 43]), command_schema=FIXTURE_SCHEMA, run_id="fx-collect-2",
                    world_preset=PRESETS[0], out_path=out, clock=clock)
        cases = (
            ("run_id ''", dict(run_id=""), "run_id"),
            ("world_preset ''", dict(world_preset=""), "world_preset"),
            ("a clock without a UTC offset", dict(clock=lambda: "2026-11-21T12:00:00"), "zero UTC offset"),
            ("a negative attempt seed", dict(attempt_seeds=np.array([41, -1, 43])), ">= 0"),
            ("a repeated attempt seed", dict(attempt_seeds=np.array([41, 42, 41])), "repeats"),
            ("float attempt seeds", dict(attempt_seeds=np.array([41.0, 42.0, 43.0])), "integer"),
            ("too few attempt seeds", dict(attempt_seeds=np.array([41, 42])), "shape"),
            ("a rerun into the same file with the same (run_id, seed)", dict(run_id="fx-collect", attempt_seeds=np.array([31, 8, 9])),
             "already holds"),
        )
        for label, override, needle in cases:
            arguments = {**good, **override}
            raises(f"collect validates {label} before the first episode", ValueError,
                   lambda arguments=arguments: collect_mod.collect(throwaway, world, situations, actions, **arguments), needle)
        broken = tmp / "gseed" / "broken.jsonl"
        broken.write_bytes(before[:-3])
        raises("collect refuses an attempt file whose last line is incomplete, before the first episode", ValueError,
               lambda: collect_mod.collect(throwaway, world, situations, actions, **{**good, "out_path": broken}), "incomplete")
        check("none of the refused calls reached an episode or changed the file",
              len(throwaway.calls) == 3 and out.read_bytes() == before)

    # ------------------------------------------------------------------------------ G2
    def g2() -> None:
        old_spec, new_spec = FixtureSpec(gain=1.0), FixtureSpec(gain=2.0)
        rows = [real_row(old_spec, index) for index in range(3)]
        path = tmp / "g2" / "attempts.jsonl"
        rows_mod.append_rows(path, rows, spec=old_spec)
        stored = path.read_bytes()
        same_error = rows_mod.rows_to_arrays(old_spec, rows, on_physics_mismatch="error")
        same_recompute = rows_mod.rows_to_arrays(old_spec, rows, on_physics_mismatch="recompute")
        check("unchanged physics: no stale row under either policy, identical arrays",
              same_error.stale_physics_rows == () and same_recompute.n_stale_physics == 0
              and all(np.array_equal(getattr(same_error, f), getattr(same_recompute, f)) for f in ("physics", "residuals")))
        back = rows_mod.read_rows(path, spec=new_spec)
        check("read_rows(spec=changed physics) still reads the file (structural validation only)", back == rows)
        check("StalePhysicsError is a ValueError", issubclass(rows_mod.StalePhysicsError, ValueError))
        raises("rows_to_arrays 'error' refuses rows written under another physics", rows_mod.StalePhysicsError,
               lambda: rows_mod.rows_to_arrays(new_spec, back, on_physics_mismatch="error"), "rows[0]")
        recomputed = rows_mod.rows_to_arrays(new_spec, back, on_physics_mismatch="recompute")
        current = np.stack([new_spec.physics_guess(s[None, :], a[None, :])[0]
                            for s, a in zip(same_error.situations, same_error.actions)])
        check("'recompute': physics is the current spec's physics, row by row (bitwise)", np.array_equal(recomputed.physics, current))
        check("'recompute': measured outcome is the stored one", np.array_equal(recomputed.measured, same_error.measured))
        check("'recompute': residual == stored measured - current physics (exact)",
              np.array_equal(recomputed.residuals, same_error.measured - current))
        check("'recompute': residuals differ from the stored (old-physics) residuals",
              not np.array_equal(recomputed.residuals, same_error.residuals))
        check("'recompute' reports how many rows differed: stale_physics_rows == (0, 1, 2)",
              recomputed.stale_physics_rows == (0, 1, 2) and recomputed.n_stale_physics == 3)
        check("the attempt file is never rewritten", path.read_bytes() == stored)
        raises("an unknown policy is refused", ValueError,
               lambda: rows_mod.rows_to_arrays(new_spec, back, on_physics_mismatch="ignore"), "on_physics_mismatch")
        raises("the policy is required", TypeError, lambda: rows_mod.rows_to_arrays(new_spec, back), "on_physics_mismatch")
        raises("'recompute' refuses a current physics that leaves the outcome bounds", ValueError,
               lambda: rows_mod.rows_to_arrays(OutOfBoundsPhysicsSpec(), back, on_physics_mismatch="recompute"),
               "leaves the outcome bounds")
        if ConstantBiasResidual is None:
            check(f"fit_residual_model part NOT RUN: ConstantBiasResidual did not import: {BASELINE_IMPORT_ERROR}", False)
            return
        new_layout = build_layout(new_spec)
        model = lambda: ConstantBiasResidual(new_layout, LEVELS, SPREAD, seed=SEED)  # noqa: E731
        raises("fit_residual_model requires on_physics_mismatch", TypeError,
               lambda: learn.fit_residual_model(new_spec, model(), back, new_layout), "on_physics_mismatch")
        raises("fit_residual_model 'error' refuses stale rows", rows_mod.StalePhysicsError,
               lambda: learn.fit_residual_model(new_spec, model(), back, new_layout, on_physics_mismatch="error"), "rows[0]")
        fitted = learn.fit_residual_model(new_spec, model(), back, new_layout, on_physics_mismatch="recompute")
        probe = np.zeros((1, new_layout.n_features))
        check("fit_residual_model 'recompute' trains on the recomputed residuals (constant mean == their mean)",
              np.array_equal(fitted.predict_residual_distribution(probe).mean[0], recomputed.residuals.mean(axis=0)))

    # ------------------------------------------------------------------------------ G3
    def g3() -> None:
        rng = stream_rng(SEED, "fixture/g3/situations")
        situations = spec.sample_situations(rng, 9)
        kw = dict(layout=layout, points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method="teacher")
        model_a = TwoRequestModel(("A0", "A1"), LEVELS, SPREAD)
        model_b = TwoRequestModel(("B0",), LEVELS, SPREAD)
        model_c = TwoRequestModel(("A1", "C0"), LEVELS, SPREAD)
        decisions = (decide_batch(spec, situations[:4], model=model_a, **kw) + decide_batch(spec, situations[4:6], model=model_b, **kw)
                     + decide_batch(spec, situations[6:8], model=model_c, **kw))
        decisions.append(dataclasses.replace(decide_batch(spec, situations[8:], model=model_a, **kw)[0],
                                             request_hash=None, request_hashes=()))
        dreams = dream_mod.dreams_from_decisions(spec, decisions, spread_levels=SPREAD)
        check("request_groups lists every request of each decision: ((A0, A1), (B0,), (A1, C0))",
              dreams.request_groups == (("A0", "A1"), ("B0",), ("A1", "C0")))
        check("request_group (n,) points every row at its group; -1 for no request",
              dreams.request_group.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, -1])
        check("request_hashes stays the flat tuple of distinct hashes in first-use order",
              dreams.request_hashes == ("A0", "A1", "B0", "C0"))
        check("request_hashes_of(row) gives each decision's own keys",
              [dreams.request_hashes_of(i) for i in (0, 4, 6, 8)] == [("A0", "A1"), ("B0",), ("A1", "C0"), ()])
        whole = dream_mod.dream(spec, model_a, layout, n_dreams=5, rng=stream_rng(SEED, "fixture/g3/dream"),
                                points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method="teacher")
        check("dream(): one predict call served by two requests records both for every row",
              whole.request_groups == (("A0", "A1"),) and whole.request_group.tolist() == [0] * 5)

        store = tmp / "g3" / "dreams.npz"
        dream_mod.save_dreams(store, dreams, {"fixture": True})
        loaded, _ = dream_mod.load_dreams(store)
        header = json.loads(store.with_suffix(".json").read_text(encoding="utf-8"))
        check("save/load keeps request_groups, request_group and request_hashes; format v2",
              loaded.request_groups == dreams.request_groups and np.array_equal(loaded.request_group, dreams.request_group)
              and loaded.request_hashes == dreams.request_hashes and header["format"] == "learned-judgment-dreamset-v2")
        threshold = float(np.median(dreams.chosen_spread))
        kept = dream_mod.keep_confident(dreams, threshold)
        mask = dreams.chosen_spread <= threshold
        check("keep_confident keeps each kept row's request keys aligned",
              [kept.request_hashes_of(i) for i in range(kept.n_rows)]
              == [dreams.request_hashes_of(i) for i in np.flatnonzero(mask)])
        rows = dream_mod.dream_rows(spec, dreams, run_id="fx-dream", seed=SEED, world_preset=PRESETS[0], timestamp_utc=TS)
        check("dream rows: teacher.request_hashes lists every request; request_hash is the last one (null for none)",
              [row.teacher["request_hashes"] for row in rows] == [list(dreams.request_hashes_of(i)) for i in range(9)]
              and [row.teacher["request_hash"] for row in rows] == ["A1"] * 4 + ["B0"] * 2 + ["C0"] * 2 + [None])
        path = tmp / "g3" / "dream_rows.jsonl"
        rows_mod.append_rows(path, rows, spec=spec)
        check("dream rows round-trip through JSON Lines", rows_mod.read_rows(path, spec=spec) == rows)
        single = rows_mod.build_dream_row(spec, run_id="fx-dream", seed=SEED, world_preset=PRESETS[0], decision=decisions[0],
                                          timestamp_utc=TS)
        check("build_dream_row records every request of the decision", single.teacher["request_hashes"] == ["A0", "A1"]
              and single.teacher["request_hash"] == "A1")
        teacher = {**single.teacher, "request_hash": "A0"}
        raises("a dream row whose request_hash is not the last of request_hashes is refused", ValueError,
               lambda: dataclasses.replace(single, teacher=teacher), "must be the last")
        bad = dataclasses.replace(decisions[0], request_hash="Z9")
        raises("a decision naming a request_hash outside its request_hashes is refused", ValueError,
               lambda: dream_mod.dreams_from_decisions(spec, [bad], spread_levels=SPREAD), "not one of its request_hashes")
        header["format"] = "learned-judgment-dreamset-v1"
        store.with_suffix(".json").write_text(json.dumps(header), encoding="utf-8")
        raises("an old v1 dream file is refused with its reason", ValueError, lambda: dream_mod.load_dreams(store),
               "last provider request")

    # ------------------------------------------------------------------------------ G5
    def g5() -> None:
        context = spec.sample_situations(stream_rng(SEED, "fixture/g5/situations"), 40)
        for fraction in (0.5, 1.0):
            def actions(n: int, fraction: float = fraction) -> np.ndarray:
                return collect_mod.exploration_actions(
                    spec, context[:n], rng=stream_rng(SEED, "fixture/g5/actions"), points_per_dim=POINTS,
                    exploration_fraction=fraction, quantile_levels=LEVELS, spread_levels=SPREAD, layout=layout)

            a20, a40 = actions(20), actions(40)
            same = int(np.sum(np.all(a20 == a40[:20], axis=1)))
            check(f"fraction {fraction}: exploration_actions(n=20) == first 20 rows of n=40 ({same}/20 rows equal)", same == 20)
        rng = stream_rng(SEED, "fixture/g5/actions")
        got = collect_mod.exploration_actions(spec, context, rng=rng, points_per_dim=POINTS, exploration_fraction=0.5,
                                              quantile_levels=LEVELS, spread_levels=SPREAD, layout=layout)
        reference = stream_rng(SEED, "fixture/g5/actions")
        uniform = reference.random((40, 2))
        decisions = decide_batch(spec, context, model=None, layout=layout, points_per_dim=POINTS, quantile_levels=LEVELS,
                                 spread_levels=SPREAD, method=PHYSICS_ONLY)
        expected = np.stack([
            d.candidates[int(np.floor(u[1] * d.candidates.shape[0]))] if u[0] < 0.5 else d.physics_action
            for d, u in zip(decisions, uniform)
        ])
        check("one draw rng.random((n, 2)): column 0 = explore flag, column 1 = candidate pick", np.array_equal(got, expected))
        check("the stream position after the call is exactly n x 2 uniforms", rng.random() == reference.random())

    # ------------------------------------------------------------------------------ G6
    def g6() -> None:
        path = tmp / "g6" / "attempts.jsonl"
        good = [real_row(spec, index) for index in range(3)]
        rows_mod.append_rows(path, good[:2], spec=spec)
        before = path.read_bytes()
        mutated = real_row(spec, 5)
        mutated.residual["fx_error"] = 123.0  # the dataclass is frozen; its dicts are not
        raises("append_rows refuses a row whose residual was changed after it was built", ValueError,
               lambda: rows_mod.append_rows(path, [good[2], mutated], spec=spec), "residual != measured_outcome - physics_guess")
        raises("... also when no spec is given (structural re-validation)", ValueError,
               lambda: rows_mod.append_rows(path, [mutated], spec=None), "residual != measured_outcome - physics_guess")
        leaky = real_row(spec, 6)
        leaky.situation["payload_mass_kg"] = 2.5  # a hidden value added after construction
        raises("append_rows(spec=...) refuses a row whose situation gained a key after it was built", ValueError,
               lambda: rows_mod.append_rows(path, [leaky], spec=spec), "extra=['payload_mass_kg']")
        nan_row = real_row(spec, 7)
        nan_row.measured_outcome["fx_error"] = float("nan")
        raises("append_rows refuses a row changed to hold NaN", ValueError,
               lambda: rows_mod.append_rows(path, [nan_row], spec=spec), "validation immediately before writing")
        check("nothing was written by any refused call (file bytes unchanged)", path.read_bytes() == before)
        raises("append_rows requires the spec argument (None for a mixed-skill file)", TypeError,
               lambda: rows_mod.append_rows(path, [good[2]]), "spec")
        rows_mod.append_rows(path, [good[2]], spec=spec)
        check("a valid row is still appended and reads back", rows_mod.read_rows(path, spec=spec) == good)

    # ------------------------------------------------------------------------------ G7
    def g7() -> None:
        path = tmp / "g7" / "attempts.jsonl"
        first = [real_row(spec, index) for index in range(3)]
        rows_mod.append_rows(path, first, spec=spec)
        before = path.read_bytes()
        twin = real_row(spec, 4, seed=first[0].seed)  # same (run_id, seed) as first[0], another attempt
        raises("append_rows refuses a batch that repeats a (run_id, seed) pair", ValueError,
               lambda: rows_mod.append_rows(tmp / "g7" / "new.jsonl", [first[1], real_row(spec, 8, seed=first[1].seed)], spec=spec),
               "repeats the attempt")
        check("... and writes nothing (the file was not created)", not (tmp / "g7" / "new.jsonl").exists())
        raises("append_rows refuses a row whose (run_id, seed) is already in the file (rerun after a crash)", ValueError,
               lambda: rows_mod.append_rows(path, [twin], spec=spec), "already holds")
        raises("append_rows refuses re-appending the same rows", ValueError,
               lambda: rows_mod.append_rows(path, first[:2], spec=spec), "already holds")
        check("nothing was appended by the refused calls", path.read_bytes() == before)
        other_run = real_row(spec, 4, run_id="fx-run-2", seed=first[0].seed)
        rows_mod.append_rows(path, [other_run], spec=spec)
        check("the same seed under another run_id is another attempt and is accepted", len(rows_mod.read_rows(path)) == 4)
        decision = decide_batch(spec, np.array([[0.5, 0.0]]), model=None, layout=layout, points_per_dim=POINTS,
                                quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)[0]
        dream_row = rows_mod.build_dream_row(spec, run_id="fx-dream", seed=SEED, world_preset=PRESETS[0], decision=decision,
                                             timestamp_utc=TS)
        rows_mod.append_rows(tmp / "g7" / "dreams.jsonl", [dream_row, dream_row, dream_row], spec=spec)
        check("dream rows carry no attempt identity: one dream run writes many rows with one (run_id, seed)",
              len(rows_mod.read_rows(tmp / "g7" / "dreams.jsonl")) == 3)
        raises("rows_to_arrays refuses a repeated (run_id, seed)", ValueError,
               lambda: rows_mod.rows_to_arrays(spec, [*first, twin], on_physics_mismatch="error"), "repeats the attempt")
        if ConstantBiasResidual is not None:
            raises("fit_residual_model refuses duplicated attempts (no double weighting)", ValueError,
                   lambda: learn.fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=SEED),
                                                    [*first, first[0]], layout, on_physics_mismatch="error"),
                   "repeats the attempt")
        raises("check_new_attempts refuses a seed already recorded under the run_id", ValueError,
               lambda: rows_mod.check_new_attempts(path, run_id="fx-run", world_preset=PRESETS[0], seeds=[999, first[2].seed],
                                                   timestamp_utc=TS), "already holds")

    # ------------------------------------------------------------------------------ G11
    def g11() -> None:
        made = adapt.world_id_layout(spec, PRESETS, hidden_value_names=HIDDEN, include_physics=True)
        reference = build_layout(spec, extra=(adapt.world_id_variable(PRESETS),), hidden_value_names=HIDDEN)
        check("world_id_layout == build_layout(spec, extra=(world_id_variable(presets),), hidden_value_names=...)",
              made == reference)
        check("the extra block is one categorical world_id column whose levels are the declared presets",
              made.names[-1] == "world_id" and made.n_extra == 1 and made.categorical[-1] == (made.n_features - 1, PRESETS))
        direct = adapt.world_id_layout(spec, PRESETS, hidden_value_names=HIDDEN, include_physics=False)
        check("include_physics=False gives the direct layout with the world_id column", direct.n_physics == 0
              and direct.names[-1] == "world_id")
        raises("hidden_value_names is required", TypeError,
               lambda: adapt.world_id_layout(spec, PRESETS, include_physics=True), "hidden_value_names")
        raises("include_physics is required", TypeError,
               lambda: adapt.world_id_layout(spec, PRESETS, hidden_value_names=HIDDEN), "include_physics")
        raises("a single string is refused (it would be read as characters)", TypeError,
               lambda: adapt.world_id_layout(spec, PRESETS, hidden_value_names="payload_mass_kg", include_physics=True),
               "not one string")
        raises("an empty list of hidden names is refused", ValueError,
               lambda: adapt.world_id_layout(spec, PRESETS, hidden_value_names=(), include_physics=True), "hidden_value_names")
        raises("an extra feature named like a hidden simulator value is refused", ValueError,
               lambda: adapt.world_id_layout(spec, PRESETS, hidden_value_names=(*HIDDEN, "world_id"), include_physics=True),
               "hidden simulator values")
        print(f"  hidden value names from config.yaml world.hidden_effects: {list(HIDDEN)}")

    # ------------------------------------------------------------------------------ G13
    def g13() -> None:
        dip = adapt.recovery_attempts({0: 3.0, 5: 1.15, 10: 2.4, 15: 2.6, 20: 2.5}, 1.0, 0.2)
        check(f"a one-off dip: first 5, sustained None, threshold 1.2 -> {dip}",
              dip == {"first": 5, "sustained": None, "threshold": 1.2, "pre_shift_error_is_zero": False})
        later = adapt.recovery_attempts({0: 3.0, 5: 1.1, 10: 1.5, 15: 1.0, 20: 1.2}, 1.0, 0.2)
        check(f"dip, relapse, then recovery (boundary 1.2 <= 1.2 counts): first 5, sustained 15 -> {later}",
              later["first"] == 5 and later["sustained"] == 15)
        unordered = adapt.recovery_attempts({20: 1.0, 10: 1.1, 0: 5.0}, 1.0, 0.2)
        check(f"keys in any order: first 10, sustained 10 -> {unordered}", unordered["first"] == 10 and unordered["sustained"] == 10)
        zero = adapt.recovery_attempts({0: 0.3, 5: 0.02, 10: 0.01, 20: 0.01}, 0.0, 0.2)
        check(f"pre-shift error 0: flagged, so 'not reached' is not read as 'not recovered' -> {zero}",
              zero == {"first": None, "sustained": None, "threshold": 0.0, "pre_shift_error_is_zero": True})
        exact = adapt.recovery_attempts({0: 0.3, 5: 0.0, 10: 0.0}, 0.0, 0.2)
        check(f"pre-shift error 0 and post-shift 0: first 5, sustained 5, flagged -> {exact}",
              exact["first"] == 5 and exact["sustained"] == 5 and exact["pre_shift_error_is_zero"] is True)

    # ------------------------------------------------------------------------------ G-index
    def g_index() -> None:
        for split in ("dev", "shift", "context"):
            small = split_situations(spec, base_seed=SEED, split=split, n=10, world_preset=PRESETS[0])
            large = split_situations(spec, base_seed=SEED, split=split, n=100, world_preset=PRESETS[0])
            equal = int(np.sum(np.all(small == large[:10], axis=1)))
            check(f"split '{split}': situation i is the same for n=10 and n=100 ({equal}/10 rows bitwise equal)", equal == 10)
        seeds_small = noise_seeds(spec, base_seed=SEED, split="dev", world_preset=PRESETS[0], n_situations=10, n_repeats=2)
        seeds_large = noise_seeds(spec, base_seed=SEED, split="dev", world_preset=PRESETS[0], n_situations=100, n_repeats=2)
        check("... and so is its per-episode seed", np.array_equal(seeds_small, seeds_large[:10]))
        d100 = dream_mod.dream(spec, None, layout, n_dreams=100, rng=stream_rng(SEED, "fixture/g_index/dreams"),
                               points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)
        d1000 = dream_mod.dream(spec, None, layout, n_dreams=1000, rng=stream_rng(SEED, "fixture/g_index/dreams"),
                                points_per_dim=POINTS, quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)
        check("the 100-dream situations are the first 100 of the 1000-dream set", np.array_equal(d100.situations, d1000.situations[:100]))

    for title, body in (
        ("G1 collect() checks the verified command schema before the world is stepped", g1),
        ("G-seed collect(): one world per run, one attempt seed per attempt, all inputs validated first", g_seed),
        ("G2 the stored physics guess is re-checked against the current spec", g2),
        ("G3 every provider request hash of every decision is recorded", g3),
        ("G5 exploration_actions: row i does not depend on n", g5),
        ("G6 append_rows re-validates every row immediately before writing", g6),
        ("G7 (run_id, seed) names one real attempt", g7),
        ("G11 world-id layout through build_layout with required hidden value names", g11),
        ("G13 recovery_attempts: first, sustained, threshold, zero pre-shift error", g13),
        ("G-index a situation index maps to the same situation at every n", g_index),
    ):
        section(title, body)


with tempfile.TemporaryDirectory(prefix="b_engine_review_") as scratch:
    SCRATCH["dir"] = str(scratch)
    run(Path(scratch))

failed = [label for label, ok in RESULTS if not ok]
print(f"\nchecks: {len(RESULTS)} run, {len(RESULTS) - len(failed)} held, {len(failed)} failed")
print("NOT RUN: collect() and the harness with Person A's Go2World (it does not exist yet; never called with a stand-in "
      "world); any TabPFN request (none has been made)")
sys.exit(1 if failed else 0)
