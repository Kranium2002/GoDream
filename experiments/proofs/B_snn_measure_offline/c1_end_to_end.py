"""Check 1: end-to-end mechanics of measuring a dream-trained spiking student.

FIXTURE CHECK - mechanics only. A FIXTURE SkillSpec defined below (not a project skill, not the
Go2, not DODGE) goes through the REAL pipeline code:

    engine.dream.dream (training DreamSet, and a held-out DreamSet from a different named stream)
    -> engine.train_in_dream.train_in_dream -> students.measure.student_record -> write_record

Variant A dreams with physics only (model=None, method=PHYSICS_ONLY). Variant B fits the real
production residual model eval.baselines.ConstantBiasResidual on FIXTURE attempt rows (made up
below; not simulator episodes) and dreams with it, so the dreamed outcomes differ from the
physics guess. The config is config.yaml with skills.fixture_measure.candidate_points_per_dim
added and students.snn.max_steps reduced (both printed below). Each record is written inside
this proof directory with a fixture notice added; nothing goes to results/.

Every number printed is a MECHANICS number on a fixture. The assertions check mechanics only:
complete records that survive a JSON round trip, consistent counts (synaptic operations =
spikes x fan-out), a latency path that never imports torch (fresh process), non-negative
regret, and estimates built from the config.yaml constants.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c1_end_to_end.py
"""

from _checks import BANNER, PROOF_DIR, REPO_ROOT, Checks

print(BANNER)

import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Mapping  # noqa: E402

import numpy as np  # noqa: E402

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.dream import DreamSet, dream  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng, stream_seed  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.eval.baselines import ConstantBiasResidual  # noqa: E402
from learned_judgment.skills.base import ActionDim, Outcome, SkillSpec, Variable  # noqa: E402
from learned_judgment.students.measure import RECORD_FORMAT, student_record, write_record  # noqa: E402

SKILL = "fixture_measure"
POINTS_PER_DIM = (7, 5)  # fixture grid: 7 side-step speeds x 5 waits = 35 candidates
BASE_SEED = 11  # fixture run seed
N_TRAIN_DREAMS = 1000  # fixture sizes
N_HELDOUT_DREAMS = 500
N_SITUATIONS = 1000
N_ATTEMPTS = 40
MAX_STEPS = 400  # reduced from config students.snn.max_steps for a short mechanics check
RECORD_KEYS = {
    "format", "label", "skill", "student", "dream_provenance", "agreement_with_heldout_dream",
    "dream_regret_difference_student_minus_physics", "latency", "operations", "seeds", "config", "versions",
    "student_meta", "out_of_scope",
}


class FixtureMeasureSpec(SkillSpec):
    """FIXTURE: an object approaches with a lateral offset; side-step at some speed after some wait.

    Best candidate: the smallest side-step (away from the object, on a floor where falls are
    likelier when wet) whose lower-quantile gap reaches the 0.4 m cap; stand still when the
    offset is already large. There is no world: episode() refuses to run.
    """

    def __init__(self, name: str = SKILL) -> None:
        self.name = name
        self.description = "FIXTURE spec for measurement mechanics checks; not a project skill"
        self.situation = (
            Variable("fx_offset", "float", "m", "fixture: lateral offset of the approaching object", low=-1.0, high=1.0),
            Variable("fx_speed", "float", "m/s", "fixture: approach speed", low=2.0, high=6.0),
            Variable("fx_floor", "categorical", "label", "fixture: floor", levels=("dry", "wet")),
        )
        self.actions = (
            ActionDim("fx_vy", "float", "m/s", "fixture side-step speed", low=-0.6, high=0.6, role="command", command_field="fixture_vy"),
            ActionDim("fx_wait", "float", "s", "fixture wait before the side-step", low=0.0, high=0.4, role="timing"),
        )
        self.outcomes = (
            Outcome("fx_gap", "regression", "m", "fixture closest gap", low=0.0),
            Outcome("fx_fall", "binary", "", "fixture fall"),
            Outcome("fx_effort", "regression", "m", "fixture side-step distance", low=0.0),
        )
        self.primary_outcome = "fx_gap"
        self.hidden_effects = ("fixture: none; this spec never runs an episode",)

    @staticmethod
    def _terms(situations: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, ...]:
        situations, actions = np.asarray(situations, dtype=np.float64), np.asarray(actions, dtype=np.float64)
        offset, speed, wet = situations[:, 0], situations[:, 1], situations[:, 2]
        vy, wait = actions[:, 0], actions[:, 1]
        moving = np.clip(2.5 / speed - wait, 0.0, None)
        grip = np.where(wet == 1.0, 0.6, 1.0)
        return offset, wet, vy, moving, grip

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        offset, wet, vy, moving, grip = self._terms(situations, actions)
        gap = np.abs(offset - vy * moving * grip)
        fall = np.clip(0.03 + np.where(wet == 1.0, 1.0, 0.25) * vy**2, 0.0, 1.0)
        return np.column_stack([gap, fall, np.abs(vy) * moving])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        lowest = float(predicted.quantile_levels[0])  # risk-aware: the lowest predictive quantile of the gap
        return (
            np.minimum(predicted.quantile_of("fx_gap", lowest), 0.4)
            - predicted.mean_of("fx_fall")
            - 0.1 * predicted.median_of("fx_effort")
        )

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, float]:
        raise RuntimeError("the fixture spec has no world; this check never runs episodes")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return measured[:, 1] + np.clip(0.4 - measured[:, 0], 0.0, None)

    def success(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return (measured[:, 0] >= 0.4) & (measured[:, 1] == 0.0)


def fixture_attempt_outcomes(spec: FixtureMeasureSpec, situations: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FIXTURE attempt outcomes (made up, NOT simulator episodes): in this fixture 'world' a side-step
    moves 70% as far as the physics guess says, falls are 1.5x likelier, and the gap is noisy."""
    offset, wet, vy, moving, grip = spec._terms(situations, actions)
    n = situations.shape[0]
    gap = np.clip(np.abs(offset - 0.7 * vy * moving * grip) + rng.normal(0.0, 0.03, n), 0.0, None)
    fall_probability = np.clip(1.5 * (0.03 + np.where(wet == 1.0, 1.0, 0.25) * vy**2), 0.0, 1.0)
    fall = (rng.uniform(size=n) < fall_probability).astype(np.float64)
    return np.column_stack([gap, fall, 0.7 * np.abs(vy) * moving])


def make_dreams(spec: SkillSpec, model: Any, method: str, stream: str, n: int) -> DreamSet:
    return dream(
        spec,
        model,
        build_layout(spec),
        n_dreams=n,
        rng=stream_rng(BASE_SEED, stream),
        points_per_dim=POINTS_PER_DIM,
        quantile_levels=require(config, "engine.quantile_levels"),
        spread_levels=tuple(require(config, "engine.spread_levels")),
        method=method,
    )


def interval_text(interval: Mapping[str, Any], scale: float = 1.0, digits: int = 1) -> str:
    return (
        f"{interval['estimate'] * scale:.{digits}f} [{interval['low'] * scale:.{digits}f}, {interval['high'] * scale:.{digits}f}] "
        f"({interval['confidence_level']:.0%} {interval['method_used']}, {interval['n_resamples']} resamples, n={interval['n']})"
    )


def print_record(record: Mapping[str, Any]) -> None:
    agreement = record["agreement_with_heldout_dream"]
    student, physics = agreement["student"], agreement["physics_only_choice"]
    difference = record["dream_regret_difference_student_minus_physics"]
    latency, operations = record["latency"], record["operations"]
    energy = operations["energy_estimate"]
    print("  MECHANICS numbers on a fixture (inside the held-out fixture dream; not the robot or the simulator):")
    print(
        f"    held-out dream: {agreement['dream']['n_rows']} rows, source={agreement['dream']['source']}; "
        f"agreement with the dream's choice {student['agreement_with_dream_choice']:.3f}, zero-regret share "
        f"{student['dream_regret']['fraction_zero']:.3f}"
    )
    for label, item in (("student", student), ("physics-only choice", physics)):
        regret = item["dream_regret"]
        over = regret["over_spread"]
        print(
            f"    dream regret of the {label}: mean {regret['mean']:.4f} p90 {regret['p90']:.4f} objective units; "
            f"/ spread: mean {over['mean']:.4f} p90 {over['p90']:.4f}"
        )
    print(
        f"    paired student - physics mean regret {difference['estimate']:.4f} [{difference['low']:.4f}, {difference['high']:.4f}] "
        f"({difference['method_used']}); rows lower/equal/higher {agreement['paired_rows']['student_regret_lower']:.3f}/"
        f"{agreement['paired_rows']['equal']:.3f}/{agreement['paired_rows']['student_regret_higher']:.3f}"
    )
    print(
        f"    latency (in this process, torch loaded by training: {latency['torch_in_sys_modules_before']}): median "
        f"{latency['median_s'] * 1e6:.1f} us p95 {latency['p95_s'] * 1e6:.1f} us over {latency['n_decisions']} decisions"
    )
    print(f"    spikes per decision {interval_text(operations['spikes_per_decision']['interval_total'])}")
    print(f"    synaptic ops per decision {interval_text(operations['synaptic_ops_per_decision']['interval'])}")
    print(
        f"    firing rate per layer {[round(v, 4) for v in operations['firing']['rate_per_layer']]} vs break-even "
        f"{operations['break_even']['rate_per_step']:.4f} per step; never fire {operations['firing']['never_fire_fraction_per_layer']}, "
        f"fire every step {operations['firing']['fire_every_step_fraction_per_layer']}"
    )
    print(
        f"    energy ESTIMATE per decision (pJ): main {interval_text(operations['energy_main_pj_interval'])}; with neuron updates "
        f"{energy['snn_with_neuron_updates_pj_per_decision']:.1f}; dense reference {energy['dense_reference_pj_per_decision']:.1f}; "
        f"ratio main/dense {energy['ratio_snn_main_to_dense']:.3f}, with-updates/dense {energy['ratio_snn_with_neuron_updates_to_dense']:.3f}"
    )


def check_record(name: str, record: dict, student: Any, heldout: DreamSet, situations: np.ndarray, source: str) -> None:
    print(f"\n  -- mechanics assertions, variant {name} --")
    checks.expect(set(record) == RECORD_KEYS and record["format"] == RECORD_FORMAT, "record has exactly the documented top-level keys")
    path = PROOF_DIR / f"c1_record_{name}.json"
    stored = {**record, "fixture_notice": BANNER}
    write_record(path, stored)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    checks.expect(loaded == stored, f"JSON round trip: {path.name} reloads equal to the record")
    checks.expect(
        path.read_text(encoding="utf-8")
        == json.dumps(stored, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        "write_record wrote canonical JSON (sorted keys, compact, trailing newline)",
    )

    operations = record["operations"]
    spikes = np.asarray(operations["per_decision"]["spikes_per_layer"], dtype=np.float64)
    fan_out = np.asarray(operations["fan_out"], dtype=np.float64)
    synops = np.asarray(operations["per_decision"]["synaptic_ops"], dtype=np.float64)
    checks.expect(operations["fan_out"] == operations["layer_sizes"][2:], "fan-out of each hidden layer = width of the layer it feeds")
    checks.expect(bool(np.array_equal(synops, spikes @ fan_out)), "synaptic ops per decision = spikes per decision x fan-out")
    checks.expect(
        math.isclose(operations["synaptic_ops_per_decision"]["mean"], float(synops.mean()), rel_tol=1e-12)
        and math.isclose(operations["spikes_per_decision"]["mean_total"], float(spikes.sum(axis=1).mean()), rel_tol=1e-12),
        "reported means equal the means of the per-decision values",
    )
    recount = student.operation_counts(situations).spikes_per_decision
    checks.expect(bool(np.array_equal(recount, spikes)), "an independent operation_counts run gives the same spikes per decision")
    widths = np.asarray(operations["layer_sizes"][1:-1], dtype=np.float64)
    checks.expect(
        np.allclose(operations["firing"]["rate_per_layer"], spikes.mean(axis=0) / (widths * operations["time_steps"]), rtol=1e-12),
        "firing rate per layer = mean spikes / (H x T)",
    )
    fractions_ok = all(
        0.0 <= never <= 1.0 and 0.0 <= every <= 1.0 and never + every <= 1.0
        for never, every in zip(
            operations["firing"]["never_fire_fraction_per_layer"], operations["firing"]["fire_every_step_fraction_per_layer"]
        )
    )
    checks.expect(fractions_ok, "never-fire and fire-every-step fractions lie in [0, 1] and do not overlap")

    energy = operations["energy_estimate"]
    e_mac, e_ac = file_config_energy["e_mac_pj"], file_config_energy["e_ac_pj"]
    checks.expect(
        energy["constants"]["e_mac_pj"] == e_mac and energy["constants"]["e_ac_pj"] == e_ac
        and energy["constants"]["source"] == file_config_energy["source"]
        and record["config"]["values_read"]["students.snn.energy.e_mac_pj"] == e_mac,
        f"the estimate used the config.yaml constants e_mac={e_mac} pJ, e_ac={e_ac} pJ and their source",
    )
    checks.expect(
        math.isclose(energy["snn_main_pj_per_decision"], e_mac * operations["first_layer_macs"] + e_ac * float(synops.mean()), rel_tol=1e-12)
        and math.isclose(energy["dense_reference_pj_per_decision"], e_mac * operations["dense_reference_macs"], rel_tol=1e-12)
        and math.isclose(operations["break_even"]["rate_per_step"], e_mac / (e_ac * operations["time_steps"]), rel_tol=1e-15),
        "main estimate = e_mac x first-layer MACs + e_ac x mean synaptic ops; dense = e_mac x dense MACs; break-even = e_mac/(e_ac T)",
    )
    checks.expect(
        operations["label"] == "estimate" and operations["caveats"] == energy["caveats"] and len(operations["caveats"]) >= 1,
        "operations labelled 'estimate' and carrying the energy caveats",
    )
    for label, interval, values in (
        ("spikes", operations["spikes_per_decision"]["interval_total"], spikes.sum(axis=1)),
        ("synaptic ops", operations["synaptic_ops_per_decision"]["interval"], synops),
        ("main energy", operations["energy_main_pj_interval"], np.asarray(energy["snn_main_pj_each"])),
    ):
        checks.expect(
            values.min() <= interval["low"] <= interval["high"] <= values.max()
            and interval["n_resamples"] == file_config["evaluation"]["bootstrap_resamples"]
            and interval["confidence_level"] == file_config["evaluation"]["bootstrap_confidence_level"]
            and interval["method_requested"] == file_config["evaluation"]["bootstrap_method"],
            f"{label} interval inside [min, max] of the per-decision values, with the config.yaml bootstrap settings",
        )

    agreement = record["agreement_with_heldout_dream"]
    student_regret = np.asarray(agreement["per_row"]["student_regret"])
    physics_regret = np.asarray(agreement["per_row"]["physics_regret"])
    checks.expect(bool(np.all(student_regret >= 0.0) and np.all(physics_regret >= 0.0)), "every dream regret is non-negative")
    decisions = student.decide_index(heldout.situations)
    checks.expect(
        agreement["per_row"]["student_choice"] == decisions.tolist()
        and agreement["student"]["agreement_with_dream_choice"] == float(np.mean(decisions == heldout.chosen)),
        "recorded decisions and agreement reproduce from student.decide_index on the held-out dream",
    )
    expected_regret = heldout.scores.max(axis=1) - heldout.scores[np.arange(heldout.n_rows), decisions]
    checks.expect(bool(np.array_equal(student_regret, expected_regret)), "student regret = row max score - score of the decision")
    checks.expect(
        record["dream_provenance"]["training_dream"]["source"] == source
        and record["dream_provenance"]["heldout_dream"]["source"] == source
        and record["dream_provenance"]["trained_in_provider_backed_dream"] is False,
        f"training and held-out dreams marked {source!r}, not provider-backed",
    )
    checks.expect(
        record["dream_provenance"]["heldout_dream"]["rows_sha256"] != record["dream_provenance"]["training_dream"]["rows_sha256"],
        "held-out dream rows differ from the training rows (digests)",
    )
    latency = record["latency"]
    checks.expect(
        latency["n_decisions"] == file_config["evaluation"]["latency_decisions"] and 0.0 < latency["median_s"] <= latency["p95_s"]
        and latency["single_row_vs_batch_decisions_differing"] == 0,
        "latency over config.yaml evaluation.latency_decisions calls; 0 < median <= p95; single-row = batched decisions",
    )
    checks.expect(
        record["student"]["network_sha256"] == student.digest == latency["network_sha256"] == operations["network_sha256"],
        "network digest is the same in every part of the record",
    )


checks = Checks()
file_config = load_config()
file_config_energy = file_config["students"]["snn"]["energy"]
config = copy.deepcopy(file_config)
config["skills"][SKILL] = {"candidate_points_per_dim": list(POINTS_PER_DIM)}
config["students"]["snn"]["max_steps"] = MAX_STEPS
print(
    f"\nconfig: config.yaml plus skills.{SKILL}.candidate_points_per_dim={list(POINTS_PER_DIM)}; "
    f"students.snn.max_steps reduced to {MAX_STEPS} (config.yaml value {file_config['students']['snn']['max_steps']})"
)
spec = FixtureMeasureSpec()
situations = spec.sample_situations(stream_rng(BASE_SEED, f"{SKILL}/measure/situations"), N_SITUATIONS)
scratch = Path(tempfile.mkdtemp(prefix="b_snn_measure_c1_"))
try:
    print("\n== variant physics_only: dreams from the physics guess alone ==")
    train_a = make_dreams(spec, None, PHYSICS_ONLY, f"{SKILL}/dreams/situations", N_TRAIN_DREAMS)
    heldout_a = make_dreams(spec, None, PHYSICS_ONLY, f"{SKILL}/heldout_dreams/situations", N_HELDOUT_DREAMS)
    print(f"  dreams: train {train_a.n_rows} x {train_a.n_candidates} candidates, held-out {heldout_a.n_rows}")
    started = time.perf_counter()
    student_a = train_in_dream(spec, train_a, config, base_seed=BASE_SEED)
    training = student_a.meta["training"]
    print(
        f"  train_in_dream: {time.perf_counter() - started:.1f} s incl. torch import; steps_run={training['steps_run']} "
        f"best_step={training['best_step']} validation loss {training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}"
    )
    started = time.perf_counter()
    record_a = student_record(student_a, spec, heldout_a, situations, config, seed=BASE_SEED)
    print(f"  student_record: {time.perf_counter() - started:.1f} s")
    print_record(record_a)
    check_record("physics_only", record_a, student_a, heldout_a, situations, "physics_only")
    checks.expect(
        bool(np.all(np.asarray(record_a["agreement_with_heldout_dream"]["per_row"]["physics_regret"]) == 0.0)),
        "physics-only dream: the physics choice is the dream's choice, so its dream regret is 0 on every row",
    )
    checks.raises(
        "student_record with the training dream as 'held-out'",
        ValueError,
        lambda: student_record(student_a, spec, train_a, situations, config, seed=BASE_SEED),
        "exactly the rows the student was trained on",
    )
    checks.raises(
        "student_record with a spec of another name",
        ValueError,
        lambda: student_record(student_a, FixtureMeasureSpec("fixture_other"), heldout_a, situations, config, seed=BASE_SEED),
        "was trained for skill",
    )

    print("\n== variant constant_bias: dreams scored with a fitted ConstantBiasResidual (eval.baselines) ==")
    layout = build_layout(spec)
    attempt_situations = spec.sample_situations(stream_rng(BASE_SEED, f"{SKILL}/attempts/situations"), N_ATTEMPTS)
    picks = stream_rng(BASE_SEED, f"{SKILL}/attempts/actions").integers(0, train_a.n_candidates, size=N_ATTEMPTS)
    attempt_actions = np.stack([spec.candidate_actions(s, POINTS_PER_DIM)[k] for s, k in zip(attempt_situations, picks)])
    physics = spec.physics_guess(attempt_situations, attempt_actions)
    measured = fixture_attempt_outcomes(spec, attempt_situations, attempt_actions, stream_rng(BASE_SEED, f"{SKILL}/attempts/noise"))
    model = ConstantBiasResidual.from_config(config, layout, seed=stream_seed(BASE_SEED, f"{SKILL}/constant_bias"))
    features = build_features(layout, attempt_situations, attempt_actions, physics=physics)
    model.fit(features, measured - physics)
    fitted_mean = model.predict_residual_distribution(features[:1]).mean[0]
    print(f"  ConstantBiasResidual fitted on {N_ATTEMPTS} FIXTURE attempt rows (made up above, not episodes): mean residual {np.round(fitted_mean, 4).tolist()}")
    train_b = make_dreams(spec, model, model.method, f"{SKILL}/model_dreams/situations", N_TRAIN_DREAMS)
    heldout_b = make_dreams(spec, model, model.method, f"{SKILL}/model_heldout_dreams/situations", N_HELDOUT_DREAMS)
    moved = float(np.mean(train_b.chosen != train_b.physics_choice))
    print(f"  dreams: train {train_b.n_rows}, held-out {heldout_b.n_rows}; share of training rows whose dream choice differs from the physics choice {moved:.3f}")
    checks.expect(moved > 0.0, "the model-backed dream changes some choices away from the physics choice")
    checks.expect(bool(np.any(train_b.chosen_mean != train_b.chosen_physics)), "the model-backed dream predicts outcomes that differ from the physics guess")
    started = time.perf_counter()
    student_b = train_in_dream(spec, train_b, config, base_seed=BASE_SEED)
    training = student_b.meta["training"]
    print(
        f"  train_in_dream: {time.perf_counter() - started:.1f} s; steps_run={training['steps_run']} best_step={training['best_step']} "
        f"validation loss {training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}"
    )
    record_b = student_record(student_b, spec, heldout_b, situations, config, seed=BASE_SEED)
    print_record(record_b)
    check_record("constant_bias", record_b, student_b, heldout_b, situations, "local_residual_model")
    checks.expect(record_b["dream_provenance"]["training_dream"]["method"] == "constant_bias", "the record names the dream's method 'constant_bias'")

    print("\n== the latency path in a FRESH process (load the saved student, numpy only) ==")
    stem = scratch / "fixture_student_b"
    student_b.save(stem)
    np.save(scratch / "situations.npy", situations)
    worker = PROOF_DIR / "c1_latency_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker), str(stem), str(scratch / "situations.npy")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    fresh = None
    for line in (result.stdout + result.stderr).strip().splitlines():
        if line.startswith("LATENCY_JSON "):
            fresh = json.loads(line[len("LATENCY_JSON "):])
        else:
            print(f"    {line.replace(str(scratch), '<tmp>')}")
    checks.expect(result.returncode == 0 and fresh is not None, f"fresh-process worker exit code {result.returncode}")
    if fresh is not None:
        print(
            f"  MECHANICS latency on a fixture student, fresh torch-free process: median {fresh['median_s'] * 1e6:.1f} us, "
            f"p95 {fresh['p95_s'] * 1e6:.1f} us over {fresh['n_decisions']} decisions; processor {fresh['processor']!r}, "
            f"cpu_count {fresh['cpu_count']}, python {fresh['python_version']}, numpy {fresh['numpy_version']}, blas {fresh['blas']}"
        )
        checks.expect(
            fresh["torch_free_process"] and not fresh["torch_in_sys_modules_after"] and "torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout,
            "loading the student and timing the deployed path never imported torch or snntorch",
        )
        checks.expect(fresh["network_sha256"] == student_b.digest, "the fresh process timed the same network (digest)")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
