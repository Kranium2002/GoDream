"""Check 1: end-to-end mechanics of measuring a dream-trained per-skill spiking student.

FIXTURE CHECK - mechanics only. A FIXTURE SkillSpec (fixture_specs.FixtureMeasureSpec: not a project
skill, not the Go2, not DODGE) goes through the REAL pipeline code:

    engine.dream.dream (training DreamSet, and a held-out DreamSet from a different named stream)
    -> engine.train_in_dream.train_in_dream -> students.measure.student_record -> write_record

Variant A dreams with physics only (model=None, method=PHYSICS_ONLY) and passes the SNNStudent to
student_record (converted by its deployed()). Variant B fits the real production residual model
eval.baselines.ConstantBiasResidual on FIXTURE attempt rows (made up; not simulator episodes),
dreams with it, and passes student.deployed(). The config is config.yaml with
skills.fixture_measure.candidate_points_per_dim added and students.snn.max_steps reduced (both
printed below). Each record is written inside this proof directory with a fixture notice added;
nothing goes to results/.

Every number printed is a MECHANICS number on a fixture. The assertions check mechanics only:
complete records that survive a JSON round trip, consistent counts (synaptic operations =
spikes x fan-out), both energy estimates with intervals and the H4 verdict consistent with their
means, a latency record through a student bound once to its spec (fresh process: never imports
torch), held-out dreams that share no situation with training, and non-negative regret.

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

from fixture_specs import FixtureMeasureSpec, fixture_attempt_outcomes, make_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.dream import DreamSet  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng, stream_seed  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.eval.baselines import ConstantBiasResidual  # noqa: E402
from learned_judgment.students.measure import H4_ENERGY_RULE, RECORD_FORMAT, HeldOutOverlapError, student_record, write_record  # noqa: E402

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


def interval_text(interval: Mapping[str, Any], digits: int = 1) -> str:
    return (
        f"{interval['estimate']:.{digits}f} [{interval['low']:.{digits}f}, {interval['high']:.{digits}f}] "
        f"({interval['confidence_level']:.0%} {interval['method_used']}, {interval['n_resamples']} resamples, "
        f"{interval['n_distinct_situations']} distinct situations in {interval['n_rows']} rows)"
    )


def print_record(record: Mapping[str, Any]) -> None:
    agreement = record["agreement_with_heldout_dream"]
    student, physics = agreement["student"], agreement["physics_only_choice"]
    difference = record["dream_regret_difference_student_minus_physics"]
    latency, operations = record["latency"], record["operations"]
    energy = operations["energy_estimate"]
    print("  MECHANICS numbers on a fixture (inside the held-out fixture dream; not the robot or the simulator):")
    print(
        f"    held-out dream: {agreement['dream']['n_rows']} rows, {agreement['dream']['n_distinct_situations']} distinct situations, "
        f"source={agreement['dream']['source']}, held_out={agreement['held_out']}; agreement with the dream's choice "
        f"{student['agreement_with_dream_choice']:.3f}, zero-regret share {student['dream_regret']['fraction_zero']:.3f}"
    )
    for label, item in (("student", student), ("physics-only choice", physics)):
        regret = item["dream_regret"]
        over = regret["over_spread"]
        print(
            f"    dream regret of the {label}: mean {regret['mean']:.4f} p90 {regret['p90']:.4f} objective units; "
            f"/ spread: mean {over['mean']:.4f} p90 {over['p90']:.4f}"
        )
    print(
        f"    student - physics mean regret {interval_text(difference, 4)}; rows lower/equal/higher "
        f"{agreement['paired_rows']['student_regret_lower']:.3f}/{agreement['paired_rows']['equal']:.3f}/"
        f"{agreement['paired_rows']['student_regret_higher']:.3f}"
    )
    print(
        f"    latency (in this process; torch loaded by training: {latency['torch_in_sys_modules_before']}): full path "
        f"median {latency['full_path']['median_s'] * 1e6:.1f} us p95 {latency['full_path']['p95_s'] * 1e6:.1f} us; index only "
        f"median {latency['index_only']['median_s'] * 1e6:.1f} us p95 {latency['index_only']['p95_s'] * 1e6:.1f} us; grid check "
        f"once at binding {latency['grid_check']['bind_seconds'] * 1e3:.2f} ms"
    )
    print(f"    spikes per decision {interval_text(operations['spikes_per_decision']['interval_total'])}")
    print(f"    synaptic ops per decision {interval_text(operations['synaptic_ops_per_decision']['interval'])}")
    print(
        f"    firing rate per layer {[round(v, 4) for v in operations['firing']['rate_per_layer']]}; never fire "
        f"{operations['firing']['never_fire_fraction_per_layer']}, fire every step {operations['firing']['fire_every_step_fraction_per_layer']}"
    )
    for name in ("main_estimate", "with_neuron_updates_estimate"):
        block = energy[name]
        print(
            f"    energy {name} (pJ per decision): {interval_text(block['estimate_pj_interval'])}; ratio to dense "
            f"{block['ratio_estimate_to_dense_reference_estimate']:.3f} [{block['ratio_interval']['low']:.3f}, "
            f"{block['ratio_interval']['high']:.3f}]; break-even rate {block['break_even']['rate_per_step']:.4f} vs weighted "
            f"measured rate {block['break_even']['measured_weighted_rate_per_step']:.4f}"
        )
    print(
        f"    dense reference estimate {energy['dense_reference_estimate']['estimate_pj_per_decision']:.1f} pJ; H4 energy "
        f"verdict from the estimates (mechanics only): {energy['h4_energy_criterion']['verdict_from_estimates']!r}"
    )


def check_record(name: str, record: dict, student: Any, heldout: DreamSet, situations: np.ndarray, source: str) -> None:
    print(f"\n  -- mechanics assertions, variant {name} --")
    deployed = student.deployed()
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
    recount = deployed.operation_counts(situations).spikes_per_decision
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
    checks.expect(
        operations["situations"]["n_rows"] == N_SITUATIONS and operations["situations"]["n_distinct_situations"] == N_SITUATIONS,
        f"operations situations: {N_SITUATIONS} rows, {N_SITUATIONS} distinct situations stated",
    )

    energy = operations["energy_estimate"]
    e_mac, e_ac = file_config_energy["e_mac_pj"], file_config_energy["e_ac_pj"]
    dense = energy["dense_reference_estimate"]["estimate_pj_per_decision"]
    main_block, updates_block = energy["main_estimate"], energy["with_neuron_updates_estimate"]
    checks.expect(
        energy["constants"]["e_mac_pj"] == e_mac and energy["constants"]["e_ac_pj"] == e_ac
        and energy["constants"]["source"] == file_config_energy["source"]
        and record["config"]["values_read"]["students.snn.energy.e_mac_pj"] == e_mac,
        f"the estimates used the config.yaml constants e_mac={e_mac} pJ, e_ac={e_ac} pJ and their source",
    )
    main_each = e_mac * operations["first_layer_macs"] + e_ac * synops
    updates_each = main_each + (e_mac + e_ac) * operations["neuron_updates"] + e_ac * spikes.sum(axis=1)
    checks.expect(
        np.allclose(main_block["per_decision_estimate_pj"], main_each, rtol=1e-12)
        and np.allclose(updates_block["per_decision_estimate_pj"], updates_each, rtol=1e-12)
        and math.isclose(main_block["mean_estimate_pj_per_decision"], float(main_each.mean()), rel_tol=1e-12)
        and math.isclose(updates_block["mean_estimate_pj_per_decision"], float(updates_each.mean()), rel_tol=1e-12)
        and math.isclose(dense, e_mac * operations["dense_reference_macs"], rel_tol=1e-12),
        "both estimates per decision and their means, and the dense reference, follow energy.py's formulas from the counts",
    )
    for label, block in (("main", main_block), ("with neuron updates", updates_block)):
        interval, values = block["estimate_pj_interval"], np.asarray(block["per_decision_estimate_pj"])
        slack = 1e-12 * float(np.max(np.abs(values)))
        checks.expect(
            values.min() - slack <= interval["low"] <= interval["high"] <= values.max() + slack
            and interval["n_resamples"] == file_config["evaluation"]["bootstrap_resamples"]
            and interval["confidence_level"] == file_config["evaluation"]["bootstrap_confidence_level"]
            and interval["method_requested"] == file_config["evaluation"]["bootstrap_method"]
            and interval["n_distinct_situations"] == N_SITUATIONS,
            f"{label} estimate: interval inside [min, max] of the per-decision values, config.yaml bootstrap settings, distinct situations resampled",
        )
        checks.expect(
            math.isclose(block["ratio_estimate_to_dense_reference_estimate"], block["mean_estimate_pj_per_decision"] / dense, rel_tol=1e-12)
            and block["break_even"]["weighted_rate_below_break_even"] == (block["mean_estimate_pj_per_decision"] < dense),
            f"{label} estimate: ratio = mean / dense; the weighted rate is below its break-even rate exactly when the mean is below the reference",
        )
    verdict = energy["h4_energy_criterion"]
    main_below, updates_below = main_block["mean_estimate_pj_per_decision"] < dense, updates_block["mean_estimate_pj_per_decision"] < dense
    expected = "supported" if main_below and updates_below else ("not supported" if not main_below and not updates_below else "inconclusive")
    checks.expect(
        verdict["verdict_from_estimates"] == expected and verdict["rule"] == H4_ENERGY_RULE,
        f"H4 energy verdict {verdict['verdict_from_estimates']!r} follows the rule from both means (stored with the rule text)",
    )
    checks.expect(
        "estimate" in operations["label"] and operations["caveats"] == energy["caveats"] and len(operations["caveats"]) >= 1,
        "operations labelled with 'estimate' and carrying the energy caveats",
    )
    for label, interval, values in (
        ("spikes", operations["spikes_per_decision"]["interval_total"], spikes.sum(axis=1)),
        ("synaptic ops", operations["synaptic_ops_per_decision"]["interval"], synops),
    ):
        checks.expect(
            values.min() <= interval["low"] <= interval["high"] <= values.max()
            and interval["n_resamples"] == file_config["evaluation"]["bootstrap_resamples"],
            f"{label} interval inside [min, max] of the per-decision values, with the config.yaml bootstrap settings",
        )

    agreement = record["agreement_with_heldout_dream"]
    student_regret = np.asarray(agreement["per_row"]["student_regret"])
    physics_regret = np.asarray(agreement["per_row"]["physics_regret"])
    checks.expect(bool(np.all(student_regret >= 0.0) and np.all(physics_regret >= 0.0)), "every dream regret is non-negative")
    decisions = deployed.decide_index(heldout.situations)
    checks.expect(
        agreement["per_row"]["student_choice"] == decisions.tolist()
        and agreement["student"]["agreement_with_dream_choice"] == float(np.mean(decisions == heldout.chosen)),
        "recorded decisions and agreement reproduce from the deployed student's decide_index on the held-out dream",
    )
    expected_regret = heldout.scores.max(axis=1) - heldout.scores[np.arange(heldout.n_rows), decisions]
    checks.expect(bool(np.array_equal(student_regret, expected_regret)), "student regret = row max score - score of the decision")
    overlap = record["dream_provenance"]["heldout_overlap_with_training"]
    checks.expect(
        agreement["held_out"] is True and overlap["rows_seen_in_training"] == 0
        and agreement["dream"]["n_distinct_situations"] == heldout.n_rows == agreement["dream"]["n_rows"],
        "held-out dream: no row's situation was seen in training; rows and distinct situations stated",
    )
    difference = record["dream_regret_difference_student_minus_physics"]
    checks.expect(
        math.isclose(difference["estimate"], float(np.mean(student_regret - physics_regret)), rel_tol=1e-12, abs_tol=1e-15)
        and difference["n_distinct_situations"] == heldout.n_rows and "distinct situation" in difference["resampled_unit"],
        "regret difference = mean(student - physics) with an interval that resamples distinct held-out situations",
    )
    checks.expect(
        record["dream_provenance"]["training_dream"]["source"] == source
        and record["dream_provenance"]["heldout_dream"]["source"] == source
        and record["dream_provenance"]["trained_in_provider_backed_dream"] is False,
        f"training and held-out dreams marked {source!r}, not provider-backed",
    )
    latency = record["latency"]
    checks.expect(
        latency["full_path"]["n_decisions"] == latency["index_only"]["n_decisions"] == file_config["evaluation"]["latency_decisions"]
        and 0.0 < latency["full_path"]["median_s"] <= latency["full_path"]["p95_s"]
        and latency["single_row_vs_batch_decisions_differing"] == 0
        and latency["full_path_actions_not_the_looked_up_candidate"] == 0,
        "latency: both paths over config.yaml evaluation.latency_decisions calls; 0 < median <= p95; single-row = batched decisions",
    )
    checks.expect(
        latency["h3_timing"] == "full_path" and latency["full_path"]["use_for_h3"] is True and latency["index_only"]["use_for_h3"] is False
        and "once" in latency["grid_check"]["ran"] and "candidate_actions" in latency["full_path"]["includes"],
        "latency record: the grid check ran once at binding; the full path (with the candidate lookup) is the H3 timing",
    )
    checks.expect(
        record["student"]["network_sha256"] == deployed.digest == latency["network_sha256"] == operations["network_sha256"]
        == agreement["network_sha256"],
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


def dreams_of(model: Any, method: str, stream: str, n: int) -> DreamSet:
    return make_dreams(spec, model, method, stream_rng(BASE_SEED, stream), n, POINTS_PER_DIM, config)


scratch = Path(tempfile.mkdtemp(prefix="b_snn_measure_c1_"))
try:
    print("\n== variant physics_only: dreams from the physics guess alone; the SNNStudent is passed as it is ==")
    train_a = dreams_of(None, PHYSICS_ONLY, f"{SKILL}/dreams/situations", N_TRAIN_DREAMS)
    heldout_a = dreams_of(None, PHYSICS_ONLY, f"{SKILL}/heldout_dreams/situations", N_HELDOUT_DREAMS)
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
        HeldOutOverlapError,
        lambda: student_record(student_a, spec, train_a, situations, config, seed=BASE_SEED),
        f"{N_TRAIN_DREAMS} of {N_TRAIN_DREAMS} dream rows",
    )
    checks.raises(
        "student_record with a spec of another name",
        ValueError,
        lambda: student_record(student_a, FixtureMeasureSpec("fixture_other"), heldout_a, situations, config, seed=BASE_SEED),
        "was trained for skill",
    )

    print("\n== variant constant_bias: dreams scored with a fitted ConstantBiasResidual (eval.baselines); student.deployed() is passed ==")
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
    print(f"  ConstantBiasResidual fitted on {N_ATTEMPTS} FIXTURE attempt rows (made up, not episodes): mean residual {np.round(fitted_mean, 4).tolist()}")
    train_b = dreams_of(model, model.method, f"{SKILL}/model_dreams/situations", N_TRAIN_DREAMS)
    heldout_b = dreams_of(model, model.method, f"{SKILL}/model_heldout_dreams/situations", N_HELDOUT_DREAMS)
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
    record_b = student_record(student_b.deployed(), spec, heldout_b, situations, config, seed=BASE_SEED)
    print_record(record_b)
    check_record("constant_bias", record_b, student_b, heldout_b, situations, "local_residual_model")
    checks.expect(record_b["dream_provenance"]["training_dream"]["method"] == "constant_bias", "the record names the dream's method 'constant_bias'")

    print("\n== the latency path in a FRESH process (load the saved student, bind it to the spec once, numpy only) ==")
    stem = scratch / "fixture_student_b"
    student_b.save(stem)
    np.save(scratch / "situations.npy", situations)
    worker = PROOF_DIR / "c1_latency_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker), str(stem), str(scratch / "situations.npy"), SKILL],
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
            f"  MECHANICS latency on a fixture student, fresh torch-free process: full path median {fresh['full_path']['median_s'] * 1e6:.1f} us, "
            f"p95 {fresh['full_path']['p95_s'] * 1e6:.1f} us; index only median {fresh['index_only']['median_s'] * 1e6:.1f} us, "
            f"p95 {fresh['index_only']['p95_s'] * 1e6:.1f} us over {fresh['full_path']['n_decisions']} decisions each; processor "
            f"{fresh['processor']!r}, cpu_count {fresh['cpu_count']}, python {fresh['python_version']}, numpy {fresh['numpy_version']}"
        )
        checks.expect(
            fresh["torch_free_process"] and not fresh["torch_in_sys_modules_after"] and "torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout,
            "loading the student, binding it and timing both deployed paths never imported torch or snntorch",
        )
        checks.expect(fresh["network_sha256"] == student_b.digest, "the fresh process timed the same network (digest)")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
