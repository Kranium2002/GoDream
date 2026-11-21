"""Check 7 (lead decision M4): records count distinct situations, and intervals resample distinct situations.

FIXTURE CHECK - mechanics only, no training and no torch. The hand-built 2-3-2 network of c2 is
wrapped in a DeployedStudent built by hand (fixture_specs.hand_deployed) for the FIXTURE spec
fixture_specs.FixtureHandSpec; its meta names a physics-only dream although its weights were
written by hand (the record is mechanics only). A physics-only dream of that spec is made by the
real engine.dream.dream. Then:

  * the same held-out dream twice over (every situation duplicated): agreement_with_dream and
    student_record state twice the rows but the same distinct situations, and the regret-difference
    interval is IDENTICAL to the one on the dream alone; the interval the old code computed
    (eval.bootstrap.paired_difference_ci over rows) is narrower on the duplicated dream;
  * measure_operations on situations given twice: identical intervals, rows and distinct situations stated;
  * unevenly repeated situations: the estimate is the row mean and the record says how many rows
    belong to how many distinct situations.

Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c7_distinct_situations.py
"""

from _checks import BANNER, Checks

print(BANNER)

import math  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import ROOT3, FixtureHandSpec, concat_dreams, hand_deployed, make_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.eval.bootstrap import paired_difference_ci  # noqa: E402
from learned_judgment.students.energy import EnergyConstants  # noqa: E402
from learned_judgment.students.measure import agreement_with_dream, measure_operations, student_record  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

checks = Checks()
config = load_config()
constants = EnergyConstants.from_config(config)
bootstrap = {
    "n_resamples": require(config, "evaluation.bootstrap_resamples"),
    "confidence_level": require(config, "evaluation.bootstrap_confidence_level"),
    "method": require(config, "evaluation.bootstrap_method"),
}
SEED = 13  # fixture seed
N_DREAMS, N_SITUATIONS = 200, 60  # fixture sizes
TRAINING = np.random.default_rng(7).uniform(-ROOT3, ROOT3, (20, 2))  # FIXTURE stand-in for "situations training saw"
network = SpikingNetwork(
    hidden_weights=(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),),
    hidden_biases=(np.array([0.0, 0.0, 2.5]),),
    betas=(0.5,),
    thresholds=(1.0,),
    readout_weight=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    readout_bias=np.array([0.0, 0.0]),
    readout_beta=0.5,
    time_steps=3,
)
spec = FixtureHandSpec("fixture_hand", 2)
student = hand_deployed(network, spec, (2,), TRAINING)
heldout = make_dreams(spec, None, PHYSICS_ONLY, np.random.default_rng(SEED), N_DREAMS, (2,), config)
twice = concat_dreams(heldout, heldout)
situations = np.random.default_rng(SEED + 1).uniform(-ROOT3, ROOT3, (N_SITUATIONS, 2))
print(f"\nheld-out FIXTURE dream: {heldout.n_rows} rows x {heldout.n_candidates} candidates; the same dream twice: {twice.n_rows} rows")

print("\n== agreement_with_dream states rows and distinct situations ==")
once_agreement, twice_agreement = agreement_with_dream(student, heldout), agreement_with_dream(student, twice)
for label, record in (("once", once_agreement), ("twice", twice_agreement)):
    print(
        f"  {label}: {record['dream']['n_rows']} rows, {record['dream']['n_distinct_situations']} distinct situations; MECHANICS agreement "
        f"{record['student']['agreement_with_dream_choice']:.3f}, mean regret {record['student']['dream_regret']['mean']:.4f}"
    )
checks.expect(
    once_agreement["dream"]["n_rows"] == N_DREAMS and twice_agreement["dream"]["n_rows"] == 2 * N_DREAMS
    and once_agreement["dream"]["n_distinct_situations"] == twice_agreement["dream"]["n_distinct_situations"] == N_DREAMS,
    f"rows {N_DREAMS} and {2 * N_DREAMS}, distinct situations {N_DREAMS} in both",
)
checks.expect(
    twice_agreement["per_row"]["situation_group"] == list(range(N_DREAMS)) * 2,
    "per_row.situation_group maps row i and row i + n to the same distinct situation",
)

print("\n== student_record: the regret-difference interval resamples distinct situations ==")
record_once = student_record(student, spec, heldout, situations, config, seed=SEED)
record_twice = student_record(student, spec, twice, situations, config, seed=SEED)
once, both = record_once["dream_regret_difference_student_minus_physics"], record_twice["dream_regret_difference_student_minus_physics"]
student_regret = np.asarray(record_twice["agreement_with_heldout_dream"]["per_row"]["student_regret"])
physics_regret = np.asarray(record_twice["agreement_with_heldout_dream"]["per_row"]["physics_regret"])
old_style = paired_difference_ci(physics_regret[:, None], student_regret[:, None], seed=both["seed"], **bootstrap)  # the old code's row bootstrap
old_once = paired_difference_ci(
    np.asarray(record_once["agreement_with_heldout_dream"]["per_row"]["physics_regret"])[:, None],
    np.asarray(record_once["agreement_with_heldout_dream"]["per_row"]["student_regret"])[:, None],
    seed=once["seed"], **bootstrap,
)
print(
    f"  MECHANICS: once [{once['low']:.5f}, {once['high']:.5f}] ({once['n_distinct_situations']} situations, {once['n_rows']} rows); "
    f"twice [{both['low']:.5f}, {both['high']:.5f}] ({both['n_distinct_situations']} situations, {both['n_rows']} rows); "
    f"a row bootstrap (the old code) gives once [{old_once['low']:.5f}, {old_once['high']:.5f}], twice [{old_style['low']:.5f}, {old_style['high']:.5f}]"
)
checks.expect(
    once["n_rows"] == N_DREAMS and both["n_rows"] == 2 * N_DREAMS and once["n_distinct_situations"] == both["n_distinct_situations"] == N_DREAMS,
    "the interval records its rows and its distinct situations",
)
checks.expect(
    (once["estimate"], once["low"], once["high"]) == (both["estimate"], both["low"], both["high"]) and once["seed"] == both["seed"],
    "duplicating every situation leaves the interval exactly unchanged (the unit resampled is the distinct situation)",
)
checks.expect(
    (old_style["high"] - old_style["low"]) < (old_once["high"] - old_once["low"]),
    "for contrast: a row bootstrap narrows the interval when the rows are duplicated",
)
checks.expect(
    record_twice["dream_provenance"]["heldout_dream"]["n_rows"] == 2 * N_DREAMS
    and record_twice["dream_provenance"]["heldout_dream"]["n_distinct_situations"] == N_DREAMS,
    "the record's held-out dream provenance states rows and distinct situations",
)

print("\n== measure_operations on situations given twice ==")
ops_once = measure_operations(student, situations, constants, seed=SEED, **bootstrap)
ops_twice = measure_operations(student, np.concatenate([situations, situations]), constants, seed=SEED, **bootstrap)
for key in ("spikes_per_decision.interval_total", "synaptic_ops_per_decision.interval",
            "energy_estimate.main_estimate.estimate_pj_interval", "energy_estimate.with_neuron_updates_estimate.estimate_pj_interval"):
    first, second = ops_once, ops_twice
    for part in key.split("."):
        first, second = first[part], second[part]
    checks.expect(
        (first["low"], first["high"]) == (second["low"], second["high"]) and first["n_distinct_situations"] == second["n_distinct_situations"] == N_SITUATIONS
        and second["n_rows"] == 2 * N_SITUATIONS,
        f"{key}: identical interval on the situations given twice ({N_SITUATIONS} distinct situations, {2 * N_SITUATIONS} rows)",
    )
checks.expect(
    ops_twice["situations"]["n_rows"] == 2 * N_SITUATIONS and ops_twice["situations"]["n_distinct_situations"] == N_SITUATIONS,
    "the operations record states rows and distinct situations",
)

print("\n== unevenly repeated situations ==")
s1, s2, s3 = [1.5, 0.0], [0.8, 0.0], [-1.0, 1.2]  # hand spikes 6, 4, 5 (c2)
uneven = np.array([s1] * 5 + [s2] + [s3] * 2)
result = measure_operations(student, uneven, constants, seed=SEED, **bootstrap)
interval = result["spikes_per_decision"]["interval_total"]
row_mean = (5 * 6 + 4 + 2 * 5) / 8
print(f"  spikes per decision: estimate {interval['estimate']} (row mean {row_mean}), interval [{interval['low']}, {interval['high']}], {interval['n_distinct_situations']} situations in {interval['n_rows']} rows")
checks.expect(
    math.isclose(interval["estimate"], row_mean, rel_tol=1e-15) and interval["n_rows"] == 8 and interval["n_distinct_situations"] == 3
    and 4.0 <= interval["low"] <= interval["high"] <= 6.0 and result["per_decision"]["situation_group"] == [0, 0, 0, 0, 0, 1, 2, 2],
    "the estimate is the row mean; 8 rows in 3 distinct situations; each row's situation group is recorded",
)
checks.expect("torch" not in sys.modules, "no torch in this process")
checks.finish()
