"""Check 2: measure_operations against hand computations on tiny hand-built deployed students.

FIXTURE CHECK - mechanics only. Every network below is written by hand and wrapped in a
DeployedStudent built by hand (fixture_specs.hand_deployed: no SNNStudent, no training, no torch).
Its spikes are worked out below by hand and compared with what learned_judgment.students.measure
reports. The energy constants and bootstrap settings are the real config.yaml values; the
expected energies are computed here from the hand spike counts with the formulas of
students/energy.py written out independently. Writes no file.

Lead decisions covered: M3 (both energy estimates with intervals, both ratios, the break-even
rate of each estimate, and the H4 verdict rule in all three outcomes), M6 (per-neuron statistics
from SpikingNetwork.forward_with_spike_trains; measure.py keeps no copy of the network loop),
M7 (the hand network is a hand-built DeployedStudent), M4 for operations (duplicated situations).

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c2_hand_network.py

Hand network "2-3-2" (encoded input x = (x_a, x_b); the encoding is the identity):
    hidden current  I = W1 x + b1,  W1 = [[1, 0], [0, 1], [0, 0]],  b1 = [0, 0, 2.5]
                    neuron A gets x_a, neuron B gets x_b, neuron C gets 2.5 for every input
    beta = 0.5, threshold = 1.0, subtractive reset taken from the previous membrane:
        U_t = (0.5 U_{t-1} + I) - [U_{t-1} > 1] * 1,   spike S_t = [U_t > 1]
    readout W_o = [[1, 0, 0], [0, 1, 0]], b_o = 0, readout beta 0.5 (fan-out of the hidden layer = 2), T = 3

    s1  = (1.5, 0.0)  A: U = 1.5 S | 0.75+1.5-1 = 1.25 S | 0.625+1.5-1 = 1.125 S  -> spikes at t1 t2 t3
                      B: U = 0, 0, 0                                             -> none
                      C: U = 2.5 S | 1.25+2.5-1 = 2.75 S | 1.375+2.5-1 = 2.875 S -> t1 t2 t3
                      per step (A+B+C): [2, 2, 2], total 6
    s2  = (0.8, 0.0)  A: U = 0.8 - | 0.4+0.8 = 1.2 S | 0.6+0.8-1 = 0.4 -          -> t2
                      B: none;  C: t1 t2 t3            per step [1, 2, 1], total 4
    s3  = (-1.0, 1.2) A: U = -1, -1.5, -1.75 (never above 1)                     -> none
                      B: U = 1.2 S | 0.6+1.2-1 = 0.8 - | 0.4+1.2 = 1.6 S          -> t1 t3
                      C: t1 t2 t3                      per step [2, 1, 2], total 5
    s1' = (1.6, 0.0)  A: U = 1.6 S | 0.8+1.6-1 = 1.4 S | 0.7+1.6-1 = 1.3 S       -> t1 t2 t3
                      B: none; C: t1 t2 t3             per step [2, 2, 2], total 6 (same as s1)

    Energy per decision with s spikes (fan-out 2, D=2, H=3, C=2, T=3):
        main            e_mac * 6 + e_ac * 2 s
        with updates    main + (e_mac + e_ac) * (3 + 2) * 3 + e_ac * s
        dense           e_mac * (2*3 + 3*2) = e_mac * 12

Hand network "supported" (T = 1, 2-3-3): W1 as above with b1 = [0, 0, 0], beta 0.5, threshold 1,
    readout identity 3x3. One step: s1 -> A fires (1.5 > 1): 1 spike; s2 -> nothing: 0; s3 -> B
    fires (1.2 > 1): 1 spike. Fan-out 3. main = e_mac*6 + e_ac*3 s, with updates = main +
    (e_mac + e_ac)*6 + e_ac s, dense = e_mac*(6 + 9).

Hand network "not supported" (T = 8, 1-2-2): W1 = [[0], [0]], b1 = [2.5, 2.5], beta 0.5,
    threshold 1, readout identity 2x2. Each neuron: U = 2.5 S | 2.75 S | 2.875 S | ... -> 3, above 1
    at every step: 16 spikes per decision for any input. main = e_mac*2 + e_ac*2*16, with updates =
    main + (e_mac + e_ac)*(2 + 2)*8 + e_ac*16, dense = e_mac*(1*2 + 2*2).
"""

from _checks import BANNER, Checks

print(BANNER)

import inspect  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import ROOT3, FixtureHandSpec, hand_deployed  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.students import measure  # noqa: E402
from learned_judgment.students.energy import CAVEATS, EnergyConstants  # noqa: E402
from learned_judgment.students.measure import (  # noqa: E402
    DREAM_SOURCES,
    H4_ENERGY_RULE,
    dream_source,
    h4_energy_criterion,
    measure_decision_latency,
    measure_operations,
)
from learned_judgment.students.snn import DeployedStudent, OutOfRangeError  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

checks = Checks()
config = load_config()
constants = EnergyConstants.from_config(config)
e_mac, e_ac = constants.e_mac_pj, constants.e_ac_pj
bootstrap = {
    "n_resamples": require(config, "evaluation.bootstrap_resamples"),
    "confidence_level": require(config, "evaluation.bootstrap_confidence_level"),
    "method": require(config, "evaluation.bootstrap_method"),
}
SEED = 5  # fixture bootstrap seed
TRAINING = np.random.default_rng(7).uniform(-ROOT3, ROOT3, (20, 2))  # FIXTURE stand-in for "situations training saw"
print(f"\nconfig.yaml: e_mac_pj={e_mac} e_ac_pj={e_ac} bootstrap {bootstrap}")

spec = FixtureHandSpec("fixture_hand", 2)
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
student = hand_deployed(network, spec, (2,), TRAINING)
s1, s2, s3, s1p = [1.5, 0.0], [0.8, 0.0], [-1.0, 1.2], [1.6, 0.0]
everything = np.array([s1, s2, s3, s1p])

print("\n== M7: the hand network is a DeployedStudent built by hand ==")
checks.expect(isinstance(student, DeployedStudent), "the measured object is a DeployedStudent (no SNNStudent involved)")
checks.expect(bool(np.array_equal(student.encode(everything), everything)), "the fixture encoding is exactly the identity")

print("\n== hand spike trains, through snn_core ==")
_, step_counts = network.forward_with_spikes(everything)
hand_steps = np.array([[[2, 1, 2, 2], [2, 2, 1, 2], [2, 1, 2, 2]]])  # (L=1, T=3, n=4): column i = situation i
print(f"  snn_core per-step counts (layer 0, rows = steps): {step_counts[0].tolist()}")
checks.expect(bool(np.array_equal(step_counts, hand_steps)), "per-step spike counts equal the hand trace")
_, trains = network.forward_with_spike_trains(everything)
hand_trains = np.zeros((3, 4, 3), dtype=np.uint8)  # (T, n, H): neuron order A, B, C
hand_trains[:, 0, 0] = 1  # s1: A at t1 t2 t3
hand_trains[1, 1, 0] = 1  # s2: A at t2
hand_trains[[0, 2], 2, 1] = 1  # s3: B at t1 t3
hand_trains[:, 3, 0] = 1  # s1': A at t1 t2 t3
hand_trains[:, :, 2] = 1  # C at every step for every input
checks.expect(bool(np.array_equal(trains[0], hand_trains)), "per-neuron spike trains equal the hand trace (A, B, C at every step)")

# Hand quantities shared by every case of the 2-3-2 network.
fan_out = 2
first_layer_macs = 2 * 3  # D * H_1
neuron_updates = (3 + 2) * 3  # (H_1 + C) * T
dense_macs = 2 * 3 + 3 * 2  # sum of fan_in * fan_out over the layers
hand_dense = e_mac * dense_macs
hand_main_break_even = e_mac / (e_ac * 3)
hand_updates_break_even = (e_mac * 3 * fan_out - (e_mac + e_ac) * neuron_updates) / (e_ac * 3 * 3 * (fan_out + 1))


def hand_energies(spikes: np.ndarray, *, d: int, h: int, c: int, t: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-decision main and with-updates estimates and the dense reference, written out by hand."""
    main = e_mac * d * h + e_ac * c * spikes
    updates = main + (e_mac + e_ac) * (h + c) * t + e_ac * spikes
    return main, updates, e_mac * (d * h + h * c)


def check_interval(label: str, interval: dict, values: np.ndarray, n_distinct: int) -> None:
    slack = 1e-12 * float(np.max(np.abs(values)))  # the bootstrap's percentile interpolation can land 1 ulp outside
    checks.expect(
        values.min() - slack <= interval["low"] <= interval["high"] <= values.max() + slack
        and interval["n_rows"] == values.size
        and interval["n_distinct_situations"] == n_distinct
        and interval["n_resamples"] == bootstrap["n_resamples"]
        and interval["seed"] == SEED
        and "distinct situation" in interval["resampled_unit"],
        f"{label} interval lies within [min, max] of the per-decision values; records rows, distinct situations, "
        "resamples, seed and the resampled unit (distinct situations)",
    )


cases = {
    # name: (situations, hand spikes per decision, hand never-fire count, hand fire-every-step count, distinct)
    "s1,s2,s3": ([s1, s2, s3], [6, 4, 5], 0, 1, 3),  # A, B, C all fire somewhere; only C fires at every step
    "s1,s2": ([s1, s2], [6, 4], 1, 1, 2),  # B never fires; A misses steps on s2; C fires at every step
    "s1,s1'": ([s1, s1p], [6, 6], 1, 2, 2),  # identical spike counts on two distinct situations (degenerate bootstrap)
    "s1,s1,s2": ([s1, s1, s2], [6, 6, 4], 1, 1, 2),  # a duplicated situation: 3 rows, 2 distinct situations
}
for name, (rows, hand_spikes, hand_never, hand_every, hand_distinct) in cases.items():
    print(f"\n== case {name} (2-3-2 network) ==")
    situations = np.array(rows)
    result = measure_operations(student, situations, constants, seed=SEED, **bootstrap)
    hand_spikes = np.array(hand_spikes, dtype=np.float64)
    hand_synops = hand_spikes * fan_out
    hand_main, hand_updates, dense_check = hand_energies(hand_spikes, d=2, h=3, c=2, t=3)
    hand_rate = hand_spikes.mean() / (3 * 3)
    energy = result["energy_estimate"]
    main_block, updates_block = energy["main_estimate"], energy["with_neuron_updates_estimate"]
    print(f"  measured spikes per decision {result['per_decision']['spikes_per_layer']} | hand {hand_spikes.tolist()}")
    print(
        f"  firing rate {result['firing']['rate_per_layer']} (hand {hand_rate:.6f}); never fire {result['firing']['never_fire_count_per_layer']} "
        f"(hand {hand_never}); fire every step {result['firing']['fire_every_step_count_per_layer']} (hand {hand_every})"
    )
    for label, block, hand in (("main", main_block, hand_main), ("with neuron updates", updates_block, hand_updates)):
        interval = block["estimate_pj_interval"]
        print(
            f"  {label} ESTIMATE {block['mean_estimate_pj_per_decision']:.4f} pJ (hand {hand.mean():.4f}) "
            f"[{interval['low']:.4f}, {interval['high']:.4f}] {interval['method_used']}; ratio to dense "
            f"{block['ratio_estimate_to_dense_reference_estimate']:.4f}; break-even rate {block['break_even']['rate_per_step']:.6f} "
            f"vs weighted measured rate {block['break_even']['measured_weighted_rate_per_step']:.6f}"
        )
    print(
        f"  dense reference ESTIMATE {energy['dense_reference_estimate']['estimate_pj_per_decision']:.4f} pJ (hand {hand_dense:.4f}); "
        f"H4 energy verdict from the estimates: {energy['h4_energy_criterion']['verdict_from_estimates']}"
    )
    checks.expect(result["per_decision"]["spikes_per_layer"] == [[v] for v in hand_spikes.tolist()], "spikes per decision = hand")
    checks.expect(result["per_decision"]["synaptic_ops"] == hand_synops.tolist(), "synaptic ops per decision = hand spikes x fan-out 2")
    checks.expect(result["fan_out"] == [fan_out] and result["layer_sizes"] == [2, 3, 2] and result["time_steps"] == 3, "fan-out, layer sizes and T")
    checks.expect(
        result["first_layer_macs"] == first_layer_macs and result["neuron_updates"] == neuron_updates
        and result["dense_reference_macs"] == dense_macs,
        f"first-layer MACs {first_layer_macs}, neuron updates {neuron_updates}, dense reference MACs {dense_macs}",
    )
    checks.expect(
        result["situations"]["n_rows"] == len(rows) and result["situations"]["n_distinct_situations"] == hand_distinct,
        f"the record states {len(rows)} rows and {hand_distinct} distinct situations",
    )
    checks.expect(math.isclose(result["firing"]["rate_per_layer"][0], hand_rate, rel_tol=1e-15), "firing rate = hand mean spikes / (H T)")
    checks.expect(
        result["firing"]["never_fire_count_per_layer"] == [hand_never]
        and result["firing"]["fire_every_step_count_per_layer"] == [hand_every]
        and math.isclose(result["firing"]["never_fire_fraction_per_layer"][0], hand_never / 3)
        and math.isclose(result["firing"]["fire_every_step_fraction_per_layer"][0], hand_every / 3)
        and "forward_with_spike_trains" in result["firing"]["source"],
        "never-fire and fire-every-step neurons = hand, from forward_with_spike_trains",
    )
    checks.expect(
        math.isclose(main_block["mean_estimate_pj_per_decision"], hand_main.mean(), rel_tol=1e-12)
        and math.isclose(updates_block["mean_estimate_pj_per_decision"], hand_updates.mean(), rel_tol=1e-12)
        and math.isclose(energy["dense_reference_estimate"]["estimate_pj_per_decision"], hand_dense, rel_tol=1e-12)
        and dense_check == hand_dense,
        "both SNN estimates and the dense reference = hand formulas with config.yaml constants",
    )
    checks.expect(
        np.allclose(main_block["per_decision_estimate_pj"], hand_main, rtol=1e-12)
        and np.allclose(updates_block["per_decision_estimate_pj"], hand_updates, rtol=1e-12),
        "per-decision values of both estimates = hand",
    )
    checks.expect(
        math.isclose(main_block["ratio_estimate_to_dense_reference_estimate"], hand_main.mean() / hand_dense, rel_tol=1e-12)
        and math.isclose(updates_block["ratio_estimate_to_dense_reference_estimate"], hand_updates.mean() / hand_dense, rel_tol=1e-12),
        "both ratios to the dense reference = hand",
    )
    for label, block in (("main estimate", main_block), ("with-neuron-updates estimate", updates_block)):
        # bounds from the record's own per-decision values (equal to the hand values up to rounding, checked above)
        check_interval(label, block["estimate_pj_interval"], np.asarray(block["per_decision_estimate_pj"]), hand_distinct)
        checks.expect(
            math.isclose(block["ratio_interval"]["low"], block["estimate_pj_interval"]["low"] / hand_dense, rel_tol=1e-12)
            and math.isclose(block["ratio_interval"]["high"], block["estimate_pj_interval"]["high"] / hand_dense, rel_tol=1e-12),
            f"{label}: ratio interval = estimate interval / dense reference",
        )
    check_interval("synaptic ops", result["synaptic_ops_per_decision"]["interval"], hand_synops, hand_distinct)
    checks.expect(
        math.isclose(main_block["break_even"]["rate_per_step"], hand_main_break_even, rel_tol=1e-15)
        and math.isclose(updates_block["break_even"]["rate_per_step"], hand_updates_break_even, rel_tol=1e-12)
        and updates_block["break_even"]["reachable"] is False
        and main_block["break_even"]["measured_rate_per_layer"] == result["firing"]["rate_per_layer"]
        and updates_block["break_even"]["measured_rate_per_layer"] == result["firing"]["rate_per_layer"],
        f"break-even rates: main e_mac/(e_ac T) = {hand_main_break_even:.4f}, with updates {hand_updates_break_even:.4f} "
        "(below 0: unreachable), each next to the measured rates",
    )
    checks.expect(
        main_block["mean_vs_dense_reference_estimate"] == "below" and updates_block["mean_vs_dense_reference_estimate"] == "above"
        and main_block["break_even"]["weighted_rate_below_break_even"] is True
        and updates_block["break_even"]["weighted_rate_below_break_even"] is False
        and energy["h4_energy_criterion"]["verdict_from_estimates"] == "inconclusive"
        and energy["h4_energy_criterion"]["rule"] == H4_ENERGY_RULE,
        "main estimate below, with-updates estimate above the reference: verdict 'inconclusive', with the rule text",
    )
    checks.expect(result["caveats"] == list(CAVEATS) and energy["caveats"] == list(CAVEATS), "carries energy.CAVEATS")
    if name == "s1,s1'":
        interval = result["synaptic_ops_per_decision"]["interval"]
        checks.expect(
            interval["method_used"] == "percentile" and interval["fallback_reason"] is not None
            and interval["low"] == interval["high"] == 12.0,
            "identical values on two distinct situations: BCa is not finite, the percentile fallback is recorded and gives [12, 12]",
        )
    if name == "s1,s1,s2":
        interval = result["spikes_per_decision"]["interval_total"]
        checks.expect(
            math.isclose(interval["estimate"], 16.0 / 3.0, rel_tol=1e-15) and 4.0 <= interval["low"] <= interval["high"] <= 6.0
            and result["per_decision"]["situation_group"] == [0, 0, 1],
            "duplicated situation: the estimate is the row mean 16/3, the two rows of s1 form one resampled situation",
        )

print("\n== every energy figure is named as an estimate ==")
energy = measure_operations(student, np.array([s1, s2, s3]), constants, seed=SEED, **bootstrap)["energy_estimate"]
pj_keys = []


def walk(value: object, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "pj" in key.lower():
                pj_keys.append(f"{path}.{key}")
            walk(item, f"{path}.{key}")


walk(energy, "energy_estimate")
print(f"  {len(pj_keys)} keys carrying pJ values, e.g. {pj_keys[:4]}")
checks.expect(
    bool(pj_keys) and all("estimate" in key.split(".")[-1] or key.split(".")[-1] in ("e_mac_pj", "e_ac_pj") for key in pj_keys),
    "every key holding an energy figure (pJ) has 'estimate' in its own name (the per-operation constants e_mac_pj, e_ac_pj aside)",
)
checks.expect("estimate" in energy["label"] and "not a measurement" in energy["label"], "the energy block's label says estimate, not a measurement")

print("\n== M3: the H4 energy verdict rule in all three outcomes, on hand-built networks ==")
supported_net = SpikingNetwork(
    hidden_weights=(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),),
    hidden_biases=(np.zeros(3),),
    betas=(0.5,),
    thresholds=(1.0,),
    readout_weight=np.eye(3),
    readout_bias=np.zeros(3),
    readout_beta=0.5,
    time_steps=1,
)
supported_spec = FixtureHandSpec("fixture_hand_sparse", 2)
busy_net = SpikingNetwork(
    hidden_weights=(np.zeros((2, 1)),),
    hidden_biases=(np.array([2.5, 2.5]),),
    betas=(0.5,),
    thresholds=(1.0,),
    readout_weight=np.eye(2),
    readout_bias=np.zeros(2),
    readout_beta=0.5,
    time_steps=8,
)
busy_spec = FixtureHandSpec("fixture_hand_busy", 1)
outcomes = {
    "supported": (
        hand_deployed(supported_net, supported_spec, (3,), TRAINING),
        np.array([s1, s2, s3]),
        np.array([1.0, 0.0, 1.0]),
        {"d": 2, "h": 3, "c": 3, "t": 1},
    ),
    "not supported": (
        hand_deployed(busy_net, busy_spec, (2,), TRAINING[:, :1]),
        np.array([[0.5], [-0.5]]),
        np.array([16.0, 16.0]),
        {"d": 1, "h": 2, "c": 2, "t": 8},
    ),
    "inconclusive": (student, np.array([s1, s2, s3]), np.array([6.0, 4.0, 5.0]), {"d": 2, "h": 3, "c": 2, "t": 3}),
}
for verdict, (deployed, situations, hand_spikes, sizes) in outcomes.items():
    result = measure_operations(deployed, situations, constants, seed=SEED, **bootstrap)
    energy = result["energy_estimate"]
    hand_main, hand_updates, hand_ref = hand_energies(hand_spikes, **sizes)
    main_block, updates_block = energy["main_estimate"], energy["with_neuron_updates_estimate"]
    print(
        f"  {deployed.skill} {result['layer_sizes']} T={result['time_steps']}: spikes {[row[0] for row in result['per_decision']['spikes_per_layer']]} "
        f"(hand {hand_spikes.tolist()}); main ESTIMATE {main_block['mean_estimate_pj_per_decision']:.2f} (hand {hand_main.mean():.2f}), "
        f"with updates {updates_block['mean_estimate_pj_per_decision']:.2f} (hand {hand_updates.mean():.2f}), dense "
        f"{energy['dense_reference_estimate']['estimate_pj_per_decision']:.2f} (hand {hand_ref:.2f}) -> "
        f"{energy['h4_energy_criterion']['verdict_from_estimates']!r}"
    )
    checks.expect(
        [row[0] for row in result["per_decision"]["spikes_per_layer"]] == hand_spikes.tolist()
        and math.isclose(main_block["mean_estimate_pj_per_decision"], hand_main.mean(), rel_tol=1e-12)
        and math.isclose(updates_block["mean_estimate_pj_per_decision"], hand_updates.mean(), rel_tol=1e-12)
        and math.isclose(energy["dense_reference_estimate"]["estimate_pj_per_decision"], hand_ref, rel_tol=1e-12),
        f"{deployed.skill}: spikes and the three estimates = hand",
    )
    hand_verdict = (
        "supported" if hand_main.mean() < hand_ref and hand_updates.mean() < hand_ref
        else "not supported" if hand_main.mean() > hand_ref and hand_updates.mean() > hand_ref
        else "inconclusive"
    )
    checks.expect(
        energy["h4_energy_criterion"]["verdict_from_estimates"] == verdict == hand_verdict,
        f"{deployed.skill}: verdict {verdict!r} (hand rule on the hand estimates gives {hand_verdict!r})",
    )
    for block in (main_block, updates_block):
        mean_below = block["mean_estimate_pj_per_decision"] < energy["dense_reference_estimate"]["estimate_pj_per_decision"]
        checks.expect(
            block["break_even"]["weighted_rate_below_break_even"] == mean_below,
            f"{deployed.skill}: the weighted firing rate is below its break-even rate exactly when that estimate is below the reference",
        )

print("\n== h4_energy_criterion on its own ==")
checks.expect(h4_energy_criterion(5.0, 6.0, 10.0)["verdict_from_estimates"] == "supported", "both below -> supported")
checks.expect(h4_energy_criterion(11.0, 12.0, 10.0)["verdict_from_estimates"] == "not supported", "both above -> not supported")
checks.expect(h4_energy_criterion(9.0, 12.0, 10.0)["verdict_from_estimates"] == "inconclusive", "disagree -> inconclusive")
equal = h4_energy_criterion(10.0, 9.0, 10.0)
checks.expect(
    equal["verdict_from_estimates"] == "inconclusive" and equal["main_estimate_vs_dense_reference_estimate"] == "equal",
    "an estimate equal to the reference is neither below nor above -> inconclusive",
)
checks.raises("dense reference 0", ValueError, lambda: h4_energy_criterion(1.0, 1.0, 0.0), "must be > 0")
checks.raises("a NaN estimate", ValueError, lambda: h4_energy_criterion(float("nan"), 1.0, 2.0), "must be finite")

print("\n== two hidden layers (random FIXTURE weights) against an independent scalar re-implementation ==")
# Not hand-traceable, so the expected spikes come from plain Python loops over neurons and steps
# written here from the dynamics in snn_core's docstring (an implementation independent of numpy).
rng = np.random.default_rng(2026)  # fixture weights
deep = SpikingNetwork(
    hidden_weights=(rng.normal(0.0, 1.2, (5, 2)), rng.normal(0.0, 0.8, (4, 5))),
    hidden_biases=(rng.normal(0.3, 0.3, 5), rng.normal(0.3, 0.3, 4)),
    betas=(0.8, 0.6),
    thresholds=(0.7, 0.9),
    readout_weight=rng.normal(0.0, 1.0, (3, 4)),
    readout_bias=np.zeros(3),
    readout_beta=0.7,
    time_steps=4,
)
deep_student = hand_deployed(deep, FixtureHandSpec("fixture_hand_deep", 2), (3,), TRAINING)
deep_situations = rng.uniform(-ROOT3, ROOT3, (40, 2))


def scalar_spikes(network: SpikingNetwork, x: list[float]) -> list[list[float]]:
    """Per-neuron spikes over the T steps for one input, with Python floats and loops."""
    membranes = [[0.0] * len(b) for b in network.hidden_biases]
    totals = [[0.0] * len(b) for b in network.hidden_biases]
    for _ in range(network.time_steps):
        previous = list(x)
        for layer, (weight, bias) in enumerate(zip(network.hidden_weights, network.hidden_biases)):
            threshold, beta = network.thresholds[layer], network.betas[layer]
            spikes = []
            for i in range(len(bias)):
                current = sum(float(weight[i, j]) * previous[j] for j in range(len(previous))) + float(bias[i])
                reset = 1.0 if membranes[layer][i] > threshold else 0.0
                membranes[layer][i] = (beta * membranes[layer][i] + current) - reset * threshold
                spike = 1.0 if membranes[layer][i] > threshold else 0.0
                totals[layer][i] += spike
                spikes.append(spike)
            previous = spikes
    return totals


expected = [scalar_spikes(deep, list(row)) for row in deep_situations]  # [n][L][H_l]
expected_layer_totals = np.array([[sum(layer) for layer in row] for row in expected])  # (n, L)
expected_neuron_totals = [np.array([row[layer] for row in expected]).sum(axis=0) for layer in range(2)]  # (H_l,)
n_deep, steps_deep = deep_situations.shape[0], deep.time_steps
expected_never = [int(np.sum(totals == 0.0)) for totals in expected_neuron_totals]
expected_every = [int(np.sum(totals == n_deep * steps_deep)) for totals in expected_neuron_totals]
deep_result = measure_operations(deep_student, deep_situations, constants, seed=SEED, **bootstrap)
deep_spikes = np.array(deep_result["per_decision"]["spikes_per_layer"])
print(
    f"  layer sizes {deep_result['layer_sizes']}, fan-out {deep_result['fan_out']}, mean spikes per layer "
    f"{[round(v, 3) for v in deep_result['spikes_per_decision']['mean_per_layer']]}, rates {[round(v, 4) for v in deep_result['firing']['rate_per_layer']]}, "
    f"never fire {deep_result['firing']['never_fire_count_per_layer']} (scalar {expected_never}), fire every step "
    f"{deep_result['firing']['fire_every_step_count_per_layer']} (scalar {expected_every})"
)
checks.expect(bool(np.array_equal(deep_spikes, expected_layer_totals)), "spikes per decision and layer = scalar re-implementation")
checks.expect(
    deep_result["fan_out"] == [4, 3] and deep_result["per_decision"]["synaptic_ops"] == (deep_spikes @ np.array([4.0, 3.0])).tolist(),
    "synaptic ops = layer-1 spikes x 4 + layer-2 spikes x 3 (fan-out = width of the layer fed)",
)
checks.expect(
    deep_result["first_layer_macs"] == 2 * 5 and deep_result["neuron_updates"] == (5 + 4 + 3) * 4
    and deep_result["dense_reference_macs"] == 2 * 5 + 5 * 4 + 4 * 3,
    "first-layer MACs 10, neuron updates 48, dense reference MACs 42",
)
checks.expect(
    deep_result["firing"]["never_fire_count_per_layer"] == expected_never
    and deep_result["firing"]["fire_every_step_count_per_layer"] == expected_every,
    "never-fire and fire-every-step counts per layer = scalar re-implementation",
)
rates = expected_layer_totals.mean(axis=0) / (np.array([5.0, 4.0]) * steps_deep)
checks.expect(np.allclose(deep_result["firing"]["rate_per_layer"], rates, rtol=1e-15), "firing rate per layer = mean spikes / (H_l T)")
weights_main, weights_updates = np.array([5.0 * 4, 4.0 * 3]), np.array([5.0 * 5, 4.0 * 4])
hand_updates_be = (e_mac * weights_main.sum() - (e_mac + e_ac) * (5 + 4 + 3) * 4) / (e_ac * 4 * weights_updates.sum())
deep_energy = deep_result["energy_estimate"]
checks.expect(
    math.isclose(deep_energy["main_estimate"]["break_even"]["measured_weighted_rate_per_step"], float(rates @ weights_main / weights_main.sum()), rel_tol=1e-12)
    and math.isclose(deep_energy["with_neuron_updates_estimate"]["break_even"]["measured_weighted_rate_per_step"], float(rates @ weights_updates / weights_updates.sum()), rel_tol=1e-12)
    and math.isclose(deep_energy["with_neuron_updates_estimate"]["break_even"]["rate_per_step"], hand_updates_be, rel_tol=1e-12),
    "two layers: weighted rates (H_l fo_l and H_l (fo_l + 1)) and the with-updates break-even rate = hand formulas",
)

print("\n== M6: measure.py keeps no copy of the network loop ==")
source = inspect.getsource(measure)
checks.expect(
    "forward_with_spike_trains" in source and "_spikes_per_neuron" not in source and "for step in range(" not in source
    and "readout_beta *" not in source,
    "measure.py uses SpikingNetwork.forward_with_spike_trains and has no loop over time steps of its own",
)

print("\n== input guards ==")
checks.raises("one distinct situation in two rows", ValueError, lambda: measure_operations(student, np.array([s1, s1]), constants, seed=SEED, **bootstrap), "distinct situation")
checks.raises("unknown bootstrap method", ValueError, lambda: measure_operations(student, everything, constants, n_resamples=10, confidence_level=0.9, method="normal", seed=SEED), "method must be one of")
checks.raises("non-finite situation", ValueError, lambda: measure_operations(student, np.array([s1, [np.nan, 0.0]]), constants, seed=SEED, **bootstrap), "non-finite")
checks.raises("situation outside the declared range (policy 'error')", OutOfRangeError, lambda: measure_operations(student, np.array([s1, [2.0, 0.0]]), constants, seed=SEED, **bootstrap), "outside its declared range")
checks.raises("wrong situation width", ValueError, lambda: measure_decision_latency(student, spec, np.zeros((3, 3)), n_decisions=5), "S=2")
checks.raises("an object that is not a student", TypeError, lambda: measure_operations(object(), everything, constants, seed=SEED, **bootstrap), "shared.deployed(skill)")


class NeedsSkill:
    def deployed(self, skill: str) -> DeployedStudent:  # stands for a shared network's view method
        raise AssertionError("never reached")


checks.raises("an object whose deployed() needs a skill", TypeError, lambda: measure_operations(NeedsSkill(), everything, constants, seed=SEED, **bootstrap), "shared.deployed(skill)")

print("\n== dream_source ==")
checks.expect(dream_source(PHYSICS_ONLY, ()) == "physics_only", "physics_only method, no requests -> physics_only")
checks.expect(dream_source("constant_bias", ()) == "local_residual_model", "a model method without requests -> local_residual_model")
checks.expect(dream_source("teacher", ("a" * 64,)) == "provider_backed_teacher", "request hashes -> provider_backed_teacher")
checks.expect(set(DREAM_SOURCES) == {"physics_only", "local_residual_model", "provider_backed_teacher"}, "three documented sources")
checks.raises("physics_only with request hashes", ValueError, lambda: dream_source(PHYSICS_ONLY, ("a" * 64,)), "cannot name provider requests")

print("\n== latency of the hand network through a bound student (mechanics only; n_decisions from config.yaml) ==")
latency = measure_decision_latency(student, spec, everything, n_decisions=require(config, "evaluation.latency_decisions"))
print(
    f"  full path median {latency['full_path']['median_s'] * 1e6:.1f} us p95 {latency['full_path']['p95_s'] * 1e6:.1f} us; "
    f"index only median {latency['index_only']['median_s'] * 1e6:.1f} us p95 {latency['index_only']['p95_s'] * 1e6:.1f} us; "
    f"{latency['full_path']['n_decisions']} decisions each; torch_free_process={latency['torch_free_process']}"
)
checks.expect(latency["torch_free_process"], "the latency path ran without torch in the process")
checks.expect(
    latency["full_path"]["n_decisions"] == latency["index_only"]["n_decisions"] == require(config, "evaluation.latency_decisions")
    and latency["full_path"]["n_inputs"] == 4,
    "n_decisions from config for both paths, cycling 4 situations",
)
checks.expect(
    0.0 < latency["full_path"]["median_s"] <= latency["full_path"]["p95_s"] and 0.0 < latency["index_only"]["median_s"] <= latency["index_only"]["p95_s"],
    "0 < median <= p95 on both paths",
)
checks.expect(
    latency["single_row_vs_batch_decisions_differing"] == 0 and latency["full_path_actions_not_the_looked_up_candidate"] == 0,
    "single-row decisions equal the batched decisions; the full path returns the looked-up candidate",
)
checks.finish()
