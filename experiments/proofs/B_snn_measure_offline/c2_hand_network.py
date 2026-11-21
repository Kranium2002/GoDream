"""Check 2: measure_operations against a hand computation on a tiny hand-built SpikingNetwork.

FIXTURE CHECK - mechanics only. The network is written by hand (2 inputs, 3 hidden neurons,
2 outputs, T = 3 steps); its spikes are worked out below by hand and compared with what
learned_judgment.students.measure reports. The energy constants and bootstrap settings are the
real config.yaml values; the expected energies are computed here from the hand spike counts
with the formulas of students/energy.py written out independently. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c2_hand_network.py

Hand network (encoded input x = (x_a, x_b); the encoder is the identity, see below):
    hidden current  I = W1 x + b1,  W1 = [[1, 0], [0, 1], [0, 0]],  b1 = [0, 0, 2.5]
                    neuron A gets x_a, neuron B gets x_b, neuron C gets 2.5 for every input
    beta = 0.5, threshold = 1.0, subtractive reset taken from the previous membrane:
        U_t = (0.5 U_{t-1} + I) - [U_{t-1} > 1] * 1,   spike S_t = [U_t > 1]
    readout W_o = [[1, 0, 0], [0, 1, 0]], b_o = 0, readout beta 0.5 (fan-out of the hidden layer = 2)

    s1 = (1.5, 0.0)   A: U = 1.5 S | 0.75+1.5-1 = 1.25 S | 0.625+1.5-1 = 1.125 S  -> 3 spikes
                      B: U = 0, 0, 0                                               -> 0
                      C: U = 2.5 S | 1.25+2.5-1 = 2.75 S | 1.375+2.5-1 = 2.875 S   -> 3
                      per step (A+B+C): [2, 2, 2], total 6
    s2 = (0.8, 0.0)   A: U = 0.8 - | 0.4+0.8 = 1.2 S | 0.6+0.8-1 = 0.4 -            -> 1
                      B: 0;  C: 3                    per step [1, 2, 1], total 4
    s3 = (-1.0, 1.2)  A: U = -1, -1.5, -1.75 (never above 1)                       -> 0
                      B: U = 1.2 S | 0.6+1.2-1 = 0.8 - | 0.4+1.2 = 1.6 S            -> 2
                      C: 3                           per step [2, 1, 2], total 5
"""

from _checks import BANNER, Checks

print(BANNER)

import math  # noqa: E402

import numpy as np  # noqa: E402

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.skills.base import Variable  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.energy import CAVEATS, EnergyConstants  # noqa: E402
from learned_judgment.students.measure import (  # noqa: E402
    DREAM_SOURCES,
    dream_source,
    measure_decision_latency,
    measure_operations,
)
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

checks = Checks()
config = load_config()
constants = EnergyConstants.from_config(config)
bootstrap = {
    "n_resamples": require(config, "evaluation.bootstrap_resamples"),
    "confidence_level": require(config, "evaluation.bootstrap_confidence_level"),
    "method": require(config, "evaluation.bootstrap_method"),
}
SEED = 5  # fixture bootstrap seed
print(f"\nconfig.yaml: e_mac_pj={constants.e_mac_pj} e_ac_pj={constants.e_ac_pj} bootstrap {bootstrap}")

# Encoder: low = -sqrt(3), high = sqrt(3) gives midpoint 0 and scale (2 sqrt 3) / sqrt 12 = 1 exactly.
root3 = math.sqrt(3.0)
encoder = InputEncoder.from_variables(
    [
        Variable("fx_a", "float", "1", "fixture hand input a", low=-root3, high=root3),
        Variable("fx_b", "float", "1", "fixture hand input b", low=-root3, high=root3),
    ]
)
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
student = SNNStudent(
    skill="fixture_hand",
    network=network,
    encoder=encoder,
    points_per_dim=(2,),
    n_candidates=2,
    meta={"kind": "fixture_hand_built_network", "student": {"action_names": ["fx_choice"]}},
)
s1, s2, s3 = [1.5, 0.0], [0.8, 0.0], [-1.0, 1.2]
everything = np.array([s1, s2, s3])
checks.expect(bool(np.array_equal(encoder.encode(everything), everything)), "the fixture encoder is exactly the identity")

print("\n== hand spike counts per step, through snn_core.forward_with_spikes ==")
_, step_counts = network.forward_with_spikes(everything)
hand_steps = np.array([[[2, 1, 2], [2, 2, 1], [2, 1, 2]]])  # (L=1, T=3, n=3): column i = situation i
print(f"  snn_core per-step counts (layer 0, rows = steps): {step_counts[0].tolist()}")
checks.expect(bool(np.array_equal(step_counts, hand_steps)), "per-step spike counts equal the hand trace")

# Hand quantities shared by every case.
fan_out = 2  # the hidden layer feeds the 2 readout outputs
first_layer_macs = 2 * 3  # D * H_1
neuron_updates = (3 + 2) * 3  # (H_1 + C) * T
dense_macs = 2 * 3 + 3 * 2  # sum of fan_in * fan_out over the layers
cases = {
    # name: (situations, hand spikes per decision, hand never-fire count, hand fire-every-step count)
    "s1,s2,s3": ([s1, s2, s3], [6, 4, 5], 0, 1),  # A, B, C all fire somewhere; only C fires at every step
    "s1,s2": ([s1, s2], [6, 4], 1, 1),  # B never fires; A misses steps on s2; C fires at every step
    "s1,s1": ([s1, s1], [6, 6], 1, 2),  # B never fires; A and C fire at every step (degenerate bootstrap)
}
for name, (rows, hand_spikes, hand_never, hand_every) in cases.items():
    print(f"\n== case {name} ==")
    situations = np.array(rows)
    result = measure_operations(student, situations, constants, seed=SEED, **bootstrap)
    hand_spikes = np.array(hand_spikes, dtype=np.float64)
    hand_synops = hand_spikes * fan_out
    hand_main = constants.e_mac_pj * first_layer_macs + constants.e_ac_pj * hand_synops
    hand_updates = hand_main + (constants.e_mac_pj + constants.e_ac_pj) * neuron_updates + constants.e_ac_pj * hand_spikes
    hand_dense = constants.e_mac_pj * dense_macs
    hand_rate = hand_spikes.mean() / (3 * 3)
    hand_break_even = constants.e_mac_pj / (constants.e_ac_pj * 3)
    energy = result["energy_estimate"]
    synops_interval = result["synaptic_ops_per_decision"]["interval"]
    energy_interval = result["energy_main_pj_interval"]
    print(f"  measured spikes per decision {result['per_decision']['spikes_per_layer']} | hand {hand_spikes.tolist()}")
    print(f"  measured synaptic ops {result['per_decision']['synaptic_ops']} | hand {hand_synops.tolist()}")
    print(
        f"  firing rate {result['firing']['rate_per_layer']} (hand {hand_rate:.6f}); break-even {result['break_even']['rate_per_step']:.6f} "
        f"(hand {hand_break_even:.6f}); never fire {result['firing']['never_fire_count_per_layer']} (hand {hand_never}); "
        f"fire every step {result['firing']['fire_every_step_count_per_layer']} (hand {hand_every})"
    )
    print(
        f"  energy (estimate, pJ): main {energy['snn_main_pj_per_decision']:.4f} (hand {hand_main.mean():.4f}); with neuron "
        f"updates {energy['snn_with_neuron_updates_pj_per_decision']:.4f} (hand {hand_updates.mean():.4f}); dense "
        f"{energy['dense_reference_pj_per_decision']:.4f} (hand {hand_dense:.4f})"
    )
    print(
        f"  synaptic-ops interval [{synops_interval['low']}, {synops_interval['high']}] method_used {synops_interval['method_used']} "
        f"fallback {synops_interval['fallback_reason']!r}"
    )
    print(f"  main-energy interval [{energy_interval['low']:.4f}, {energy_interval['high']:.4f}] method_used {energy_interval['method_used']}")
    checks.expect(result["per_decision"]["spikes_per_layer"] == [[v] for v in hand_spikes.tolist()], "spikes per decision = hand")
    checks.expect(result["per_decision"]["synaptic_ops"] == hand_synops.tolist(), "synaptic ops per decision = hand spikes x fan-out 2")
    checks.expect(result["fan_out"] == [fan_out] and result["layer_sizes"] == [2, 3, 2] and result["time_steps"] == 3, "fan-out, layer sizes and T")
    checks.expect(
        result["first_layer_macs"] == first_layer_macs and result["neuron_updates"] == neuron_updates
        and result["dense_reference_macs"] == dense_macs,
        f"first-layer MACs {first_layer_macs}, neuron updates {neuron_updates}, dense reference MACs {dense_macs}",
    )
    checks.expect(math.isclose(result["firing"]["rate_per_layer"][0], hand_rate, rel_tol=1e-15), "firing rate = hand mean spikes / (H T)")
    checks.expect(
        result["firing"]["never_fire_count_per_layer"] == [hand_never]
        and result["firing"]["fire_every_step_count_per_layer"] == [hand_every]
        and math.isclose(result["firing"]["never_fire_fraction_per_layer"][0], hand_never / 3)
        and math.isclose(result["firing"]["fire_every_step_fraction_per_layer"][0], hand_every / 3),
        "never-fire and fire-every-step neurons = hand",
    )
    checks.expect(math.isclose(result["break_even"]["rate_per_step"], hand_break_even, rel_tol=1e-15), "break-even rate = e_mac / (e_ac T)")
    checks.expect(
        math.isclose(energy["snn_main_pj_per_decision"], hand_main.mean(), rel_tol=1e-12)
        and math.isclose(energy["snn_with_neuron_updates_pj_per_decision"], hand_updates.mean(), rel_tol=1e-12)
        and math.isclose(energy["dense_reference_pj_per_decision"], hand_dense, rel_tol=1e-12)
        and math.isclose(energy["ratio_snn_main_to_dense"], hand_main.mean() / hand_dense, rel_tol=1e-12),
        "the three energy estimates and the main/dense ratio = hand formulas with config.yaml constants",
    )
    checks.expect(
        energy["constants"]["e_mac_pj"] == require(config, "students.snn.energy.e_mac_pj")
        and energy["constants"]["e_ac_pj"] == require(config, "students.snn.energy.e_ac_pj"),
        "the estimate used the config.yaml constants",
    )
    for label, interval, values in (
        ("synaptic ops", synops_interval, hand_synops),
        ("main energy", energy_interval, hand_main),
    ):
        checks.expect(
            values.min() <= interval["low"] <= interval["high"] <= values.max() and interval["n"] == len(rows)
            and interval["n_resamples"] == bootstrap["n_resamples"] and interval["seed"] == SEED,
            f"{label} interval lies within [min, max] of the per-decision values and records n, resamples, seed",
        )
    checks.expect(result["label"] == "estimate" and result["caveats"] == list(CAVEATS), "labelled 'estimate' with energy.CAVEATS")
    checks.expect(result["replica_check"]["max_abs_logit_difference"] == 0.0, "per-neuron replica logits identical to snn_core")
    if name == "s1,s1":
        checks.expect(
            synops_interval["method_used"] == "percentile" and synops_interval["fallback_reason"] is not None
            and synops_interval["low"] == synops_interval["high"] == 12.0,
            "identical decisions: BCa is not finite, the percentile fallback is recorded and gives [12, 12]",
        )

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
deep_student = SNNStudent(
    skill="fixture_hand_deep", network=deep, encoder=encoder, points_per_dim=(3,), n_candidates=3,
    meta={"kind": "fixture_hand_built_network", "student": {"action_names": ["fx_choice"]}},
)
deep_situations = rng.uniform(-root3, root3, (40, 2))


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
checks.expect(
    np.allclose(deep_result["firing"]["rate_per_layer"], expected_layer_totals.mean(axis=0) / (np.array([5.0, 4.0]) * steps_deep), rtol=1e-15),
    "firing rate per layer = mean spikes / (H_l T)",
)

print("\n== input guards ==")
checks.raises("one situation (bootstrap needs two)", ValueError, lambda: measure_operations(student, np.array([s1]), constants, seed=SEED, **bootstrap), "n >= 2")
checks.raises("unknown bootstrap method", ValueError, lambda: measure_operations(student, everything, constants, n_resamples=10, confidence_level=0.9, method="normal", seed=SEED), "method must be one of")
checks.raises("non-finite situation", ValueError, lambda: measure_operations(student, np.array([s1, [np.nan, 0.0]]), constants, seed=SEED, **bootstrap), "non-finite")
checks.raises("wrong situation width", ValueError, lambda: measure_decision_latency(student, np.zeros((3, 3)), n_decisions=5), "S=2")

print("\n== dream_source ==")
checks.expect(dream_source(PHYSICS_ONLY, ()) == "physics_only", "physics_only method, no requests -> physics_only")
checks.expect(dream_source("constant_bias", ()) == "local_residual_model", "a model method without requests -> local_residual_model")
checks.expect(dream_source("teacher", ("a" * 64,)) == "provider_backed_teacher", "request hashes -> provider_backed_teacher")
checks.expect(set(DREAM_SOURCES) == {"physics_only", "local_residual_model", "provider_backed_teacher"}, "three documented sources")
checks.raises("physics_only with request hashes", ValueError, lambda: dream_source(PHYSICS_ONLY, ("a" * 64,)), "cannot name provider requests")

print("\n== latency of the hand network (mechanics only; n_decisions from config.yaml) ==")
latency = measure_decision_latency(student, everything, n_decisions=require(config, "evaluation.latency_decisions"))
print(
    f"  median {latency['median_s'] * 1e6:.1f} us p95 {latency['p95_s'] * 1e6:.1f} us over {latency['n_decisions']} decisions; "
    f"torch_free_process={latency['torch_free_process']} single_row_vs_batch_differing={latency['single_row_vs_batch_decisions_differing']}"
)
checks.expect(latency["torch_free_process"], "the latency path ran without torch in the process")
checks.expect(latency["n_decisions"] == require(config, "evaluation.latency_decisions") and latency["n_inputs"] == 3, "n_decisions from config, cycling 3 situations")
checks.expect(0.0 < latency["median_s"] <= latency["p95_s"], "0 < median <= p95")
checks.expect(latency["single_row_vs_batch_decisions_differing"] == 0, "single-row decisions equal the batched decisions")
checks.finish()
