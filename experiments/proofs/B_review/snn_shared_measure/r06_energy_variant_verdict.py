"""R06: can the H4 energy comparison flip with the estimate variant, and which variant gets an interval?

energy.estimate_energy reports a main estimate and a "with neuron updates" estimate next to the same
dense reference; measure_operations attaches a bootstrap interval to the main estimate only
(energy_main_pj_interval). PREREGISTRATION.md H4 (amendment) says only "lower estimated energy per
decision" than the dense reference. This runs measure_operations on a HAND-BUILT network of the config
size (4 encoded inputs, 64 hidden, 35 outputs, T = 10; FIXTURE weights tuned to a ~0.3 firing rate) with
the config.yaml energy constants and bootstrap settings. numpy only.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\snn_shared_measure\\r06_energy_variant_verdict.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import numpy as np  # noqa: E402

from _review_common import POINTS, fixture_spec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.energy import EnergyConstants  # noqa: E402
from learned_judgment.students.measure import measure_operations  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

config = load_config()
constants = EnergyConstants.from_config(config)
boot = {
    "n_resamples": require(config, "evaluation.bootstrap_resamples"),
    "confidence_level": require(config, "evaluation.bootstrap_confidence_level"),
    "method": require(config, "evaluation.bootstrap_method"),
}
spec = fixture_spec("fx_alpha")
encoder = InputEncoder.from_variables(spec.situation)
situations = spec.sample_situations(stream_rng(5, "fx_alpha/review/energy"), 300)
for bias in (0.25, 0.35, 0.45):
    rng = np.random.default_rng(5)  # fixture weights
    network = SpikingNetwork(
        hidden_weights=(rng.normal(0, 0.5, (64, encoder.width)),), hidden_biases=(np.full(64, bias) + rng.normal(0, 0.1, 64),),
        betas=(0.9,), thresholds=(1.0,), readout_weight=rng.normal(0, 1, (35, 64)), readout_bias=np.zeros(35),
        readout_beta=0.9, time_steps=10,
    )
    student = SNNStudent(
        skill="fx_alpha", network=network, encoder=encoder, points_per_dim=POINTS["fx_alpha"], n_candidates=35,
        meta={"kind": "review_hand_network", "student": {"action_names": list(spec.action_names)}},
    )
    result = measure_operations(student, situations, constants, seed=3, **boot)
    energy = result["energy_estimate"]
    interval = result["energy_main_pj_interval"]
    print(
        f"bias {bias}: firing rate {result['firing']['rate_per_layer'][0]:.3f}; main {energy['snn_main_pj_per_decision']:.0f} pJ "
        f"[{interval['low']:.0f}, {interval['high']:.0f}] ratio to dense {energy['ratio_snn_main_to_dense']:.3f}; with neuron updates "
        f"{energy['snn_with_neuron_updates_pj_per_decision']:.0f} pJ (no interval) ratio {energy['ratio_snn_with_neuron_updates_to_dense']:.3f}; "
        f"dense {energy['dense_reference_pj_per_decision']:.0f} pJ"
    )
print(f"operations keys holding an interval: {[k for k, v in result.items() if isinstance(v, dict) and 'low' in v]} "
      f"+ spikes/synaptic-ops sub-intervals; energy_estimate keys: {sorted(energy)}")
