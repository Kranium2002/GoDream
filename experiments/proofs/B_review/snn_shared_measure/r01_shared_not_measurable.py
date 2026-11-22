"""R01: can the SHARED spiking student be measured (latency, operations, firing, energy, in-dream agreement)?

03_METRICS.md asks for median/p95 latency of "teacher network time, per-skill SNN, and shared SNN", and
06_PERSON_B.md for spikes, synaptic operations and latency of "per-skill and shared SNNs". This trains a
tiny FIXTURE shared student (one fixture skill, 60 steps) and hands it to every public function of
learned_judgment.students.measure, then tries to wrap its per-skill view in an SNNStudent.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\snn_shared_measure\\r01_shared_not_measurable.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import POINTS, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import measure  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.energy import EnergyConstants, estimate_energy  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

config = load_config()
check_config = fixture_config(config)
settings = dataclasses.replace(SNNSettings.from_config(check_config), max_steps=60, evaluate_every_steps=20)
shared = SharedSNNStudent(
    skills=require(check_config, "students.snn.shared_skills"),
    feature_width=require(check_config, "students.snn.shared_feature_width"),
    settings=settings,
    rehearsal_fraction=require(check_config, "students.snn.rehearsal_fraction"),
)
spec = fixture_spec("fx_alpha")
dreams = fixture_dreams(spec, config, 300)
shared.add_skill(spec, dreams, points_per_dim=POINTS["fx_alpha"], base_seed=3, rehearsal=True)
situations = spec.sample_situations(stream_rng(3, "fx_alpha/review/situations"), 20)
constants = EnergyConstants.from_config(config)
boot = {"n_resamples": 99, "confidence_level": 0.95, "method": "percentile", "seed": 1}
print(f"shared student holds {shared.skills_held}; view layer sizes {shared.network_view('fx_alpha').layer_sizes}")


def attempt(label, function):
    try:
        function()
        print(f"  {label}: ACCEPTED")
    except Exception as error:  # noqa: BLE001
        print(f"  {label}: {type(error).__name__}: {str(error)[:150]}")


print("\nmeasure.* on the shared student:")
attempt("measure_decision_latency(shared, ...)", lambda: measure.measure_decision_latency(shared, situations, n_decisions=10))
attempt("measure_operations(shared, ...)", lambda: measure.measure_operations(shared, situations, constants, **boot))
attempt("agreement_with_dream(shared, dreams)", lambda: measure.agreement_with_dream(shared, dreams))
attempt("student_record(shared, ...)", lambda: measure.student_record(shared, spec, dreams, situations, config, seed=1))

print("\nwrapping the shared view of fx_alpha in an SNNStudent (the only type measure accepts):")
view = shared.network_view("fx_alpha")
attempt(
    "SNNStudent(network=view, encoder=fx_alpha's own encoder)",
    lambda: SNNStudent(
        skill="fx_alpha", network=view, encoder=InputEncoder.from_variables(spec.situation),
        points_per_dim=POINTS["fx_alpha"], n_candidates=view.n_outputs,
        meta={"student": {"action_names": list(spec.action_names)}},
    ),
)

print("\nwhat remains for the shared student: operation_counts + energy.estimate_energy (no latency, no firing")
print("statistics, no never/always-fire fractions, no bootstrap intervals, no provenance record):")
counts = shared.operation_counts("fx_alpha", situations)
energy = estimate_energy(counts, constants)
print(f"  layer sizes {counts.layer_sizes} first-layer MACs {counts.first_layer_macs} dense MACs {counts.dense_reference_macs}")
print(f"  keys of estimate_energy: {sorted(energy)[:6]} ... (no latency key anywhere)")
print(f"  hasattr(shared, 'digest')={hasattr(shared, 'digest')} hasattr(shared, 'meta')={hasattr(shared, 'meta')}")
