"""R03: training "succeeds" for a student whose decisions do not depend on the situation.

fit() raises TrainingError only when the restored validation loss is not below its value at
initialisation. A network whose hidden layer emits no spike still lowers that loss by moving its
readout bias towards the marginal target distribution, so train_in_dream returns a student that
picks the same candidate for every situation; silent_layers is recorded in meta but nothing
refuses. Two ways to get there on the FIXTURE dream (real train_in_dream, 300 steps):
  * spike_rate_penalty > 0 (the energy knob that H4 invites tuning);
  * a threshold the input current never reaches.
For each run: was a student returned, silent layers, distinct decisions on 500 new situations,
in-dream validation agreement, and the energy-relevant spike count per decision.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

import _review_common  # noqa: E402,F401
from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402

config = load_config()
base = dataclasses.replace(SNNSettings.from_config(config), max_steps=300, evaluate_every_steps=50)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 1000)
situations = spec.sample_situations(stream_rng(4, "review/r03"), 500)

for label, settings in (
    ("control (config values, penalty 0)", base),
    ("spike_rate_penalty 5", dataclasses.replace(base, spike_rate_penalty=5.0)),
    ("spike_rate_penalty 50", dataclasses.replace(base, spike_rate_penalty=50.0)),
    ("threshold 1000", dataclasses.replace(base, threshold=1000.0)),
):
    try:
        student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=9)
    except TrainingError as error:
        print(f"{label}: TrainingError (refused): {str(error)[:160]}")
        continue
    training = student.meta["training"]
    decisions = student.decide_index(situations)
    counts = student.operation_counts(situations)
    print(
        f"{label}: STUDENT RETURNED; best_step={training['best_step']} "
        f"val loss {training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}; "
        f"silent_layers={training['silent_layers']} firing_rate={[round(v, 4) for v in training['firing_rate_per_layer']]}; "
        f"distinct decisions on 500 situations = {len(set(decisions.tolist()))}; "
        f"agreement in dream = {student.meta['validation_in_dream']['agreement_with_dream_choice']:.3f}; "
        f"mean spikes/decision = {float(counts.spikes_per_decision.sum(axis=1).mean()):.2f}"
    )
