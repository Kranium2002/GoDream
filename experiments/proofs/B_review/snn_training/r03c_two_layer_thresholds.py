"""R03c: R03 with moderate thresholds and two hidden layers (hidden_sizes [64, 64], a config option
DECISIONS/config list for comparison on dev). Does a hidden layer go silent while train_in_dream
still returns a student? FIXTURE dream, 300 steps, everything else config values.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import _review_common  # noqa: E402,F401
from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402

config = load_config()
base = dataclasses.replace(SNNSettings.from_config(config), max_steps=300, evaluate_every_steps=50, hidden_sizes=(64, 64))
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 1000)
situations = spec.sample_situations(stream_rng(4, "review/r03"), 500)
for threshold in (1.0, 2.0, 3.0, 5.0):
    settings = dataclasses.replace(base, threshold=threshold)
    try:
        student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=9)
    except TrainingError as error:
        print(f"threshold {threshold}: TrainingError (refused): {str(error)[:160]}")
        continue
    training = student.meta["training"]
    decisions = student.decide_index(situations)
    print(
        f"threshold {threshold}: STUDENT RETURNED; best_step={training['best_step']} val KL "
        f"{training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}; initial firing "
        f"{[round(v, 4) for v in training['initial_firing_rate_per_layer']]} final {[round(v, 4) for v in training['firing_rate_per_layer']]}; "
        f"silent_layers={training['silent_layers']}; distinct decisions on 500 situations = {len(set(decisions.tolist()))}; "
        f"agreement {student.meta['validation_in_dream']['agreement_with_dream_choice']:.3f} zero-regret "
        f"{student.meta['validation_in_dream']['fraction_with_zero_dream_regret']:.3f}"
    )
