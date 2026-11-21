"""R03b: R03 with learn_threshold / learn_beta and a spike-rate penalty (config knobs only).

Same FIXTURE dream and question as R03: does train_in_dream return a student whose decisions are
(nearly) independent of the situation, and how does the in-dream validation KL of the restored
network compare with the KL of the best CONSTANT prediction (the mean training target, which a
silent network can reach through its readout bias alone)?
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
from learned_judgment.students.targets import score_spread, soft_targets  # noqa: E402

config = load_config()
base = dataclasses.replace(SNNSettings.from_config(config), max_steps=300, evaluate_every_steps=50)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 1000)
situations = spec.sample_situations(stream_rng(4, "review/r03"), 500)

# best constant prediction on the same split train_in_dream uses
n = dreams.n_rows
n_val = max(1, int(round(base.validation_fraction * n)))
order = stream_rng(9, f"{spec.name}/snn/validation_split").permutation(n)
val_rows, train_rows = np.sort(order[:n_val]), np.sort(order[n_val:])
targets = soft_targets(dreams.scores, temperature=base.target_temperature, spread=score_spread(dreams.scores[train_rows]))
constant = targets[train_rows].mean(axis=0)
q = targets[val_rows]
constant_kl = float(np.mean(np.sum(np.where(q > 0, q * (np.log(np.where(q > 0, q, 1.0)) - np.log(constant)), 0.0), axis=1)))
print(f"validation KL of the best constant prediction (mean training target): {constant_kl:.4f}")

for label, settings in (
    ("learn_threshold, penalty 50", dataclasses.replace(base, learn_threshold=True, spike_rate_penalty=50.0)),
    ("learn_threshold+learn_beta, penalty 200", dataclasses.replace(base, learn_threshold=True, learn_beta=True, spike_rate_penalty=200.0)),
    ("threshold 1000", dataclasses.replace(base, threshold=1000.0)),
):
    try:
        student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=9)
    except TrainingError as error:
        print(f"{label}: TrainingError (refused): {str(error)[:160]}")
        continue
    training = student.meta["training"]
    decisions = student.decide_index(situations)
    print(
        f"{label}: STUDENT RETURNED; best_step={training['best_step']} val KL {training['initial_validation_loss']:.4f} -> "
        f"{training['best_validation_loss']:.4f} (constant predictor {constant_kl:.4f}); "
        f"silent_layers={training['silent_layers']} firing_rate={[round(v, 4) for v in training['firing_rate_per_layer']]}; "
        f"thresholds={[round(t, 3) for t in student.network.thresholds]} betas={[round(b, 3) for b in student.network.betas]}; "
        f"distinct decisions on 500 situations = {len(set(decisions.tolist()))}; "
        f"agreement in dream = {student.meta['validation_in_dream']['agreement_with_dream_choice']:.3f}"
    )
