"""Check 19: the per-evaluation observer of fit() only reads.

FIXTURE CHECK - mechanics only. One fixture spec, 400 fixture dreams. Checked: training with the
observer gives bit-for-bit the same network and the same report curves as training without it;
the observer is called at step 0 and after every validation evaluation with the documented keys
and with the values the report holds; an exception raised by the observer propagates and leaves
the torch thread and determinism settings as they were. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c19_on_evaluation.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import torch  # noqa: E402

from fixture_snn_spec import Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402
from learned_judgment.students.targets import score_spread, soft_targets  # noqa: E402

GRID = (7, 5)  # fixture grid: 35 candidates
SEED = 47  # fixture seed
KEYS = {
    "step", "max_steps", "train_loss", "validation_loss", "validation_loss_per_head", "best_validation_loss",
    "best_step", "firing_rate_per_layer",
}

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=300, evaluate_every_steps=50)
spec = FixtureSnnSpec(name="fx_observed")
dreams = fixture_dreams(spec, config, 400, points_per_dim=GRID)
encoder = InputEncoder.from_variables(spec.situation)
encoded = encoder.encode(dreams.situations)
targets = soft_targets(
    dreams.scores, temperature=settings.target_temperature, target_spread="global_median",
    spread=score_spread(dreams.scores[:360]),
)
tasks = {spec.name: (encoded[:360], targets[:360])}
validation = {spec.name: (encoded[360:], targets[360:])}
heads = {spec.name: targets.shape[1]}


def train(observer):  # noqa: ANN001, ANN201
    model = snn_train.build_model(encoder.width, heads, settings, seed=SEED)
    report = snn_train.fit(
        model, tasks, validation=validation, sampling={spec.name: 1.0}, settings=settings, seed=SEED,
        on_evaluation=observer,
    )
    return snn_train.export_network(model, spec.name), report


events: list[dict] = []
plain_network, plain_report = train(None)
observed_network, observed_report = train(events.append)
print(f"\nsteps_run {plain_report.steps_run}; best_step {plain_report.best_step}; observer calls {len(events)}")

checks.expect(observed_network.digest() == plain_network.digest(), "the observed training exports the same network, bit for bit")
checks.expect(
    observed_report.validation_total_curve == plain_report.validation_total_curve
    and observed_report.train_loss_curve == plain_report.train_loss_curve
    and observed_report.best_step == plain_report.best_step,
    "the observed training reports the same curves and best step",
)
checks.expect(all(set(event) == KEYS for event in events), "every event holds exactly the documented keys")
checks.expect(
    [event["step"] for event in events] == [step for step, _ in observed_report.validation_total_curve],
    "one event at step 0 and one per validation evaluation, in order",
)
checks.expect(events[0]["train_loss"] is None, "the event at step 0 has no training loss")
checks.expect(
    [event["validation_loss"] for event in events] == [total for _, total in observed_report.validation_total_curve],
    "each event's validation loss is the report's value at that step",
)
checks.expect(
    [event["train_loss"] for event in events[1:]] == [loss for _, loss in observed_report.train_loss_curve],
    "each event's training loss is the report's value at that step",
)
checks.expect(
    events[-1]["best_step"] == observed_report.best_step
    and events[-1]["best_validation_loss"] == observed_report.best_validation_loss,
    "the last event names the restored step and its loss",
)
checks.expect(
    all(len(event["firing_rate_per_layer"]) == len(settings.hidden_sizes)
        and all(0.0 <= rate <= 1.0 for rate in event["firing_rate_per_layer"]) for event in events),
    "every event holds one firing rate in [0, 1] per hidden layer",
)
checks.expect(all(event["max_steps"] == settings.max_steps for event in events), "every event names max_steps")


class Halt(RuntimeError):
    """Raised by the observer below to stop the training."""


def halting(event: dict) -> None:
    if event["step"] >= 100:
        raise Halt(f"stopped by the observer at step {event['step']}")


threads_before = torch.get_num_threads()
deterministic_before = torch.are_deterministic_algorithms_enabled()
checks.raises("an exception raised by the observer propagates", Halt, lambda: train(halting), "stopped by the observer at step 100")
checks.expect(
    torch.get_num_threads() == threads_before and torch.are_deterministic_algorithms_enabled() == deterministic_before,
    "the torch thread count and determinism flag are restored after the observer stopped the training",
)

checks.finish()
