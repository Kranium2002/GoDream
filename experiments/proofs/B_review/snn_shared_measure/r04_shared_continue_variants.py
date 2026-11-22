"""R04: shared student in the settings the existing checks do not cover.

B_snn_shared_offline/c1 checks save/load/continue with the config settings (one hidden layer of 64,
fixed beta and threshold, no spike penalty, feature width above every skill's encoding). Here:
two hidden layers (16, 12), learn_beta=True, learn_threshold=True, spike_rate_penalty=0.05, and
feature width 5 = fx_beta's encoded width (no padding for that skill). Questions:
  1. after save -> load (in-process) -> add a third skill, are all views bitwise identical to the
     never-saved student's (same seeds)?
  2. are the rehearsal inputs/targets of an earlier skill bitwise identical when it is first trained,
     when it is rehearsed, after load, and are the inputs equal to the decision-time encoding?
  3. do the batches actually contain the stated shares (instrumented _batch_counts)?
  4. hand check of operation counts for two hidden layers.
The script wraps snn_train.fit and snn_train._batch_counts only to record their arguments/results; it
calls the real functions unchanged.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\snn_shared_measure\\r04_shared_continue_variants.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import POINTS, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

config = load_config()
settings = dataclasses.replace(
    SNNSettings.from_config(fixture_config(config)),
    hidden_sizes=(16, 12), learn_beta=True, learn_threshold=True, spike_rate_penalty=0.05,
    max_steps=120, evaluate_every_steps=20,
)
print(f"settings: {settings.describe()}")
SKILLS = ["fx_alpha", "fx_beta", "fx_gamma"]
specs = {name: fixture_spec(name) for name in SKILLS}
dreams = {name: fixture_dreams(specs[name], config, 400) for name in SKILLS}

captured: list[dict] = []
real_fit, real_counts = snn_train.fit, snn_train._batch_counts


def recording_fit(model, tasks, *, validation, sampling, settings, seed):  # noqa: ANN001
    entry = {"tasks": {h: (x.copy(), q.copy()) for h, (x, q) in tasks.items()}, "sampling": dict(sampling), "counts": []}
    captured.append(entry)
    return real_fit(model, tasks, validation=validation, sampling=sampling, settings=settings, seed=seed)


def recording_counts(weights, batch_size, generator):  # noqa: ANN001
    counts = real_counts(weights, batch_size, generator)
    captured[-1]["counts"].append(counts.copy())
    return counts


snn_train.fit, snn_train._batch_counts = recording_fit, recording_counts


def new_student() -> SharedSNNStudent:
    return SharedSNNStudent(skills=SKILLS, feature_width=5, settings=settings, rehearsal_fraction=0.5)


def add(student: SharedSNNStudent, name: str) -> dict:
    return student.add_skill(specs[name], dreams[name], points_per_dim=POINTS[name], base_seed=13, rehearsal=True)


scratch = Path(tempfile.mkdtemp(prefix="review_shared_r04_"))
try:
    memory = new_student()
    add(memory, "fx_alpha")  # captured[0]
    add(memory, "fx_beta")  # captured[1]
    memory.save(scratch / "student")
    loaded = SharedSNNStudent.load(scratch / "student")
    record_memory = add(memory, "fx_gamma")  # captured[2]
    record_loaded = add(loaded, "fx_gamma")  # captured[3]
finally:
    snn_train.fit, snn_train._batch_counts = real_fit, real_counts
    shutil.rmtree(scratch, ignore_errors=True)

print("\n1. continue after load")
print(f"   steps_run memory {record_memory['steps_run']} loaded {record_loaded['steps_run']}; best_step "
      f"{record_memory['training']['best_step']} / {record_loaded['training']['best_step']}")
same_digests = memory.network_digests() == loaded.network_digests()
print(f"   all three views identical (network digests) after adding fx_gamma: {same_digests}")
strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_seconds", "training")}  # noqa: E731
print(f"   records identical except wall time / training wall time: {strip(record_memory) == strip(record_loaded)}")
view = memory.network_view("fx_gamma")
print(f"   learned betas {view.betas} thresholds {view.thresholds} (moved from {settings.beta}/{settings.threshold})")

print("\n2. inputs and targets of an earlier skill")
for skill, uses in (("fx_alpha", [(0, "first trained"), (1, "rehearsed at add fx_beta"), (2, "rehearsed at add fx_gamma (memory)"), (3, "rehearsed at add fx_gamma (loaded)")]),
                    ("fx_beta", [(1, "first trained"), (2, "rehearsed (memory)"), (3, "rehearsed (loaded)")])):
    x0, q0 = captured[uses[0][0]]["tasks"][skill]
    for index, label in uses[1:]:
        x, q = captured[index]["tasks"][skill]
        print(f"   {skill} {label}: inputs identical {np.array_equal(x, x0)}, targets identical {np.array_equal(q, q0)}")
    held = memory._held[skill]  # decision-time encoding of the same training rows
    decision_inputs = memory._encode_situations(held, held.dreams.situations[held.train_rows])
    print(f"   {skill}: decision-time encoding of the training rows equals the training inputs: {np.array_equal(decision_inputs, x0)}")

print("\n3. batch composition (instrumented _batch_counts; batch_size", settings.batch_size, ")")
for index, label in ((1, "add fx_beta"), (2, "add fx_gamma")):
    entry = captured[index]
    counts = np.array(entry["counts"])
    heads = list(entry["tasks"])
    shares = counts.sum(axis=0) / counts.sum()
    print(f"   {label}: heads {heads} stated {[round(entry['sampling'][h], 4) for h in heads]} realised {[round(float(s), 4) for s in shares]} "
          f"over {counts.shape[0]} batches; min/max rows per batch {counts.min(axis=0).tolist()}/{counts.max(axis=0).tolist()}")

print("\n4. two hidden layers, hand check of the operation counts (fx_gamma view)")
situations = specs["fx_gamma"].sample_situations(stream_rng(13, "fx_gamma/review/situations"), 40)
counts = memory.operation_counts("fx_gamma", situations)
W, H1, H2, C, T = view.n_inputs, 16, 12, view.n_outputs, view.time_steps
_, trains = view.forward_with_spike_trains(memory._encode_situations(memory._held["fx_gamma"], situations))
per_decision = np.stack([train.sum(axis=(0, 2)) for train in trains], axis=1).astype(np.float64)  # (n, L)
print(f"   layer sizes {counts.layer_sizes} fan-out {counts.fan_out} (hand ({H2}, {C}))")
print(f"   spikes per decision equal spike-train sums: {np.array_equal(per_decision, counts.spikes_per_decision)}")
print(f"   synaptic ops = s1*{H2} + s2*{C}: {np.array_equal(counts.synaptic_ops_per_decision, per_decision @ np.array([H2, C], float))}")
print(f"   first-layer MACs {counts.first_layer_macs} (hand W*H1 = {W * H1}); neuron updates {counts.neuron_updates} (hand {(H1 + H2 + C) * T}); "
      f"dense MACs {counts.dense_reference_macs} (hand {W * H1 + H1 * H2 + H2 * C})")
