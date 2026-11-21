"""B2 research probe (MECHANICS ONLY, synthetic inputs; no simulator, no dream, no TabPFN; not a result).

Questions: (1) does today's snn_train.fit train a two-head network (threat: 2 outputs, action: 4 outputs) on the SAME
rows of 1,152 inputs in [0, 1] without code changes; (2) wall-clock per 1,000 optimizer steps at D = 1,152 with the
configured students.snn settings; (3) does concatenating the two exported heads' readouts into ONE numpy
SpikingNetwork with 6 outputs reproduce each head's logits (same trunk, same readout_beta); (4) numpy per-decision
latency and operation counts at D = 1,152.

Synthetic data: inputs uniform [0, 1]; each head's soft target is a softmax of a few fixed input pixels (threat:
4 * (x0 - x1, x1 - x0); action: 4 * (x2..x5 - 0.5)), so the heads are learnable. Edits no project file.

    .venv/Scripts/python.exe experiments/proofs/B2_research/snn_multihead_probe.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from learned_judgment.env import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork, count_operations  # noqa: E402

D, N, N_VAL = 8 * 12 * 12, 2000, 200


def main() -> None:
    config = load_config()
    base = snn_train.SNNSettings.from_config(config)
    settings = dataclasses.replace(base, max_steps=1000, patience_evaluations=1000)
    print("settings:", json.dumps(settings.describe()))
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(N + N_VAL, D))

    def soft(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    q_threat = soft(4.0 * np.column_stack([x[:, 0] - x[:, 1], x[:, 1] - x[:, 0]]))
    q_action = soft(4.0 * (x[:, 2:6] - 0.5))
    tr, va = slice(0, N), slice(N, N + N_VAL)
    model = snn_train.build_model(D, {"threat": 2, "action": 4}, settings, seed=1)
    t0 = time.perf_counter()
    report = snn_train.fit(model, {"threat": (x[tr], q_threat[tr]), "action": (x[tr], q_action[tr])},
                           validation={"threat": (x[va], q_threat[va]), "action": (x[va], q_action[va])},
                           sampling={"threat": 0.5, "action": 0.5}, settings=settings, seed=2)
    wall = time.perf_counter() - t0
    print("two-head fit on the same rows: steps_run", report.steps_run, "best_step", report.best_step,
          "| wall_seconds", round(report.wall_seconds, 2), "| per 1000 steps", round(report.wall_seconds / report.steps_run * 1000, 2),
          "| outer wall", round(wall, 2))
    print("validation KL per head (initial -> best):", report.validation_loss_curve[0][1], "->",
          report.best_validation_loss_per_head, "| constant predictor:", report.constant_predictor_validation_loss)
    print("firing rate per layer:", report.firing_rate_per_layer)

    # single-head reference timing, same D and settings
    model1 = snn_train.build_model(D, {"action": 4}, settings, seed=1)
    r1 = snn_train.fit(model1, {"action": (x[tr], q_action[tr])}, validation={"action": (x[va], q_action[va])},
                       sampling={"action": 1.0}, settings=settings, seed=2)
    print("one-head fit: steps_run", r1.steps_run, "| per 1000 steps", round(r1.wall_seconds / r1.steps_run * 1000, 2))

    nets = {h: snn_train.export_network(model, h) for h in ("threat", "action")}
    for h, net in nets.items():
        parity = snn_train.check_parity(model, h, net, x, logit_tolerance=settings.parity_logit_tolerance)
        print(f"check_parity head {h!r}: max |logit diff| {parity['max_abs_logit_difference']:.3g}, "
              f"spike mismatches {parity['spike_train_mismatches_per_layer']}, decisions differing {parity['decisions_differing']}")
    t, a = nets["threat"], nets["action"]
    combined = SpikingNetwork(hidden_weights=a.hidden_weights, hidden_biases=a.hidden_biases, betas=a.betas,
                              thresholds=a.thresholds,
                              readout_weight=np.vstack([t.readout_weight, a.readout_weight]),
                              readout_bias=np.concatenate([t.readout_bias, a.readout_bias]),
                              readout_beta=a.readout_beta, time_steps=a.time_steps)
    same_trunk = all(np.array_equal(p, q) for p, q in zip(t.hidden_weights, a.hidden_weights))
    lc = combined.forward(x)
    print("combined network layer_sizes", combined.layer_sizes, "| trunks identical:", same_trunk,
          "| readout_beta threat/action:", t.readout_beta, a.readout_beta)
    print("max |combined[:, :2] - threat|", float(np.abs(lc[:, :2] - t.forward(x)).max()),
          "| max |combined[:, 2:] - action|", float(np.abs(lc[:, 2:] - a.forward(x)).max()))
    wrong = int(np.sum(combined.decide(x) != (2 + a.decide(x))))
    print("argmax over all 6 outputs vs 2 + action argmax: differs on", wrong, "of", x.shape[0], "rows (any agreement depends on the two heads' logit scales; one argmax over 6 outputs mixes the heads and is not a defined decision)")

    times = []
    for i in range(1000):
        row = x[i % x.shape[0]]
        s = time.perf_counter_ns()
        combined._simulate(row[None, :], None, None)
        times.append(time.perf_counter_ns() - s)
    times = np.array(times) / 1e6
    print("numpy forward of one row (D=1152, H=64, C=6, T=10): median ms", round(float(np.median(times)), 4),
          "p95 ms", round(float(np.quantile(times, 0.95)), 4))
    ops = count_operations(combined, x[:500])
    print("operation counts: layer_sizes", ops.layer_sizes, "first_layer_macs", ops.first_layer_macs,
          "neuron_updates", ops.neuron_updates, "dense_reference_macs", ops.dense_reference_macs,
          "mean spikes per decision", float(ops.spikes_per_decision.mean()), "mean synaptic ops", float(ops.synaptic_ops_per_decision.mean()))


if __name__ == "__main__":
    main()
