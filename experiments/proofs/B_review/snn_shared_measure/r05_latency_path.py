"""R05: what does measure_decision_latency time, compared with a decision that yields a command?

measure_decision_latency times student.decide_index(situation[None, :]) and states that the action
lookup spec.candidate_actions is not timed (_DECISION_PATH). A decision usable by the robot is
SNNStudent.decide(spec, situation): spec check, decide_index, spec.candidate_actions(situation, grid),
action_in_bounds. The teacher's decision (engine.decide.decide_batch) builds the candidate grid inside
its timed path. This times both student paths on a HAND-BUILT network of the config size (4 encoded
inputs, 64 hidden, 35 outputs, T = 10; random FIXTURE weights) with fx_alpha's (7, 5) grid, numpy only.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\snn_shared_measure\\r05_latency_path.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import POINTS, fixture_spec  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.eval.harness import measure_latency  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.measure import measure_decision_latency  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

spec = fixture_spec("fx_alpha")
encoder = InputEncoder.from_variables(spec.situation)
rng = np.random.default_rng(5)  # fixture weights
network = SpikingNetwork(
    hidden_weights=(rng.normal(0, 0.8, (64, encoder.width)),), hidden_biases=(rng.normal(0.5, 0.3, 64),),
    betas=(0.9,), thresholds=(1.0,), readout_weight=rng.normal(0, 1, (35, 64)), readout_bias=np.zeros(35),
    readout_beta=0.9, time_steps=10,
)
student = SNNStudent(
    skill="fx_alpha", network=network, encoder=encoder, points_per_dim=POINTS["fx_alpha"], n_candidates=35,
    meta={"kind": "review_hand_network", "student": {"action_names": list(spec.action_names)}},
)
situations = spec.sample_situations(stream_rng(5, "fx_alpha/review/latency"), 200)
N = 3000
for repeat in range(2):
    reported = measure_decision_latency(student, situations, n_decisions=N)
    full = measure_latency(lambda s: student.decide(spec, s[None, :]), situations, N)
    grid_only = measure_latency(lambda s: spec.candidate_actions(s, POINTS["fx_alpha"]), situations, N)
    print(
        f"repeat {repeat}: measure_decision_latency median {reported['median_s'] * 1e6:.1f} us p95 {reported['p95_s'] * 1e6:.1f} us | "
        f"decide(spec, situation) -> command median {full['median_s'] * 1e6:.1f} us p95 {full['p95_s'] * 1e6:.1f} us "
        f"(x{full['median_s'] / reported['median_s']:.2f}) | candidate_actions alone median {grid_only['median_s'] * 1e6:.1f} us"
    )
print(f"torch imported: {'torch' in sys.modules}")
