"""Check 8 worker: the OLD measure.py (committed version) accepts each defect the lead's decisions fix.

FIXTURE CHECK - mechanics only. Run ONLY by c8_old_behaviour.py inside a temporary overlay whose
learned_judgment/students/measure.py is the committed (HEAD) version; refuses to run otherwise.
Every scenario goes through the old API and prints "OLD <decision>: DEFECT ..." when the old code
shows the defect; an assertion holds when it does. A FIXTURE per-skill student is trained by the
real engine step (config.yaml with skills.fixture_measure.candidate_points_per_dim added and
students.snn.max_steps reduced; printed). Writes no file.
"""

from _checks import BANNER, Checks

print(BANNER)

import copy  # noqa: E402
import inspect  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import ROOT3, FixtureHandSpec, FixtureMeasureSpec, concat_dreams, dream_rows, hand_deployed, make_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.students import measure  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.energy import EnergyConstants  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

if hasattr(measure, "HeldOutOverlapError"):
    raise SystemExit("c8_old_probe.py must run on the committed (old) measure.py; start it through c8_old_behaviour.py")

SKILL = "fixture_measure"
POINTS_PER_DIM = (7, 5)  # fixture grid
BASE_SEED = 23  # fixture run seed (as c4)
MAX_STEPS = 400  # reduced from config students.snn.max_steps for a short mechanics check
checks = Checks()
file_config = load_config()
config = copy.deepcopy(file_config)
config["skills"][SKILL] = {"candidate_points_per_dim": list(POINTS_PER_DIM)}
config["students"]["snn"]["max_steps"] = MAX_STEPS
constants = EnergyConstants.from_config(config)
print(f"\nold measure.py: {measure.__file__}; config: config.yaml plus skills.{SKILL}.candidate_points_per_dim, students.snn.max_steps={MAX_STEPS}")


def outcome(function):  # returns ("ok", result) or ("error", "Type: message")
    try:
        return "ok", function()
    except Exception as error:  # noqa: BLE001
        return "error", f"{type(error).__name__}: {str(error)[:160]}"


print("\n== M1 / M7: a hand-built DeployedStudent (the form a shared network's view takes) ==")
network = SpikingNetwork(
    hidden_weights=(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),), hidden_biases=(np.array([0.0, 0.0, 2.5]),),
    betas=(0.5,), thresholds=(1.0,), readout_weight=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    readout_bias=np.zeros(2), readout_beta=0.5, time_steps=3,
)
hand_spec = FixtureHandSpec("fixture_hand", 2)
hand = hand_deployed(network, hand_spec, (2,), np.random.default_rng(7).uniform(-ROOT3, ROOT3, (20, 2)))
status, detail = outcome(lambda: measure.measure_operations(hand, np.array([[1.5, 0.0], [0.8, 0.0]]), constants, n_resamples=99, confidence_level=0.95, method="BCa", seed=1))
print(f"  old measure_operations(DeployedStudent): {detail if status == 'error' else 'accepted'}")
checks.expect(status == "error" and "SNNStudent" in detail, "OLD M1: DEFECT the old functions refuse a DeployedStudent, so a shared network's view cannot be measured")
status, detail = outcome(lambda: SNNStudent(
    skill="fixture_hand", network=network, encoder=InputEncoder.from_variables(hand_spec.situation), points_per_dim=(2,), n_candidates=2,
    meta={"kind": "fixture_hand_built_network", "student": {"action_names": ["fx_choice"]}},
))
print(f"  the old c2's hand-built SNNStudent: {detail if status == 'error' else 'constructed'}")
checks.expect(status == "error", "OLD M7: DEFECT the old hand-network check cannot build its student, and the old functions take no DeployedStudent")

print("\n== a trained FIXTURE student for the dream scenarios ==")
spec = FixtureMeasureSpec()
train = make_dreams(spec, None, PHYSICS_ONLY, stream_rng(BASE_SEED, f"{SKILL}/dreams/situations"), 1000, POINTS_PER_DIM, config)
fresh = make_dreams(spec, None, PHYSICS_ONLY, stream_rng(BASE_SEED, f"{SKILL}/heldout_dreams/situations"), 500, POINTS_PER_DIM, config)
situations = spec.sample_situations(stream_rng(BASE_SEED, f"{SKILL}/measure/situations"), 200)
student = train_in_dream(spec, train, config, base_seed=BASE_SEED)
record_fresh = measure.student_record(student, spec, fresh, situations, config, seed=BASE_SEED)

print("\n== M2: subsets and supersets of the training dream passed as held out ==")
for name, dreams in (
    ("subset (first 150 training rows)", dream_rows(train, np.arange(150))),
    ("superset (training dream + 500 fresh rows)", concat_dreams(train, fresh)),
    ("fresh rows + one training row", concat_dreams(fresh, dream_rows(train, [7]))),
):
    status, result = outcome(lambda dreams=dreams: measure.student_record(student, spec, dreams, situations, config, seed=BASE_SEED))
    if status == "ok":
        print(
            f"  old student_record accepted the {name} as held out: agreement {result['agreement_with_heldout_dream']['student']['agreement_with_dream_choice']:.3f} "
            f"(fresh rows: {record_fresh['agreement_with_heldout_dream']['student']['agreement_with_dream_choice']:.3f})"
        )
    checks.expect(status == "ok", f"OLD M2: DEFECT the {name} is accepted as held out, with no overlap count")

print("\n== M3: the two energy estimates ==")
operations = record_fresh["operations"]
with_interval = sorted(key for key, value in operations.items() if isinstance(value, dict) and "low" in value)
print(f"  old operations keys holding an interval: {with_interval}; energy keys: {sorted(operations['energy_estimate'])}")
checks.expect(
    with_interval == ["energy_main_pj_interval"] and "h4_energy_criterion" not in str(operations)
    and all("below_break_even" in item for item in operations["break_even"]["per_layer"]),
    "OLD M3: DEFECT only the main estimate has an interval and a break-even; no H4 verdict rule",
)

print("\n== M4: the same held-out dream twice over ==")
record_twice = measure.student_record(student, spec, concat_dreams(fresh, fresh), situations, config, seed=BASE_SEED)
once, twice = record_fresh["dream_regret_difference_student_minus_physics"], record_twice["dream_regret_difference_student_minus_physics"]
heldout_block = record_twice["agreement_with_heldout_dream"]["dream"]
print(
    f"  old interval once [{once['low']:.5f}, {once['high']:.5f}] (n={once['n_situations']}), twice [{twice['low']:.5f}, {twice['high']:.5f}] "
    f"(n={twice['n_situations']}); held-out dream block keys: {sorted(heldout_block)}"
)
checks.expect(
    (twice["high"] - twice["low"]) < (once["high"] - once["low"]) and "n_distinct_situations" not in heldout_block
    and "n_distinct_situations" not in twice,
    "OLD M4: DEFECT duplicated rows narrow the interval, and the held-out dream and the interval state no distinct-situation count",
)

print("\n== M5: what the old latency times ==")
latency = record_fresh["latency"]
print(f"  old decision path: {latency['decision_path']}")
checks.expect(
    "full_path" not in latency and "not timed" in latency["decision_path"]
    and "spec" not in inspect.signature(measure.measure_decision_latency).parameters,
    "OLD M5: DEFECT the timed path leaves out the candidate lookup (no spec, no binding, no full-path timing)",
)

print("\n== M6: the old per-neuron statistics ==")
source = inspect.getsource(measure)
checks.expect(
    "_spikes_per_neuron" in source and "for step in range(" in source,
    "OLD M6: DEFECT measure.py keeps its own copy of the network loop (_spikes_per_neuron)",
)
checks.finish()
