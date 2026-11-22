"""Check 6 (lead decision M1): a record for a DeployedStudent with a padded shared-layout encoding.

FIXTURE CHECK - mechanics only. This stands for one skill's view of the shared spiking network
without depending on shared_snn.py: a per-skill student is trained by the real engine step on a
physics-only dream of the FIXTURE spec fixture_specs.FixtureMeasureSpec, and its network is
padded by hand to the shared input layout of students.encoding.SharedInputEncoder

    [ the skill's own encoding (4 columns), zero-padded to the feature width | one-hot skill slot ]

with config.yaml's students.snn.shared_skill_slots slots, of which this skill takes the first and
the others stay empty, and a FIXTURE feature width (config.yaml students.snn.shared_feature_width
is null until the specs exist; the value used is printed). The first-layer weights of the padding
and slot columns are 0, so the view must decide exactly like the per-skill student. A DeployedStudent
is built by hand around the padded network with the shared encode functions and the per-skill
student's grid and situation records, and its meta is a read-only mapping (types.MappingProxyType).
student_record must accept it and give the same decisions, regrets and spikes as the per-skill
record, with the network's wider input counted in the first layer and the dense reference.
Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c6_shared_layout.py
"""

from _checks import BANNER, Checks

print(BANNER)

import copy  # noqa: E402
import math  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import FixtureMeasureSpec, make_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.train_in_dream import train_in_dream  # noqa: E402
from learned_judgment.students.encoding import InputEncoder, SharedInputEncoder  # noqa: E402
from learned_judgment.students.measure import HeldOutOverlapError, agreement_with_dream, student_record  # noqa: E402
from learned_judgment.students.snn import DeployedStudent  # noqa: E402
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

SKILL = "fixture_measure"
POINTS_PER_DIM = (7, 5)  # fixture grid
BASE_SEED = 31  # fixture run seed
N_TRAIN, N_HELDOUT, N_SITUATIONS = 1000, 400, 300  # fixture sizes
MAX_STEPS = 400  # reduced from config students.snn.max_steps for a short mechanics check
FEATURE_WIDTH = 6  # FIXTURE shared feature width (config.yaml students.snn.shared_feature_width is null)

checks = Checks()
file_config = load_config()
config = copy.deepcopy(file_config)
config["skills"][SKILL] = {"candidate_points_per_dim": list(POINTS_PER_DIM)}
config["students"]["snn"]["max_steps"] = MAX_STEPS
n_slots = require(config, "students.snn.shared_skill_slots")
print(
    f"\nconfig: config.yaml plus skills.{SKILL}.candidate_points_per_dim={list(POINTS_PER_DIM)}; students.snn.max_steps reduced to "
    f"{MAX_STEPS} (config.yaml value {file_config['students']['snn']['max_steps']}); shared_skill_slots={n_slots} (config.yaml); "
    f"FIXTURE feature width {FEATURE_WIDTH} (config.yaml shared_feature_width is {file_config['students']['snn']['shared_feature_width']})"
)
spec = FixtureMeasureSpec()
train = make_dreams(spec, None, PHYSICS_ONLY, stream_rng(BASE_SEED, f"{SKILL}/dreams/situations"), N_TRAIN, POINTS_PER_DIM, config)
heldout = make_dreams(spec, None, PHYSICS_ONLY, stream_rng(BASE_SEED, f"{SKILL}/heldout_dreams/situations"), N_HELDOUT, POINTS_PER_DIM, config)
situations = spec.sample_situations(stream_rng(BASE_SEED, f"{SKILL}/measure/situations"), N_SITUATIONS)
student = train_in_dream(spec, train, config, base_seed=BASE_SEED)
per_skill = student.deployed()

encoder = InputEncoder.from_variables(spec.situation)
shared = SharedInputEncoder(n_slots=n_slots, feature_width=FEATURE_WIDTH).with_skill(SKILL)
first = per_skill.network.hidden_weights[0]
padded_first = np.zeros((first.shape[0], shared.width))
padded_first[:, : encoder.width] = first
padded = SpikingNetwork(
    hidden_weights=(padded_first, *per_skill.network.hidden_weights[1:]),
    hidden_biases=per_skill.network.hidden_biases,
    betas=per_skill.network.betas,
    thresholds=per_skill.network.thresholds,
    readout_weight=per_skill.network.readout_weight,
    readout_bias=per_skill.network.readout_bias,
    readout_beta=per_skill.network.readout_beta,
    time_steps=per_skill.network.time_steps,
)
view = DeployedStudent(
    skill=SKILL,
    network=padded,
    encode=lambda rows: shared.encode(SKILL, encoder, rows),
    encode_one=lambda row: shared.encode_one(SKILL, encoder, row),
    situation_variables=per_skill.situation_variables,
    points_per_dim=per_skill.points_per_dim,
    n_candidates=per_skill.n_candidates,
    action_records=per_skill.action_records,
    grid_probe=per_skill.grid_probe,
    grid_sha256=per_skill.grid_sha256,
    training_situation_digests=per_skill.training_situation_digests,
    out_of_range=per_skill.out_of_range,
    meta=types.MappingProxyType(
        {
            "kind": "fixture_padded_shared_layout_view",
            "label": "FIXTURE: a per-skill network padded by hand to the shared input layout; stands for a shared view",
            "dream": student.meta["dream"],
            "settings": student.meta["settings"],
            "encoder": {"shared": shared.describe(), "skill": encoder.describe()},
        }
    ),
)
print(
    f"  shared layout: width {shared.width} = feature width {shared.feature_width} + {shared.n_slots} slots; skills {list(shared.skills)}, "
    f"free slots {shared.free_slots}; view layers {view.network.layer_sizes} vs per-skill {per_skill.network.layer_sizes}"
)
encoded = view.encode(situations[:5])
checks.expect(
    shared.free_slots == n_slots - 1 and encoded.shape == (5, FEATURE_WIDTH + n_slots)
    and bool(np.all(encoded[:, FEATURE_WIDTH] == 1.0)) and bool(np.all(encoded[:, FEATURE_WIDTH + 1 :] == 0.0))
    and bool(np.all(encoded[:, encoder.width : FEATURE_WIDTH] == 0.0)),
    f"one skill in slot 0, {n_slots - 1} slots empty; padding columns and empty slots are 0",
)

print("\n== student_record for the per-skill student and for the padded shared-layout view ==")
record_skill = student_record(student, spec, heldout, situations, config, seed=BASE_SEED)
record_view = student_record(view, spec, heldout, situations, config, seed=BASE_SEED)
skill_ops, view_ops = record_skill["operations"], record_view["operations"]
for label, record in (("per-skill", record_skill), ("shared-layout view", record_view)):
    ops, agreement = record["operations"], record["agreement_with_heldout_dream"]
    energy = ops["energy_estimate"]
    print(
        f"  MECHANICS {label}: n_inputs {record['student']['n_inputs']}, layers {ops['layer_sizes']}, agreement "
        f"{agreement['student']['agreement_with_dream_choice']:.3f}, mean spikes {ops['spikes_per_decision']['mean_total']:.2f}, first-layer "
        f"MACs {ops['first_layer_macs']}, main ESTIMATE {energy['main_estimate']['mean_estimate_pj_per_decision']:.1f} pJ, with updates "
        f"{energy['with_neuron_updates_estimate']['mean_estimate_pj_per_decision']:.1f} pJ, dense {energy['dense_reference_estimate']['estimate_pj_per_decision']:.1f} pJ, "
        f"verdict {energy['h4_energy_criterion']['verdict_from_estimates']!r}; latency full path median {record['latency']['full_path']['median_s'] * 1e6:.1f} us"
    )
checks.expect(
    record_view["student"]["n_inputs"] == FEATURE_WIDTH + n_slots and record_skill["student"]["n_inputs"] == encoder.width
    and record_view["student"]["kind"] == "fixture_padded_shared_layout_view"
    and record_view["student_meta"]["encoder"]["shared"]["n_slots"] == n_slots,
    "the view's record states its input width, its kind and the shared layout from its meta (a read-only mapping)",
)
checks.expect(
    record_view["student"]["situation_encoding"] == record_skill["student"]["situation_encoding"] == encoder.describe(),
    "both records state the skill's own situation encoding",
)
checks.expect(
    record_view["agreement_with_heldout_dream"]["per_row"]["student_choice"] == record_skill["agreement_with_heldout_dream"]["per_row"]["student_choice"],
    "the view decides exactly like the per-skill student on every held-out row",
)
checks.expect(
    view_ops["per_decision"]["spikes_per_layer"] == skill_ops["per_decision"]["spikes_per_layer"]
    and record_view["agreement_with_heldout_dream"]["student"] == record_skill["agreement_with_heldout_dream"]["student"],
    "same spikes per decision and the same in-dream agreement and regret",
)
diff_view, diff_skill = record_view["dream_regret_difference_student_minus_physics"], record_skill["dream_regret_difference_student_minus_physics"]
checks.expect(
    all(diff_view[key] == diff_skill[key] for key in ("estimate", "low", "high", "n_distinct_situations", "method_used")),
    "the same regret-difference interval (same regrets, same seed stream)",
)
extra_columns = FEATURE_WIDTH + n_slots - encoder.width
hidden = view.network.hidden_sizes[0]
e_mac = record_view["config"]["values_read"]["students.snn.energy.e_mac_pj"]
view_energy, skill_energy = view_ops["energy_estimate"], skill_ops["energy_estimate"]
checks.expect(
    view_ops["first_layer_macs"] - skill_ops["first_layer_macs"] == extra_columns * hidden
    and view_ops["dense_reference_macs"] - skill_ops["dense_reference_macs"] == extra_columns * hidden
    and math.isclose(
        view_energy["main_estimate"]["mean_estimate_pj_per_decision"] - skill_energy["main_estimate"]["mean_estimate_pj_per_decision"],
        e_mac * extra_columns * hidden, rel_tol=1e-9,
    )
    and math.isclose(
        view_energy["dense_reference_estimate"]["estimate_pj_per_decision"] - skill_energy["dense_reference_estimate"]["estimate_pj_per_decision"],
        e_mac * extra_columns * hidden, rel_tol=1e-9,
    ),
    f"the {extra_columns} extra input columns are counted in the first layer and in the dense reference ({extra_columns} x {hidden} MACs each)",
)
checks.expect(
    record_view["latency"]["network_sha256"] == view.digest != per_skill.digest
    and record_view["latency"]["h3_timing"] == "full_path" and record_view["latency"]["full_path_actions_not_the_looked_up_candidate"] == 0,
    "the view's latency was timed on its own (padded) network through the full path",
)

print("\n== the view is held to the same held-out rule ==")
checks.raises(
    "student_record for the view with the training dream as held out",
    HeldOutOverlapError,
    lambda: student_record(view, spec, train, situations, config, seed=BASE_SEED),
    f"{N_TRAIN} of {N_TRAIN} dream rows",
)
checks.expect(agreement_with_dream(view, heldout)["held_out"] is True, "agreement_with_dream accepts the view on fresh dreams")
checks.finish()
