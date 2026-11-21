"""Check (a): a shared student holding ONE skill, every other slot empty, then a second skill.

FIXTURE CHECK - mechanics only. The student is created from the check config (config.yaml in memory
with the fixture shared_feature_width and max_steps reduced, printed below), so the number of slots
is config.yaml's students.snn.shared_skill_slots. fx_alpha is added alone: it must train, export,
pass parity on every row of its dream, decide in-bounds grid actions through the batched and the
per-decision paths, decide identically through deployed(skill) and through a hand-built shared
encoding, and count operations over the full padded input (feature columns + every slot). Then
fx_beta takes the next slot, and the stage-0 view of fx_alpha still gives fx_alpha's original
decisions. Old behaviour (committed shared_snn.py): the network had one input per NAMED skill and no
empty slots, no deployed() and no stages (c7_old_behaviour.py shows it). Every number printed is a
MECHANICS number on a fixture dream, not evidence about the robot, the simulator, TabPFN or a model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c2_one_skill.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402

from fixture_shared_specs import FIXTURE_FEATURE_WIDTH, POINTS, Checks, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn import DeployedStudent, situation_digests  # noqa: E402
from learned_judgment.students.targets import dream_regret  # noqa: E402

MAX_STEPS = 300  # reduced from config students.snn.max_steps for a short mechanics check
BASE_SEED = 7  # fixture run seed
CHECK_SITUATIONS = 300

checks = Checks()
config = load_config()
check_config = fixture_config(config)
check_config["students"]["snn"]["max_steps"] = MAX_STEPS
slots_config = require(config, "students.snn.shared_skill_slots")
print(
    f"\ncheck config: config.yaml with students.snn.shared_feature_width={FIXTURE_FEATURE_WIDTH} (fixture; config.yaml "
    f"{config['students']['snn']['shared_feature_width']!r}) and students.snn.max_steps={MAX_STEPS} (config.yaml "
    f"{config['students']['snn']['max_steps']}); every other value from config.yaml"
)

print("\n== create(config): an empty student with reserved slots ==")
student = SharedSNNStudent.create(check_config)
settings = student.settings
print(f"  settings: {settings.describe()}")
print(
    f"  n_slots={student.n_slots} feature_width={student.feature_width} input_width={student.input_width} "
    f"retention_fraction={student.retention_fraction} rehearsal_fraction={student.rehearsal_fraction}"
)
checks.expect(
    student.n_slots == slots_config and student.feature_width == FIXTURE_FEATURE_WIDTH
    and student.input_width == FIXTURE_FEATURE_WIDTH + slots_config and student.free_slots == slots_config,
    f"create reads students.snn.shared_skill_slots ({slots_config}) and shared_feature_width: W = {student.input_width}",
)
checks.expect(
    student.retention_fraction == float(require(config, "students.snn.shared_retention_fraction"))
    and student.rehearsal_fraction == float(require(config, "students.snn.rehearsal_fraction")),
    "create reads students.snn.shared_retention_fraction and rehearsal_fraction",
)
checks.expect(student.skills_held == () and student.n_stages == 0 and student.slots == {}, "a created student holds no skill and no stage")

alpha, beta = fixture_spec("fx_alpha"), fixture_spec("fx_beta")
dreams_alpha, dreams_beta = fixture_dreams(alpha, config), fixture_dreams(beta, config)
own_alpha = InputEncoder.from_variables(alpha.situation)
print(f"  fixture fx_alpha: S={alpha.n_situation} encoded D={own_alpha.width} C={dreams_alpha.n_candidates} dream rows={dreams_alpha.n_rows}")

print(f"\n== add fx_alpha: one skill held, {slots_config - 1} slots empty ==")
record = student.add_skill(alpha, dreams_alpha, points_per_dim=POINTS["fx_alpha"], base_seed=BASE_SEED, rehearsal=True)
training = record["training"]
print(
    f"  steps_run={record['steps_run']} best_step={training['best_step']} validation {training['initial_validation_loss']:.4f} -> "
    f"{training['best_validation_loss']:.4f} (constant prediction {training['constant_predictor_validation_loss']}); "
    f"firing rate {training['firing_rate_per_layer']}; add wall {record['wall_seconds']:.1f} s"
)
print(f"  split: {({k: v for k, v in record['split'].items() if k.startswith('n_')})}")
checks.expect(
    record["slot"] == 0 and record["slots"] == {"fx_alpha": 0} and record["free_slots"] == slots_config - 1
    and student.slots == {"fx_alpha": 0} and student.free_slots == slots_config - 1 and student.n_stages == 1,
    f"fx_alpha took slot 0; {student.free_slots} slots stay free; one stage",
)
checks.expect(
    record["rehearsal_requested"] and not record["rehearsal_applied"] and record["sampling"] == {"fx_alpha": 1.0}
    and record["heads_trained"] == ["fx_alpha"],
    "rehearsal requested but nothing to rehearse: every batch is fx_alpha's",
)
checks.expect(
    training["best_validation_loss"] < training["initial_validation_loss"]
    and training["best_validation_loss_per_head"]["fx_alpha"] < training["constant_predictor_validation_loss"]["fx_alpha"]
    and training["silent_layers"] == [],
    "trained: restored validation loss below initialisation and below the best constant prediction; no silent layer",
)
parity = record["parity"]["fx_alpha"]
print(f"  parity: { {k: parity[k] for k in ('n_inputs_checked', 'max_abs_logit_difference', 'spike_train_mismatches_per_layer', 'decisions_differing', 'decide_one_vs_batch_differing', 'decide_one_vs_torch_differing')} }")
checks.expect(
    parity["n_inputs_checked"] == dreams_alpha.n_rows and all(m == 0 for m in parity["spike_train_mismatches_per_layer"])
    and parity["decisions_differing"] == 0 and parity["decide_one_vs_batch_differing"] == 0
    and parity["decide_one_vs_torch_differing"] == 0 and parity["max_abs_logit_difference"] <= settings.parity_logit_tolerance,
    f"parity on all {dreams_alpha.n_rows} dream rows: spike trains, decisions, decide_one, logits within {settings.parity_logit_tolerance}",
)
train_rows, early_rows, retention_rows = (student.training_rows("fx_alpha"), student.early_stopping_rows("fx_alpha"), student.retention_rows("fx_alpha"))
checks.expect(
    bool(np.array_equal(np.sort(np.concatenate([train_rows, early_rows, retention_rows])), np.arange(dreams_alpha.n_rows))),
    f"training {train_rows.size} + early-stopping {early_rows.size} + retention {retention_rows.size} rows partition the dream",
)

situations = alpha.sample_situations(stream_rng(BASE_SEED, "fx_alpha/check/situations"), CHECK_SITUATIONS)
index = student.decide_index("fx_alpha", situations)
actions = student.decide(alpha, situations)
grid = np.stack([alpha.candidate_actions(s, POINTS["fx_alpha"])[i] for s, i in zip(situations, index)])
checks.expect(
    index.dtype == np.int64 and index.shape == (CHECK_SITUATIONS,) and bool(np.all((index >= 0) & (index < dreams_alpha.n_candidates)))
    and actions.shape == (CHECK_SITUATIONS, alpha.n_action) and bool(np.all(alpha.action_in_bounds(actions))) and bool(np.array_equal(actions, grid)),
    f"decide -> {actions.shape} in-bounds actions = candidate_actions(situation, {POINTS['fx_alpha']})[decide_index]; "
    f"{len(set(index.tolist()))} distinct candidates chosen",
)
one_index = np.array([student.decide_index_one("fx_alpha", s) for s in situations])
one_actions = np.stack([student.decide_one(alpha, s) for s in situations[:50]])
checks.expect(
    bool(np.array_equal(one_index, index)) and bool(np.array_equal(one_actions, actions[:50])),
    "the per-decision path (decide_index_one, decide_one) gives the batched decisions",
)

deployed = student.deployed("fx_alpha")
checks.expect(isinstance(deployed, DeployedStudent) and student.deployed("fx_alpha") is deployed and student.deployed("fx_alpha", 0) is deployed, "deployed('fx_alpha') is one DeployedStudent per (skill, stage)")
checks.expect(
    bool(np.array_equal(deployed.decide_index(situations), index)) and bool(np.array_equal(deployed.decide(alpha, situations), actions)),
    "deployed(skill) decides exactly as the student",
)
manual = np.zeros((CHECK_SITUATIONS, FIXTURE_FEATURE_WIDTH + slots_config))
manual[:, : own_alpha.width] = own_alpha.encode(situations)
manual[:, FIXTURE_FEATURE_WIDTH + 0] = 1.0
encoded = deployed.encode(situations)
checks.expect(
    bool(np.array_equal(encoded, manual)) and bool(np.all(encoded[:, own_alpha.width:FIXTURE_FEATURE_WIDTH] == 0.0))
    and bool(np.all(encoded[:, FIXTURE_FEATURE_WIDTH + 1:] == 0.0)),
    f"shared layout: {own_alpha.width} own columns, {FIXTURE_FEATURE_WIDTH - own_alpha.width} zero padding, slot 0 = 1, "
    f"{slots_config - 1} empty slots = 0",
)
checks.expect(
    bool(np.array_equal(student.network_view("fx_alpha").decide(manual), index)),
    "the network view on the hand-built encoding gives the same decisions",
)
counts = student.operation_counts("fx_alpha", situations)
hidden = settings.hidden_sizes
expected_macs = student.input_width * hidden[0]
dense = sum(a * b for a, b in zip((student.input_width, *hidden), (*hidden, dreams_alpha.n_candidates)))
print(
    f"  operation counts: layer_sizes={counts.layer_sizes} first_layer_macs={counts.first_layer_macs} "
    f"dense_reference_macs={counts.dense_reference_macs} neuron_updates={counts.neuron_updates} "
    f"mean spikes per layer={[round(float(v), 2) for v in counts.mean_spikes_per_layer]} (MECHANICS)"
)
checks.expect(
    counts.n_decisions == CHECK_SITUATIONS and counts.layer_sizes == (student.input_width, *hidden, dreams_alpha.n_candidates)
    and counts.first_layer_macs == expected_macs and counts.dense_reference_macs == dense,
    f"operation counts include the padded inputs: first layer {student.input_width} x {hidden[0]} = {expected_macs} MACs "
    f"(own encoding alone: {own_alpha.width * hidden[0]}); dense reference {dense}",
)

meta = deployed.meta
checks.expect(
    meta["kind"] == "shared_snn_view" and meta["stage"] == 0 and meta["slot"] == 0 and meta["n_slots"] == slots_config
    and meta["student"]["action_records"] and meta["student"]["candidate_grid"]["sha256"] == deployed.grid_sha256
    and meta["network_sha256"] == deployed.digest == record["network_digests"]["fx_alpha"],
    "the view's meta names kind, stage, slot, student record (action records, grid digest) and network digest",
)
seen = meta["student"]["situations_seen"]
checks.expect(
    deployed.training_situation_digests == frozenset(seen["optimiser_rows"]) | frozenset(seen["early_stopping_rows"])
    and frozenset(seen["optimiser_rows"]) == situation_digests(dreams_alpha.situations[train_rows])
    and frozenset(seen["early_stopping_rows"]) == situation_digests(dreams_alpha.situations[early_rows]),
    "recorded situation digests: the optimiser saw exactly the training rows, early stopping exactly the early-stopping rows",
)
checks.expect(
    not bool(np.any(deployed.seen_in_training(dreams_alpha.situations[retention_rows])))
    and bool(np.all(deployed.seen_in_training(dreams_alpha.situations[train_rows]))),
    "seen_in_training: no retention situation, every training situation",
)
held_index = deployed.decide_index(dreams_alpha.situations[retention_rows])
regret = dream_regret(dreams_alpha.scores[retention_rows], held_index)
stored = record["retention_in_dream"]["fx_alpha"]
print(f"  retention numbers (MECHANICS, fixture dream): {stored}")
checks.expect(
    stored["n_rows"] == retention_rows.size
    and float(np.mean(held_index == dreams_alpha.chosen[retention_rows])) == stored["agreement_with_dream_choice"]
    and float(np.mean(regret)) == stored["mean_dream_regret"],
    "the record's retention numbers are reproduced through deployed(skill) on the retention rows",
)
matrix = student.retention_matrix()
checks.expect(
    matrix["skills"] == ["fx_alpha"] and matrix["agreement_with_dream_choice"] == [[stored["agreement_with_dream_choice"]]]
    and all(value is None for value in matrix["backward_transfer"].values()),
    "retention matrix 1 x 1; backward transfer null with one skill",
)
stage0_index = index.copy()
stage0_actions = actions.copy()
stage0_digest = deployed.digest

print("\n== add fx_beta: it takes the next slot; stage 0 keeps fx_alpha's original view ==")
record_beta = student.add_skill(beta, dreams_beta, points_per_dim=POINTS["fx_beta"], base_seed=BASE_SEED, rehearsal=True)
print(f"  steps_run={record_beta['steps_run']} sampling={record_beta['sampling']} add wall {record_beta['wall_seconds']:.1f} s")
checks.expect(
    student.slots == {"fx_alpha": 0, "fx_beta": 1} and student.free_slots == slots_config - 2 and student.n_stages == 2
    and student.stage_skills(0) == ("fx_alpha",) and student.stage_skills(1) == ("fx_alpha", "fx_beta"),
    f"fx_beta took slot 1; stages {[student.stage_skills(i) for i in range(student.n_stages)]}",
)
checks.expect(
    all(
        all(m == 0 for m in report["spike_train_mismatches_per_layer"]) and report["decisions_differing"] == 0
        and report["n_inputs_checked"] == {"fx_alpha": dreams_alpha.n_rows, "fx_beta": dreams_beta.n_rows}[skill]
        for skill, report in record_beta["parity"].items()
    ),
    "after the second add both views pass parity on every row of their dreams",
)
early = student.deployed("fx_alpha", stage=0)
latest = student.deployed("fx_alpha")
checks.expect(
    early.digest == stage0_digest and student.network_view("fx_alpha", 0).digest() == record["network_digests"]["fx_alpha"]
    and student.network_digests(0) == record["network_digests"],
    "stage 0's view of fx_alpha is the network fx_alpha had before fx_beta was added (same digest)",
)
checks.expect(
    bool(np.array_equal(early.decide_index(situations), stage0_index)) and bool(np.array_equal(early.decide(alpha, situations), stage0_actions)),
    "deployed('fx_alpha', stage=0) gives fx_alpha's original decisions",
)
changed = int(np.sum(latest.decide_index(situations) != stage0_index))
checks.expect(latest.digest != stage0_digest, f"the latest fx_alpha view differs (the trunk moved); {changed} of {CHECK_SITUATIONS} decisions changed (MECHANICS)")
checks.raises("deployed('fx_beta', stage=0)", ValueError, lambda: student.deployed("fx_beta", stage=0), "was added at stage 1; stage 0 holds only ['fx_alpha']")
checks.raises("deployed('fx_alpha', stage=2)", ValueError, lambda: student.deployed("fx_alpha", stage=2), "stage must be an integer in [0, 2)")
checks.raises("deployed('fx_alpha', stage=True)", ValueError, lambda: student.deployed("fx_alpha", stage=True), "stage must be an integer")
retention_beta_stage = record_beta["retention_in_dream"]["fx_alpha"]
stage1_index = latest.decide_index(dreams_alpha.situations[retention_rows])
checks.expect(
    float(np.mean(stage1_index == dreams_alpha.chosen[retention_rows])) == retention_beta_stage["agreement_with_dream_choice"]
    and student.retention_matrix()["agreement_with_dream_choice"][1][0] == retention_beta_stage["agreement_with_dream_choice"],
    "R[1][0] is reproduced by the stage-1 view of fx_alpha on its retention rows",
)
checks.finish()
