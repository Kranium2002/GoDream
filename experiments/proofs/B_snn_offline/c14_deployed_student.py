"""Check 14 (decision M1 groundwork): every decision goes through one DeployedStudent object.

FIXTURE CHECK - mechanics only. Old behaviour (review finding "the shared SNN cannot be measured",
reproduction experiments/proofs/B_review/snn_shared_measure/r01): deciding existed only as
methods of SNNStudent with its own InputEncoder, so a network whose input layout differs (a shared
network's view) had no object to decide or be measured through. Now snn.DeployedStudent holds a
skill's network, its encode / encode_one functions, grid record, situation digests and meta, and
decides (decide_index, decide_index_one, decide, decide_one, operation_counts); SNNStudent.deployed()
returns one and SNNStudent's own decide methods go through it. Also checked: the stored in-dream
numbers are labelled as early-stopping rows, the old private helper names still import, and a
student whose meta lacks the record is refused. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c14_deployed_student.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402
import importlib  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import snn  # noqa: E402
from learned_judgment.students.snn import DeployedStudent, SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

BASE_SEED = 31  # fixture run seed

checks = Checks()
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=150, evaluate_every_steps=50)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 400)
student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 200)

print("\n== SNNStudent.deployed() ==")
deployed = student.deployed()
checks.expect(isinstance(deployed, DeployedStudent) and student.deployed() is deployed, "deployed() returns one DeployedStudent per student (the same object each call)")
record = student.meta["student"]
checks.expect(
    deployed.skill == spec.name and deployed.network is student.network and deployed.points_per_dim == FIXTURE_POINTS_PER_DIM
    and deployed.n_candidates == dreams.n_candidates and deployed.digest == student.digest,
    "skill, network, points_per_dim, n_candidates and digest are the student's",
)
checks.expect(
    [dict(item) for item in deployed.action_records] == record["action_records"] and deployed.grid_sha256 == record["candidate_grid"]["sha256"]
    and deployed.grid_probe.tolist() == record["candidate_grid"]["probe_situations"],
    "action records, grid probe and grid digest come from the stored record",
)
checks.expect(
    deployed.training_situation_digests == frozenset(record["situations_seen"]["optimiser_rows"] + record["situations_seen"]["early_stopping_rows"]),
    "training situation digests = optimiser + early-stopping digests",
)
checks.expect(deployed.meta is student.meta and deployed.out_of_range == settings.out_of_range, "meta and the out_of_range policy are carried")
encoded = deployed.encode(situations)
checks.expect(encoded.shape == (200, student.network.n_inputs), f"encode: (n, S) -> {encoded.shape}")
checks.expect(bool(np.array_equal(deployed.encode_one(situations[0]), encoded[0])), "encode_one: (S,) -> (D,), bitwise the row of encode")

print("\n== SNNStudent decides through it ==")
index = deployed.decide_index(situations)
checks.expect(bool(np.array_equal(student.decide_index(situations), index)), "SNNStudent.decide_index = DeployedStudent.decide_index")
checks.expect(bool(np.array_equal(student.decide(spec, situations), deployed.decide(spec, situations))), "SNNStudent.decide = DeployedStudent.decide")
checks.expect(
    all(student.decide_index_one(s) == deployed.decide_index_one(s) == i for s, i in zip(situations, index)),
    "decide_index_one (student and deployed) = batched index on every situation",
)
checks.expect(bool(np.array_equal(student.decide_one(spec, situations[5]), deployed.decide_one(spec, situations[5]))), "decide_one agrees")
counts_student, counts_deployed = student.operation_counts(situations), deployed.operation_counts(situations)
checks.expect(bool(np.array_equal(counts_student.spikes_per_decision, counts_deployed.spikes_per_decision)), "operation_counts agree")
log = student.out_of_range_log.as_dict()
checks.expect(log["decisions"] > 0 and student.out_of_range_log is deployed.out_of_range_log, f"one out_of_range log for both paths: {log}")

print("\n== a DeployedStudent built by hand from other parts (as a shared network's view would be) ==")
# FIXTURE: an input layout with one extra constant column; the network is widened by a zero
# input weight, so it computes exactly what the student's network computes.
network = student.network
widened = dataclasses.replace(network, hidden_weights=(np.hstack([network.hidden_weights[0], np.zeros((network.hidden_sizes[0], 1))]), *network.hidden_weights[1:]))
pad = lambda rows: np.hstack([student.encoder.encode(rows), np.ones((np.asarray(rows).shape[0], 1))])  # noqa: E731
pad_one = lambda row: np.append(student.encoder.encode_one(row), 1.0)  # noqa: E731
view = DeployedStudent(
    skill=spec.name, network=widened, encode=pad, encode_one=pad_one, situation_variables=spec.situation,
    points_per_dim=FIXTURE_POINTS_PER_DIM, n_candidates=dreams.n_candidates, action_records=tuple(record["action_records"]),
    grid_probe=np.asarray(record["candidate_grid"]["probe_situations"]), grid_sha256=record["candidate_grid"]["sha256"],
    training_situation_digests=deployed.training_situation_digests, out_of_range="error", meta={"kind": "fixture_view"},
)
checks.expect(bool(np.array_equal(view.decide_index(situations), index)), "the hand-built view decides like the student")
checks.expect(bool(np.array_equal(view.decide(spec, situations), student.decide(spec, situations))), "and maps to the same commands through the same grid checks")
fields = dict(
    skill=spec.name, network=widened, encode=pad, encode_one=pad_one, situation_variables=spec.situation,
    points_per_dim=FIXTURE_POINTS_PER_DIM, n_candidates=dreams.n_candidates, action_records=tuple(record["action_records"]),
    grid_probe=np.asarray(record["candidate_grid"]["probe_situations"]), grid_sha256=record["candidate_grid"]["sha256"],
    training_situation_digests=deployed.training_situation_digests, out_of_range="error", meta={},
)
checks.raises("encode that does not fit the network", ValueError, lambda: DeployedStudent(**{**fields, "encode": student.encoder.encode}), "the network takes 5 inputs")
checks.raises("encode_one that differs from encode", ValueError, lambda: DeployedStudent(**{**fields, "encode_one": lambda row: pad_one(row) * 2.0}), "encode_one(situation) differs")
checks.raises("no action records", ValueError, lambda: DeployedStudent(**{**fields, "action_records": ()}), "action_records is empty")
checks.raises("an action record without command_values", ValueError, lambda: DeployedStudent(**{**fields, "action_records": tuple({k: v for k, v in r.items() if k != "command_values"} for r in record["action_records"])}), "must have exactly the keys")
checks.raises("an unknown out_of_range policy", ValueError, lambda: DeployedStudent(**{**fields, "out_of_range": "ignore"}), "out_of_range must be one of")
checks.raises("n_candidates that differs from the network", ValueError, lambda: DeployedStudent(**{**fields, "n_candidates": 34}), "differs from the network's 35 outputs")
checks.raises("a 17-row grid probe", ValueError, lambda: DeployedStudent(**{**fields, "grid_probe": np.repeat(fields["grid_probe"][:1], 17, axis=0)}), "grid_probe must have shape (1..16, 3)")

print("\n== labels, refusals and old names ==")
checks.expect("validation_in_dream" not in student.meta and "NOT held out" in student.meta["early_stopping_rows_in_dream"]["label"], "in-dream numbers are stored as early-stopping rows, labelled NOT held out")
checks.expect("early_stopping_rows_in_dream" in student.meta["label"] and "not held-out" in student.meta["label"], "the student's label says so too")
checks.expect(
    snn._check_dreams is snn.check_dreams and snn._dream_content_sha256 is snn.dream_content_sha256
    and snn._points_per_dim is snn.check_points_per_dim and snn._variable_from_record is snn.variable_from_record
    and SNNStudent._student_record is snn.student_record,
    "old private names are aliases of the public helpers",
)
for module in ("learned_judgment.students.measure", "learned_judgment.students.shared_snn"):
    try:
        importlib.import_module(module)
        checks.expect(True, f"{module} still imports (it uses the old names)")
    except Exception as error:  # noqa: BLE001
        checks.expect(False, f"{module} import failed: {type(error).__name__}: {error}")
partial = {key: value for key, value in student.meta.items() if key != "student"}
checks.raises("SNNStudent with a meta lacking the student record", ValueError, lambda: dataclasses.replace(student, meta=partial), "meta['student'] is missing")
checks.raises(
    "SNNStudent whose record names other points_per_dim",
    ValueError,
    lambda: dataclasses.replace(student, meta={**student.meta, "student": {**record, "points_per_dim": [5, 7]}}),
    "meta['student']['points_per_dim'] is [5, 7]",
)
checks.finish()
