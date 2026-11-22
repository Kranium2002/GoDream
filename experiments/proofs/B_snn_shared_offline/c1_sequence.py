"""Check (b): three skills with and without rehearsal; retention numbers come only from retention rows.

FIXTURE CHECK - mechanics only. Three fixture skills (fixture_shared_specs.py: different situation
widths, different candidate counts) whose DreamSets are built by the real engine (physics only).
Settings: config.yaml students.snn with max_steps reduced (printed); slots, fractions and every
other value from config.yaml, shared_feature_width fixture.

1. fx_alpha -> fx_beta -> fx_gamma with rehearsal=True and the same sequence, seeds and settings with
   rehearsal=False. Both complete; their seeds, splits and settings are identical; every view passes
   parity on every row of its dream after every add.
2. Proof that the retention rows were never in a training batch and never in an early-stopping set:
   (i) from the recorded digests: each skill's retention digests (recomputed from the stored dream)
   are disjoint from the optimiser and early-stopping digests in its student record, and every add
   record's digest-set sha256 of what it passed to fit equals the record's sets; (ii) from what fit
   actually received: the check wraps snn_train.fit (check-script instrumentation, not production
   code) and compares, row by row, every input array passed to it (training batches are drawn only
   from these) with the encoded retention, training and early-stopping rows.
3. Every R[i][j] of both retention matrices is reproduced through deployed(skill_j, stage=i) on skill
   j's retention rows; the two matrices are printed side by side as MECHANICS numbers. No claim
   about which run is better is made or asserted.
4. A dream whose every situation appears twice: all copies of a retention situation are retention
   rows (split by distinct situation). The old row split of the committed code on the same dream is
   computed here for comparison.

Old behaviour (committed shared_snn.py): the held-out rows of the retention matrix WERE the
early-stopping rows passed to fit, and rows were split by row (c7_old_behaviour.py shows both).

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c1_sequence.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from fixture_shared_specs import (  # noqa: E402
    FIXTURE_FEATURE_WIDTH,
    POINTS,
    SEQUENCE,
    Checks,
    fixture_config,
    fixture_dreams,
    fixture_spec,
    repeated_dreams,
)
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn import situation_row_digests  # noqa: E402
from learned_judgment.students.targets import dream_regret  # noqa: E402

MAX_STEPS = 300  # reduced from config students.snn.max_steps for a short mechanics check
REPEATED_MAX_STEPS = 100  # the repeated-situation dream only exercises the split
BASE_SEED = 7  # fixture run seed
KEYS = ("agreement_with_dream_choice", "fraction_with_zero_dream_regret", "mean_dream_regret", "mean_dream_regret_over_spread")

checks = Checks()
config = load_config()
check_config = fixture_config(config)
check_config["students"]["snn"]["max_steps"] = MAX_STEPS
fraction = require(check_config, "students.snn.rehearsal_fraction")
print(
    f"\ncheck config: config.yaml with students.snn.shared_feature_width={FIXTURE_FEATURE_WIDTH} (fixture) and "
    f"students.snn.max_steps={MAX_STEPS} (config.yaml {config['students']['snn']['max_steps']}); shared_skill_slots="
    f"{require(config, 'students.snn.shared_skill_slots')}, shared_retention_fraction="
    f"{require(config, 'students.snn.shared_retention_fraction')}, validation_fraction="
    f"{require(config, 'students.snn.validation_fraction')}, rehearsal_fraction={fraction} (config.yaml)"
)

specs = {name: fixture_spec(name) for name in SEQUENCE}
dreams = {name: fixture_dreams(spec, config) for name, spec in specs.items()}
for name, spec in specs.items():
    print(
        f"  fixture {name}: S={spec.n_situation} encoded D={InputEncoder.from_variables(spec.situation).width} "
        f"C={dreams[name].n_candidates} grid={POINTS[name]} dream rows={dreams[name].n_rows} method={dreams[name].method}"
    )

# Check-script instrumentation: remember every (tasks, validation) pair fit receives.
fit_calls: list[dict] = []
real_fit = snn_train.fit


def recording_fit(model, tasks, *, validation, sampling, settings, seed):  # noqa: ANN001, ANN201
    fit_calls.append(
        {
            "tasks": {head: np.array(pair[0], copy=True) for head, pair in tasks.items()},
            "validation": {head: np.array(pair[0], copy=True) for head, pair in validation.items()},
        }
    )
    return real_fit(model, tasks, validation=validation, sampling=sampling, settings=settings, seed=seed)


def row_set(array: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in np.ascontiguousarray(array, dtype=np.float64)}


def run_sequence(rehearsal: bool) -> tuple[SharedSNNStudent, list[dict], list[dict]]:
    student = SharedSNNStudent.create(check_config)
    records, calls = [], []
    for name in SEQUENCE:
        before = len(fit_calls)
        record = student.add_skill(specs[name], dreams[name], points_per_dim=POINTS[name], base_seed=BASE_SEED, rehearsal=rehearsal)
        calls.append(fit_calls[before])
        training = record["training"]
        print(
            f"  + {name} (slot {record['slot']}): steps_run={record['steps_run']} best_step={training['best_step']} "
            f"validation total {training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}; "
            f"add wall {record['wall_seconds']:.1f} s; heads trained {record['heads_trained']} "
            f"sampling={ {k: round(v, 4) for k, v in record['sampling'].items()} }"
        )
        records.append(record)
    return student, records, calls


def parity_clean(record: dict) -> bool:
    return list(record["parity"]) == record["skills_held"] and all(
        all(m == 0 for m in report["spike_train_mismatches_per_layer"])
        and report["decisions_differing"] == 0
        and report["decide_one_vs_batch_differing"] == 0
        and report["decide_one_vs_torch_differing"] == 0
        and report["max_abs_logit_difference"] <= report["logit_tolerance"]
        and report["n_inputs_checked"] == dreams[skill].n_rows
        for skill, report in record["parity"].items()
    )


snn_train.fit = recording_fit
try:
    print("\n== 1. fx_alpha -> fx_beta -> fx_gamma with rehearsal=True ==")
    started = time.perf_counter()
    student_r, records_r, calls_r = run_sequence(True)
    print(f"  wall {time.perf_counter() - started:.1f} s incl. torch import; versions {records_r[-1]['versions']}")
    print("\n== 1. the same sequence, seeds and settings with rehearsal=False ==")
    student_n, records_n, calls_n = run_sequence(False)
finally:
    snn_train.fit = real_fit

runs = {"rehearsal=True": (student_r, records_r, calls_r), "rehearsal=False": (student_n, records_n, calls_n)}
checks.expect(student_r.skills_held == SEQUENCE and student_n.skills_held == SEQUENCE, "both runs complete")
checks.expect(
    student_r.slots == student_n.slots == {"fx_alpha": 0, "fx_beta": 1, "fx_gamma": 2}
    and student_r.free_slots == student_r.n_slots - 3,
    f"slots in add order: {student_r.slots}; {student_r.free_slots} free",
)
checks.expect(
    all(a["seeds"] == b["seeds"] and a["split"] == b["split"] and a["settings"] == b["settings"] for a, b in zip(records_r, records_n))
    and all(np.array_equal(student_r.retention_rows(s), student_n.retention_rows(s)) and np.array_equal(student_r.training_rows(s), student_n.training_rows(s)) for s in SEQUENCE),
    "same seeds, splits and settings in both runs",
)
checks.expect(records_r[0]["network_digests"] == records_n[0]["network_digests"], "first add identical in both runs (nothing to rehearse)")
expected_sampling = [
    {"fx_alpha": 1.0},
    {"fx_beta": 1.0 - fraction, "fx_alpha": fraction},
    {"fx_gamma": 1.0 - fraction, "fx_alpha": fraction / 2, "fx_beta": fraction / 2},
]
checks.expect(
    all(
        set(record["sampling"]) == set(expected) and all(math.isclose(record["sampling"][k], v, abs_tol=1e-12) for k, v in expected.items())
        for record, expected in zip(records_r, expected_sampling)
    )
    and all(record["sampling"] == {record["skill"]: 1.0} for record in records_n),
    "batch shares: with rehearsal new skill 1 - rehearsal_fraction and earlier skills split the rest; without, the new skill only",
)
for label, (student, records, _) in runs.items():
    checks.expect(all(parity_clean(record) for record in records), f"{label}: after every add every view passes parity on every row of its dream")

print("\n== 2(i). retention rows vs the recorded situation digests ==")
for label, (student, records, _) in runs.items():
    for skill in SEQUENCE:
        stored = student.stored_dreams(skill)
        digests = situation_row_digests(stored.situations)
        retention = {digests[row] for row in student.retention_rows(skill)}
        seen = student.deployed(skill).meta["student"]["situations_seen"]
        optimiser, early = set(seen["optimiser_rows"]), set(seen["early_stopping_rows"])
        used = [record["situations_used"][skill] for record in records if skill in record["situations_used"]]
        checks.expect(
            retention == set(student.retention_situation_digests(skill)) and not retention & (optimiser | early)
            and len(retention) == records[SEQUENCE.index(skill)]["split"]["n_rows_retention"]
            and all(item["retention_situations_passed_to_fit"] == 0 for item in used)
            and len({(item["optimiser_situations_sha256"], item["early_stopping_situations_sha256"]) for item in used}) == 1,
            f"{label} {skill}: {len(retention)} retention situations, none among the {len(optimiser)} optimiser and "
            f"{len(early)} early-stopping situations recorded; the {len(used)} add(s) that trained {skill} passed the same recorded sets",
        )

print("\n== 2(ii). retention rows vs the arrays snn_train.fit actually received ==")
for label, (student, records, calls) in runs.items():
    clean = True
    details = []
    for stage, call in enumerate(calls):
        checks_ok = set(call["tasks"]) == set(records[stage]["heads_trained"]) == set(call["validation"])
        for skill in call["tasks"]:
            deployed = student.deployed(skill)
            stored = student.stored_dreams(skill)
            encoded = {
                part: row_set(deployed.encode(stored.situations[rows]))
                for part, rows in (
                    ("training", student.training_rows(skill)),
                    ("early_stopping", student.early_stopping_rows(skill)),
                    ("retention", student.retention_rows(skill)),
                )
            }
            task_rows, validation_rows = row_set(call["tasks"][skill]), row_set(call["validation"][skill])
            ok = (
                task_rows == encoded["training"] and validation_rows == encoded["early_stopping"]
                and not (task_rows | validation_rows) & encoded["retention"]
            )
            checks_ok = checks_ok and ok
            details.append(f"stage {stage} {skill}: batches from {len(task_rows)} rows, early stopping on {len(validation_rows)}, retention overlap {len((task_rows | validation_rows) & encoded['retention'])}")
        clean = clean and checks_ok
    for line in details:
        print(f"    {label} {line}")
    checks.expect(
        clean,
        f"{label}: at every add, fit's training inputs are exactly the training rows and its validation inputs exactly the "
        "early-stopping rows of each trained skill; no retention row was passed to fit",
    )
checks.expect(
    [sorted(call["tasks"]) for call in calls_r] == [["fx_alpha"], ["fx_alpha", "fx_beta"], ["fx_alpha", "fx_beta", "fx_gamma"]]
    and [sorted(call["tasks"]) for call in calls_n] == [["fx_alpha"], ["fx_beta"], ["fx_gamma"]],
    "rehearsal draws earlier skills' training rows only in the rehearsal run",
)

print("\n== 3. retention matrices from the retention rows only ==")
matrices = {}
for label, (student, records, _) in runs.items():
    matrix = student.retention_matrix()
    matrices[label] = matrix
    reproduced = True
    for i in range(3):
        for j, skill in enumerate(SEQUENCE[: i + 1]):
            rows = student.retention_rows(skill)
            stored = student.stored_dreams(skill)
            index = student.deployed(skill, stage=i).decide_index(stored.situations[rows])
            regret = dream_regret(stored.scores[rows], index)
            reproduced = reproduced and (
                float(np.mean(index == stored.chosen[rows])) == matrix["agreement_with_dream_choice"][i][j]
                and float(np.mean(regret)) == matrix["mean_dream_regret"][i][j]
                and float(np.mean(regret == 0.0)) == matrix["fraction_with_zero_dream_regret"][i][j]
                and not bool(np.any(student.deployed(skill, stage=i).seen_in_training(stored.situations[rows])))
            )
    checks.expect(reproduced, f"{label}: every R[i][j] is reproduced by deployed(skill_j, stage=i) on skill j's retention rows, none of which it saw in training")
    shape_ok = all(
        len(matrix[key]) == 3 and all((value is None) == (j > i) for i, row in enumerate(matrix[key]) for j, value in enumerate(row))
        for key in KEYS
    )
    checks.expect(
        shape_ok
        and all(
            math.isclose(matrix["backward_transfer"][key], float(np.mean([matrix[key][2][j] - matrix[key][j][j] for j in range(2)])), rel_tol=0.0, abs_tol=0.0)
            for key in KEYS
        ),
        f"{label}: matrices 3 x 3 with null above the diagonal; backward_transfer = mean over j < last of (R[last][j] - R[j][j])",
    )
alpha_before, alpha_after = student_n.network_view("fx_alpha", 0), student_n.network_view("fx_alpha", 2)
checks.expect(
    np.array_equal(alpha_before.readout_weight, alpha_after.readout_weight) and np.array_equal(alpha_before.readout_bias, alpha_after.readout_bias)
    and not np.array_equal(alpha_before.hidden_weights[0], alpha_after.hidden_weights[0]),
    "no rehearsal: fx_alpha's readout is untouched by later adds while the shared trunk moved (stage 0 vs stage 2 views)",
)
checks.expect(
    not np.array_equal(student_r.network_view("fx_alpha", 0).readout_weight, student_r.network_view("fx_alpha", 2).readout_weight),
    "rehearsal: fx_alpha's readout is trained again at later adds",
)
print("\n  MECHANICS numbers on FIXTURE dreams, inside the dream, on RETENTION rows only")
print(f"  (max_steps={MAX_STEPS}; no claim about which run is better is made or asserted)")
for key in ("agreement_with_dream_choice", "mean_dream_regret_over_spread", "fraction_with_zero_dream_regret"):
    print(f"  {key}:")
    header = "".join(f"{skill:>10s}" for skill in SEQUENCE)
    print(f"    {'stage':10s}{'rehearsal=True':>30s}   {'rehearsal=False':>30s}")
    print(f"    {'':10s}{header}   {header}")
    for i, skill in enumerate(SEQUENCE):
        cells = []
        for label in ("rehearsal=True", "rehearsal=False"):
            cells.append("".join(f"{'-':>10s}" if v is None else f"{v:10.4f}" for v in matrices[label][key][i]))
        print(f"    {i} {skill:8s}{cells[0]}   {cells[1]}")
for key in KEYS:
    print(
        f"  backward_transfer {key}: rehearsal=True {matrices['rehearsal=True']['backward_transfer'][key]:+.4f}   "
        f"rehearsal=False {matrices['rehearsal=False']['backward_transfer'][key]:+.4f}"
    )
print("  early-stopping rows (NOT held out: they chose the restored weights), agreement at the skill's own add, for contrast:")
for label, (_, records, _) in runs.items():
    print(f"    {label}: " + ", ".join(f"{r['skill']} {r['early_stopping_rows_in_dream'][r['skill']]['agreement_with_dream_choice']:.4f}" for r in records))
checks.expect(
    all(
        records[i]["early_stopping_rows_in_dream_label"].endswith("NOT held out")
        and records[i]["retention_in_dream_label"].endswith("(held out)")
        for _, records, _ in runs.values()
        for i in range(3)
    ),
    "the records label early-stopping numbers NOT held out and retention numbers held out",
)

print(f"\n== 4. a dream with every situation twice (fx_beta, max_steps={REPEATED_MAX_STEPS}) ==")
doubled = repeated_dreams(dreams["fx_beta"], 2)
short_config = fixture_config(config)
short_config["students"]["snn"]["max_steps"] = REPEATED_MAX_STEPS
repeated = SharedSNNStudent.create(short_config)
record_doubled = repeated.add_skill(specs["fx_beta"], doubled, points_per_dim=POINTS["fx_beta"], base_seed=BASE_SEED, rehearsal=True)
digests = situation_row_digests(doubled.situations)
parts = {
    "training": repeated.training_rows("fx_beta"),
    "early_stopping": repeated.early_stopping_rows("fx_beta"),
    "retention": repeated.retention_rows("fx_beta"),
}
sets = {part: {digests[row] for row in rows} for part, rows in parts.items()}
n = dreams["fx_beta"].n_rows
retention_pairs = all((row + n) % (2 * n) in set(parts["retention"].tolist()) for row in parts["retention"])
print(f"  {doubled.n_rows} rows, {record_doubled['split']['n_distinct_situations']} distinct situations; rows per part { {k: int(v.size) for k, v in parts.items()} }")
checks.expect(
    retention_pairs and not sets["retention"] & (sets["training"] | sets["early_stopping"]) and not sets["training"] & sets["early_stopping"],
    f"both copies of each of the {len(sets['retention'])} retention situations are retention rows; no situation is in two parts",
)
validation_fraction = require(config, "students.snn.validation_fraction")
old_rows = np.sort(
    stream_rng(BASE_SEED, "fx_beta/snn/validation_split").permutation(doubled.n_rows)[: max(1, int(round(validation_fraction * doubled.n_rows)))]
)
old_train = set(np.setdiff1d(np.arange(doubled.n_rows), old_rows).tolist())
duplicated = sum(1 for row in old_rows if (row + n) % (2 * n) in old_train)
print(
    f"  for comparison, the committed code's row split (stream fx_beta/snn/validation_split, validation_fraction "
    f"{validation_fraction} of ROWS) on this dream: {duplicated} of {old_rows.size} held-out rows have a training copy"
)
checks.finish()
