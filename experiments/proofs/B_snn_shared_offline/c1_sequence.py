"""Checks 1, 2 and 4 of the shared spiking student: sequence with and without rehearsal, save/load.

FIXTURE CHECK - mechanics only. Three fixture skills (fixture_shared_specs.py: different situation
widths, different candidate counts) whose DreamSets are built by the real engine (physics only).
The check config is config.yaml loaded in memory with students.snn.shared_feature_width filled in
and students.snn.shared_skills replaced by fixture names (see fixture_shared_specs.py); settings are
that config's students.snn values except max_steps, reduced explicitly (printed below). Every
agreement / regret number printed here is a MECHANICS number on a fixture dream, measured inside
the dream; it says nothing about the robot, the simulator, TabPFN or a project student.

Check 1: fx_alpha -> fx_beta -> fx_gamma with rehearsal=True. Every per-skill view reproduces the
    torch model after every add (zero spike-count mismatches, identical decisions); decide() returns
    in-bounds grid actions for every skill; the held-out numbers are reproduced through the public
    decide path; the retention matrix has the documented shape and BWT.
Check 2: the same sequence and seeds with rehearsal=False; both retention matrices printed side by
    side. Asserted only: both runs complete; the first add is identical (nothing to rehearse yet);
    later networks differ; without rehearsal an earlier head's readout is untouched while the trunk moves.
Check 4: save; load in-process; load in a FRESH process (c4_load_worker.py) and decide for every skill
    with torch absent from sys.modules; then add fx_delta to the loaded student there and to the
    in-memory student here: identical network digests. Files go to a temporary directory only.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c1_sequence.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_shared_specs import (  # noqa: E402
    FIXTURE_FEATURE_WIDTH,
    FIXTURE_SHARED_SKILLS,
    POINTS,
    REPO_ROOT,
    SEQUENCE,
    Checks,
    fixture_config,
    fixture_dreams,
    fixture_spec,
)
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402
from learned_judgment.students.targets import dream_regret  # noqa: E402

MAX_STEPS = 300  # reduced from config students.snn.max_steps for a short mechanics check
BASE_SEED = 7  # fixture run seed
CHECK_SITUATIONS = 300
KEYS = ("agreement_with_dream_choice", "fraction_with_zero_dream_regret", "mean_dream_regret", "mean_dream_regret_over_spread")

checks = Checks()
config = load_config()
check_config = fixture_config(config)
config_settings = SNNSettings.from_config(check_config)
settings = dataclasses.replace(config_settings, max_steps=MAX_STEPS)
skills = require(check_config, "students.snn.shared_skills")
width = require(check_config, "students.snn.shared_feature_width")
fraction = require(check_config, "students.snn.rehearsal_fraction")
print(
    f"\ncheck config: config.yaml with students.snn.shared_feature_width={width} (fixture; config.yaml value "
    f"{config['students']['snn']['shared_feature_width']!r}) and students.snn.shared_skills={skills} (fixture names; "
    f"config.yaml value {config['students']['snn']['shared_skills']})"
)
print(f"settings: students.snn with max_steps={MAX_STEPS} passed explicitly (config value {config_settings.max_steps})")
print(f"  {settings.describe()}")
print(f"  rehearsal_fraction={fraction} (config)")

print("\n== create(config) ==")
created = SharedSNNStudent.create(check_config)
checks.expect(
    created.skills == tuple(FIXTURE_SHARED_SKILLS) and created.feature_width == FIXTURE_FEATURE_WIDTH
    and created.input_width == FIXTURE_FEATURE_WIDTH + len(FIXTURE_SHARED_SKILLS),
    f"create reads shared_skills and shared_feature_width; input width W = {created.input_width}",
)
checks.expect(created.settings == config_settings, "create reads every students.snn setting (SNNSettings.from_config)")
checks.expect(created.rehearsal_fraction == float(fraction), "create reads students.snn.rehearsal_fraction")
matrix = created.retention_matrix()
checks.expect(created.skills_held == () and matrix["skills"] == [] and matrix["backward_transfer"]["mean_dream_regret"] is None, "a created student is empty")

specs = {name: fixture_spec(name) for name in (*SEQUENCE, "fx_delta")}
dreams = {name: fixture_dreams(spec, config) for name, spec in specs.items()}
for name, spec in specs.items():
    print(
        f"  fixture {name}: S={spec.n_situation} encoded D={InputEncoder.from_variables(spec.situation).width} "
        f"C={dreams[name].n_candidates} grid={POINTS[name]} dream rows={dreams[name].n_rows} method={dreams[name].method}"
    )


def new_student() -> SharedSNNStudent:
    return SharedSNNStudent(skills=skills, feature_width=width, settings=settings, rehearsal_fraction=fraction)


def run_sequence(rehearsal: bool) -> tuple[SharedSNNStudent, list[dict], list[dict]]:
    student = new_student()
    records, snapshots = [], []
    for name in SEQUENCE:
        record = student.add_skill(specs[name], dreams[name], points_per_dim=POINTS[name], base_seed=BASE_SEED, rehearsal=rehearsal)
        training = record["training"]
        print(
            f"  + {name}: steps_run={record['steps_run']} best_step={training['best_step']} "
            f"validation total {training['initial_validation_loss']:.4f} -> {training['best_validation_loss']:.4f}; "
            f"add_skill wall {record['wall_seconds']:.1f} s (fit {training['wall_seconds']:.1f} s); "
            f"rehearsal_applied={record['rehearsal_applied']} sampling={ {k: round(v, 4) for k, v in record['sampling'].items()} }"
        )
        for skill, report in record["parity"].items():
            print(
                f"      parity {skill}: rows {report['n_inputs_checked']} spike-count mismatches "
                f"{report['spike_count_mismatches_per_layer']} decisions differing {report['decisions_differing']} "
                f"max|logit diff| {report['max_abs_logit_difference']:.1e}"
            )
        records.append(record)
        snapshots.append({skill: student.network_view(skill) for skill in student.skills_held})
    return student, records, snapshots


def parity_clean(record: dict, student: SharedSNNStudent) -> bool:
    return all(
        all(m == 0 for m in report["spike_count_mismatches_per_layer"])
        and report["decisions_differing"] == 0
        and report["n_inputs_checked"] == student.held_out_rows(skill).size
        for skill, report in record["parity"].items()
    ) and list(record["parity"]) == record["skills_held"]


def print_matrix(label: str, matrix: dict, key: str) -> None:
    print(f"    {label}")
    header = "".join(f"{skill:>12s}" for skill in matrix["skills"])
    print(f"      {'after adding':14s}{header}")
    for i, row in enumerate(matrix[key]):
        cells = "".join(f"{'-':>12s}" if value is None else f"{value:12.4f}" for value in row)
        print(f"      {matrix['skills'][i]:14s}{cells}")


print("\n== check 1: fx_alpha -> fx_beta -> fx_gamma with rehearsal=True ==")
started = time.perf_counter()
student_r, records_r, snapshots_r = run_sequence(True)
print(f"  sequence wall time incl. torch import: {time.perf_counter() - started:.1f} s; versions {records_r[-1]['versions']}")
for record in records_r:
    checks.expect(parity_clean(record, student_r), f"after adding {record['skill']}: every view {record['skills_held']} reproduces the torch model on its held-out rows (zero spike-count mismatches, identical decisions)")
expected_sampling = [
    {"fx_alpha": 1.0},
    {"fx_beta": 1.0 - fraction, "fx_alpha": fraction},
    {"fx_gamma": 1.0 - fraction, "fx_alpha": fraction / 2, "fx_beta": fraction / 2},
]
checks.expect(
    all(
        set(record["sampling"]) == set(expected) and all(math.isclose(record["sampling"][k], v, abs_tol=1e-12) for k, v in expected.items())
        for record, expected in zip(records_r, expected_sampling)
    ),
    f"batch shares: new skill 1 - rehearsal_fraction, earlier skills rehearsal_fraction split evenly: {[r['sampling'] for r in records_r]}",
)
checks.expect([r["rehearsal_applied"] for r in records_r] == [False, True, True], "rehearsal applied from the second skill on (nothing to rehearse for the first)")
checks.expect(
    len({r["seeds"]["trunk_init_seed"] for r in records_r}) == 1 and [r["seeds"]["trunk_initialised_here"] for r in records_r] == [True, False, False],
    "the trunk is initialised once (first add) and continued afterwards",
)
views = {skill: student_r.network_view(skill) for skill in SEQUENCE}
first = views[SEQUENCE[0]]
checks.expect(
    all(
        all(np.array_equal(a, b) for a, b in zip(view.hidden_weights, first.hidden_weights))
        and all(np.array_equal(a, b) for a, b in zip(view.hidden_biases, first.hidden_biases))
        and view.betas == first.betas and view.thresholds == first.thresholds
        for view in views.values()
    ),
    "all per-skill views share one trunk (identical hidden weights, biases, betas, thresholds)",
)
checks.expect(
    all(views[s].layer_sizes == (student_r.input_width, *settings.hidden_sizes, dreams[s].n_candidates) for s in SEQUENCE),
    f"view layer sizes (W, H, C_skill): {[views[s].layer_sizes for s in SEQUENCE]}",
)
checks.expect(
    not np.array_equal(snapshots_r[0]["fx_alpha"].readout_weight, snapshots_r[2]["fx_alpha"].readout_weight),
    "with rehearsal the earlier head fx_alpha is trained again (its readout changed)",
)

situations = {}
indices = {}
for name in SEQUENCE:
    spec = specs[name]
    situations[name] = spec.sample_situations(stream_rng(BASE_SEED, f"{name}/check/situations"), CHECK_SITUATIONS)
    indices[name] = student_r.decide_index(name, situations[name])
    actions = student_r.decide(spec, situations[name])
    grid = np.stack([spec.candidate_actions(s, POINTS[name])[i] for s, i in zip(situations[name], indices[name])])
    checks.expect(
        indices[name].dtype == np.int64 and indices[name].shape == (CHECK_SITUATIONS,)
        and bool(np.all((indices[name] >= 0) & (indices[name] < dreams[name].n_candidates)))
        and actions.shape == (CHECK_SITUATIONS, spec.n_action) and bool(np.all(spec.action_in_bounds(actions)))
        and bool(np.array_equal(actions, grid)),
        f"{name}: decide -> {actions.shape} in-bounds actions = candidate_actions(situation, {POINTS[name]})[decide_index]",
    )
    counts = student_r.operation_counts(name, situations[name])
    checks.expect(
        counts.n_decisions == CHECK_SITUATIONS and counts.layer_sizes == views[name].layer_sizes,
        f"{name}: operation_counts layer sizes {counts.layer_sizes}, mean spikes per layer {[round(float(v), 1) for v in counts.mean_spikes_per_layer]}",
    )
    rows = student_r.held_out_rows(name)
    n = dreams[name].n_rows
    recipe = np.sort(stream_rng(BASE_SEED, f"{name}/snn/validation_split").permutation(n)[: max(1, int(round(settings.validation_fraction * n)))])
    checks.expect(bool(np.array_equal(rows, recipe)), f"{name}: held-out rows ({rows.size}) are the per-skill student's validation rows for this seed")
    held = student_r.decide_index(name, dreams[name].situations[rows])
    regret = dream_regret(dreams[name].scores[rows], held)
    stored = records_r[-1]["held_out_in_dream"][name]
    checks.expect(
        float(np.mean(held == dreams[name].chosen[rows])) == stored["agreement_with_dream_choice"]
        and float(np.mean(regret)) == stored["mean_dream_regret"],
        f"{name}: decide_index on the held-out rows reproduces the last record's agreement and regret",
    )

retention_r = student_r.retention_matrix()
shape_ok = all(
    len(retention_r[key]) == 3
    and all(len(row) == 3 for row in retention_r[key])
    and all((value is None) == (j > i) for i, row in enumerate(retention_r[key]) for j, value in enumerate(row))
    for key in KEYS
)
checks.expect(shape_ok and retention_r["skills"] == list(SEQUENCE), "retention matrices are 3 x 3, numbers for j <= i and None above the diagonal")
checks.expect(
    all(retention_r[key][i][i] == records_r[i]["held_out_in_dream"][SEQUENCE[i]][key] for key in KEYS for i in range(3)),
    "diagonal R[i][i] = the new skill's own held-out numbers in its add record",
)
checks.expect(
    all(
        math.isclose(retention_r["backward_transfer"][key], float(np.mean([retention_r[key][2][j] - retention_r[key][j][j] for j in range(2)])), rel_tol=0.0, abs_tol=0.0)
        for key in KEYS
    ),
    "backward_transfer = mean over j < last of (R[last][j] - R[j][j]) for every matrix",
)

print("\n== check 2: the same sequence and seeds with rehearsal=False (no-rehearsal control) ==")
student_n, records_n, snapshots_n = run_sequence(False)
checks.expect(len(records_n) == 3 and student_n.skills_held == SEQUENCE, "the no-rehearsal run completes")
checks.expect(
    all(not r["rehearsal_applied"] and r["sampling"] == {r["skill"]: 1.0} for r in records_n),
    "no rehearsal: every batch holds the new skill's rows only",
)
for record in records_n:
    checks.expect(parity_clean(record, student_n), f"no rehearsal, after adding {record['skill']}: every view reproduces the torch model")
checks.expect(records_r[0]["network_digests"] == records_n[0]["network_digests"], "first add identical in both runs (same seeds, nothing to rehearse)")
checks.expect(
    all(records_r[i]["network_digests"][s] != records_n[i]["network_digests"][s] for i in (1, 2) for s in records_r[i]["network_digests"]),
    "after the second and third add every network differs between the runs",
)
alpha_before, alpha_after = snapshots_n[0]["fx_alpha"], snapshots_n[2]["fx_alpha"]
checks.expect(
    np.array_equal(alpha_before.readout_weight, alpha_after.readout_weight) and np.array_equal(alpha_before.readout_bias, alpha_after.readout_bias)
    and np.array_equal(snapshots_n[1]["fx_beta"].readout_weight, snapshots_n[2]["fx_beta"].readout_weight),
    "no rehearsal: earlier heads' readouts are untouched by later training (no gradient reaches them)",
)
checks.expect(not np.array_equal(alpha_before.hidden_weights[0], alpha_after.hidden_weights[0]), "no rehearsal: the shared trunk moved under the earlier heads")
retention_n = student_n.retention_matrix()
print("\n  MECHANICS numbers on FIXTURE dreams, measured inside the dream on held-out dream rows")
print(f"  (max_steps={MAX_STEPS}; no claim about which run is better is made or asserted)")
for key in ("agreement_with_dream_choice", "mean_dream_regret_over_spread"):
    print(f"  {key}:")
    print_matrix("rehearsal=True", retention_r, key)
    print_matrix("rehearsal=False", retention_n, key)
for key in KEYS:
    print(f"  backward_transfer {key}: rehearsal=True {retention_r['backward_transfer'][key]:+.4f}   rehearsal=False {retention_n['backward_transfer'][key]:+.4f}")
checks.expect(retention_r != retention_n, "the two retention matrices differ")

print("\n== check 4: save, load, decide without torch in a fresh process, add a fourth skill ==")
scratch = Path(tempfile.mkdtemp(prefix="b_snn_shared_c1_"))
checks.redactions[str(scratch)] = "<tmp>"
try:
    directory = scratch / "shared_student"
    student_r.save(directory)
    files = sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())
    print(f"  saved files: {files}")
    expected_files = sorted(
        ["manifest.json", "model_state.npz"]
        + [f"networks/{s}.{ext}" for s in SEQUENCE for ext in ("json", "npz")]
        + [f"dreams/{s}.{ext}" for s in SEQUENCE for ext in ("json", "npz")]
    )
    checks.expect(files == expected_files, "layout: manifest, model state, one network view and one stored dream set per skill")
    checks.expect(not any(path.endswith((".pt", ".pkl", ".pickle")) for path in files), "no pickle or torch.save file")
    loaded = SharedSNNStudent.load(directory)
    checks.expect(loaded.skills_held == student_r.skills_held and loaded.network_digests() == student_r.network_digests(), "in-process load: same skills and network digests")
    checks.expect(loaded.retention_matrix() == retention_r and loaded.records == student_r.records, "in-process load: same records and retention matrix")
    checks.expect(loaded.settings == settings and loaded.rehearsal_fraction == student_r.rehearsal_fraction, "in-process load: same settings")
    checks.expect(
        all(np.array_equal(loaded.decide_index(s, situations[s]), indices[s]) for s in SEQUENCE)
        and all(np.array_equal(loaded.held_out_rows(s), student_r.held_out_rows(s)) for s in SEQUENCE),
        "in-process load: same decisions and held-out rows for every skill",
    )
    for name in SEQUENCE:
        np.save(scratch / f"{name}_situations.npy", situations[name])
        np.save(scratch / f"{name}_expected.npy", indices[name])
    (scratch / "expected_digests.json").write_text(json.dumps(student_r.network_digests()), encoding="utf-8")
    worker = Path(__file__).resolve().parent / "c4_load_worker.py"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(worker), str(directory), str(scratch), str(BASE_SEED)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    print(f"  fresh-process worker (exit {result.returncode}, {time.perf_counter() - started:.1f} s):")
    worker_result = None
    for line in (result.stdout + result.stderr).strip().splitlines():
        if line.startswith("RESULT_JSON "):
            worker_result = json.loads(line[len("RESULT_JSON "):])
            print(f"    RESULT_JSON skills_held={worker_result['skills_held']} steps_run={worker_result['steps_run']} parity_clean={worker_result['parity_clean']}")
        else:
            print(f"    {checks.clean(line)}")
    checks.expect(result.returncode == 0 and worker_result is not None, f"fresh process: worker exit code {result.returncode}")
    checks.expect("digests_match=True" in result.stdout and "decisions_identical=True" in result.stdout, "fresh process: loaded digests and decisions identical for every skill")
    checks.expect(
        "before adding a skill: torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout,
        "fresh process: load, decide, decide_index and operation_counts did not import torch",
    )
    record_delta = student_r.add_skill(specs["fx_delta"], dreams["fx_delta"], points_per_dim=POINTS["fx_delta"], base_seed=BASE_SEED, rehearsal=True)
    print(f"  in-memory add of fx_delta: steps_run={record_delta['steps_run']} sampling={ {k: round(v, 4) for k, v in record_delta['sampling'].items()} }")
    checks.expect(
        worker_result is not None and worker_result["skills_held"] == [*SEQUENCE, "fx_delta"] and worker_result["parity_clean"],
        "fresh process: fx_delta added to the loaded student, parity clean on all four views",
    )
    checks.expect(
        worker_result is not None and worker_result["digests"] == record_delta["network_digests"],
        "adding fx_delta to the loaded student (fresh process) and to the in-memory student gives identical digests for all four views",
    )
    retention_4 = student_r.retention_matrix()
    checks.expect(
        len(retention_4["agreement_with_dream_choice"]) == 4 and retention_4["rehearsal_applied"] == [False, True, True, True],
        "retention matrix grows to 4 x 4",
    )
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
