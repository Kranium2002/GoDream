"""Check (d): save, load in a fresh process without torch, decide for every skill and stage, add a skill.

FIXTURE CHECK - mechanics only. fx_alpha -> fx_beta -> fx_gamma (rehearsal=True; max_steps reduced,
printed) are added in memory, every (skill, stage) decides on fixture situations, and the student is
saved to a temporary directory. Then:

1. layout: manifest, model state, one stored dream set per skill and one network view per skill per
   stage (stage i holds the skills added up to i); no pickle or torch.save file;
2. in-process load: same slots, rows, records, retention matrix, stage digests and decisions;
3. fresh process (c4_load_worker.py): load and decide for EVERY skill at EVERY stage (batched,
   per-decision path, decide with the spec, operation counts) with torch and snntorch absent from
   sys.modules; then add fx_delta to the loaded student there, save it again and reload it;
4. the same fx_delta add on the in-memory student gives identical digests for every view of the new
   stage, and the earlier stages are unchanged in both.

Old behaviour (committed shared_snn.py): save wrote only the latest views, so an earlier stage could
not be rebuilt (c7_old_behaviour.py). Files go to a temporary directory that is removed at the end.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c4_save_load.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_shared_specs import FIXTURE_FEATURE_WIDTH, POINTS, REPO_ROOT, SEQUENCE, Checks, fixture_config, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402

MAX_STEPS = 200  # reduced from config students.snn.max_steps for a short mechanics check
BASE_SEED = 7  # fixture run seed
CHECK_SITUATIONS = 200
ONE_BY_ONE = 20  # situations decided through the per-decision path

checks = Checks()
config = load_config()
check_config = fixture_config(config)
check_config["students"]["snn"]["max_steps"] = MAX_STEPS
print(
    f"\ncheck config: config.yaml with students.snn.shared_feature_width={FIXTURE_FEATURE_WIDTH} (fixture) and "
    f"students.snn.max_steps={MAX_STEPS} (config.yaml {config['students']['snn']['max_steps']}); every other value from config.yaml"
)
specs = {name: fixture_spec(name) for name in (*SEQUENCE, "fx_delta")}
dreams = {name: fixture_dreams(spec, config) for name, spec in specs.items()}

print("\n== train in memory: fx_alpha -> fx_beta -> fx_gamma (rehearsal=True) ==")
student = SharedSNNStudent.create(check_config)
for name in SEQUENCE:
    record = student.add_skill(specs[name], dreams[name], points_per_dim=POINTS[name], base_seed=BASE_SEED, rehearsal=True)
    print(f"  + {name} (slot {record['slot']}): steps_run={record['steps_run']} add wall {record['wall_seconds']:.1f} s")
situations = {name: specs[name].sample_situations(stream_rng(BASE_SEED, f"{name}/check/situations"), CHECK_SITUATIONS) for name in SEQUENCE}
expected = {
    f"{name}@{stage}": student.deployed(name, stage).decide_index(situations[name])
    for stage in range(student.n_stages)
    for name in student.stage_skills(stage)
}
print(f"  decisions recorded for {len(expected)} (skill, stage) views: {sorted(expected)}")
stage_digests = [student.network_digests(stage) for stage in range(student.n_stages)]

scratch = Path(tempfile.mkdtemp(prefix="b_snn_shared_c4_"))
checks.redactions[str(scratch)] = "<tmp>"
try:
    directory = scratch / "shared_student"
    student.save(directory)
    files = sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())
    print("\n== 1. layout ==")
    print(f"  saved files: {files}")
    expected_files = sorted(
        ["manifest.json", "model_state.npz"]
        + [f"dreams/{s}.{ext}" for s in SEQUENCE for ext in ("json", "npz")]
        + [f"networks/stage_{i}/{s}.{ext}" for i in range(3) for s in SEQUENCE[: i + 1] for ext in ("json", "npz")]
    )
    checks.expect(files == expected_files, "manifest, model state, one stored dream set per skill, one view per skill per stage (1 + 2 + 3 views)")
    checks.expect(not any(path.endswith((".pt", ".pkl", ".pickle")) for path in files), "no pickle or torch.save file")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checks.expect(
        manifest["slots"] == {"fx_alpha": 0, "fx_beta": 1, "fx_gamma": 2} and manifest["n_slots"] == student.n_slots
        and manifest["free_slots"] == student.n_slots - 3
        and [stage["skills_held"] for stage in manifest["stages"]] == [list(SEQUENCE[: i + 1]) for i in range(3)],
        f"the manifest records the slot assignment {manifest['slots']} and every stage's views",
    )

    print("\n== 2. in-process load ==")
    loaded = SharedSNNStudent.load(directory)
    checks.expect(
        loaded.skills_held == student.skills_held and loaded.slots == student.slots and loaded.n_slots == student.n_slots
        and loaded.settings == student.settings and loaded.retention_fraction == student.retention_fraction,
        "same skills, slots and settings",
    )
    checks.expect(
        [loaded.network_digests(stage) for stage in range(loaded.n_stages)] == stage_digests
        and loaded.records == student.records and loaded.retention_matrix() == student.retention_matrix(),
        "same digests at every stage, same records and retention matrix",
    )
    checks.expect(
        all(
            np.array_equal(getattr(loaded, part)(s), getattr(student, part)(s))
            for s in SEQUENCE
            for part in ("training_rows", "early_stopping_rows", "retention_rows")
        ),
        "same training, early-stopping and retention rows for every skill",
    )
    checks.expect(
        all(np.array_equal(loaded.deployed(key.split("@")[0], int(key.split("@")[1])).decide_index(situations[key.split("@")[0]]), value) for key, value in expected.items()),
        "same decisions for every skill at every stage",
    )

    print("\n== 3. fresh process: load, decide everywhere without torch, then add fx_delta ==")
    for name in SEQUENCE:
        np.save(scratch / f"{name}_situations.npy", situations[name])
    for key, value in expected.items():
        np.save(scratch / f"expected_{key.replace('@', '_stage_')}.npy", value)
    (scratch / "expected_stages.json").write_text(json.dumps(stage_digests), encoding="utf-8")
    worker = Path(__file__).resolve().parent / "c4_load_worker.py"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(worker), str(directory), str(scratch), str(BASE_SEED), str(ONE_BY_ONE)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    print(f"  fresh-process worker (exit {result.returncode}, {time.perf_counter() - started:.1f} s):")
    worker_result = None
    for line in (result.stdout + result.stderr).strip().splitlines():
        if line.startswith("RESULT_JSON "):
            worker_result = json.loads(line[len("RESULT_JSON "):])
            print(f"    RESULT_JSON skills_held={worker_result['skills_held']} slots={worker_result['slots']} n_stages={worker_result['n_stages']}")
        else:
            print(f"    {checks.clean(line)}")
    checks.expect(result.returncode == 0 and worker_result is not None, f"fresh process: worker exit code {result.returncode}")
    checks.expect(
        "loaded: stage_digests_match=True" in result.stdout and "decisions_identical_every_skill_and_stage=True" in result.stdout,
        "fresh process: loaded digests and decisions identical for every skill at every stage",
    )
    checks.expect(
        "before adding a skill: torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout,
        "fresh process: load, deployed(), decide_index, decide_index_one, decide, decide_one and operation_counts did not import torch",
    )

    print("\n== 4. the same add of fx_delta in memory ==")
    record_delta = student.add_skill(specs["fx_delta"], dreams["fx_delta"], points_per_dim=POINTS["fx_delta"], base_seed=BASE_SEED, rehearsal=True)
    print(f"  in-memory add of fx_delta (slot {record_delta['slot']}): steps_run={record_delta['steps_run']} sampling={ {k: round(v, 4) for k, v in record_delta['sampling'].items()} }")
    checks.expect(
        worker_result is not None and worker_result["skills_held"] == [*SEQUENCE, "fx_delta"]
        and worker_result["slots"] == {"fx_alpha": 0, "fx_beta": 1, "fx_gamma": 2, "fx_delta": 3} and worker_result["parity_clean"],
        "fresh process: fx_delta took slot 3 on the loaded student; parity clean on all four views",
    )
    checks.expect(
        worker_result is not None and worker_result["stages"][3] == record_delta["network_digests"]
        and worker_result["stages"][:3] == stage_digests == [student.network_digests(stage) for stage in range(3)],
        "adding fx_delta to the loaded student (fresh process) and to the in-memory student gives identical views at stage 3; "
        "stages 0-2 unchanged in both",
    )
    checks.expect(
        worker_result is not None and worker_result["reloaded_stages"] == worker_result["stages"],
        "fresh process: the grown student saved again and reloaded has the same four stages",
    )
    checks.expect(
        worker_result is not None and worker_result["retention_matrix"] == student.retention_matrix(),
        "fresh process and in-memory student: identical 4 x 4 retention matrices",
    )
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
