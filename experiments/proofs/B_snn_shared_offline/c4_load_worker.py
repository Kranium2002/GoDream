"""Check (d) worker: load a saved shared student in a FRESH process, decide for every skill at every
stage, report whether torch was imported, then add fx_delta to the loaded student, save it again
and reload it. Started by c4_save_load.py with paths inside its temporary directory.

FIXTURE CHECK - mechanics only. Usage:
    python c4_load_worker.py <student directory> <scratch directory> <base_seed> <n one by one>
Prints RESULT_JSON with the digests of every stage after the add. Exit code 0 when the loaded
digests and decisions are identical and neither torch nor snntorch was imported before the add.
Writes only inside <scratch directory>.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_shared_specs import POINTS, fixture_dreams, fixture_spec  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students.shared_snn import SharedSNNStudent  # noqa: E402

directory, scratch, base_seed, one_by_one = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
student = SharedSNNStudent.load(directory)
expected_stages = json.loads((scratch / "expected_stages.json").read_text(encoding="utf-8"))
stages_ok = [student.network_digests(stage) for stage in range(student.n_stages)] == expected_stages
identical = True
views = 0
for stage in range(student.n_stages):
    for name in student.stage_skills(stage):
        spec = fixture_spec(name)
        situations = np.load(scratch / f"{name}_situations.npy", allow_pickle=False)
        expected = np.load(scratch / f"expected_{name}_stage_{stage}.npy", allow_pickle=False)
        deployed = student.deployed(name, stage)
        index = deployed.decide_index(situations)
        single = np.array([deployed.decide_index_one(s) for s in situations[:one_by_one]])
        actions = deployed.decide(spec, situations)
        one_actions = np.stack([deployed.decide_one(spec, s) for s in situations[:one_by_one]])
        counts = deployed.operation_counts(situations)
        identical = (
            identical
            and bool(np.array_equal(index, expected))
            and bool(np.array_equal(single, expected[:one_by_one]))
            and bool(np.array_equal(one_actions, actions[:one_by_one]))
            and actions.shape[0] == counts.n_decisions == index.shape[0]
            and counts.layer_sizes[0] == student.input_width
            and deployed.meta["stage"] == stage
            and deployed.meta["slot"] == student.slot_of(name)
        )
        views += 1
torch_loaded, snntorch_loaded = "torch" in sys.modules, "snntorch" in sys.modules
print(f"worker: loaded skills_held={list(student.skills_held)} slots={student.slots} n_stages={student.n_stages}")
print(f"worker: loaded: stage_digests_match={stages_ok}")
print(f"worker: {views} (skill, stage) views decided: decisions_identical_every_skill_and_stage={identical}")
print(f"worker: before adding a skill: torch_in_sys_modules={torch_loaded} snntorch_in_sys_modules={snntorch_loaded}")

spec = fixture_spec("fx_delta")
record = student.add_skill(spec, fixture_dreams(spec, load_config()), points_per_dim=POINTS["fx_delta"], base_seed=base_seed, rehearsal=True)
print(f"worker: after add_skill(fx_delta): torch_in_sys_modules={'torch' in sys.modules}; slot {record['slot']}")
parity_clean = all(
    all(m == 0 for m in report["spike_train_mismatches_per_layer"])
    and report["decisions_differing"] == 0
    and report["decide_one_vs_batch_differing"] == 0
    and report["max_abs_logit_difference"] <= report["logit_tolerance"]
    for report in record["parity"].values()
)
stages = [student.network_digests(stage) for stage in range(student.n_stages)]
grown = scratch / "grown_student"
student.save(grown)
reloaded = SharedSNNStudent.load(grown)
print(
    "RESULT_JSON "
    + json.dumps(
        {
            "skills_held": list(student.skills_held),
            "slots": student.slots,
            "n_stages": student.n_stages,
            "stages": stages,
            "reloaded_stages": [reloaded.network_digests(stage) for stage in range(reloaded.n_stages)],
            "retention_matrix": student.retention_matrix(),
            "parity_clean": parity_clean,
        }
    )
)
raise SystemExit(0 if stages_ok and identical and not torch_loaded and not snntorch_loaded else 1)
