"""Check 5 (lead decision M5): latency is timed through a student bound once to its spec.

FIXTURE CHECK - mechanics only. The hand-built 2-3-2 network of c2 is wrapped in a DeployedStudent
built by hand (fixture_specs.hand_deployed) for the FIXTURE spec fixture_specs.FixtureHandSpec,
which counts its candidate_actions calls. Checked:

  * DeployedStudent.bind(spec) runs check_spec once (one candidate_actions call per stored probe
    situation); BoundStudent.decide_one then makes exactly one candidate_actions call per decision
    (the lookup), while the unbound DeployedStudent.decide_one repeats the grid check every call;
  * both bound paths decide like the unbound ones; a spec that differs is refused at binding, and
    a spec whose actions are replaced after binding is refused at the next decision;
  * measure_decision_latency binds once (candidate_actions calls during the whole measurement =
    probe rows + one per timed full-path call + two per situation for the after-timing comparison)
    and records both timings, which of them is for H3, and that the grid check ran at binding;
  * this process never imports torch, and a FRESH process that loads a saved hand-built student
    (snn.student_record v2 record, SNNStudent.save) and times it stays torch-free too.

Writes only inside a temporary directory, which is removed.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c5_latency_bound.py
"""

from _checks import BANNER, PROOF_DIR, REPO_ROOT, Checks

print(BANNER)

import dataclasses  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_specs import HAND_META_NOTE, ROOT3, FixtureHandSpec, hand_deployed  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.measure import measure_decision_latency  # noqa: E402
from learned_judgment.students.snn import (  # noqa: E402
    BoundStudent,
    GridMismatchError,
    OutOfRangeError,
    SNNStudent,
    situation_row_digests,
    student_record,
)
from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402

checks = Checks()
config = load_config()
N_DECISIONS = require(config, "evaluation.latency_decisions")
TRAINING = np.random.default_rng(7).uniform(-ROOT3, ROOT3, (20, 2))  # FIXTURE stand-in for "situations training saw"
SITUATIONS = np.random.default_rng(8).uniform(-ROOT3, ROOT3, (25, 2))  # FIXTURE situations to decide on
network = SpikingNetwork(
    hidden_weights=(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),),
    hidden_biases=(np.array([0.0, 0.0, 2.5]),),
    betas=(0.5,),
    thresholds=(1.0,),
    readout_weight=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    readout_bias=np.array([0.0, 0.0]),
    readout_beta=0.5,
    time_steps=3,
)
spec = FixtureHandSpec("fixture_hand", 2)
deployed = hand_deployed(network, spec, (2,), TRAINING)
k = deployed.grid_probe.shape[0]
n = SITUATIONS.shape[0]
print(f"\nhand student: {deployed.network.layer_sizes}, {k} stored probe situations, {n} FIXTURE situations; evaluation.latency_decisions={N_DECISIONS}")

print("\n== binding runs the grid check once ==")
before = spec.grid_calls
bound = deployed.bind(spec)
checks.expect(isinstance(bound, BoundStudent) and spec.grid_calls - before == k, f"bind: {k} candidate_actions calls (one per probe situation), i.e. one check_spec")
before = spec.grid_calls
bound_actions = np.stack([bound.decide_one(row) for row in SITUATIONS])
bound_calls = spec.grid_calls - before
before = spec.grid_calls
unbound_actions = np.stack([deployed.decide_one(spec, row) for row in SITUATIONS])
unbound_calls = spec.grid_calls - before
print(f"  candidate_actions calls for {n} decisions: bound {bound_calls}, unbound DeployedStudent.decide_one {unbound_calls}")
checks.expect(bound_calls == n, f"BoundStudent.decide_one: exactly one candidate_actions call per decision (the lookup), {n} in total")
checks.expect(unbound_calls == n * (k + 1), f"DeployedStudent.decide_one keeps its per-call grid check: {k} + 1 calls per decision")
checks.expect(bool(np.array_equal(bound_actions, unbound_actions)), "bound and unbound decide_one return the same command values")
indices = [bound.decide_index_one(row) for row in SITUATIONS]
checks.expect(indices == [deployed.decide_index_one(row) for row in SITUATIONS], "bound and unbound decide_index_one return the same indices")
checks.expect(
    all(np.array_equal(action, spec.candidate_actions(row, (2,))[index]) for action, row, index in zip(bound_actions, SITUATIONS, indices)),
    "the bound full path returns spec.candidate_actions(situation, points_per_dim)[index]",
)
checks.expect(bound.skill == deployed.skill and bound.digest == deployed.digest, "the bound student names the same skill and network digest")

print("\n== a spec that differs is refused at binding; a replaced spec is refused at the next decision ==")
checks.raises("bind to a spec with other action bounds", GridMismatchError, lambda: deployed.bind(FixtureHandSpec("fixture_hand", 2, choice_low=-0.6)), "low")
checks.raises("bind to a spec of another skill", ValueError, lambda: deployed.bind(FixtureHandSpec("fixture_other", 2)), "was trained for skill")
checks.raises("BoundStudent around something that is not a DeployedStudent", TypeError, lambda: BoundStudent(object(), spec), "must be a DeployedStudent")
replaced = FixtureHandSpec("fixture_hand", 2)
bound_replaced = deployed.bind(replaced)
replaced.actions = (dataclasses.replace(replaced.actions[0], high=0.6),)
checks.raises("decide_one after spec.actions was replaced", GridMismatchError, lambda: bound_replaced.decide_one(SITUATIONS[0]), "bind the student again")
checks.raises("decide_index_one after spec.actions was replaced", GridMismatchError, lambda: bound_replaced.decide_index_one(SITUATIONS[0]), "bind the student again")
checks.raises("bound decide_one outside the declared range (policy 'error')", OutOfRangeError, lambda: bound.decide_one(np.array([2.0, 0.0])), "outside its declared range")
log_before = deployed.out_of_range_log.decisions
bound.decide_one(SITUATIONS[0])
bound.decide_index_one(SITUATIONS[1])
checks.expect(deployed.out_of_range_log.decisions == log_before + 2, "bound decisions are counted in the student's out_of_range log")

print("\n== measure_decision_latency binds once and records both paths ==")
before = spec.grid_calls
latency = measure_decision_latency(deployed, spec, SITUATIONS, n_decisions=N_DECISIONS)
calls = spec.grid_calls - before
expected_calls = k + N_DECISIONS + 2 * n
print(
    f"  candidate_actions calls during the measurement: {calls} (= {k} at binding + {N_DECISIONS} timed full-path lookups + "
    f"2 x {n} after timing); a per-call grid check would have made at least {N_DECISIONS * (k + 1)}"
)
print(
    f"  MECHANICS timings on a fixture: full path median {latency['full_path']['median_s'] * 1e6:.1f} us p95 "
    f"{latency['full_path']['p95_s'] * 1e6:.1f} us; index only median {latency['index_only']['median_s'] * 1e6:.1f} us p95 "
    f"{latency['index_only']['p95_s'] * 1e6:.1f} us; bind {latency['grid_check']['bind_seconds'] * 1e3:.3f} ms"
)
checks.expect(calls == expected_calls, f"the grid check ran once: {calls} candidate_actions calls = {k} + {N_DECISIONS} + 2 x {n}")
checks.expect(
    latency["grid_check"]["ran"] == "once, at binding, before timing" and "check_spec" in latency["grid_check"]["definition"]
    and latency["grid_check"]["bind_seconds"] > 0.0,
    "the record says the grid check ran once at binding and how long binding took",
)
checks.expect(
    latency["h3_timing"] == "full_path" and latency["full_path"]["use_for_h3"] is True and latency["index_only"]["use_for_h3"] is False
    and "H3" in latency["h3_note"] and "full_path" in latency["h3_note"],
    "the record names full_path as the H3 timing",
)
checks.expect(
    "candidate_actions" in latency["full_path"]["includes"] and "command values" in latency["full_path"]["includes"]
    and "not included" in latency["index_only"]["includes"],
    "each timing states what it includes (full path: up to the command values; index only: no lookup)",
)
checks.expect(
    latency["full_path"]["n_decisions"] == latency["index_only"]["n_decisions"] == N_DECISIONS
    and 0.0 < latency["full_path"]["median_s"] <= latency["full_path"]["p95_s"]
    and 0.0 < latency["index_only"]["median_s"] <= latency["index_only"]["p95_s"]
    and latency["single_row_vs_batch_decisions_differing"] == 0 and latency["full_path_actions_not_the_looked_up_candidate"] == 0
    and latency["situations"]["n_rows"] == n and latency["situations"]["n_distinct_situations"] == n,
    "both timings over evaluation.latency_decisions calls, 0 < median <= p95; decisions consistent; rows and distinct situations stated",
)
checks.raises("latency with a spec of other bounds", GridMismatchError, lambda: measure_decision_latency(deployed, FixtureHandSpec("fixture_hand", 2, choice_high=0.6), SITUATIONS, n_decisions=5), "high")
checks.expect(latency["torch_free_process"] and "torch" not in sys.modules, "this process never imported torch (the record agrees)")

print("\n== a FRESH process loads a saved hand-built student, binds it and times it without torch ==")
encoder = InputEncoder.from_variables(spec.situation)
record = student_record(
    spec.name, encoder, deployed.points_per_dim, deployed.n_candidates, spec.actions,
    grid_probe=deployed.grid_probe, grid_sha256=deployed.grid_sha256,
    optimiser_situations=situation_row_digests(TRAINING[:16]), early_stopping_situations=situation_row_digests(TRAINING[16:]),
    out_of_range="error",
)
saved = SNNStudent(
    skill=spec.name, network=network, encoder=encoder, points_per_dim=deployed.points_per_dim, n_candidates=deployed.n_candidates,
    meta={"label": HAND_META_NOTE, "student": record, "dream": {"method": PHYSICS_ONLY, "request_hashes": [], "note": HAND_META_NOTE}},
)
scratch = Path(tempfile.mkdtemp(prefix="b_snn_measure_c5_"))
try:
    stem = scratch / "fixture_hand_student"
    saved.save(stem)
    np.save(scratch / "situations.npy", SITUATIONS)
    result = subprocess.run(
        [sys.executable, str(PROOF_DIR / "c1_latency_worker.py"), str(stem), str(scratch / "situations.npy"), "fixture_hand"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    fresh = None
    for line in (result.stdout + result.stderr).strip().splitlines():
        if line.startswith("LATENCY_JSON "):
            fresh = json.loads(line[len("LATENCY_JSON "):])
        else:
            print(f"    {line.replace(str(scratch), '<tmp>')}")
    checks.expect(result.returncode == 0 and fresh is not None, f"fresh-process worker exit code {result.returncode}")
    if fresh is not None:
        print(
            f"  MECHANICS latency, fresh process: full path median {fresh['full_path']['median_s'] * 1e6:.1f} us, index only median "
            f"{fresh['index_only']['median_s'] * 1e6:.1f} us; torch_free_process={fresh['torch_free_process']}"
        )
        checks.expect(
            fresh["torch_free_process"] and "torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout
            and fresh["h3_timing"] == "full_path" and fresh["grid_check"]["ran"] == "once, at binding, before timing",
            "the worker loaded, bound and timed the student without importing torch or snntorch",
        )
        checks.expect(fresh["network_sha256"] == deployed.digest, "the fresh process timed the same network (digest)")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
