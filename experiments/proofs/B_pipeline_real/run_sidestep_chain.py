"""DEV RUN: the whole pipeline chain on the REAL Go2 simulator with the sidestep_reach dev skill.

DEV RUN on the real Go2 simulator (Person A's Go2World through pipeline.go2_world_factory, preset
default, no ball). The skill is the development skill of experiments/proofs/B_sidestep_dev/
(sidestep_reach_spec.py, "reach a given sideways displacement"), which is not a registered project
skill. Nothing here is tuned, and no number here is an evaluation result or enters a report or
table: every file goes to experiments/proofs/B_pipeline_real/out/<UTC stamp>/.

Steps, all through learned_judgment.pipeline.SkillRun (Python API):
    collect   pipeline.default_attempts real attempts (context split, exploration actions)
    teacher   residual method "gradient_boosting" (a local baseline of eval.baselines: nothing is
              sent to Prior Labs)
    dream     engine.dreams_per_skill_initial dreams with that model
    train     the per-skill spiking student, trained in the dream (students.snn settings unchanged)
    evaluate  physics_only, gradient_boosting, snn_per_skill on the dev split,
              evaluation.dev_situations_per_skill situations x pipeline.evaluation_repeats repeats
    decide    decide_with for each brain on dev situations 0, 1 and 2

Config copy (printed below): every output directory, the teacher cache and call log and the
test-access log are redirected into the out directory; world.save_traces is set from
dev_checks.sidestep_reach.save_traces; skills.sidestep_reach.candidate_points_per_dim mirrors
dev_checks.sidestep_reach.candidate_points_per_dim (the pipeline reads a skill's grid from
skills.<skill>); tabpfn.project_token_budget is set to null and every billable tabpfn_client entry
point is replaced by a tripwire, so no request can be sent even by mistake.

Checks (the script exits 1 if any fails):
  * attempt rows: situations, actions and attempt seeds equal an independent recomputation from the
    documented streams (eval.splits, engine.collect.exploration_actions), world preset "default";
  * every evaluation used the run's world seed and the dev split's situations and noise seeds, the
    same for every method;
  * the 40 attempts and the first 60 physics_only dev episodes equal, bit for bit, the rows of the
    same situations, actions and attempt seeds in B_sidestep_dev c3 (out/c3_20261121T172039Z), which
    ran outside the pipeline on one world built from another world seed. This holds only while
    Go2World has no hidden effects and draws no noise (HANDOFF_PERSON_A item 8; B_sidestep_dev c2);
  * decide_with of each brain chooses, on dev situations 0-2, the action its evaluation ran;
  * a fresh process (run_sidestep_chain_reload_worker.py) reopens the run from the manifest (identity
    and config digest checked), reloads every run result, the dreams and the student, makes the same
    decide_with decisions, and leaves the manifest bytes unchanged;
  * no tripwire was hit, no provider object was built, the redirected call log has no dispatch.

Run from the repository root (takes several minutes; it imports torch):
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_real\\run_sidestep_chain.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SIDESTEP_DIR = ROOT / "experiments" / "proofs" / "B_sidestep_dev"
C3_DIR = SIDESTEP_DIR / "out" / "c3_20261121T172039Z"  # the sidestep builder's final c3 run (committed)
for entry in (ROOT, SIDESTEP_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

LABEL = (
    "DEV RUN on the real Go2 simulator (Person A's Go2World, preset default, no ball) - pipeline chain of the "
    "sidestep_reach dev skill; not an evaluation result, not in any report or table"
)
print(LABEL, flush=True)

import numpy as np  # noqa: E402
import tabpfn_client  # noqa: E402
import yaml  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402
from tabpfn_client.service_wrapper import InferenceClient  # noqa: E402

TRIPWIRE_HITS: list[str] = []


def install_tripwire(owner, name: str) -> None:  # noqa: ANN001
    label = f"{owner.__name__}.{name}"

    def tripwire(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        TRIPWIRE_HITS.append(label)
        raise AssertionError(f"billable entry point {label} was called")

    setattr(owner, name, tripwire)


installed = 0
for owner, names in (
    (tabpfn_client.TabPFNRegressor, ("fit", "_fit", "predict", "_predict", "submit_fit", "submit_predict", "wait_for_predict",
                                     "cross_val_predict", "submit_cross_val_predict", "cross_val_predict_splits")),
    (InferenceClient, ("fit", "fit_with_result", "predict", "predict_chunks", "submit_predict")),
    (ServiceClient, ("fit", "fit_with_result", "_fit", "_submit_fit_job", "_upload_train_set", "_upload_fold_ids", "predict",
                     "predict_chunks", "submit_predict", "_predict", "_start_predict", "_sync_predict", "_async_predict",
                     "_submit_predict_job", "_upload_test_set", "_upload_to_gcs", "_upload_single_chunk",
                     "_submit_cross_validate_job")),
):
    for name in names:
        if hasattr(owner, name):
            install_tripwire(owner, name)
            installed += 1
print(f"tripwires installed on {installed} billable tabpfn_client entry points", flush=True)

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.collect import exploration_actions  # noqa: E402
from learned_judgment.engine.dream import load_dreams  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.eval.harness import read_manifest  # noqa: E402
from learned_judgment.eval.report import read_run_result  # noqa: E402
from learned_judgment.eval.splits import noise_seed, noise_seeds, split_situations  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from sidestep_reach_spec import ROOT as SPEC_ROOT  # noqa: E402
from sidestep_reach_spec import for_world  # noqa: E402

FAILURES: list[str] = []
PASSED = [0]


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED[0] += 1
    else:
        FAILURES.append(label)
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""), flush=True)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_key(row: dict) -> tuple:
    """(situation, action, attempt seed, measured outcome) of a stored row, as exact JSON values."""
    return (json.dumps(row["situation"], sort_keys=True), json.dumps(row["action"], sort_keys=True), int(row["seed"]),
            json.dumps(row["measured_outcome"], sort_keys=True))


# ----- config copy -----------------------------------------------------------------------------
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = HERE / "out" / stamp
out.mkdir(parents=True, exist_ok=False)
base = load_config()
config = copy.deepcopy(base)
changes = {
    "engine.attempts_dir": (out / "attempts").as_posix(),
    "engine.dreams_dir": (out / "dreams").as_posix(),
    "pipeline.manifests_dir": (out / "manifests").as_posix(),
    "pipeline.students_dir": (out / "students").as_posix(),
    "pipeline.evaluation_rows_dir": (out / "evaluation_rows").as_posix(),
    "pipeline.run_results_dir": (out / "run_results").as_posix(),
    "tabpfn.cache_dir": (out / "tabpfn_cache").as_posix(),
    "tabpfn.call_log": (out / "tabpfn_calls.jsonl").as_posix(),
    "evaluation.test_access_log": (out / "test_access_log.jsonl").as_posix(),
    "tabpfn.project_token_budget": None,
    "world.save_traces": require(base, f"{SPEC_ROOT}.save_traces"),
}
for dotted, value in changes.items():
    section, key = dotted.split(".")
    config[section][key] = value
config["skills"]["sidestep_reach"] = {"candidate_points_per_dim": list(require(base, f"{SPEC_ROOT}.candidate_points_per_dim"))}
config_path = out / "config_copy.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
config = load_config(config_path)  # the run uses exactly what the reload worker will read
rel = lambda path: Path(path).relative_to(ROOT).as_posix()  # noqa: E731
print(f"output directory {rel(out)}")
print("config copy (config.yaml with these keys changed):")
for dotted, value in changes.items():
    print(f"  {dotted}: {value if not isinstance(value, str) or not value.startswith(out.as_posix()) else rel(value)}")
print(f"  skills.sidestep_reach.candidate_points_per_dim: {config['skills']['sidestep_reach']['candidate_points_per_dim']} "
      f"(mirrors {SPEC_ROOT}.candidate_points_per_dim)")

base_seed = int(require(config, "runtime.seed"))
preset = str(require(config, f"{SPEC_ROOT}.world_preset"))
n_attempts = int(require(config, "pipeline.default_attempts"))
n_dreams = int(require(config, "engine.dreams_per_skill_initial"))
n_dev = int(require(config, "evaluation.dev_situations_per_skill"))
n_repeats = int(require(config, "pipeline.evaluation_repeats"))
method = "gradient_boosting"
methods = ["physics_only", method, "snn_per_skill"]
run_id = "sidestep-chain"
git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
pipeline_sha = hashlib.sha256((ROOT / "learned_judgment" / "pipeline.py").read_bytes()).hexdigest()
import mujoco  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

versions = {"python": sys.version.split()[0], "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "mujoco": mujoco.__version__, "torch": torch.__version__, "platform": platform.platform()}
print(f"versions {versions}")
print(f"git HEAD {git_head}; learned_judgment/pipeline.py sha256 {pipeline_sha} (working tree)")
print(f"runtime.seed {base_seed} | preset {preset} | attempts {n_attempts} (pipeline.default_attempts) | dreams {n_dreams} "
      f"(engine.dreams_per_skill_initial) | dev situations {n_dev} (evaluation.dev_situations_per_skill) x repeats "
      f"{n_repeats} (pipeline.evaluation_repeats) | residual method {method} | config digest {config_digest(config)[:12]}")

# ----- open the run ----------------------------------------------------------------------------
wall: dict[str, float] = {}


def on_event(event: dict) -> None:
    if event["event"] != "progress" or (event["done"] or 0) % 25 == 0:
        print(f"  [{event['step']:<8} {event['event']:<8} {event['done']}/{event['total']}] {event['message']}", flush=True)


began = time.perf_counter()
factory = P.go2_world_factory(config)
world_seed = P.world_seed_for(base_seed, preset)
probe = factory(world_seed, preset)
spec = for_world(config, probe)
run = P.SkillRun(config, spec, world_factory=factory, run_id=run_id, world_preset=preset, base_seed=base_seed,
                 clock=P.utc_now, progress=on_event)
wall["open (probe world + spec + manifest)"] = time.perf_counter() - began
print(f"world seed {world_seed} (stream '{run.streams['world_seed']}'), world: {run.world_description}")

print("\n== collect ==", flush=True)
began = time.perf_counter()
rows = run.collect_attempts(n_attempts)
wall["collect"] = time.perf_counter() - began

print("\n== teacher (gradient_boosting, local) ==", flush=True)
began = time.perf_counter()
model = run.residual_model(method)
wall["teacher"] = time.perf_counter() - began

print("\n== dream ==", flush=True)
began = time.perf_counter()
dreams = run.dream(method, n_dreams)
wall["dream"] = time.perf_counter() - began
print(f"  dreamed {run.manifest['steps']['dream']['counts']['n_dreamed']}, kept {dreams.n_rows} x {dreams.n_candidates} candidates; "
      f"threshold {run.manifest['steps']['dream']['settings']['confidence_threshold']} "
      f"({run.manifest['steps']['dream']['settings']['confidence_threshold_reason']})")

print("\n== train (in the dream) ==", flush=True)
began = time.perf_counter()
student = run.train_student()
wall["train"] = time.perf_counter() - began
print(f"  network sha256 {student.digest}")

print("\n== evaluate on dev ==", flush=True)
began = time.perf_counter()
evaluation = run.evaluate(methods, split="dev", n_situations=n_dev, n_repeats=n_repeats)
wall["evaluate"] = time.perf_counter() - began

dev = split_situations(spec, base_seed=base_seed, split="dev", n=n_dev, world_preset=preset)
print("\n== decide_with on dev situations 0, 1, 2 ==", flush=True)
decisions: dict[str, list[dict]] = {}
began = time.perf_counter()
for brain in methods:
    decisions[brain] = []
    for index in range(3):
        answer = run.decide_with(brain, spec.decode_situation(dev[index]))
        decisions[brain].append(answer)
        summary = {"chosen_index": answer["chosen_index"], "action": answer["action"], "physics_index": answer["physics_index"],
                   "physics_guess": answer["physics_guess"]}
        if answer["predicted_outcome"] is not None:
            summary["predicted_mean"] = answer["predicted_outcome"]["mean"]
            summary["predicted_spread"] = answer["predicted_outcome"]["spread"]
        summary["score_kind"] = answer["score_kind"]
        summary["chosen_score"] = answer["scores"][answer["chosen_index"]]
        print(f"  {brain:<18} dev {index} situation {answer['situation']}")
        print(f"  {'':<18}        -> {json.dumps(summary, sort_keys=True)}")
wall["decide_with (9 decisions)"] = time.perf_counter() - began

# ----- results as stored ------------------------------------------------------------------------
print("\n== per-method dev metrics, as the run-result files hold them ==")
results = {}
for name in methods:
    item = evaluation["methods"][name]
    stored = read_run_result(item["run_result"])
    results[name] = stored
    print(f"  {name:<18} {rel(item['run_result'])}")
    print(f"  {'':<18} {json.dumps(stored['metrics'], sort_keys=True)}")

print("\n== wall time of each step ==")
for label, seconds in wall.items():
    print(f"  {label:<40} {seconds:9.1f} s  (this script, time.perf_counter)")
manifest = P.read_run_manifest(run.paths.manifest)
print("  manifest history (wall_s of each recorded call):")
for entry in manifest["history"]:
    print(f"    {entry['step']:<9} {entry['state']:<8} {entry['wall_s']:9.1f} s  {entry['started_utc']} -> {entry['finished_utc']}")
episodes = n_attempts + len(methods) * n_dev * n_repeats
print(f"  simulator episodes {episodes}: collect {manifest['steps']['collect']['wall_s']:.1f} s for {n_attempts} "
      f"({manifest['steps']['collect']['wall_s'] / n_attempts:.2f} s each), evaluate "
      f"{manifest['steps']['evaluate']['wall_s']:.1f} s for {len(methods) * n_dev * n_repeats} including the decisions")

# ----- checks -----------------------------------------------------------------------------------
print("\n== checks ==")
stored_rows = jsonl(run.paths.attempts)
context = split_situations(spec, base_seed=base_seed, split="context", n=n_attempts, world_preset=preset)
from learned_judgment.engine.features import build_layout  # noqa: E402

planned_actions = exploration_actions(
    spec, context, rng=stream_rng(base_seed, f"{spec.name}/context/{preset}/exploration"),
    points_per_dim=config["skills"]["sidestep_reach"]["candidate_points_per_dim"],
    exploration_fraction=require(config, "engine.attempt_exploration_fraction"),
    quantile_levels=require(config, "engine.quantile_levels"), spread_levels=tuple(require(config, "engine.spread_levels")),
    layout=build_layout(spec),
)
check("attempt rows: situations equal split 'context' 0..n-1",
      np.array_equal(np.array([spec.encode_situation(r["situation"]) for r in stored_rows]), context))
check("attempt rows: actions equal engine.collect.exploration_actions on the documented stream",
      np.array_equal(np.array([spec.encode_action(r["action"]) for r in stored_rows]), planned_actions))
check("attempt rows: attempt seeds equal noise_seed(split 'context', index i, repeat 0)",
      [int(r["seed"]) for r in stored_rows] == [noise_seed(spec, base_seed=base_seed, split="context", world_preset=preset,
                                                            situation_index=i, repeat=0) for i in range(n_attempts)])
check("attempt rows: world preset and run id", all(r["world_preset"] == preset and r["run_id"] == run_id and r["source"] == "real"
                                                   for r in stored_rows))
dev_seeds = noise_seeds(spec, base_seed=base_seed, split="dev", world_preset=preset, n_situations=n_dev, n_repeats=n_repeats)
harness = {name: read_manifest(evaluation["methods"][name]["harness_manifest"]) for name in methods}
check("every evaluation ran in the run's world seed", all(h["world_seed"] == manifest["world_seed"] == world_seed for h in harness.values()))
check("every method met the dev split's noise seeds in the same order",
      all([e[2] for e in h["episodes"]] == [int(s) for s in dev_seeds[:, 0]] for h in harness.values()))
eval_rows = {name: jsonl(evaluation["methods"][name]["attempts"]) for name in methods}
check("every method ran the dev split's situations",
      all(np.array_equal(np.array([spec.encode_situation(r["situation"]) for r in eval_rows[name]]), dev) for name in methods))

if C3_DIR.exists():
    c3 = jsonl(C3_DIR / "attempts.jsonl")
    c3_explore = [r for r in c3 if r["run_id"].endswith("/context/explore")]
    c3_dev = [r for r in c3 if r["run_id"].endswith("/dev/physics_only")]
    same_context = [episode_key(a) == episode_key(b) for a, b in zip(stored_rows, c3_explore)]
    same_dev = [episode_key(a) == episode_key(b) for a, b in zip(eval_rows["physics_only"][: len(c3_dev)], c3_dev)]
    check(f"the {n_attempts} attempts equal c3's context/explore rows bit for bit (situation, action, seed, outcome)",
          len(c3_explore) == n_attempts and all(same_context), f"{sum(same_context)} of {len(c3_explore)} equal")
    check(f"physics_only dev episodes 0..{len(c3_dev) - 1} equal c3's dev/physics_only rows bit for bit",
          len(c3_dev) > 0 and all(same_dev), f"{sum(same_dev)} of {len(c3_dev)} equal")
else:
    print(f"NOT RUN  comparison with B_sidestep_dev c3: {rel(C3_DIR)} does not exist")

for name in methods:
    ran = [eval_rows[name][i]["action"] for i in range(3)]
    chosen = [decisions[name][i]["action"] for i in range(3)]
    check(f"decide_with({name!r}) chooses on dev 0-2 the action its evaluation ran", ran == chosen, f"{chosen} vs {ran}")

manifest_sha = hashlib.sha256(run.paths.manifest.read_bytes()).hexdigest()
request = {
    "config": config_path.as_posix(), "run_id": run_id, "preset": preset, "base_seed": base_seed, "methods": methods,
    "situations": [spec.decode_situation(dev[i]) for i in range(3)],
    "run_results": {name: evaluation["methods"][name]["run_result"] for name in methods},
}
request_path = out / "reload_request.json"
request_path.write_text(json.dumps(request, default=float), encoding="utf-8")
worker = subprocess.run([sys.executable, str(HERE / "run_sidestep_chain_reload_worker.py"), str(request_path)],
                        cwd=ROOT, capture_output=True, text=True, timeout=1800)
reloaded = json.loads(worker.stdout.strip().splitlines()[-1]) if worker.returncode == 0 and worker.stdout.strip() else {}
check("fresh process: reopened the run (identity and config digest) and reloaded every file", worker.returncode == 0,
      f"exit {worker.returncode} {worker.stderr[-600:]}")
check("fresh process: run results reload with the same metrics",
      reloaded.get("metrics") == {name: results[name]["metrics"] for name in methods})
check("fresh process: same decide_with decisions for every brain",
      reloaded.get("actions") == {name: [d["action"] for d in decisions[name]] for name in methods})
check("fresh process: student reloads with the same network digest", reloaded.get("student_digest") == student.digest)
check("fresh process: dream set reloads with the same rows", reloaded.get("n_dreams") == dreams.n_rows)
check("fresh process: the manifest bytes are unchanged by reopening and deciding", reloaded.get("manifest_sha256") == manifest_sha)
check("no billable tabpfn_client entry point was called", TRIPWIRE_HITS == [], str(TRIPWIRE_HITS))
check("no provider object was built by the run", run._provider is None)  # noqa: SLF001
call_log = Path(changes["tabpfn.call_log"])
check("the redirected call log holds no dispatch", not call_log.exists() or not any(r.get("billable_dispatch") for r in jsonl(call_log)))

print("\n== the run manifest ==")
print(json.dumps(manifest, indent=1, sort_keys=True))
summary = {"label": LABEL, "stamp": stamp, "versions": versions, "git_head": git_head, "pipeline_sha256": pipeline_sha,
           "wall_s": wall, "metrics": {name: results[name]["metrics"] for name in methods},
           "checks_passed": PASSED[0], "checks_failed": FAILURES}
(out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(f"\n{PASSED[0]} passed, {len(FAILURES)} failed {FAILURES or ''}")
sys.exit(1 if FAILURES else 0)
