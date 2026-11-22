"""Worker of run_sidestep_chain.py: reopen the DEV RUN in a fresh process and reload everything.

Reads the request JSON written by run_sidestep_chain.py, loads its config copy, builds one Go2World
(the run's world seed) for the sidestep_reach spec, reopens the SkillRun from its manifest (the
identity and the config digest are checked by SkillRun; nothing is written), reloads every run
result, the dream set and the student, and repeats decide_with for each brain on the given dev
situations. Prints one JSON line: metrics, actions, student digest, dream rows, manifest sha256.
Runs no episode. DEV RUN helper; not an evaluation result.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "experiments" / "proofs" / "B_sidestep_dev"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.dream import load_dreams  # noqa: E402
from learned_judgment.eval.report import read_run_result  # noqa: E402
from sidestep_reach_spec import for_world  # noqa: E402

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = load_config(request["config"])
factory = P.go2_world_factory(config)
world = factory(P.world_seed_for(request["base_seed"], request["preset"]), request["preset"])
spec = for_world(config, world)
manifest_path = P.RunPaths(config, request["run_id"]).manifest
before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
run = P.SkillRun(config, spec, world_factory=factory, run_id=request["run_id"], world_preset=request["preset"],
                 base_seed=request["base_seed"], clock=P.utc_now)
metrics = {name: read_run_result(path)["metrics"] for name, path in request["run_results"].items()}
actions = {name: [run.decide_with(name, situation)["action"] for situation in request["situations"]] for name in request["methods"]}
dreams, _ = load_dreams(run.paths.dreams)
after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
print(json.dumps({
    "metrics": metrics,
    "actions": actions,
    "student_digest": run.student().digest,
    "n_dreams": dreams.n_rows,
    "manifest_sha256": after if after == before else f"changed: {before} -> {after}",
}))
