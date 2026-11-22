"""Worker of c1 (PLUMBING CHECK): reload a pipeline run from disk in a fresh process.

Reads the paths file written by c1_chain.py and prints one JSON line with what it reloaded:
step states, history, evaluations, attempt rows, the dream set, the student's digest and one
decision, and the metrics of every run-result file (eval.report.read_run_result re-checks the
digests of the harness manifests and attempt files). Fixture data only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import FixturePipelineSpec  # noqa: E402

from learned_judgment.engine.dream import load_dreams  # noqa: E402
from learned_judgment.engine.rows import read_rows  # noqa: E402
from learned_judgment.eval.report import read_run_result  # noqa: E402
from learned_judgment.pipeline import STEPS, read_run_manifest  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402

paths = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
spec = FixturePipelineSpec()
manifest = read_run_manifest(paths["manifest"])
rows = read_rows(Path(paths["attempts"]), spec=spec)
dreams, _ = load_dreams(Path(paths["dreams"]))
student = SNNStudent.load(paths["student"])
situation = spec.encode_situation({"fx_lateral": 0.25, "fx_speed": 4.0, "fx_surface": "slip"})
print(
    json.dumps(
        {
            "steps": {step: manifest["steps"][step]["state"] for step in STEPS},
            "history_steps": [entry["step"] for entry in manifest["history"]],
            "evaluations": [entry["evaluation_id"] for entry in manifest["evaluations"]],
            "n_attempts": len(rows),
            "attempt_seeds": [row.seed for row in rows],
            "n_dreams": dreams.n_rows,
            "student_digest": student.digest,
            "student_decision": int(student.decide_index_one(situation)),
            "metrics": [read_run_result(path)["metrics"] for path in paths["results"]],
        }
    )
)
