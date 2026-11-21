"""Review reproduction: what build_final_table accepts as the 'artifact behind a number'.

The table should only show a number that was computed from a saved run artifact of that skill
and method. Below, hand-typed numbers are attached to files that have nothing to do with them.
Nothing is written outside a temporary directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from learned_judgment.eval.report import build_final_table, write_final_table  # noqa: E402

print("REVIEW REPRODUCTION - fixture, not evidence")

records = [
    # 1) a hand-typed number "backed" by the project config file
    {"skill": "dodge", "method": "teacher", "metrics": {"mean_abs_error": {"value": 0.0123, "artifact": "config.yaml"}}},
    # 2) a number "backed" by a DIFFERENT skill's / method's proof output
    {
        "skill": "dodge",
        "method": "gaussian_process",
        "metrics": {
            "mean_abs_error": {
                "value": 0.5,
                "artifact": "experiments/proofs/B_eval_offline/c6_report.out.txt",
            }
        },
    },
    # 3) a number "backed" by this very reproduction script
    {"skill": "dodge", "method": "physics_only", "metrics": {"mean_abs_error": {"value": 0.9, "artifact": str(Path(__file__).resolve())}}},
]
table = build_final_table(
    records,
    skills=["dodge"],
    methods=["physics_only", "gaussian_process", "teacher"],
    artifact_root=REPO,
)
print(table[["skill", "method", "status", "mean_abs_error", "mean_abs_error__artifact"]].to_string(index=False))

# A required method simply left out of `methods` gets no row at all (not a NOT RUN row).
table2 = build_final_table([], skills=["dodge"], methods=["teacher"], artifact_root=REPO)
print("\nmethods=['teacher'] only -> rows:", table2[["method", "status"]].values.tolist(),
      "| B2 constant_bias / B5 mlp_on_attempts / oracle rows present:",
      any(m in set(table2["method"]) for m in ("constant_bias", "mlp_on_attempts", "oracle")))

with tempfile.TemporaryDirectory() as tmp:
    write_final_table(table, Path(tmp) / "t.csv", Path(tmp) / "t.md")
    print("\nwritten markdown:\n" + (Path(tmp) / "t.md").read_text(encoding="utf-8"))
