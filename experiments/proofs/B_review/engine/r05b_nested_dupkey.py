"""R05b: a duplicate key nested inside measured_outcome is rejected by read_rows."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import TS, ReviewSpec  # noqa: E402

from learned_judgment.engine.rows import build_real_row, read_rows  # noqa: E402

spec = ReviewSpec()
r = build_real_row(spec, run_id="rv", seed=3, world_preset="default", situation=np.array([0.1, 1.0]),
                   action=np.array([0.5, 0.1]), measured_outcome={"rv_sep": 0.3, "rv_hit": 0.0}, timestamp_utc=TS)
line = r.to_json_line().decode()
target = '"measured_outcome":{"rv_hit":0.0,'
print("target present:", target in line)
bad = line.replace(target, '"measured_outcome":{"rv_hit":1.0,"rv_hit":0.0,', 1)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "k.jsonl"
    path.write_text(bad, encoding="utf-8", newline="\n")
    try:
        print("accepted", len(read_rows(path)))
    except ValueError as error:
        print("nested duplicate key ->", str(error).split(": ", 1)[1])
