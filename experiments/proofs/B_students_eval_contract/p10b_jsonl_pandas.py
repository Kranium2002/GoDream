"""p10b: reading a JSONL attempt log back with pandas: float precision of pd.read_json vs json.loads + json_normalize. FIXTURE rows."""
import inspect, io, json, struct
from pathlib import Path
import numpy as np
import pandas as pd

SCR = Path(__file__).resolve().parent / "p10_scratch"; SCR.mkdir(exist_ok=True)
bits = lambda x: struct.unpack("<Q", struct.pack("<d", float(x)))[0]
rng = np.random.default_rng(1)
vals = (rng.normal(size=5000) * 10.0 ** rng.integers(-20, 20, 5000)).tolist()
rows = [{"run_id": "fixture", "seed": i, "skill": "dodge", "source": "real",
         "situation": {"ball_vx": v, "ball_x": -v / 3}, "action": {"side": i % 2, "t_cmd": v * 1e-3},
         "physics_guess": [v, 2 * v], "measured_outcome": [v + 1e-9, 2 * v], "residual": [1e-9, 0.0]} for i, v in enumerate(vals)]
p = SCR / "attempts_fixture.jsonl"
with open(p, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

print("pd.read_json precise_float default:", inspect.signature(pd.read_json).parameters["precise_float"].default)
d_default = pd.read_json(p, lines=True)
d_precise = pd.read_json(p, lines=True, precise_float=True)
src = [r["physics_guess"][0] for r in rows]
mis_default = sum(bits(a) != bits(b[0]) for a, b in zip(src, d_default["physics_guess"]))
mis_precise = sum(bits(a) != bits(b[0]) for a, b in zip(src, d_precise["physics_guess"]))
print(f"read_json(lines=True) default: {mis_default}/{len(src)} values not bitwise equal; precise_float=True: {mis_precise}/{len(src)}")
print("read_json nested dict column dtype:", d_default["situation"].dtype, "type of cell:", type(d_default["situation"][0]).__name__)
with open(p, encoding="utf-8") as f:
    recs = [json.loads(ln) for ln in f]
flat = pd.json_normalize(recs, sep=".")
mis_norm = sum(bits(a) != bits(b) for a, b in zip(src, flat["situation.ball_vx"]))
print("json.loads + pd.json_normalize: columns", [c for c in flat.columns][:8], "... | situation.ball_vx bitwise mismatches:", mis_norm)
print("json_normalize leaves lists as object cells:", type(flat["physics_guess"][0]).__name__)
