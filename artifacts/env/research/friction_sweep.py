"""Dev sweep for the slippery floor type: the foot-floor sliding friction at which the dog still trots at 0.6 m/s
(20 s after the 2 s settle) and stops from 0.9 m/s without falling. Only the friction effect is on, without per-run
variation; seed 5, attempt 0. Measurements as in learned_judgment/env/effects.py.
Run from the repository root: uv run python -X utf8 artifacts/env/research/friction_sweep.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from learned_judgment.env import Go2World, load_config  # noqa: E402
from learned_judgment.env.effects import EFFECTS, measure  # noqa: E402

FRICTIONS = [0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
config = load_config()
m = config["world"]["hidden_effects"]["measure"]
rows = []
print("friction  trot_0.6_speed  trot_fall  stop_0.9_distance  stop_fall")
for f in FRICTIONS:
    cfg = copy.deepcopy(config)
    he = cfg["world"]["hidden_effects"]
    he["enabled"] = {name: name == "friction" for name in EFFECTS}
    he["surface_friction_by_type"]["sweep"], he["friction_per_run_variation"] = f, 0.0
    cfg["world"]["presets"]["default"]["floor_type"] = "sweep"
    cfg["world"]["save_traces"] = False
    speed, trot_fall = measure(Go2World(cfg, 5), "steady_speed_m_s", m | {"duration_s": 20.0})
    distance, stop_fall = measure(Go2World(cfg, 5), "stopping_distance_m", m)
    rows.append({"friction": f, "trot_0.6_speed_m_s": speed, "trot_fall": trot_fall,
                 "stop_0.9_distance_m": distance, "stop_fall": stop_fall})
    print(f"{f:8.3f}  {speed:14.4f}  {trot_fall:9d}  {distance:17.4f}  {stop_fall:9d}", flush=True)
out = Path("artifacts/env/friction-sweep/sweep.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seed": 5, "attempt_seed": 0, "trot_duration_s": 20.0, "measure": m, "rows": rows}, indent=2))
print("saved", out.as_posix())
