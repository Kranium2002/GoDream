"""Dev check for the heavy_payload preset's offset: a 3 kg payload centred, at each end of the trunk's length and at
the rear-left corner of payload_off_center_range_m, trotting 10 s at 0.6 and 0.9 m/s after the 2 s settle. Only the
payload effect is on; seed 5, attempt 0. Speed along the heading and yaw rate are means over the last 5 s.
Run from the repository root: uv run python -X utf8 artifacts/env/research/payload_offset_check.py
"""
import copy
import sys

import numpy as np

sys.path.insert(0, ".")
from learned_judgment.env import Go2World, load_config  # noqa: E402
from learned_judgment.env.effects import EFFECTS, heading_speed  # noqa: E402

config = load_config()
print("offset_m  commanded_m_s  speed_m_s  yaw_rate_rad_s  fall")
for offset in ([0.0, 0.0], [-0.1881, 0.0], [0.1881, 0.0], [-0.1881, 0.04675]):
    for speed in (0.6, 0.9):
        cfg = copy.deepcopy(config)
        cfg["world"]["hidden_effects"]["enabled"] = {name: name == "payload" for name in EFFECTS}
        cfg["world"]["presets"]["heavy_payload"]["payload_offset_m"] = offset
        cfg["world"]["save_traces"] = False
        w = Go2World(cfg, 5, preset="heavy_payload")
        w.reset({"dog_forward_speed_m_s": speed})
        obs = [w.step({"vel_x": speed, "vel_y": 0.0, "yaw_rate": 0.0}, 0.02) for _ in range(500)]
        fall = w.finish_episode()["fall"]
        mean_speed = np.mean([heading_speed(o) for o in obs[-250:]])
        yaw_rate = np.mean([o["base_ang_vel_rad_s"][2] for o in obs[-250:]])
        print(offset, speed, round(float(mean_speed), 4), round(float(yaw_rate), 4), fall, flush=True)
