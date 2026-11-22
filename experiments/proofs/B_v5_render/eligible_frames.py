"""v5 check (Person B): at vision.frame_rate_hz, how many of 500 sampled scenarios have NO eligible decision frame
before the object's straight-line arrival (dodge_vision._frames_before_arrival falls back to the earliest frame)."""
import copy
from collections import Counter

import numpy as np

from learned_judgment.config import load_config
from learned_judgment.skills import dodge_vision as dv

N = 500


def count(config, split):
    eligible = dv._eligible_frames(config, 8)
    times = dv.frame_times(config)
    scenarios, _ = dv._split_plan(config, split, N)
    aim_x = config["skills"]["dodge_vision"]["punch"]["aim_x_m"]
    margin = config["skills"]["dodge_vision"]["pre_contact_margin_s"]
    none, approaching = Counter(), Counter()
    arrivals = []
    for s in scenarios:
        sit = s["situation"]
        if "ball_vel_x_m_s" not in sit or sit["ball_vel_x_m_s"] >= -1e-9:
            continue
        approaching[s["kind"]] += 1
        arrival = (sit["ball_rel_x_m"] - aim_x) / -sit["ball_vel_x_m_s"]
        arrivals.append(arrival)
        if times[eligible[0]] > arrival - margin + 1e-9:
            none[s["kind"]] += 1
    return sum(none.values()), sum(approaching.values()), dict(none), dict(approaching), np.array(arrivals)


print("module", dv.__file__)
config = load_config()
for rate in (15.0, 30.0):
    c = copy.deepcopy(config)
    c["vision"]["frame_rate_hz"] = rate
    for split in ("dev", "dream"):
        k, a, none, appr, arrivals = count(c, split)
        first = dv.frame_times(c)[dv._eligible_frames(c, 8)[0]]
        print(f"{rate:g} Hz {split}: first eligible frame {first:.2f} s; no pre-contact frame {k}/{N} "
              f"({100 * k / N:.1f} % of all, {100 * k / a:.1f} % of {a} approaching) by kind {none}")
# Arrival-time distribution of approaching scenarios (dev, 15 Hz) and what a config change would give.
k, a, none, appr, arrivals = count(config, "dev")
print("arrival quantiles (s) 5/10/25/50:", np.round(np.quantile(arrivals, [0.05, 0.1, 0.25, 0.5]), 3))
for margin in (0.15, 0.1, 0.05):
    for dlow in (0.75, 0.85, 0.95):
        c = copy.deepcopy(config)
        c["skills"]["dodge_vision"]["pre_contact_margin_s"] = margin
        c["skills"]["dodge_vision"]["scenarios"]["distance_m"] = [dlow, 1.4]
        k2, a2, *_ = count(c, "dev")
        print(f"  pre_contact_margin_s {margin}, distance_m [{dlow}, 1.4]: no pre-contact frame {k2}/{N} ({100 * k2 / N:.1f} %)")
