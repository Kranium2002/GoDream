"""Measure the baseline on the incline: a constant forward command, without and then with a constant yaw_rate trim.

    python -m learned_judgment.env.incline_sweep

Settings: config.yaml world.incline.sweep. Every episode: reset({"dog_forward_speed_m_s": vel_x, "incline_deg": a},
attempt_seed=k) on the default preset, then the command held for duration_s. The trim is measured here, not chosen: the
mean yaw drift per second of the untrimmed flat (0 degree) episodes, with its sign reversed. Prints one JSON line per
episode, then a table per mode. A development measurement, not an evaluation result.
"""
import json
import math

import numpy as np

from learned_judgment.env import Go2World, load_config


def episode(world: Go2World, deg: float, seed: int, command: dict, duration_s: float) -> dict:
    inc = world.config["world"]["incline"]
    start = world.reset({"dog_forward_speed_m_s": command["vel_x"], "incline_deg": deg}, attempt_seed=seed)
    w, x, y, z = start["base_quat_wxyz"]
    yaw0 = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    ahead, left = np.array([math.cos(yaw0), math.sin(yaw0)]), np.array([-math.sin(yaw0), math.cos(yaw0)])
    rise = inc["length_m"] * math.sin(math.radians(deg))
    end = inc["start_x_m"] + inc["length_m"] * math.cos(math.radians(deg))
    seen = {"plateau_t_s": None, "max_abs_sideways_m": 0.0}

    def on_period(obs):
        travel = obs["base_pos_m"][:2] - start["base_pos_m"][:2]
        seen["max_abs_sideways_m"] = max(seen["max_abs_sideways_m"], abs(float(travel @ left)))
        # On the plateau: 0.3 m past the slope's end and within 5 cm of the rise (on a flat floor: 0.3 m past "end").
        if seen["plateau_t_s"] is None and travel @ ahead >= end + 0.3 and \
                obs["base_pos_m"][2] - start["base_pos_m"][2] >= rise - 0.05:
            seen["plateau_t_s"] = obs["t_s"]

    last = world.step(command, duration_s, on_period=on_period)
    out = world.finish_episode()
    return {"incline_deg": deg, "attempt_seed": seed, "yaw_rate": command["yaw_rate"], "fall": out["fall"],
            "reached_plateau": int(seen["plateau_t_s"] is not None), "plateau_t_s": seen["plateau_t_s"],
            "height_gain_m": float(last["base_pos_m"][2] - start["base_pos_m"][2]), "rise_m": rise,
            "final_forward_m": out["final_forward_m"], "final_sideways_m": out["final_sideways_m"],
            "max_abs_sideways_m": seen["max_abs_sideways_m"], "final_yaw_change_rad": out["final_yaw_change_rad"],
            "final_speed_m_s": out["final_speed_m_s"], "foot_slip_m": out["foot_slip_m"],
            "off_the_side": int(seen["max_abs_sideways_m"] > inc["width_m"] / 2)}


def table(rows: list[dict]) -> str:
    lines = ["| incline deg | yaw_rate trim | reached plateau | falls | mean time to plateau s | mean height gain m "
             "(rise) | mean final sideways m | mean yaw change deg | mean foot slip m |", "|" + "---|" * 9]
    for deg in sorted({r["incline_deg"] for r in rows}):
        rs = [r for r in rows if r["incline_deg"] == deg]
        times = [r["plateau_t_s"] for r in rs if r["plateau_t_s"] is not None]
        lines.append(f"| {deg:g} | {rs[0]['yaw_rate']:+.3f} | {sum(r['reached_plateau'] for r in rs)} of {len(rs)} | "
                     f"{sum(r['fall'] for r in rs)} | {np.mean(times) if times else float('nan'):.2f} | "
                     f"{np.mean([r['height_gain_m'] for r in rs]):.3f} ({rs[0]['rise_m']:.3f}) | "
                     f"{np.mean([r['final_sideways_m'] for r in rs]):+.2f} | "
                     f"{math.degrees(np.mean([r['final_yaw_change_rad'] for r in rs])):+.1f} | "
                     f"{np.mean([r['foot_slip_m'] for r in rs]):.3f} |")
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    sw = config["world"]["incline"]["sweep"]
    world = Go2World(config, config["runtime"]["seed"])
    zero = {name: 0.0 for name in world.command_schema["fields"]}
    tables = {}
    for mode in ("untrimmed", "trimmed"):
        if mode == "trimmed":
            flat = [r for r in rows if r["incline_deg"] == 0.0]
            trim = -float(np.mean([r["final_yaw_change_rad"] for r in flat])) / sw["duration_s"]
        command = zero | {"vel_x": sw["vel_x"], "yaw_rate": 0.0 if mode == "untrimmed" else round(trim, 3)}
        rows = []
        for deg in sw["angles_deg"]:
            for seed in sw["attempt_seeds"]:
                rows.append(episode(world, float(deg), seed, command, sw["duration_s"]))
                print(json.dumps({"mode": mode, **rows[-1]}), flush=True)
        tables[mode] = table(rows)
    for mode, text in tables.items():
        print(f"\n{mode} (vel_x {sw['vel_x']}, {sw['duration_s']} s, seed {config['runtime']['seed']}, default preset)")
        print(text)


if __name__ == "__main__":
    main()
