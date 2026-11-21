"""Each hidden effect measured on its own, through Go2World: the world Person B drives."""
import copy
import json
from pathlib import Path

import numpy as np

from learned_judgment.env.world import Go2World

EFFECTS = ("payload", "friction", "slope", "motor_strength", "latency")
QUANTITY = {"payload": "steady_speed_m_s", "friction": "stopping_distance_m", "slope": "steady_speed_m_s",
            "motor_strength": "steady_speed_m_s", "latency": "time_to_half_speed_s"}


def heading(obs) -> np.ndarray:
    """Unit vector of the trunk's heading on the floor (its x axis projected), world frame."""
    w, x, y, z = obs["base_quat_wxyz"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw), np.sin(yaw)])


def heading_speed(obs) -> float:
    return float(obs["base_lin_vel_m_s"][:2] @ heading(obs))


def measure(world: Go2World, quantity: str, m: dict) -> tuple[float | None, int]:
    """One episode measuring `quantity` as config.yaml world.hidden_effects.measure defines it, and its fall (0 or 1).
    A fall is a measurement: the value is still taken, or None if the episode never reaches it."""
    dt, zero = world.policy.control_dt, {name: 0.0 for name in world.command_schema["fields"]}
    periods = round(m["duration_s"] / dt)
    if quantity == "steady_speed_m_s":
        world.reset({"dog_forward_speed_m_s": m["steady_speed_m_s"]})
        obs = [world.step(zero | {"vel_x": m["steady_speed_m_s"]}, dt) for _ in range(periods)]
        value = float(np.mean([heading_speed(o) for o in obs[-round(m["steady_window_s"] / dt):]]))
    elif quantity == "stopping_distance_m":
        start = world.reset({"dog_forward_speed_m_s": m["stop_from_m_s"]})
        end = world.step(zero, m["duration_s"])
        value = float((end["base_pos_m"] - start["base_pos_m"])[:2] @ heading(start))
    else:  # time_to_half_speed_s: from standing, the first period whose speed along the heading reaches half
        world.reset()
        obs = [world.step(zero | {"vel_x": m["start_to_m_s"]}, dt) for _ in range(periods)]
        value = next((o["t_s"] for o in obs if heading_speed(o) >= m["start_to_m_s"] / 2), None)
    return value, world.finish_episode()["fall"]


def measure_effects(config: dict, seed: int, out_dir) -> Path:
    """Saves out_dir/effects.json: every quantity with all effects off in the default preset ("baseline") and in a
    preset holding every effect at the demanding end of its range ("all_off"), then each effect switched on alone in
    that preset, measured on its quantity, with its hidden values and the latency that attempt had. Returns the file's
    path."""
    he = config["world"]["hidden_effects"]
    frictions = he["surface_friction_by_type"]
    demanding = {
        "payload_mass_kg": he["payload_mass_kg"][1], "payload_offset_m": [0.0, 0.0],  # centred: the mass alone
        "floor_type": min(frictions, key=frictions.get), "slope_deg": he["slope_deg"][1],
        "motor_strength_fraction": he["motor_strength_fraction"][0], "command_latency_s": he["command_latency_s"][1]}

    def run(preset: dict, on: set, quantity: str):
        cfg = copy.deepcopy(config)
        cfg["world"]["save_traces"] = False
        cfg["world"]["hidden_effects"]["enabled"] = {name: name in on for name in EFFECTS}
        cfg["world"]["presets"]["measure"] = preset
        world = Go2World(cfg, seed, preset="measure")
        value, fall = measure(world, quantity, he["measure"])
        return value, fall, world._hidden, world._delay * world.policy.control_dt  # the latency this attempt had

    result = {"seed": seed, "demanding_preset": demanding}
    for name, preset in (("baseline", config["world"]["presets"]["default"]), ("all_off", demanding)):
        runs = {q: run(preset, set(), q) for q in sorted(set(QUANTITY.values()))}
        result[name] = {q: r[0] for q, r in runs.items()}
        result[f"{name}_fall"] = {q: r[1] for q, r in runs.items()}
    for effect in EFFECTS:
        measured, fall, hidden, latency = run(demanding, {effect}, QUANTITY[effect])
        result[effect] = {"quantity": QUANTITY[effect], "measured": measured, "fall": fall, "hidden": hidden,
                          "attempt_latency_s": latency}
    out = Path(out_dir) / "effects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out
