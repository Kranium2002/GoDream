"""The simulated Go2 world the learning engine drives: reset, step and finish one episode at a time."""
import itertools
import json
import math
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from learned_judgment.env.acceptance import trunk_on_floor
from learned_judgment.env.commands import validate_command
from learned_judgment.env.policy import WalkingPolicy
from learned_judgment.env.scene import build_model

STREAMS = ("situation", "world", "ball", "noise")
SITUATION_KEYS = ("dog_forward_speed_m_s", "settle_s")  # the six ball keys follow with the ball (Task 5)
FEET = ("FL", "FR", "RL", "RR")  # go2.xml foot geoms, in the model's body order
BAD_STATE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
TRACE_KEYS = ("t_s", "base_pos_m", "base_quat_wxyz", "base_lin_vel_m_s", "base_ang_vel_rad_s", "joint_pos_rad",
              "joint_vel_rad_s", "policy_obs", "foot_contact", "fallen")


def make_streams(seed: int, attempt_seed: int | None = None) -> dict[str, np.random.Generator]:
    """Independent generators under the keys of STREAMS, fixed by seed alone or, for one attempt, by
    (seed, attempt_seed). The attempt family sits on its own SeedSequence spawn key, so it never repeats a stream of
    the seed-only family (SeedSequence([s, 0]) would equal SeedSequence(s))."""
    key = () if attempt_seed is None else (len(STREAMS), attempt_seed)
    children = np.random.SeedSequence(seed, spawn_key=key).spawn(len(STREAMS))
    return dict(zip(STREAMS, map(np.random.default_rng, children)))


class Go2World:
    """One world: the pinned Go2 and its frozen walking policy on a flat floor, with a preset's hidden effects.

    Seeds. `seed` and `preset` fix the hidden world for the life of the object. reset(attempt_seed=k) fixes one
    episode's noise: `attempt_streams` is make_streams(seed, k), so two worlds with the same seed reset with the same
    k face the same noise whatever ran before; k=None is the episode index (0 for this object's first reset). The
    hidden values and the declared payload are drawn at construction from make_streams(seed)["world"]; each reset draws
    its latency jitter from a child of attempt_streams["noise"] (Generator.spawn), so attempt_streams reach their user
    untouched. The simulator and the policy are deterministic: repeated runs are bit-identical.

    Hidden effects (config.yaml world.hidden_effects; each applied only if switched on in its `enabled`): a payload box
    on the trunk; the feet's sliding friction on the floor, by the preset's floor_type with a per-world variation; a
    slope, as tilted gravity (the policy senses true gravity, as an IMU would); motor strength, scaling every joint
    torque and its limit; command latency, a delay line of whole control periods with a per-attempt jitter. A preset
    that makes the dog fall is a measurement: the fall is reported, nothing raises.

    Episode. reset(situation) spawns the dog, runs the policy's settle and soft start, trots at
    dog_forward_speed_m_s (default 0.0, a vel_x command) for settle_s (default world.settle_s) and returns the
    observation at the decision instant, t_s 0.0. step(command, duration_s) holds a command, validated strictly
    against command_schema, for duration_s rounded to whole control periods (at least one), any number of times.
    finish_episode() ends the episode and returns the outcome; reset may follow for a new episode. A reset during an
    episode abandons it without a trace.

    Observation, one per control period (0.02 s). World frame: x, y on the floor, z up (normal to the floor; a slope
    tilts gravity instead), origin on the floor under the spawn point (the dog starts facing +x).
        t_s                 float, s. Episode time, 0.0 at the decision instant.
        base_pos_m          (3,) m. Trunk (base_link) origin, world frame.
        base_quat_wxyz      (4,) Trunk orientation, body to world, w x y z.
        base_lin_vel_m_s    (3,) m/s. Trunk linear velocity, world frame.
        base_ang_vel_rad_s  (3,) rad/s. Trunk angular velocity, world frame (MuJoCo gives it in the body frame; it is
                            rotated here). z > 0 turns anticlockwise seen from above.
        joint_pos_rad       (12,) rad and joint_vel_rad_s (12,) rad/s. Leg joints in the model's order FL, FR, RL,
                            RR, each hip, thigh, calf (the policy's own order is config.yaml action_schema).
        policy_obs          (45,) float32. What the policy read at its last step, scaled as in config.yaml
                            observation_schema; NaN before its first step (settle_s 0).
        foot_contact        (4,) bool, FL, FR, RL, RR: that foot geom has a contact with the floor.
        fallen              bool. A fall (below) has happened this episode; once true it stays true.
        ball                None until the ball is added (Task 5).
        command_active      {field: float}. The command the policy is receiving now: the one issued the latency
                            earlier (the reset's trot until then).
        declared_payload_kg float, kg, and declared_surface str: what the judge declares, fixed for the world's life
                            and also its read-only attributes of the same names. The payload is the true mass times
                            (1 + an error drawn per world in declared_mass_error_fraction); the surface is the
                            preset's floor_type.

    Outcome of finish_episode():
        elapsed_s           float, s. t_s at the end.
        fall                0 or 1. A trunk geom touched the floor or the trunk tilted beyond robot.fall.max_tilt_deg,
                            at any physics step from the spawn on.
        fall_time_s         t_s of the first such physics step (negative: before the decision instant), or None.
        sim_unstable        0 or 1. 1 if MuJoCo flagged a bad state this episode (warning mjWARN_BADQPOS, BADQVEL or
                            BADQACC: it then resets the state itself) or the final state is not finite.
        commands_received   [(t_s, command)], one per step() call, at the t_s it was issued. Re-sending the same
                            command in shorter steps gives the same episode: the delay line is not restarted.
        final_base_pos_m    [x, y, z], m, world frame.
        declared_payload_kg, declared_surface as in the observation.
        trace_path          The episode's .npz (world.trace_dir), or None if world.save_traces is false. It holds
                            each TRACE_KEYS array stacked per control period from the decision instant on, and
                            command_active as (N, 3) in command_schema field order. The hidden values (the effects
                            switched on and this attempt's latency) are only in <trace>.hidden.json beside it, never
                            in an observation or outcome.
    """

    def __init__(self, config: dict, seed: int, preset: str = "default"):
        presets = config["world"]["presets"]
        if preset not in presets:
            raise ValueError(f"unknown preset {preset!r}; config.yaml world.presets has {list(presets)}")
        self.config, self.seed, self.preset = config, seed, preset
        self.command_schema = config["robot"]["walking_policy"]["command_schema"]
        self._streams = make_streams(seed)
        self._hidden, self._declared = self._hidden_effects()
        self.model = build_model(config, self._hidden)
        self.policy = WalkingPolicy(config, self.model, self._hidden.get("motor_strength", 1.0))
        self._base = self.model.body("base_link").id
        self._floor = self.model.geom("floor").id
        self._feet = [self.model.geom(name).id for name in FEET]
        self._cos_max_tilt = math.cos(math.radians(config["robot"]["fall"]["max_tilt_deg"]))
        self._episodes = 0
        self._active = False
        self._renderer = None

    def _hidden_effects(self) -> tuple[dict, dict]:
        """The true hidden values of this world, for each effect switched on in world.hidden_effects.enabled, and the
        declared values, both fixed by the preset and self._streams["world"]. The draws are made whatever is switched
        on, so switching one effect never changes another's value."""
        he, p = self.config["world"]["hidden_effects"], self.config["world"]["presets"][self.preset]
        variation = self._streams["world"].uniform(-1.0, 1.0) * he["friction_per_run_variation"]
        mass_error = self._streams["world"].uniform(*he["declared_mass_error_fraction"])
        hidden = {
            "payload": {"mass_kg": p["payload_mass_kg"], "offset_m": p["payload_offset_m"]},
            "friction": he["surface_friction_by_type"][p["floor_type"]] * (1.0 + variation),
            "slope": p["slope_deg"],
            "motor_strength": p["motor_strength_fraction"],
            "latency": p["command_latency_s"],  # nominal; each attempt adds its jitter (reset)
        }
        declared = {"declared_payload_kg": p["payload_mass_kg"] * (1.0 + mass_error),
                    "declared_surface": p["floor_type"]}
        return {k: v for k, v in hidden.items() if he["enabled"][k]}, declared

    @property
    def declared_payload_kg(self) -> float:
        return self._declared["declared_payload_kg"]

    @property
    def declared_surface(self) -> str:
        return self._declared["declared_surface"]

    def reset(self, situation: dict | None = None, attempt_seed: int | None = None) -> dict:
        situation = situation or {}
        unknown = sorted(set(situation) - set(SITUATION_KEYS), key=repr)  # repr: keys may be of mixed types
        if unknown:
            raise ValueError(f"unknown situation key(s) {unknown}; Go2World accepts {list(SITUATION_KEYS)}")
        trot = {name: 0.0 for name in self.command_schema["fields"]} | {
            "vel_x": situation.get("dog_forward_speed_m_s", 0.0)}
        native = validate_command(trot, self.command_schema)
        settle_s = float(situation.get("settle_s", self.config["world"]["settle_s"]))
        if not 0.0 <= settle_s < math.inf:
            raise ValueError(f"settle_s must be finite and >= 0, got {settle_s}")

        self._active = False
        self._attempt_seed = self._episodes if attempt_seed is None else attempt_seed
        self.attempt_streams = make_streams(self.seed, self._attempt_seed)
        self._episodes += 1
        dt, self._delay = self.policy.control_dt, 0  # control periods from issuing a command to the policy receiving it
        if "latency" in self._hidden:
            # From a child of the attempt's noise stream (Generator.spawn), so that stream reaches its user untouched.
            jitter = round(self.config["world"]["hidden_effects"]["command_latency_jitter_s"] / dt)
            draw = self.attempt_streams["noise"].spawn(1)[0].integers(-jitter, jitter + 1)
            self._delay = max(0, round(self._hidden["latency"] / dt) + int(draw))
        self._n, self._fall_n = 0, None  # physics steps since the spawn; the step of the first fall
        self.policy.reset(on_substep=self._watch)
        for _ in range(round(settle_s / self.policy.control_dt)):
            self.policy.step(native, on_substep=self._watch)
        self._n0 = self._n  # the decision instant
        self._command = {name: float(v) for name, v in trot.items()}
        self._queue = deque([(native, self._command)] * self._delay)  # the trot, issued long enough ago
        self._received, self._trace = [], []
        self._active = True
        return self._observe()

    def step(self, command: dict, duration_s: float) -> dict:
        if not self._active:
            raise RuntimeError("no episode in progress: call reset() first")
        native = validate_command(command, self.command_schema)
        if not 0.0 < duration_s < math.inf:
            raise ValueError(f"duration_s must be finite and > 0, got {duration_s}")
        issued = {name: float(command[name]) for name in self.command_schema["fields"]}
        self._received.append((self._t(), dict(issued)))
        for _ in range(max(1, round(duration_s / self.policy.control_dt))):
            self._queue.append((native, issued))
            active, self._command = self._queue.popleft()  # the command issued self._delay control periods ago
            self.policy.step(active, on_substep=self._watch)
            obs = self._observe()
        return obs

    def finish_episode(self) -> dict:
        if not self._active:
            raise RuntimeError("no episode in progress: call reset() first")
        self._active = False
        d = self.policy.data
        # MuJoCo checks qpos, qvel and qacc at every step, so a bad state before the last step left a warning count.
        unstable = any(d.warning[w].number for w in BAD_STATE) or not (
            np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        last = self._trace[-1]
        return {
            "elapsed_s": last["t_s"],
            "fall": int(self._fall_n is not None),
            "fall_time_s": None if self._fall_n is None else (self._fall_n - self._n0) * self.model.opt.timestep,
            "sim_unstable": int(unstable),
            "commands_received": self._received,
            "final_base_pos_m": last["base_pos_m"].tolist(),
            **self._declared,
            "trace_path": self._write_trace() if self.config["world"]["save_traces"] else None,
        }

    def render_frame(self) -> np.ndarray:
        """An H x W x 3 uint8 image of the current state from a camera tracking the trunk. The offscreen renderer is
        made on the first call, so episodes that never render never need OpenGL."""
        if self._renderer is None:
            size = self.config["world"]["render"]
            self._renderer = mujoco.Renderer(self.model, height=size["height"], width=size["width"])
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._camera)
            self._camera.type, self._camera.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, self._base
        self._renderer.update_scene(self.policy.data, camera=self._camera)
        return self._renderer.render()

    def _t(self) -> float:
        return (self._n - self._n0) * self.model.opt.timestep

    def _watch(self) -> None:
        """After every physics step from the spawn on: count it and latch the first fall."""
        self._n += 1
        d = self.policy.data
        # xmat[base, 8] is world z . trunk z, the cosine of the trunk's tilt
        if self._fall_n is None and (d.xmat[self._base, 8] < self._cos_max_tilt or trunk_on_floor(self.model, d)):
            self._fall_n = self._n

    def _observe(self) -> dict:
        d = self.policy.data
        ang_vel = np.empty(3)
        mujoco.mju_rotVecQuat(ang_vel, d.qvel[3:6], d.qpos[3:7])
        pairs = d.contact.geom
        obs = {
            "t_s": self._t(),
            "base_pos_m": d.qpos[0:3].copy(),
            "base_quat_wxyz": d.qpos[3:7].copy(),
            "base_lin_vel_m_s": d.qvel[0:3].copy(),
            "base_ang_vel_rad_s": ang_vel,
            "joint_pos_rad": d.qpos[7:19].copy(),
            "joint_vel_rad_s": d.qvel[6:18].copy(),
            "policy_obs": self.policy.last_obs.copy(),
            "foot_contact": np.isin(self._feet, pairs[(pairs == self._floor).any(axis=1)]),
            "fallen": self._fall_n is not None,
            "ball": None,
            "command_active": dict(self._command),
            **self._declared,
        }
        self._trace.append(obs)
        return obs

    def _write_trace(self) -> str:
        out = Path(self.config["world"]["trace_dir"])
        out.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.array([o[k] for o in self._trace]) for k in TRACE_KEYS}
        arrays["command_active"] = np.array([list(o["command_active"].values()) for o in self._trace])
        for i in itertools.count():
            path = out / f"{self.preset}-seed{self.seed}-attempt{self._attempt_seed}-{i}.npz"
            try:
                f = open(path, "xb")  # exclusive create: an existing trace is never overwritten
            except FileExistsError:
                continue
            with f:
                np.savez_compressed(f, **arrays)
            break
        hidden = {"preset": self.preset, "seed": self.seed, "attempt_seed": self._attempt_seed,
                  "hidden_effects": self._hidden, "attempt_latency_s": self._delay * self.policy.control_dt}
        path.with_suffix(".hidden.json").write_text(json.dumps(hidden, indent=2, default=lambda o: o.tolist()))
        return path.as_posix()
