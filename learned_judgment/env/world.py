"""The simulated Go2 world the learning engine drives: reset, step and finish one episode at a time."""
import copy
import itertools
import json
import math
import numbers
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from learned_judgment.env.acceptance import trunk_on_floor
from learned_judgment.env.commands import validate_command
from learned_judgment.env.policy import WalkingPolicy
from learned_judgment.env.incline import place as place_incline
from learned_judgment.env.scene import build_model, is_ground
from learned_judgment.env.throws import BALL_KEYS, sample_dodge_situations

STREAMS = ("situation", "world", "ball", "noise")
SITUATION_KEYS = ("dog_forward_speed_m_s", "settle_s", *BALL_KEYS, "object_kind", "incline_deg")
OBJECT_KINDS = ("ball", "punch")  # object_kind: config.yaml `ball` and `punch`
FEET = ("FL", "FR", "RL", "RR")  # go2.xml foot geoms, in the model's body order
BAD_STATE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
TRACE_KEYS = ("t_s", "base_pos_m", "base_quat_wxyz", "base_lin_vel_m_s", "base_ang_vel_rad_s", "joint_pos_rad",
              "joint_vel_rad_s", "foot_contact", "fallen")


def _yaw(quat_wxyz) -> float:
    """Heading of a body-to-world quaternion: its rotation about world z, rad."""
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class RecoveryClock:
    """recovery() one sample at a time, so an episode need not keep its samples."""

    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self._first = self._start = self._last = self._result = None  # sample times

    def add(self, t_s: float, within: bool) -> None:
        if self._result is not None:
            return
        self._last = t_s
        if self._first is None:
            self._first = None if within else t_s
            return
        self._start = None if not within else t_s if self._start is None else self._start
        if self._start is not None and t_s - self._start >= self.hold_s - 1e-9:  # 1e-9: times are sums of 0.002 s
            self._result = float(self._start - self._first), 1

    def result(self) -> tuple[float, int]:
        if self._first is None:
            return 0.0, 1
        return self._result or (float(self._last - self._first), 0)


def recovery(t_s: np.ndarray, within: np.ndarray, hold_s: float) -> tuple[float, int]:
    """(recovery_time_s, recovered) from per-sample times and whether each sample is within the recovery limits:
    (0.0, 1) if every sample is; else the time from the first sample out to the first sample from which they hold
    for hold_s, and 1; or, if they never hold that long, the time to the last sample, and 0."""
    clock = RecoveryClock(hold_s)
    for t, w in zip(t_s, within):
        clock.add(t, w)
    return clock.result()


def _number(key: str, value) -> float:
    """A situation value as a float; ValueError naming the key unless it is a finite Python or numpy int or float."""
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)) or not math.isfinite(value):
        raise ValueError(f"situation key {key!r} must be a finite number, got {value!r}")
    return float(value)


def make_streams(seed: int, attempt_seed: int | None = None) -> dict[str, np.random.Generator]:
    """Independent generators under the keys of STREAMS, fixed by seed alone or, for one attempt, by
    (seed, attempt_seed). The attempt family sits on its own SeedSequence spawn key, so it never repeats a stream of
    the seed-only family (SeedSequence([s, 0]) would equal SeedSequence(s))."""
    key = () if attempt_seed is None else (4, attempt_seed)  # 4 is fixed (DECISIONS.md), whatever STREAMS holds
    children = np.random.SeedSequence(seed, spawn_key=key).spawn(len(STREAMS))
    return dict(zip(STREAMS, map(np.random.default_rng, children)))


class Go2World:
    """One world: the pinned Go2 and its frozen walking policy on a floor or an incline, with a preset's hidden effects.

    Seeds. `seed` and `preset` fix the hidden world for the life of the object. reset(attempt_seed=k) fixes one
    episode's noise: `attempt_streams` is make_streams(seed, k), so two worlds with the same seed reset with the same
    k face the same noise whatever ran before; k=None is the episode index (0 for this object's first reset). The
    hidden values and the declared payload are drawn at construction from make_streams(seed)["world"], and
    sample_dodge_situation draws from the world's own `ball` stream; each reset draws its latency jitter from a child
    of attempt_streams["noise"] (Generator.spawn), so attempt_streams reach their user untouched. The simulator and the
    policy are deterministic: repeated runs are bit-identical.

    Hidden effects (config.yaml world.hidden_effects; each applied only if switched on in its `enabled`): a payload box
    on the trunk; the feet's sliding friction on the floor, by the preset's floor_type with a per-world variation; a
    slope, as tilted gravity (the policy senses true gravity, as an IMU would); motor strength, scaling every joint
    torque and its limit; command latency, a delay line of whole control periods with a per-attempt jitter. A preset
    that makes the dog fall is a measurement: the fall is reported, nothing raises.

    Episode. reset(situation) spawns the dog, runs the policy's settle and soft start, trots at
    dog_forward_speed_m_s (default 0.0, a vel_x command) for settle_s (default world.settle_s) and returns the
    observation at the decision instant, t_s 0.0. step(command, duration_s, on_period=None) holds a command, validated
    strictly against command_schema, for duration_s rounded to whole control periods (at least one), any number of
    times, and returns the last observation; on_period, if given, is called after every control period with a copy of
    that period's observation, and an exception it raises ends step after that period with the episode still open.
    finish_episode() ends the episode and returns the outcome; reset may follow for a new episode. A reset during an
    episode abandons it without a trace. With world.save_traces false nothing is kept per control period: the outcome
    comes from the last observation and measures updated as the episode runs.

    Ball (DODGE). A situation with all six BALL_KEYS throws the ball; with some of them reset raises ValueError naming
    the missing ones, and with none the episode simulates the scene without a ball body. Their frame is fixed at the
    decision instant: origin at the trunk origin, x along the dog's heading (yaw only), y to its left, z up.
    ball_rel_x_m, ball_rel_y_m, ball_rel_z_m (m) are the ball's position in it and ball_vel_x_m_s, ball_vel_y_m_s,
    ball_vel_z_m_s (m/s) its velocity over the ground on those axes, not relative to the dog. The ball (config.yaml
    `ball`: an IHF size 3 handball) is a free MuJoCo body that collides with nothing until the decision instant, is then
    placed there with that velocity and no spin, and is never touched again. sample_dodge_situation() draws a situation
    (the six keys and dog_forward_speed_m_s) from the world's own `ball` stream with sample_dodge_situations
    (learned_judgment/env/throws.py). object_kind (default "ball") chooses the thrown object: "punch" (config.yaml
    `punch`) is a sphere outside gravity (MuJoCo gravcomp) that flies straight at constant speed until it touches
    something; it takes the ball's place in the observation's `ball` entry (radius_m is its own), the trace arrays and
    the outcome keys, and its kind is only in <trace>.hidden.json. Any other kind raises ValueError.

    Incline (RAMP). incline_deg (degrees, 0 to world.incline.max_deg; absent or 0: the plain flat scene, the same
    episode as without the key) places, at the decision instant, a slope rising at that angle world.incline.start_x_m
    ahead of the trunk origin along the heading then, and a level plateau after it (learned_judgment/env/incline.py).
    Ramp and plateau are ground like the floor: foot contact, the trunk-on-ground fall check and foot_slip_m count
    them. The angle is hidden: it is only in <trace>.hidden.json. sample_ramp_situations(config, rng, n) draws angles
    from world.incline.range_deg.

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
        foot_contact        (4,) bool, FL, FR, RL, RR: that foot geom has a contact with the ground (floor or incline).
        fallen              bool. A fall (below) has happened this episode; once true it stays true.
        ball                None without a ball, else {"pos_m": (3,) m, "vel_m_s": (3,) m/s, "radius_m": float}: the
                            ball's centre and linear velocity, world frame.
        command_active      {field: float}. The command the caller issued last (the reset's trot until the first
                            step), whatever the latency. What the policy receives, delayed by the latency, is hidden.
        declared_payload_kg float, kg, and declared_surface str: what the judge declares, fixed for the world's life
                            and also its read-only attributes of the same names. The payload is the true mass times
                            (1 + an error drawn per world in declared_mass_error_fraction); the surface is the
                            preset's floor_type.

    Outcome of finish_episode():
        elapsed_s           float, s. t_s at the end.
        fall                0 or 1. A trunk geom touched the ground (floor or incline) or the trunk tilted beyond
                            robot.fall.max_tilt_deg, at any physics step from the spawn on.
        fall_time_s         t_s of the first such physics step (negative: before the decision instant), or None.
        sim_unstable        0 or 1. 1 if MuJoCo flagged a bad state this episode (warning mjWARN_BADQPOS, BADQVEL or
                            BADQACC: it then resets the state itself) or the final state is not finite.
        commands_received   [(t_s, command)]: each command that differs from the one before it, at the t_s it was
                            issued. Re-sending the same command, in steps of any length, adds no entry and gives the
                            same episode: the delay line is not restarted.
        final_base_pos_m    [x, y, z], m, world frame.
        declared_payload_kg, declared_surface as in the observation.
        collision           0 or 1. 1 if the ball had a contact with a robot geom (any body but the ground and the
                            ball) at a physics step from the decision instant on. Ball and floor is not a collision.
        first_collision_time_s  t_s of the first such physics step, or None.
        min_separation_m    m. Smallest surface-to-surface distance between the ball and the robot's collision geoms
                            over every physics step from the decision instant on (mj_geomDistance), never below 0, and
                            0 at a step with a contact. None without a ball; then collision is 0.
        recovery_time_s     s, always a number. 0.0 if trunk tilt and tilt rate stayed within robot.recovery at every
                            control period from the decision instant on; otherwise the time from first leaving them to
                            the sample from which they hold for robot.recovery.hold_s, or to the end of the episode.
        recovered           0 or 1. 0 if they never held for hold_s after first leaving them.
        final_forward_m, final_sideways_m  m. Trunk travel from the decision instant to the end, along the dog's
                            heading at the decision instant (yaw only) and to its left.
        final_yaw_change_rad  rad. Heading at the end minus heading at the decision instant, wrapped to [-pi, pi);
                            + is anticlockwise seen from above.
        final_speed_m_s     m/s. Trunk speed over the ground (horizontal) at the end.
        ramp_start_m, ramp_end_m  m, for scoring. Floor distance along the decision-instant heading from the trunk
                            origin then to where the slope begins and ends; None without an incline.
        foot_slip_m         m. Sum over the physics steps of step() and over the feet with a ground contact at that
                            step (the first such contact of each foot) of the speed of the foot's material point at
                            the contact point along the contact plane, times the physics timestep: how far the feet slid
                            on the ground, rolling of the foot spheres excluded.
        trace_path          The episode's .npz (world.trace_dir), or None if world.save_traces is false. It holds
                            each TRACE_KEYS array stacked per control period from the decision instant on,
                            command_active as (N, 3) in command_schema field order and, with a ball, ball_pos_m and
                            ball_vel_m_s as (N, 3). The hidden values (the effects switched on, this attempt's
                            latency and incline_deg) and policy_command, the command the policy received at each of
                            those control periods (N lists in command_schema field order), are only in
                            <trace>.hidden.json beside it, never in an observation or outcome.
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
        self._feet = [self.model.geom(name).id for name in FEET]
        self._cos_max_tilt = math.cos(math.radians(config["robot"]["fall"]["max_tilt_deg"]))
        limits = config["robot"]["recovery"]
        self._recovery_limits = math.cos(math.radians(limits["max_tilt_deg"])), limits["max_tilt_rate_rad_s"]
        self._episodes = 0
        self._active = False
        self._renderer = None
        # (object kind or None, incline) -> (model, policy); a scene with an object or the incline is built for its
        # first episode
        self._scenes = {(None, False): (self.model, self.policy)}
        self._ground = is_ground(self.model)
        self._ball_live = False

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

    def sample_dodge_situation(self) -> dict:
        """One DODGE situation from this world's `ball` stream (throws.sample_dodge_situations with n=1)."""
        return sample_dodge_situations(self.config, self._streams["ball"], 1)[0]

    def reset(self, situation: dict | None = None, attempt_seed: int | None = None) -> dict:
        situation = situation or {}
        unknown = sorted(set(situation) - set(SITUATION_KEYS), key=repr)  # repr: keys may be of mixed types
        if unknown:
            raise ValueError(f"unknown situation key(s) {unknown}; Go2World accepts {list(SITUATION_KEYS)}")
        ball = any(k in situation for k in BALL_KEYS)
        missing = [k for k in BALL_KEYS if k not in situation]
        if ball and missing:
            raise ValueError(f"missing ball key(s) {missing}: give all six of {list(BALL_KEYS)} or none")
        throw = [_number(k, situation[k]) for k in BALL_KEYS] if ball else None
        kind = situation.get("object_kind", "ball")
        if kind not in OBJECT_KINDS:
            raise ValueError(f"object_kind must be one of {list(OBJECT_KINDS)}, got {kind!r}")
        incline = _number("incline_deg", situation.get("incline_deg", 0.0))
        if not 0.0 <= incline <= self.config["world"]["incline"]["max_deg"]:
            raise ValueError(f"incline_deg must be from 0 to world.incline.max_deg "
                             f"({self.config['world']['incline']['max_deg']}), got {incline}")
        trot = {name: 0.0 for name in self.command_schema["fields"]} | {
            "vel_x": situation.get("dog_forward_speed_m_s", 0.0)}
        native = validate_command(trot, self.command_schema)
        settle_s = _number("settle_s", situation.get("settle_s", self.config["world"]["settle_s"]))
        if settle_s < 0.0:
            raise ValueError(f"settle_s must be finite and >= 0, got {settle_s}")
        attempt = self._episodes if attempt_seed is None else attempt_seed
        streams = make_streams(self.seed, attempt)  # raises on a bad attempt_seed before the open episode is abandoned

        self._active = self._ball_live = False
        self._attempt_seed, self.attempt_streams = attempt, streams
        self._episodes += 1
        self._object = kind if ball else None
        self._use_scene(self._object, incline > 0.0)
        self._incline = incline
        dt, self._delay = self.policy.control_dt, 0  # control periods from issuing a command to the policy receiving it
        if "latency" in self._hidden:
            # From a child of the attempt's noise stream (Generator.spawn), so that stream reaches its user untouched.
            jitter = round(self.config["world"]["hidden_effects"]["command_latency_jitter_s"] / dt)
            draw = self.attempt_streams["noise"].spawn(1)[0].integers(-jitter, jitter + 1)
            self._delay = max(0, round(self._hidden["latency"] / dt) + int(draw))
        self._n, self._fall_n = 0, None  # physics steps since the spawn; the step of the first fall
        if incline:  # parked out of the spawn's way until the decision instant
            self.policy.data.mocap_pos[:] = [0.0, 0.0, self.config["world"]["incline"]["park_height_m"]]
            self.policy.data.mocap_quat[:] = [1.0, 0.0, 0.0, 0.0]
        self.policy.reset(on_substep=self._watch)
        for _ in range(round(settle_s / self.policy.control_dt)):
            self.policy.step(native, on_substep=self._watch)
        self._n0 = self._n  # the decision instant
        self._hit_n, self._min_sep = None, math.inf  # the step of the first ball contact; the smallest separation
        if ball:
            self._throw(throw)
        self._start_pos, self._start_yaw = self.policy.data.qpos[0:2].copy(), _yaw(self.policy.data.qpos[3:7])
        self._ramp = None  # (ramp_start_m, ramp_end_m)
        if incline:
            self._ramp = place_incline(self.model, self.policy.data, self.config["world"]["incline"], incline,
                                       self._start_pos, self._start_yaw)
            mujoco.mj_kinematics(self.model, self.policy.data)  # so a frame at t_s 0 shows it; the policy reads qpos
        self._slip = 0.0
        self._issued = self._command = {name: float(v) for name, v in trot.items()}  # last issued; policy's (delayed)
        self._queue = deque([(native, self._command)] * self._delay)  # the trot, issued long enough ago
        # With traces off only the last observation and the running measures are kept, never every period's.
        self._received, self._trace, self._policy_commands, self._last = [], [], [], None
        self._recovery = RecoveryClock(self.config["robot"]["recovery"]["hold_s"])
        self._active = True
        return copy.deepcopy(self._observe())

    def step(self, command: dict, duration_s: float, on_period=None) -> dict:
        if not self._active:
            raise RuntimeError("no episode in progress: call reset() first")
        native = validate_command(command, self.command_schema)
        if not 0.0 < duration_s < math.inf:
            raise ValueError(f"duration_s must be finite and > 0, got {duration_s}")
        issued = {name: float(command[name]) for name in self.command_schema["fields"]}
        if not self._received or self._received[-1][1] != issued:  # recorded when it changes, at its issue time
            self._received.append((self._t(), dict(issued)))
        self._issued = issued
        for _ in range(max(1, round(duration_s / self.policy.control_dt))):
            self._queue.append((native, issued))
            active, self._command = self._queue.popleft()  # the command issued self._delay control periods ago
            self.policy.step(active, on_substep=self._watch)
            obs = self._observe()
            if on_period is not None:
                on_period(copy.deepcopy(obs))
        return copy.deepcopy(obs)

    def finish_episode(self) -> dict:
        if not self._active:
            raise RuntimeError("no episode in progress: call reset() first")
        self._active = False
        d = self.policy.data
        # MuJoCo checks qpos, qvel and qacc at every step, so a bad state before the last step left a warning count.
        unstable = any(d.warning[w].number for w in BAD_STATE) or not (
            np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        last = self._last
        recovery_time_s, recovered = self._recovery.result()
        travel, yaw0 = last["base_pos_m"][:2] - self._start_pos, self._start_yaw
        return {
            "elapsed_s": last["t_s"],
            "fall": int(self._fall_n is not None),
            "fall_time_s": None if self._fall_n is None else (self._fall_n - self._n0) * self.model.opt.timestep,
            "sim_unstable": int(unstable),
            "commands_received": self._received,
            "final_base_pos_m": last["base_pos_m"].tolist(),
            **self._declared,
            "collision": int(self._hit_n is not None),
            "first_collision_time_s": None if self._hit_n is None else (self._hit_n - self._n0) * self.model.opt.timestep,
            "min_separation_m": self._min_sep if self._ball_live else None,
            "recovery_time_s": recovery_time_s,
            "recovered": recovered,
            "final_forward_m": float(travel @ [math.cos(yaw0), math.sin(yaw0)]),
            "final_sideways_m": float(travel @ [-math.sin(yaw0), math.cos(yaw0)]),
            "final_yaw_change_rad": (_yaw(last["base_quat_wxyz"]) - yaw0 + math.pi) % (2 * math.pi) - math.pi,
            "final_speed_m_s": float(np.hypot(*last["base_lin_vel_m_s"][:2])),
            "ramp_start_m": None if self._ramp is None else self._ramp[0],
            "ramp_end_m": None if self._ramp is None else self._ramp[1],
            "foot_slip_m": self._slip,
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
            self._camera.distance = self.config["world"]["render"]["tracking_distance_m"]
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
        if self._ball_live:
            self._watch_ball()
        if self._active:
            self._watch_feet()

    def _watch_feet(self) -> None:
        """After every physics step of step(): add each foot's slip at this step to foot_slip_m (see the outcome)."""
        d, g = self.policy.data, self.policy.data.contact.geom
        ground = self._ground[g]
        velocity = np.empty(6)
        for foot in self._feet:
            rows = np.flatnonzero(((g[:, 0] == foot) & ground[:, 1]) | ((g[:, 1] == foot) & ground[:, 0]))
            if rows.size:  # its first ground contact: one per foot, also where the slope meets the floor
                c = d.contact[rows[0]]
                mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_GEOM, foot, velocity, 0)  # rot, lin
                v = velocity[3:] + np.cross(velocity[:3], c.pos - d.geom_xpos[foot])  # at the contact point
                normal = c.frame[:3]
                self._slip += float(np.linalg.norm(v - (v @ normal) * normal)) * self.model.opt.timestep

    # The ball. Its scene is the plain one plus a free body after the robot (scene.build_model), so the robot's qpos,
    # qvel, body and geom indices are the same in both scenes.

    def _use_scene(self, ball: str | None, incline: bool = False) -> None:
        """Point model and policy at the scene with the thrown object of kind `ball` (or none) and with or without
        the incline, building it on first use. The object collides with nothing until _throw. Object and incline come
        after the robot, the object first, so the robot's and the object's indices are the same in every scene."""
        key = (ball, incline)
        if key not in self._scenes:
            model = build_model(self.config, self._hidden, ball=ball or False, incline=incline)
            self._scenes[key] = (model, WalkingPolicy(self.config, model, self._hidden.get("motor_strength", 1.0)))
        if ball:
            model = self._scenes[key][0]
            body = model.body("ball")
            self._ball_geom, self._ball_q, self._ball_v = model.geom("ball").id, model.jnt_qposadr[body.jntadr[0]], \
                body.dofadr[0]
            self._ball_masks = model.geom_contype[self._ball_geom], model.geom_conaffinity[self._ball_geom]
            self._robot_geoms = np.array([g for g in range(model.ngeom) if not is_ground(model)[g]
                                          and model.geom_bodyid[g] != body.id
                                          and (model.geom_contype[g] or model.geom_conaffinity[g])])
            # Centres closer than this can mean surfaces closer than 0 (bounding spheres); contacts start within margin.
            self._reach = model.geom_rbound[self._robot_geoms] + model.geom_rbound[self._ball_geom]
            self._margin = model.geom_margin[[*self._robot_geoms, self._ball_geom]].max()
            self._ball_radius = float(model.geom_size[self._ball_geom][0])
        if self.model is not self._scenes[key][0]:
            self.model, self.policy = self._scenes[key]
            self._ground = is_ground(self.model)
            self._renderer = None  # a renderer belongs to one model
        if ball:
            self.model.geom_contype[self._ball_geom] = self.model.geom_conaffinity[self._ball_geom] = 0

    def _throw(self, throw: list[float]) -> None:
        """Place the ball at the decision instant: the situation's frame turned into the world frame."""
        d = self.policy.data
        yaw = _yaw(d.qpos[3:7])
        turn = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        d.qpos[self._ball_q:self._ball_q + 7] = [*(d.qpos[0:3] + turn @ throw[:3]), 1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_v:self._ball_v + 6] = [*(turn @ throw[3:]), 0.0, 0.0, 0.0]
        self.model.geom_contype[self._ball_geom], self.model.geom_conaffinity[self._ball_geom] = self._ball_masks
        self._ball_live = True

    def _watch_ball(self) -> None:
        """After every physics step from the decision instant on: latch the first ball-robot contact and keep the
        smallest ball-robot separation (0 at a step with contact). Exact, but only for robot geoms whose cheap lower
        bound (centre distance less both bounding radii) could beat the smallest separation so far or touch."""
        if self._hit_n is not None:  # the separation is 0 and the first contact is latched: nothing left to measure
            return
        d = self.policy.data
        bound = np.linalg.norm(d.geom_xpos[self._robot_geoms] - d.geom_xpos[self._ball_geom], axis=1) - self._reach
        near = self._robot_geoms[bound < max(self._min_sep, self._margin)]
        if near.size == 0:
            return
        pairs = d.contact.geom
        if np.isin(pairs[(pairs == self._ball_geom).any(axis=1)], self._robot_geoms).any():
            self._hit_n, self._min_sep = self._n, 0.0
        else:  # distances are capped at distmax, the smallest so far
            self._min_sep = max(0.0, min(self._min_sep, *(
                mujoco.mj_geomDistance(self.model, d, self._ball_geom, int(g), self._min_sep, None) for g in near)))

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
            "foot_contact": np.isin(self._feet, pairs[self._ground[pairs].any(axis=1)]),
            "fallen": self._fall_n is not None,
            "ball": None if not self._ball_live else {"pos_m": d.qpos[self._ball_q:self._ball_q + 3].copy(),
                                                      "vel_m_s": d.qvel[self._ball_v:self._ball_v + 3].copy(),
                                                      "radius_m": self._ball_radius},
            "command_active": dict(self._issued),
            **self._declared,
        }
        q, cos_tilt, max_rate = obs["base_quat_wxyz"], *self._recovery_limits
        self._recovery.add(obs["t_s"], bool(1 - 2 * (q[1] ** 2 + q[2] ** 2) >= cos_tilt
                                            and np.hypot(ang_vel[0], ang_vel[1]) <= max_rate))
        self._last = obs
        if self.config["world"]["save_traces"]:
            self._trace.append(obs)
            self._policy_commands.append(list(self._command.values()))
        return obs

    def _write_trace(self) -> str:
        out = Path(self.config["world"]["trace_dir"])
        out.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.array([o[k] for o in self._trace]) for k in TRACE_KEYS}
        arrays["command_active"] = np.array([list(o["command_active"].values()) for o in self._trace])
        if self._ball_live:
            arrays["ball_pos_m"] = np.array([o["ball"]["pos_m"] for o in self._trace])
            arrays["ball_vel_m_s"] = np.array([o["ball"]["vel_m_s"] for o in self._trace])
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
                  "hidden_effects": self._hidden, "attempt_latency_s": self._delay * self.policy.control_dt,
                  "incline_deg": self._incline, "object_kind": self._object,
                  "policy_command": self._policy_commands}
        path.with_suffix(".hidden.json").write_text(json.dumps(hidden, indent=2, default=lambda o: o.tolist()))
        return path.as_posix()
