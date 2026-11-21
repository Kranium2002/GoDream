"""DODGE: the trotting Go2 sidesteps a thrown ball (the primary skill).

Frame and units
---------------
Every position and velocity is in the dog's heading frame at the decision instant (the
moment ``world.reset`` returns): x forward, y left, z up, origin at the base; SI units.

Situation (S = 9), in this order::

    ball_rel_x_m, ball_rel_y_m, ball_rel_z_m        ball centre relative to the base
    ball_vel_x_m_s, ball_vel_y_m_s, ball_vel_z_m_s  ball velocity
    dog_forward_speed_m_s                           forward speed the dog trots at (sent as the forward command)
    declared_payload_kg                             payload label declared by the world (float)
    declared_surface                                floor label declared by the world (categorical)

Action (A = 3), in this order::

    side_velocity_m_s  sideways command through the verified side field; its sign picks the side, 0 = stay
    start_delay_s      wait after the decision instant before the sideways command starts
    hold_duration_s    how long the sideways command is held

Outcomes (K = 5), in this fixed order (the residual order)::

    min_separation_m   closest ball-to-body distance, 0 on contact             read from the world
    collision          1 if the ball touched the dog, else 0                    read from the world
    fall               1 if the dog fell, else 0                                read from the world
    recovery_time_s    time until the dog is stable again, 0 if never upset     read from the world
    lateral_command_m  |side velocity| x hold duration                          exact, from the commands issued

Episode (one attempt)::

    world.reset(<the seven ball/dog keys>, attempt_seed)
    world.step(nominal, start_delay_s)                     skipped when the delay is 0
    world.step(dodge, hold_duration_s)
    world.step(nominal, episode_duration_s - delay - hold) skipped when nothing remains
    world.finish_episode()

    nominal = {forward field: dog_forward_speed_m_s, side field: 0.0,               yaw field: 0.0}
    dodge   = {forward field: dog_forward_speed_m_s, side field: side_velocity_m_s, yaw field: 0.0}

Every constant comes from config.yaml (``skills.dodge``, ``ball.radius_m``) and from the
verified command schema (``robot.walking_policy.command_schema``); nothing is defaulted, and
a null key stops the build with a :class:`~learned_judgment.config.ConfigError` naming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np

from learned_judgment.config import ConfigError, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.skills.base import (
    ActionDim,
    CommandSchema,
    Outcome,
    SkillSpec,
    Variable,
    World,
    check_command_binding,
    encode_row,
)

__all__ = [
    "BALL_DOG_KEYS",
    "DECLARED_KEYS",
    "DodgeObjectiveWeights",
    "DodgePhysics",
    "DodgeSpec",
    "SPEC_CLASS",
    "ThrowSampler",
    "WORLD_OUTCOME_KEYS",
    "declared_values",
    "for_world",
    "throw_sampler_from",
]

ThrowSampler = Callable[[np.random.Generator, int], np.ndarray]
"""``(rng, n) -> float64 (n, 7)``: n throws, columns in :data:`BALL_DOG_KEYS` order (Person A's throw model)."""

BALL_DOG_KEYS: tuple[str, ...] = (
    "ball_rel_x_m",
    "ball_rel_y_m",
    "ball_rel_z_m",
    "ball_vel_x_m_s",
    "ball_vel_y_m_s",
    "ball_vel_z_m_s",
    "dog_forward_speed_m_s",
)
"""The situation keys requested from the world in ``reset`` (situation columns 0..6)."""

DECLARED_KEYS: tuple[str, ...] = ("declared_payload_kg", "declared_surface")
"""Declared world values: read FROM the world (situation columns 7, 8), never requested."""

WORLD_OUTCOME_KEYS: tuple[str, ...] = ("min_separation_m", "collision", "fall", "recovery_time_s")
"""Outcome keys read from ``world.finish_episode()``; every other key of that mapping is ignored."""

_ROOT = "skills.dodge"
# Structural, not tunable: a duration within this of zero is float rounding of zero (for
# example 1.6 - 0.6 - 1.0), so its step is skipped instead of sent as a 1e-16 s step.
_TIME_TOLERANCE_S = 1e-9


# --------------------------------------------------------------------------------------
# Constants read from config.yaml
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DodgePhysics:
    """Constants of the physics guess: ``skills.dodge.physics.*`` and ``ball.radius_m``.

    The guess is a straight-trotting dog with a first-order sideways step and a ballistic
    ball that stops where it first reaches the floor; the residual model absorbs the rest.
    """

    gravity_m_s2: float
    assumed_latency_s: float
    response_time_s: float
    time_step_s: float
    collision_softness_m: float
    body_half_length_m: float
    body_radius_m: float
    base_height_m: float
    ball_radius_m: float


@dataclass(frozen=True)
class DodgeObjectiveWeights:
    """Weights of the risk-aware objective: ``skills.dodge.objective_weights.*``."""

    collision: float
    fall: float
    separation: float
    separation_cap_m: float
    contact_margin_m: float
    motion: float
    recovery: float


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(config: Mapping[str, Any], key: str, *, positive: bool = False, non_negative: bool = False) -> float:
    value = require(config, key)
    if not _is_number(value):
        raise ConfigError(f"config.yaml key {key!r} must be a finite number; got {value!r}")
    if positive and not value > 0:
        raise ConfigError(f"config.yaml key {key!r} must be > 0; got {value!r}")
    if non_negative and value < 0:
        raise ConfigError(f"config.yaml key {key!r} must be >= 0; got {value!r}")
    return float(value)


def _interval(config: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = require(config, key)
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_number(item) for item in value)
        and value[0] < value[1]
    ):
        raise ConfigError(f"config.yaml key {key!r} must be [low, high] with finite numbers low < high; got {value!r}")
    return float(value[0]), float(value[1])


def _levels(config: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = require(config, key)
    if (
        not isinstance(value, (list, tuple))
        or len(value) < 1
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ConfigError(f"config.yaml key {key!r} must list >= 1 unique non-empty level names; got {value!r}")
    return tuple(value)


def _command_field(config: Mapping[str, Any], key: str, schema: CommandSchema) -> str:
    value = require(config, key)
    if not isinstance(value, str) or value not in schema.fields:
        raise ConfigError(
            f"config.yaml key {key!r} names {value!r}, which is not a verified command field; "
            f"verified fields: {sorted(schema.fields)}"
        )
    return value


def _rows(label: str, values: Any, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (n, {width}); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contain non-finite values")
    return array


# --------------------------------------------------------------------------------------
# The SkillSpec
# --------------------------------------------------------------------------------------


class DodgeSpec(SkillSpec):
    """DODGE SkillSpec: choose a sideways dodge command, when to start it and how long to hold it.

    Args:
        config:              the parsed config.yaml.
        command_schema:      the verified command schema of the selected walking policy.
        throw_sampler:       Person A's throw model, ``(rng, n) -> float64 (n, 7)`` in
                             :data:`BALL_DOG_KEYS` order. Needed only by :meth:`sample_situations`
                             (dreams, splits, the spec-validation probe).
        declared_payload_kg: the current world's declared payload (required with a throw_sampler).
        declared_surface:    the current world's declared surface level (required with a throw_sampler).

    Raises:
        ConfigError: a required key is missing, null or malformed (the message names the key),
                     or the action timings do not fit inside the episode.
        UnverifiedCommandSchemaError / SkillSpecError: from the command-binding check.
        ValueError:  inconsistent sampler / declared-value arguments.
    """

    skill_name: ClassVar[str] = "dodge"

    def __init__(
        self,
        config: Mapping[str, Any],
        command_schema: CommandSchema,
        *,
        throw_sampler: ThrowSampler | None = None,
        declared_payload_kg: float | None = None,
        declared_surface: str | None = None,
    ) -> None:
        if not isinstance(command_schema, CommandSchema):
            raise TypeError(f"command_schema must be a CommandSchema; got {type(command_schema).__name__}")
        self.command_schema = command_schema

        # ----- situation ------------------------------------------------------------------
        ranges = {key: _interval(config, f"{_ROOT}.situation_ranges.{key}") for key in BALL_DOG_KEYS}
        payload_range = _interval(config, f"{_ROOT}.situation_ranges.declared_payload_kg")
        surfaces = _levels(config, f"{_ROOT}.declared_surfaces")

        # ----- command fields and action ranges ------------------------------------------
        self.side_field = _command_field(config, f"{_ROOT}.action.side_velocity_field", command_schema)
        self.forward_field = _command_field(config, f"{_ROOT}.action.forward_velocity_field", command_schema)
        self.yaw_field = _command_field(config, f"{_ROOT}.action.yaw_rate_field", command_schema)
        delay_range = _interval(config, f"{_ROOT}.action.start_delay_s")
        hold_range = _interval(config, f"{_ROOT}.action.hold_duration_s")
        self.episode_duration_s = _number(config, f"{_ROOT}.episode_duration_s", positive=True)

        points = require(config, f"{_ROOT}.candidate_points_per_dim")
        if not (
            isinstance(points, (list, tuple))
            and len(points) == 3
            and all(isinstance(p, int) and not isinstance(p, bool) and p >= 2 for p in points)
        ):
            raise ConfigError(
                f"config.yaml key '{_ROOT}.candidate_points_per_dim' must list 3 integers >= 2 "
                f"(side velocity, start delay, hold duration); got {points!r}"
            )
        self.candidate_points_per_dim: tuple[int, ...] = tuple(int(p) for p in points)

        # ----- physics guess and objective ------------------------------------------------
        physics = f"{_ROOT}.physics"
        self.physics = DodgePhysics(
            gravity_m_s2=_number(config, f"{physics}.gravity_m_s2", positive=True),
            assumed_latency_s=_number(config, f"{physics}.assumed_latency_s", non_negative=True),
            response_time_s=_number(config, f"{physics}.response_time_s", positive=True),
            time_step_s=_number(config, f"{physics}.time_step_s", positive=True),
            collision_softness_m=_number(config, f"{physics}.collision_softness_m", positive=True),
            body_half_length_m=_number(config, f"{physics}.body_half_length_m", non_negative=True),
            body_radius_m=_number(config, f"{physics}.body_radius_m", positive=True),
            base_height_m=_number(config, f"{physics}.base_height_m", positive=True),
            ball_radius_m=_number(config, "ball.radius_m", positive=True),
        )
        weights = f"{_ROOT}.objective_weights"
        self.weights = DodgeObjectiveWeights(
            collision=_number(config, f"{weights}.collision", non_negative=True),
            fall=_number(config, f"{weights}.fall", non_negative=True),
            separation=_number(config, f"{weights}.separation", non_negative=True),
            separation_cap_m=_number(config, f"{weights}.separation_cap_m", positive=True),
            contact_margin_m=_number(config, f"{weights}.contact_margin_m", non_negative=True),
            motion=_number(config, f"{weights}.motion", non_negative=True),
            recovery=_number(config, f"{weights}.recovery", non_negative=True),
        )

        # ----- build-time consistency -----------------------------------------------------
        self._check_fields(ranges["dog_forward_speed_m_s"])
        if delay_range[1] + hold_range[1] > self.episode_duration_s + _TIME_TOLERANCE_S:
            raise ConfigError(
                f"the longest start delay ({delay_range[1]} s, '{_ROOT}.action.start_delay_s') plus the longest "
                f"hold ({hold_range[1]} s, '{_ROOT}.action.hold_duration_s') exceeds "
                f"'{_ROOT}.episode_duration_s' ({self.episode_duration_s} s)"
            )
        if self.physics.time_step_s > self.episode_duration_s:
            raise ConfigError(
                f"'{physics}.time_step_s' ({self.physics.time_step_s} s) is longer than "
                f"'{_ROOT}.episode_duration_s' ({self.episode_duration_s} s)"
            )

        # ----- declarative attributes -----------------------------------------------------
        side = command_schema.field(self.side_field)
        self.name = self.skill_name
        self.description = (
            "Dodge a thrown ball while trotting: choose the sideways speed command (its sign picks the side), "
            "when to start it and how long to hold it, so the ball misses the dog and the dog does not fall."
        )
        frame = "in the dog's heading frame at the decision instant (x forward, y left, z up, origin at the base)"
        descriptions = {
            "ball_rel_x_m": f"ball centre x relative to the base, {frame}",
            "ball_rel_y_m": f"ball centre y relative to the base, {frame}",
            "ball_rel_z_m": f"ball centre z relative to the base, {frame}",
            "ball_vel_x_m_s": f"ball velocity along x, {frame}",
            "ball_vel_y_m_s": f"ball velocity along y, {frame}",
            "ball_vel_z_m_s": f"ball velocity along z, {frame}",
            "dog_forward_speed_m_s": (
                f"forward speed the dog trots at before the decision instant; sent as {self.forward_field!r} "
                "for the whole episode"
            ),
        }
        units = {key: ("m" if key.endswith("_m") else "m/s") for key in BALL_DOG_KEYS}
        self.situation = (
            *(
                Variable(key, "float", units[key], descriptions[key], low=ranges[key][0], high=ranges[key][1])
                for key in BALL_DOG_KEYS
            ),
            Variable(
                "declared_payload_kg",
                "float",
                "kg",
                "payload mass the world declares (a label that can be wrong; never the true mass)",
                low=payload_range[0],
                high=payload_range[1],
            ),
            Variable(
                "declared_surface",
                "categorical",
                "label",
                "floor type the world declares (a label; never its friction)",
                levels=surfaces,
            ),
        )
        self.actions = (
            ActionDim(
                "side_velocity_m_s",
                "float",
                side.unit,
                f"sideways speed sent through {self.side_field!r} while the dodge is held; positive steps left, "
                "negative steps right, 0 does not dodge",
                low=side.low,
                high=side.high,
                role="command",
                command_field=self.side_field,
            ),
            ActionDim(
                "start_delay_s",
                "float",
                "s",
                "time after the decision instant before the sideways command starts",
                low=delay_range[0],
                high=delay_range[1],
                role="timing",
            ),
            ActionDim(
                "hold_duration_s",
                "float",
                "s",
                "how long the sideways command is held before the nominal command returns",
                low=hold_range[0],
                high=hold_range[1],
                role="timing",
            ),
        )
        self.outcomes = (
            Outcome(
                "min_separation_m",
                "regression",
                "m",
                "closest distance between the ball and the dog's body during the episode, 0 on contact (world)",
                low=0.0,
            ),
            Outcome("collision", "binary", "flag", "1 if the ball touched the dog's body, else 0 (world)"),
            Outcome("fall", "binary", "flag", "1 if the dog fell during the episode, else 0 (world)"),
            Outcome(
                "recovery_time_s",
                "regression",
                "s",
                "time until the dog is stable again, 0 if it was never destabilised (world)",
                low=0.0,
            ),
            Outcome(
                "lateral_command_m",
                "regression",
                "m",
                "commanded sideways travel |side velocity| x hold duration, exact from the commands issued",
                low=0.0,
            ),
        )
        self.primary_outcome = "min_separation_m"
        self.hidden_effects = (
            "Payload: the true mass (world.hidden_effects.payload_mass_kg) sits off centre, and the declared label "
            "declared_payload_kg can be wrong by world.hidden_effects.declared_mass_error_fraction; it changes how "
            "fast the trunk moves sideways. The physics guess ignores the payload.",
            "Floor friction differs by surface and varies per run; declared_surface names the surface, never its "
            "friction. A slippery floor slows the sideways step and can cause a fall. The physics guess ignores it.",
            "Slope (world.hidden_effects.slope_deg): the floor is not level, so the dog drifts and the ball lands at "
            "another height than the flat floor of the physics guess.",
            "Motor strength (world.hidden_effects.motor_strength_fraction): weaker motors slow the step response.",
            "Command latency (world.hidden_effects.command_latency_s) plus jitter: the physics guess assumes "
            "skills.dodge.physics.assumed_latency_s for every command.",
            "A sideways command also turns the dog (Person A's standstill vel_y traces, messages log 2026-11-21); "
            "the physics guess keeps the heading fixed.",
            "Left and right dodges are not mirror images (different steady sideways speeds and yaw coupling in the "
            "same traces); the physics guess is mirror symmetric.",
            "Forward speed under-delivers at low commands (Gate 1, 0.3 m/s commanded) and with zero yaw commanded "
            "the dog drifts anticlockwise; the physics guess moves the base straight at the commanded speed.",
            "The forward command at trot speed together with a sideways command has not been measured: every "
            "command proof run moved one field alone.",
            "The sideways step response was measured from standstill only; assumed_latency_s and response_time_s "
            "come from those traces, not from a trot.",
            "The ball can bounce and roll after it lands; the physics guess stops it where it first reaches the "
            "floor (base_height_m below the base).",
            "Body shape: legs, head and the gait's vertical bob lie outside the trunk capsule of the physics guess, "
            "so a ball touching a leg is a collision the guess does not predict.",
        )

        # ----- throws and declared values for sample_situations ---------------------------
        self._throw_sampler: ThrowSampler | None = None
        self._declared_row: np.ndarray | None = None
        if throw_sampler is None:
            if declared_payload_kg is not None or declared_surface is not None:
                raise ValueError(
                    "declared_payload_kg / declared_surface are used only to complete sampled throws; "
                    "pass them together with a throw_sampler"
                )
        else:
            if not callable(throw_sampler):
                raise TypeError(f"throw_sampler must be callable (rng, n) -> (n, 7); got {type(throw_sampler).__name__}")
            if declared_payload_kg is None or declared_surface is None:
                raise ValueError(
                    "a throw_sampler needs the current world's declared_payload_kg and declared_surface: "
                    "sampled situations carry the declared values of the world they are run in"
                )
            payload_row = encode_row(
                self.situation[len(BALL_DOG_KEYS) :],
                {"declared_payload_kg": declared_payload_kg, "declared_surface": declared_surface},
            )
            self._throw_sampler = throw_sampler
            self._declared_row = payload_row

        check_command_binding(self, command_schema)

    # ----- build-time helpers ----------------------------------------------------------------

    def _check_fields(self, forward_range: tuple[float, float]) -> None:
        """The three command fields are distinct, together form the whole verified schema, the
        two velocity fields are in m/s, and every value the episode can send is verified."""
        schema = self.command_schema
        fields = (self.side_field, self.forward_field, self.yaw_field)
        if len(set(fields)) != 3:
            raise ConfigError(f"'{_ROOT}.action.*_field' must name three different command fields; got {fields}")
        if set(fields) != set(schema.fields):
            raise ConfigError(
                f"'{_ROOT}.action.*_field' name {sorted(fields)}, but every command must carry exactly the verified "
                f"fields {sorted(schema.fields)}"
            )
        for key, name in (("side_velocity_field", self.side_field), ("forward_velocity_field", self.forward_field)):
            if schema.field(name).unit != "m/s":
                raise ConfigError(
                    f"'{_ROOT}.action.{key}' names {name!r} with unit {schema.field(name).unit!r}; the physics guess "
                    "needs a velocity in m/s"
                )
        for name in (self.side_field, self.yaw_field):
            field = schema.field(name)
            if not field.low <= 0.0 <= field.high:
                raise ConfigError(f"the nominal command sends {name}=0.0, outside its verified range [{field.low}, {field.high}]")
        forward = schema.field(self.forward_field)
        if forward_range[0] < forward.low or forward_range[1] > forward.high:
            raise ConfigError(
                f"'{_ROOT}.situation_ranges.dog_forward_speed_m_s' {list(forward_range)} is sent as "
                f"{self.forward_field!r} but exceeds its verified range [{forward.low}, {forward.high}]"
            )

    # ----- situations ------------------------------------------------------------------------

    def sample_situations(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` situations (n, 9): ``throw_sampler(rng, n)`` gives the seven ball/dog
        columns and the current world's declared values are appended to every row.

        Row ``i`` depends on ``n`` only if the throw sampler's row ``i`` does (Person A's
        sampler draws row by row, as the base contract requires).

        Raises RuntimeError without a throw_sampler, and ValueError when the sampler returns
        the wrong shape, a non-finite value, or a value outside the declared situation ranges.
        """
        if self._throw_sampler is None or self._declared_row is None:
            raise RuntimeError(
                "DodgeSpec.sample_situations needs a throw_sampler (Person A's throw model, (rng, n) -> (n, 7)) and "
                "the world's declared values: an independent box over the ball values would mostly miss the dog, "
                "so DODGE has no default sampler"
            )
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"n must be an int >= 0; got {n!r}")
        n = int(n)
        try:
            throws = np.asarray(self._throw_sampler(rng, n), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"throw_sampler must return float64 (n, 7); conversion failed: {error}") from error
        width = len(BALL_DOG_KEYS)
        if throws.shape != (n, width):
            raise ValueError(f"throw_sampler must return shape ({n}, {width}) in {BALL_DOG_KEYS} order; got {throws.shape}")
        if not np.all(np.isfinite(throws)):
            raise ValueError("throw_sampler returned non-finite values")
        for column, variable in enumerate(self.situation[:width]):
            values = throws[:, column]
            outside = (values < variable.low) | (values > variable.high)
            if np.any(outside):
                raise ValueError(
                    f"throw_sampler returned {int(outside.sum())} of {n} rows with {variable.name} outside its declared "
                    f"range [{variable.low}, {variable.high}] (first: {float(values[outside][0])})"
                )
        return np.concatenate([throws, np.repeat(self._declared_row[None, :], n, axis=0)], axis=1)

    # ----- physics guess ---------------------------------------------------------------------

    def _time_grid(self) -> np.ndarray:
        """(T,) times 0, dt, 2 dt, ... up to and including the episode duration."""
        dt, duration = self.physics.time_step_s, self.episode_duration_s
        count = int(math.floor(duration / dt + _TIME_TOLERANCE_S))
        times = np.minimum(dt * np.arange(count + 1, dtype=np.float64), duration)
        if duration - times[-1] > _TIME_TOLERANCE_S:
            times = np.append(times, duration)
        return times

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Outcome guess (n, 5) from the formula alone, in outcome order.

        Args:
            situations: (n, 9) encoded situations.
            actions:    (n, 3) actions inside the declared bounds.

        Model, per row, at every time ``t`` of a grid of ``physics.time_step_s`` from 0 to the
        episode duration (a running minimum; memory O(n), no (n, T) array):

        * ball: ``p(t) = p0 + v0 t - (0, 0, g t^2 / 2)``, stopped at the first time its centre
          reaches ``ball_radius_m - base_height_m`` (the floor lies ``base_height_m`` below the
          base); a ball starting at or below that height stays where it starts.
        * dog base: ``x = forward speed * t``; sideways speed is 0 until ``start_delay +
          assumed_latency_s``, rises first-order (time constant ``response_time_s``) toward the
          side velocity while the command is held, and decays the same way back to 0 from
          ``start_delay + hold + assumed_latency_s`` (latency delays both edges). Closed form.
        * trunk: a capsule along x of half length ``body_half_length_m`` and radius
          ``body_radius_m`` centred on the base, heading fixed.
        * clearance = distance(ball centre, capsule axis segment) - body_radius_m - ball_radius_m.

        Outcomes: min_separation_m = max(min clearance, 0); collision = logistic(-min clearance /
        collision_softness_m) (0.5 at zero clearance); fall = 0; recovery_time_s = 0;
        lateral_command_m = |side velocity| x hold duration (exact).
        """
        situations = _rows("situations", situations, self.n_situation)
        actions = _rows("actions", actions, self.n_action)
        if situations.shape[0] != actions.shape[0]:
            raise ValueError(f"situations {situations.shape} and actions {actions.shape} must have the same rows")
        if not np.all(self.action_in_bounds(actions)):
            raise ValueError("actions outside the declared action bounds")
        c = self.physics
        p_x, p_y, p_z, v_x, v_y, v_z, forward = (situations[:, column] for column in range(len(BALL_DOG_KEYS)))
        side, delay, hold = actions[:, 0], actions[:, 1], actions[:, 2]
        g, tau = c.gravity_m_s2, c.response_time_s

        height = p_z - (c.ball_radius_m - c.base_height_m)  # ball centre above its resting height
        landing = np.where(
            height > 0.0, (v_z + np.sqrt(v_z * v_z + 2.0 * g * np.maximum(height, 0.0))) / g, 0.0
        )
        start = delay + c.assumed_latency_s  # the sideways speed starts to change
        stop = start + hold  # ... and starts to decay back to 0
        reach = c.body_radius_m + c.ball_radius_m
        half = c.body_half_length_m

        closest = np.full(situations.shape[0], np.inf)
        for t in self._time_grid():
            flight = np.minimum(t, landing)
            ball_x = p_x + v_x * flight
            ball_y = p_y + v_y * flight
            ball_z = p_z + v_z * flight - 0.5 * g * flight * flight
            rise = np.clip(t - start, 0.0, hold)  # time spent rising toward the side velocity
            decay = np.maximum(t - stop, 0.0)  # time since the held command ended
            rise_left = np.exp(-rise / tau)
            speed_at_end = side * (1.0 - rise_left)
            dog_y = side * (rise - tau * (1.0 - rise_left)) + speed_at_end * tau * (1.0 - np.exp(-decay / tau))
            along = ball_x - forward * t
            off_axis = along - np.clip(along, -half, half)
            distance = np.sqrt(off_axis * off_axis + (ball_y - dog_y) ** 2 + ball_z * ball_z)
            np.minimum(closest, distance - reach, out=closest)

        guess = np.empty((situations.shape[0], self.n_outcome), dtype=np.float64)
        guess[:, self.outcome_index("min_separation_m")] = np.maximum(closest, 0.0)
        guess[:, self.outcome_index("collision")] = 0.5 * (1.0 + np.tanh(-closest / (2.0 * c.collision_softness_m)))
        guess[:, self.outcome_index("fall")] = 0.0
        guess[:, self.outcome_index("recovery_time_s")] = 0.0
        guess[:, self.outcome_index("lateral_command_m")] = np.abs(side) * hold
        return guess

    # ----- objective, loss, success ------------------------------------------------------------

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        """Risk-aware utility (n,), higher is better (weights: ``skills.dodge.objective_weights``)::

            - collision  * max(mean collision, P(min_separation_m <= contact_margin_m))
            - fall       * mean fall
            + separation * min(lowest-level quantile of min_separation_m, separation_cap_m)
            - motion     * mean lateral_command_m
            - recovery   * mean recovery_time_s

        The lowest level is ``predicted.quantile_levels[0]``, so a wider predictive spread of
        the separation lowers the score.
        """
        if tuple(predicted.names) != self.outcome_names:
            raise ValueError(f"predicted outcomes {predicted.names} are not the DODGE outcomes {self.outcome_names}")
        w = self.weights
        contact = np.maximum(
            predicted.mean_of("collision"), predicted.prob_below("min_separation_m", w.contact_margin_m)
        )
        low_separation = predicted.quantile_of("min_separation_m", float(predicted.quantile_levels[0]))
        return (
            -w.collision * contact
            - w.fall * predicted.mean_of("fall")
            + w.separation * np.minimum(low_separation, w.separation_cap_m)
            - w.motion * predicted.mean_of("lateral_command_m")
            - w.recovery * predicted.mean_of("recovery_time_s")
        )

    def _failed(self, measured: np.ndarray) -> np.ndarray:
        measured = _rows("measured", measured, self.n_outcome)
        collision = measured[:, self.outcome_index("collision")]
        fall = measured[:, self.outcome_index("fall")]
        return (collision == 1.0) | (fall == 1.0)

    def loss(self, measured: np.ndarray) -> np.ndarray:
        """Preregistered primary loss (n,) of measured outcomes (n, 5): 1.0 where the ball
        touched the dog or the dog fell (collision or fall is 1), else 0.0."""
        return self._failed(measured).astype(np.float64)

    def success(self, measured: np.ndarray) -> np.ndarray:
        """bool (n,): no collision and no fall (PREREGISTRATION.md, operational definitions for DODGE)."""
        return ~self._failed(measured)

    # ----- one real episode ------------------------------------------------------------------

    def _command(self, forward: float, side: float) -> dict[str, float]:
        return {self.forward_field: float(forward), self.side_field: float(side), self.yaw_field: 0.0}

    def episode(
        self, world: World, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int
    ) -> Mapping[str, float]:
        """Run one attempt in ``world`` and return the five outcomes, in outcome order.

        Args:
            world:        Person A's Go2World (``reset`` / ``step`` / ``finish_episode``).
            situation:    all nine situation keys; the seven ball/dog keys are requested from the
                          world, the two declared keys must equal what the world declares.
            action:       the three action keys, inside the declared bounds.
            attempt_seed: int >= 0 seeding this attempt's noise in the world.

        Everything is checked before the world is touched: the command binding, the situation
        and action against their declared ranges, and every command against the verified
        schema. The first four outcomes are read from ``world.finish_episode()`` (other keys
        ignored); ``lateral_command_m`` is |side velocity| x hold duration of the commands issued.
        """
        check_command_binding(self, self.command_schema)
        if isinstance(attempt_seed, bool) or not isinstance(attempt_seed, (int, np.integer)) or attempt_seed < 0:
            raise ValueError(f"attempt_seed must be an int >= 0; got {attempt_seed!r}")
        request = self.reset_situation(situation)
        schedule = self.commands(situation, action)
        side, _, hold = (float(value) for value in self.encode_action(action))

        observation = world.reset(request, int(attempt_seed))
        self._check_declared(observation, situation)
        for command, duration in schedule:
            world.step(dict(command), float(duration))
        measured = self._read_outcome(world.finish_episode())
        measured["lateral_command_m"] = abs(side) * hold
        return {name: measured[name] for name in self.outcome_names}

    def reset_situation(self, situation: Mapping[str, Any]) -> dict[str, Any]:
        """The seven ball and dog values requested from the world for ``situation`` (all nine keys).

        The two declared keys are not requested: the world reports them, and :meth:`episode`
        checks that they equal the situation's.
        """
        situation_vector = self.encode_situation(situation)
        return {key: float(situation_vector[column]) for column, key in enumerate(BALL_DOG_KEYS)}

    def commands(
        self, situation: Mapping[str, Any], action: Mapping[str, Any]
    ) -> list[tuple[dict[str, float], float]]:
        """The commands of one attempt, in the order sent, as ``[(command, duration_s), ...]``.

        Trot at the situation's forward speed for ``start_delay_s``, the same trot plus the
        sideways command for ``hold_duration_s``, then the trot again until
        ``episode_duration_s``. A phase of zero length is left out. Every command names all
        three verified fields and is checked against the verified schema.
        """
        situation_vector = self.encode_situation(situation)
        action_vector = self.encode_action(action)
        forward = float(situation_vector[self.situation_names.index("dog_forward_speed_m_s")])
        side, delay, hold = (float(value) for value in action_vector)
        remaining = self.episode_duration_s - delay - hold
        if remaining < -_TIME_TOLERANCE_S:
            raise ValueError(
                f"start_delay_s + hold_duration_s = {delay + hold} s exceeds the episode duration {self.episode_duration_s} s"
            )
        nominal = self._command(forward, 0.0)
        dodge = self._command(forward, side)
        for command in (nominal, dodge):
            self.command_schema.check_command(command)
        return [
            (dict(command), float(duration))
            for command, duration in ((nominal, delay), (dodge, hold), (nominal, remaining))
            if duration > _TIME_TOLERANCE_S
        ]

    @staticmethod
    def _check_declared(observation: Mapping[str, Any], situation: Mapping[str, Any]) -> None:
        if not isinstance(observation, Mapping):
            raise TypeError(f"world.reset must return a mapping; got {type(observation).__name__}")
        payload = observation.get("declared_payload_kg")
        if payload is not None and float(payload) != float(situation["declared_payload_kg"]):
            raise ValueError(
                f"the world declares declared_payload_kg={payload!r} but the situation has "
                f"{situation['declared_payload_kg']!r}: declared values are read from the world, not requested"
            )
        surface = observation.get("declared_surface")
        if surface is not None and surface != situation["declared_surface"]:
            raise ValueError(
                f"the world declares declared_surface={surface!r} but the situation has "
                f"{situation['declared_surface']!r}: declared values are read from the world, not requested"
            )

    def _read_outcome(self, outcome: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(outcome, Mapping):
            raise TypeError(f"world.finish_episode must return a mapping; got {type(outcome).__name__}")
        missing = [key for key in WORLD_OUTCOME_KEYS if key not in outcome]
        if missing:
            raise ValueError(f"the world's outcome lacks {missing}; it has {sorted(outcome)}")
        world_outcomes = tuple(o for o in self.outcomes if o.name in WORLD_OUTCOME_KEYS)
        vector = encode_row(world_outcomes, {o.name: outcome[o.name] for o in world_outcomes})  # finite; binaries 0/1
        for outcome_spec, value in zip(world_outcomes, vector):
            low, high = outcome_spec.bounds
            if not low <= value <= high:
                raise ValueError(f"the world measured {outcome_spec.name}={value}, outside [{low}, {high}]")
        return {outcome_spec.name: float(value) for outcome_spec, value in zip(world_outcomes, vector)}


def declared_values(world: Any) -> tuple[float, str]:
    """(declared_payload_kg, declared_surface) of ``world``.

    Reads the attributes ``world.declared_payload_kg`` and ``world.declared_surface`` when the
    world has them (agreed with Person A). Until they exist, reads them from the observation of
    ``world.reset()``; that reset starts an episode which the next ``reset`` abandons.
    """
    payload = getattr(world, "declared_payload_kg", None)
    surface = getattr(world, "declared_surface", None)
    if payload is None or surface is None:
        observation = world.reset()
        payload, surface = observation["declared_payload_kg"], observation["declared_surface"]
    return float(payload), str(surface)


def throw_sampler_from(
    sample: Callable[[Mapping[str, Any], np.random.Generator, int], Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Adapt a throw model ``sample(config, rng, n) -> n mappings`` to the spec's sampler.

    Each mapping must hold exactly the seven ball and dog keys (:data:`BALL_DOG_KEYS`); the
    result is the float64 array (n, 7) in that key order. Nothing is filled in or clipped: a
    missing key, an extra key or a wrong count raises ValueError.
    """

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        throws = sample(config, rng, n)
        if len(throws) != n:
            raise ValueError(f"the throw model returned {len(throws)} situations for n={n}")
        rows = np.empty((n, len(BALL_DOG_KEYS)), dtype=np.float64)
        for index, throw in enumerate(throws):
            if set(throw) != set(BALL_DOG_KEYS):
                raise ValueError(
                    f"throw {index} must hold exactly the keys {list(BALL_DOG_KEYS)}; got {sorted(throw)}"
                )
            rows[index] = [float(throw[key]) for key in BALL_DOG_KEYS]
        return rows

    return sampler


def for_world(config: Mapping[str, Any], world: Any) -> "DodgeSpec":
    """The DODGE spec for ``world``: its declared values, and throws from Person A's throw model.

    The throw model is ``learned_judgment.env.sample_dodge_situations(config, rng, n)``; it is
    imported here, at call time, so this module still imports while the model does not exist.
    The spec is built and validated through :func:`learned_judgment.skills.load_skill`.
    """
    from learned_judgment.env import sample_dodge_situations  # Person A's throw model
    from learned_judgment.skills import load_skill

    payload, surface = declared_values(world)
    return load_skill(
        "dodge",
        config,
        throw_sampler=throw_sampler_from(sample_dodge_situations, config),
        declared_payload_kg=payload,
        declared_surface=surface,
    )


SPEC_CLASS = DodgeSpec
"""The registry in :mod:`learned_judgment.skills` builds this class for ``load_skill("dodge", ...)``."""
