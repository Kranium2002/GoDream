"""SIDESTEP_REACH: a development skill, "reach a given sideways displacement", with no ball.

Purpose
-------
DODGE cannot run until Person A's simulated ball exists. Half of the DODGE physics guess is
the formula for how far the dog moves sideways after a sideways command. This skill isolates
exactly that half, so the whole learning chain (attempts, residual models, decisions) can run
on the REAL Go2 simulator today. It is a development check: it lives under
``experiments/proofs/B_sidestep_dev/``, is not registered in ``learned_judgment.skills``, and
none of its numbers is an evaluation result.

Frame and units
---------------
Positions are in the dog's heading frame at the decision instant (the moment ``world.reset``
returns): x forward, y left, origin at the trunk; SI units. The heading is the trunk's yaw in
the observation that ``world.reset`` returns.

Situation (S = 4), in this order::

    dog_forward_speed_m_s   forward speed the dog trots at; requested from the world in reset and
                            sent as the forward command for the whole episode
    target_sideways_m       sideways displacement to reach at the end of the episode, + left
    declared_payload_kg     payload label the world declares (float), read from the world
    declared_surface        floor label the world declares (categorical), read from the world

Action (A = 2), in this order::

    side_velocity_m_s   sideways command through DODGE's verified side field, over that field's
                        verified range; its sign picks the side
    hold_duration_s     how long the sideways command is held, from the decision instant

Outcomes (K = 3), in this fixed order (the residual order)::

    sideways_error_m    measured sideways travel of the trunk at the end of the episode, in the
                        heading frame of the decision instant, minus the target      (world)
    fall                1 if the dog fell, else 0                                    (world)
    lateral_command_m   |side velocity| x hold duration, exact from the commands issued

The task named two outcomes. ``lateral_command_m`` is the third because
:meth:`SkillSpec.objective` sees only the predicted outcomes, never the action, so a cost on
commanded motion must be an outcome; DODGE uses the same construction (its own
``lateral_command_m``). Its residual is exactly 0.

Episode (one attempt)::

    world.reset({"dog_forward_speed_m_s": forward}, attempt_seed)
    world.step(sidestep, hold_duration_s)                    whole 0.02 s control periods
    world.step(nominal, episode_duration_s - hold)           skipped when nothing remains
    world.finish_episode()

    sidestep = {forward field: forward, side field: side_velocity_m_s, yaw field: 0.0}
    nominal  = {forward field: forward, side field: 0.0,               yaw field: 0.0}

The command fields are DODGE's (``skills.dodge.action.*_field``), so this skill sends exactly
the commands DODGE sends with a start delay of 0.

Physics guess
-------------
The same first-order sideways response DodgeSpec assumes, with the same constants
(``skills.dodge.physics.assumed_latency_s`` and ``response_time_s``): sideways speed is 0 until
``assumed_latency_s`` after the command starts, rises first order (time constant
``response_time_s``) toward the commanded speed while it is held, and decays the same way back
to 0 from ``hold + assumed_latency_s``. :func:`sideways_travel` states it in closed form, with
the operations in the same order as the inline code of ``DodgeSpec.physics_guess``; check c0
compares the two. Fall is guessed 0.

Why these situation ranges (``dev_checks.sidestep_reach.situation_ranges``)
-------------------------------------------------------------------------
* Action bounds: side velocity -1 to 1 m/s (the verified ``vel_y`` range), hold 0.2 to 0.8 s,
  episode 1.8 s. After the speed has decayed, the first-order formula delivers exactly
  side x hold of travel; with the tail of at least 1.0 s it leaves at most about 0.02 m
  undelivered. The candidate grid therefore spans formula travels of about 0.05 m
  (0.25 m/s x 0.2 s) up to about 0.79 m on each side.
* The lead's first run on the real Go2World (experiments/proofs/B_go2world_contact/
  sidestep_dev_run.out.txt; 0.2 s trot, 0.6 s hold, 1.0 s trot, forward 0.3/0.6/0.9 m/s):
  travel at 1.8 s was 0.45 to 0.63 m left and 0.36 to 0.47 m right at full command,
  0.31 to 0.41 m left and 0.21 to 0.23 m right at half command, and 0.094 m left at zero
  command (the dog drifts and turns left). The formula said 0.589 and 0.295 m.
* Target -0.4 to 0.4 m: reachable on both sides by the simulator (the weaker right side
  already reached 0.36 to 0.47 m with a 0.6 s hold; holds up to 0.8 s add margin) and by the
  formula with margin, while formula and robot disagree by up to about 0.2 m inside it, which is
  the gap the residual models are meant to close. Targets near 0 are kept on purpose: the zero
  command drifts left by about 0.1 m, which the formula does not know.
* Forward speed 0.3 to 0.9 m/s: the speeds of Gate 1 and of the lead's run.

Every constant comes from config.yaml (``dev_checks.sidestep_reach``, ``skills.dodge.action``
for the three command field names, ``skills.dodge.physics`` for the two formula constants) and
from the verified command schema; nothing is defaulted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np

from learned_judgment.config import ConfigError, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.skills import _validation_settings
from learned_judgment.skills.base import (
    ActionDim,
    CommandSchema,
    Outcome,
    SkillSpec,
    Variable,
    World,
    check_command_binding,
    encode_row,
    validate_spec,
)
from learned_judgment.skills.dodge import (
    _TIME_TOLERANCE_S,
    DodgeSpec,
    _command_field,
    _interval,
    _number,
    _rows,
)

__all__ = [
    "ROOT",
    "SidestepObjectiveWeights",
    "SidestepReachSpec",
    "declared_values",
    "for_world",
    "heading_rad",
    "sideways_travel",
]

ROOT = "dev_checks.sidestep_reach"
_DODGE_ACTION = "skills.dodge.action"
_DODGE_PHYSICS = "skills.dodge.physics"
_FORWARD_KEY = "dog_forward_speed_m_s"
_TARGET_KEY = "target_sideways_m"
DECLARED_KEYS: tuple[str, ...] = ("declared_payload_kg", "declared_surface")


# --------------------------------------------------------------------------------------
# The formula and the frame (pure functions)
# --------------------------------------------------------------------------------------


def sideways_travel(
    side_velocity_m_s: np.ndarray,
    hold_duration_s: np.ndarray,
    time_s: np.ndarray | float,
    *,
    start_delay_s: np.ndarray | float,
    assumed_latency_s: float,
    response_time_s: float,
) -> np.ndarray:
    """Sideways travel (m, + left) at ``time_s`` after the decision instant, first-order model.

    The sideways speed is 0 until ``start_delay_s + assumed_latency_s``, rises toward
    ``side_velocity_m_s`` with time constant ``response_time_s`` while the command is held
    (``hold_duration_s``), and decays the same way back to 0 afterwards (the latency delays both
    edges). Closed form; the operations are in the order of ``DodgeSpec.physics_guess`` so that
    the same inputs give the same float64 bits. Arguments broadcast against each other.
    """
    side = np.asarray(side_velocity_m_s, dtype=np.float64)
    hold = np.asarray(hold_duration_s, dtype=np.float64)
    delay = np.asarray(start_delay_s, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    tau = response_time_s
    start = delay + assumed_latency_s  # the sideways speed starts to change
    stop = start + hold  # ... and starts to decay back to 0
    rise = np.clip(t - start, 0.0, hold)  # time spent rising toward the side velocity
    decay = np.maximum(t - stop, 0.0)  # time since the held command ended
    rise_left = np.exp(-rise / tau)
    speed_at_end = side * (1.0 - rise_left)
    return side * (rise - tau * (1.0 - rise_left)) + speed_at_end * tau * (1.0 - np.exp(-decay / tau))


def heading_rad(quat_wxyz: Any) -> float:
    """Yaw (rad, anticlockwise from the world x axis) of a body-to-world quaternion w x y z."""
    w, x, y, z = (float(value) for value in np.asarray(quat_wxyz, dtype=np.float64))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SidestepObjectiveWeights:
    """Weights of the risk-aware objective: ``dev_checks.sidestep_reach.objective_weights.*``."""

    fall: float
    abs_error: float
    spread: float
    motion: float


def _surface_levels(config: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = require(config, key)
    if (
        not isinstance(value, (list, tuple))
        or len(value) < 1
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ConfigError(f"config.yaml key {key!r} must list >= 1 unique non-empty level names; got {value!r}")
    return tuple(value)


def _points(config: Mapping[str, Any], key: str, count: int) -> tuple[int, ...]:
    value = require(config, key)
    if not (
        isinstance(value, (list, tuple))
        and len(value) == count
        and all(isinstance(p, int) and not isinstance(p, bool) and p >= 2 for p in value)
    ):
        raise ConfigError(f"config.yaml key {key!r} must list {count} integers >= 2; got {value!r}")
    return tuple(int(p) for p in value)


# --------------------------------------------------------------------------------------
# The SkillSpec
# --------------------------------------------------------------------------------------


class SidestepReachSpec(SkillSpec):
    """Choose a sideways command and how long to hold it so the trunk ends ``target_sideways_m``
    to the side of where it was at the decision instant, without falling.

    Args:
        config:              the parsed config.yaml.
        command_schema:      the verified command schema of the selected walking policy.
        declared_payload_kg: the current world's declared payload (read from the world).
        declared_surface:    the current world's declared surface level (read from the world).

    Raises:
        ConfigError: a required key is missing, null or malformed (the message names the key), or
                     the durations do not fit whole control periods / the episode.
        UnverifiedCommandSchemaError / SkillSpecError: from the command-binding check.
        ValueError:  a declared value outside its declared range or levels.
    """

    skill_name: ClassVar[str] = "sidestep_reach"

    def __init__(
        self,
        config: Mapping[str, Any],
        command_schema: CommandSchema,
        *,
        declared_payload_kg: float,
        declared_surface: str,
    ) -> None:
        if not isinstance(command_schema, CommandSchema):
            raise TypeError(f"command_schema must be a CommandSchema; got {type(command_schema).__name__}")
        self.command_schema = command_schema
        self.control_period_s = command_schema.update_period_s

        # ----- command fields: DODGE's binding -------------------------------------------------
        self.side_field = _command_field(config, f"{_DODGE_ACTION}.side_velocity_field", command_schema)
        self.forward_field = _command_field(config, f"{_DODGE_ACTION}.forward_velocity_field", command_schema)
        self.yaw_field = _command_field(config, f"{_DODGE_ACTION}.yaw_rate_field", command_schema)

        # ----- ranges, durations, grid --------------------------------------------------------
        forward_range = _interval(config, f"{ROOT}.situation_ranges.{_FORWARD_KEY}")
        target_range = _interval(config, f"{ROOT}.situation_ranges.{_TARGET_KEY}")
        payload_range = _interval(config, f"{ROOT}.situation_ranges.declared_payload_kg")
        surfaces = _surface_levels(config, f"{ROOT}.declared_surfaces")
        hold_range = _interval(config, f"{ROOT}.hold_duration_s")
        self.episode_duration_s = _number(config, f"{ROOT}.episode_duration_s", positive=True)
        self.min_tail_s = _number(config, f"{ROOT}.min_tail_s", positive=True)
        self.candidate_points_per_dim = _points(config, f"{ROOT}.candidate_points_per_dim", 2)
        self.success_tolerance_m = _number(config, f"{ROOT}.success_tolerance_m", positive=True)

        # ----- physics guess and objective ----------------------------------------------------
        self.assumed_latency_s = _number(config, f"{_DODGE_PHYSICS}.assumed_latency_s", non_negative=True)
        self.response_time_s = _number(config, f"{_DODGE_PHYSICS}.response_time_s", positive=True)
        weights = f"{ROOT}.objective_weights"
        self.weights = SidestepObjectiveWeights(
            fall=_number(config, f"{weights}.fall", non_negative=True),
            abs_error=_number(config, f"{weights}.abs_error", non_negative=True),
            spread=_number(config, f"{weights}.spread", non_negative=True),
            motion=_number(config, f"{weights}.motion", non_negative=True),
        )

        # ----- build-time consistency ---------------------------------------------------------
        self._check_fields(forward_range)
        self.episode_periods = self._whole_periods(self.episode_duration_s, f"'{ROOT}.episode_duration_s'")
        if hold_range[0] < self.control_period_s:
            raise ConfigError(
                f"'{ROOT}.hold_duration_s' starts at {hold_range[0]} s, shorter than one control period "
                f"({self.control_period_s} s)"
            )
        if self.episode_duration_s - hold_range[1] < self.min_tail_s - _TIME_TOLERANCE_S:
            raise ConfigError(
                f"'{ROOT}.episode_duration_s' ({self.episode_duration_s} s) minus the longest hold ({hold_range[1]} s, "
                f"'{ROOT}.hold_duration_s') leaves a tail shorter than '{ROOT}.min_tail_s' ({self.min_tail_s} s)"
            )
        hold_grid = np.linspace(hold_range[0], hold_range[1], self.candidate_points_per_dim[1])
        for value in hold_grid:
            self._whole_periods(float(value), f"hold grid point {float(value)} s ('{ROOT}.candidate_points_per_dim')")

        # ----- declarative attributes ---------------------------------------------------------
        side = command_schema.field(self.side_field)
        self.name = self.skill_name
        self.description = (
            "Reach a given sideways displacement while trotting: choose the sideways speed command (its sign picks "
            "the side) and how long to hold it, so the trunk ends at the target and the dog does not fall."
        )
        self.situation = (
            Variable(
                _FORWARD_KEY,
                "float",
                "m/s",
                f"forward speed the dog trots at before the decision instant; sent as {self.forward_field!r} for the "
                "whole episode",
                low=forward_range[0],
                high=forward_range[1],
            ),
            Variable(
                _TARGET_KEY,
                "float",
                "m",
                "sideways displacement of the trunk to reach at the end of the episode, in the heading frame of the "
                "decision instant, + left",
                low=target_range[0],
                high=target_range[1],
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
                f"sideways speed sent through {self.side_field!r} while it is held; positive steps left, negative "
                "steps right",
                low=side.low,
                high=side.high,
                role="command",
                command_field=self.side_field,
            ),
            ActionDim(
                "hold_duration_s",
                "float",
                "s",
                "how long the sideways command is held from the decision instant before the nominal trot returns",
                low=hold_range[0],
                high=hold_range[1],
                role="timing",
            ),
        )
        self.outcomes = (
            Outcome(
                "sideways_error_m",
                "regression",
                "m",
                "measured sideways travel of the trunk at the end of the episode, in the heading frame of the decision "
                "instant, minus the target (world observations)",
            ),
            Outcome("fall", "binary", "flag", "1 if the dog fell during the episode, else 0 (world)"),
            Outcome(
                "lateral_command_m",
                "regression",
                "m",
                "commanded sideways travel |side velocity| x hold duration, exact from the commands issued",
                low=0.0,
            ),
        )
        self.primary_outcome = "sideways_error_m"
        self.hidden_effects = (
            "A sideways command also turns the dog (left commands by 11 to 17 degrees in the lead's first run); trotting "
            "forward on the new heading adds sideways travel in the frame of the decision instant. The formula keeps "
            "the heading fixed and ignores the forward speed.",
            "Left and right are not mirror images (different steady sideways speeds and yaw coupling); the formula is "
            "mirror symmetric.",
            "With zero yaw commanded the dog drifts anticlockwise while trotting (Gate 1), so even a zero sideways "
            "command ends about 0.1 m to the left; the formula gives 0.",
            "The latency and response time come from Person A's standstill vel_y step traces, not from a trot; the "
            "first-order shape is an assumption.",
            "Gait oscillation: the trunk sways sideways and in yaw every step, so the heading read at the decision "
            "instant and the final position each carry a gait-phase offset.",
            "Payload, floor friction, slope, motor strength and command latency (world.hidden_effects) are not active "
            "in today's default Go2World; once they are, each changes the sideways response, and the formula ignores "
            "all of them.",
        )

        # ----- declared values of the world, appended to every sampled situation ----------------
        self._declared_row = encode_row(
            self.situation[2:],
            {"declared_payload_kg": declared_payload_kg, "declared_surface": declared_surface},
        )
        check_command_binding(self, command_schema)

    # ----- build-time helpers ------------------------------------------------------------------

    def _check_fields(self, forward_range: tuple[float, float]) -> None:
        """The three command fields are distinct, together form the whole verified schema, the two
        velocity fields are in m/s, and every value an episode can send is verified."""
        schema = self.command_schema
        fields = (self.side_field, self.forward_field, self.yaw_field)
        if len(set(fields)) != 3 or set(fields) != set(schema.fields):
            raise ConfigError(
                f"'{_DODGE_ACTION}.*_field' name {sorted(fields)}; every command must carry exactly the three distinct "
                f"verified fields {sorted(schema.fields)}"
            )
        for name in (self.side_field, self.forward_field):
            if schema.field(name).unit != "m/s":
                raise ConfigError(f"command field {name!r} has unit {schema.field(name).unit!r}; the spec needs m/s")
        for name in (self.side_field, self.yaw_field):
            field = schema.field(name)
            if not field.low <= 0.0 <= field.high:
                raise ConfigError(f"the nominal command sends {name}=0.0, outside its verified range [{field.low}, {field.high}]")
        forward = schema.field(self.forward_field)
        if forward_range[0] < forward.low or forward_range[1] > forward.high:
            raise ConfigError(
                f"'{ROOT}.situation_ranges.{_FORWARD_KEY}' {list(forward_range)} is sent as {self.forward_field!r} but "
                f"exceeds its verified range [{forward.low}, {forward.high}]"
            )

    def _whole_periods(self, duration_s: float, label: str) -> int:
        """Number of control periods in ``duration_s``; raises ConfigError unless it is a whole number >= 1."""
        periods = round(duration_s / self.control_period_s)
        if periods < 1 or abs(periods * self.control_period_s - duration_s) > _TIME_TOLERANCE_S:
            raise ConfigError(f"{label} must be a whole number (>= 1) of {self.control_period_s} s control periods")
        return int(periods)

    # ----- situations ----------------------------------------------------------------------------

    def sample_situations(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` situations (n, 4): forward speed and target independent and uniform over their
        declared ranges (one ``rng.random((n, 2))`` draw, row by row, so row ``i`` does not depend on
        ``n``), and the world's declared values appended to every row."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"n must be an int >= 0; got {n!r}")
        n = int(n)
        draws = rng.random((n, 2))
        forward, target = self.situation[0], self.situation[1]
        columns = [
            forward.low + draws[:, 0] * (forward.high - forward.low),
            target.low + draws[:, 1] * (target.high - target.low),
        ]
        return np.concatenate([np.column_stack(columns), np.repeat(self._declared_row[None, :], n, axis=0)], axis=1)

    # ----- physics guess --------------------------------------------------------------------------

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Outcome guess (n, 3) from the formula alone, in outcome order.

        sideways_error_m = sideways_travel(side, hold, episode_duration_s, start delay 0) - target;
        fall = 0; lateral_command_m = |side| x hold (exact). The forward speed and the declared
        values are not used by the formula.
        """
        situations = _rows("situations", situations, self.n_situation)
        actions = _rows("actions", actions, self.n_action)
        if situations.shape[0] != actions.shape[0]:
            raise ValueError(f"situations {situations.shape} and actions {actions.shape} must have the same rows")
        if not np.all(self.action_in_bounds(actions)):
            raise ValueError("actions outside the declared action bounds")
        side, hold = actions[:, 0], actions[:, 1]
        target = situations[:, self.situation_names.index(_TARGET_KEY)]
        travel = sideways_travel(
            side,
            hold,
            self.episode_duration_s,
            start_delay_s=0.0,
            assumed_latency_s=self.assumed_latency_s,
            response_time_s=self.response_time_s,
        )
        guess = np.empty((situations.shape[0], self.n_outcome), dtype=np.float64)
        guess[:, self.outcome_index("sideways_error_m")] = travel - target
        guess[:, self.outcome_index("fall")] = 0.0
        guess[:, self.outcome_index("lateral_command_m")] = np.abs(side) * hold
        return guess

    # ----- objective, loss, success -----------------------------------------------------------------

    @staticmethod
    def _quantile_weights(levels: np.ndarray) -> np.ndarray:
        """Probability mass each quantile level stands for: the interval between the midpoints to its
        neighbours, with 0 and 1 as the outer ends. The weights sum to 1."""
        edges = np.concatenate([[0.0], 0.5 * (levels[1:] + levels[:-1]), [1.0]])
        return np.diff(edges)

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        """Risk-aware utility (n,), higher is better (weights: ``dev_checks.sidestep_reach.objective_weights``)::

            - fall      * mean fall                                  (predicted fall probability)
            - abs_error * E|sideways_error_m|
            - spread    * predictive spread of sideways_error_m      (half width between engine.spread_levels)
            - motion    * mean lateral_command_m

        E|e| is the sum over the predictive quantiles of |quantile| x the probability mass the
        level stands for (:meth:`_quantile_weights`); for a point mass it is |e| up to float rounding.
        """
        if tuple(predicted.names) != self.outcome_names:
            raise ValueError(f"predicted outcomes {predicted.names} are not the {self.name} outcomes {self.outcome_names}")
        w = self.weights
        column = predicted.index("sideways_error_m")
        mass = self._quantile_weights(predicted.quantile_levels)
        expected_abs = np.abs(predicted.quantiles[:, column, :]) @ mass
        return (
            -w.fall * predicted.mean_of("fall")
            - w.abs_error * expected_abs
            - w.spread * predicted.spread_of("sideways_error_m")
            - w.motion * predicted.mean_of("lateral_command_m")
        )

    def loss(self, measured: np.ndarray) -> np.ndarray:
        """Per-episode loss (n,) of measured outcomes (n, 3): |sideways_error_m| in metres."""
        measured = _rows("measured", measured, self.n_outcome)
        return np.abs(measured[:, self.outcome_index("sideways_error_m")])

    def success(self, measured: np.ndarray) -> np.ndarray:
        """bool (n,): |sideways_error_m| <= ``success_tolerance_m`` and no fall."""
        measured = _rows("measured", measured, self.n_outcome)
        error = np.abs(measured[:, self.outcome_index("sideways_error_m")])
        return (error <= self.success_tolerance_m) & (measured[:, self.outcome_index("fall")] == 0.0)

    # ----- one real episode -----------------------------------------------------------------------

    def _command(self, forward: float, side: float) -> dict[str, float]:
        return {self.forward_field: float(forward), self.side_field: float(side), self.yaw_field: 0.0}

    def reset_situation(self, situation: Mapping[str, Any]) -> dict[str, Any]:
        """The mapping passed to ``world.reset`` for ``situation`` (all four keys): the forward speed
        only. The target is not the world's business; the declared values are reported by the world
        and checked in :meth:`episode`."""
        situation_vector = self.encode_situation(situation)
        return {_FORWARD_KEY: float(situation_vector[self.situation_names.index(_FORWARD_KEY)])}

    def commands(self, situation: Mapping[str, Any], action: Mapping[str, Any]) -> list[tuple[dict[str, float], float]]:
        """The command schedule of one episode: [(command, duration_s), ...], every duration a whole
        number of control periods, the durations summing to ``episode_duration_s``."""
        situation_vector = self.encode_situation(situation)
        side, hold = (float(value) for value in self.encode_action(action))
        forward = float(situation_vector[self.situation_names.index(_FORWARD_KEY)])
        try:
            hold_periods = self._whole_periods(hold, f"hold_duration_s={hold}")
        except ConfigError as error:
            raise ValueError(str(error)) from None
        tail_periods = self.episode_periods - hold_periods
        schedule = [(self._command(forward, side), hold_periods * self.control_period_s)]
        if tail_periods > 0:
            schedule.append((self._command(forward, 0.0), tail_periods * self.control_period_s))
        for command, _ in schedule:
            self.command_schema.check_command(command)
        return schedule

    def episode(
        self, world: World, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int
    ) -> Mapping[str, float]:
        """Run one attempt in ``world`` and return the three outcomes, in outcome order.

        Args:
            world:        Person A's Go2World (``reset`` / ``step`` / ``finish_episode``).
            situation:    the four situation keys; the forward speed is requested from the world,
                          the two declared keys must equal what the world declares, the target is
                          used only to compute the error.
            action:       the two action keys, inside the declared bounds; the hold must be a whole
                          number of control periods.
            attempt_seed: int >= 0 seeding this attempt's noise in the world.

        The reset request and the command schedule are :meth:`reset_situation` and :meth:`commands`,
        so a replay outside this method sends the world the same thing. The command binding,
        situation, action and every command are checked before the world is touched. The sideways
        travel is measured from the trunk position and heading in the
        observation ``world.reset`` returns and ``final_base_pos_m`` of ``world.finish_episode()``;
        the fall flag is the world's. Raises RuntimeError if the world reports ``sim_unstable`` or
        did not simulate exactly the scheduled whole control periods.
        """
        check_command_binding(self, self.command_schema)
        if isinstance(attempt_seed, bool) or not isinstance(attempt_seed, (int, np.integer)) or attempt_seed < 0:
            raise ValueError(f"attempt_seed must be an int >= 0; got {attempt_seed!r}")
        request = self.reset_situation(situation)
        schedule = self.commands(situation, action)
        side, hold = (float(value) for value in self.encode_action(action))
        target = float(self.encode_situation(situation)[self.situation_names.index(_TARGET_KEY)])

        observation = world.reset(request, int(attempt_seed))
        DodgeSpec._check_declared(observation, situation)
        start = np.asarray(observation["base_pos_m"], dtype=np.float64)
        heading = heading_rad(observation["base_quat_wxyz"])
        for command, duration in schedule:
            world.step(dict(command), float(duration))
        outcome = world.finish_episode()

        if not isinstance(outcome, Mapping):
            raise TypeError(f"world.finish_episode must return a mapping; got {type(outcome).__name__}")
        if int(outcome["sim_unstable"]) != 0:
            raise RuntimeError("the world reports sim_unstable=1 for this episode; its measurement is not usable")
        elapsed = float(outcome["elapsed_s"])
        if abs(elapsed - self.episode_periods * self.control_period_s) > _TIME_TOLERANCE_S:
            raise RuntimeError(
                f"the world simulated {elapsed} s after the decision instant; the schedule was "
                f"{self.episode_periods} control periods ({self.episode_periods * self.control_period_s} s)"
            )
        final = np.asarray(outcome["final_base_pos_m"], dtype=np.float64)
        dx, dy = float(final[0] - start[0]), float(final[1] - start[1])
        travel = -math.sin(heading) * dx + math.cos(heading) * dy  # + is to the dog's left at the decision instant
        measured = {
            "sideways_error_m": travel - target,
            "fall": float(outcome["fall"]),
            "lateral_command_m": abs(side) * hold,
        }
        self.encode_outcome(measured)  # finite, fall 0/1, bounds
        return {name: measured[name] for name in self.outcome_names}


# --------------------------------------------------------------------------------------
# Building the spec for one world
# --------------------------------------------------------------------------------------


def declared_values(world: Any) -> tuple[float, str]:
    """(declared_payload_kg, declared_surface) of ``world``.

    Reads the attributes ``world.declared_payload_kg`` and ``world.declared_surface`` when the
    world has them (agreed with Person A, A-4). Until they exist, reads them from the observation
    of ``world.reset()``, the interim agreed in the same exchange; that reset starts an episode
    that the next ``reset`` abandons without a trace.
    """
    payload = getattr(world, "declared_payload_kg", None)
    surface = getattr(world, "declared_surface", None)
    if payload is not None and surface is not None:
        return float(payload), str(surface)
    observation = world.reset()
    return float(observation["declared_payload_kg"]), str(observation["declared_surface"])


def for_world(config: Mapping[str, Any], world: Any) -> SidestepReachSpec:
    """The spec built with ``world``'s declared values and validated with the project's settings.

    Validation is :func:`learned_judgment.skills.base.validate_spec` with the settings
    ``load_skill`` uses (``world.hidden_effects`` names, ``engine.quantile_levels``,
    ``runtime.seed``, ``engine.spec_validation.*``) against ``CommandSchema.from_config(config)``.
    """
    command_schema = CommandSchema.from_config(config)
    payload, surface = declared_values(world)
    spec = SidestepReachSpec(config, command_schema, declared_payload_kg=payload, declared_surface=surface)
    validate_spec(spec, command_schema, **_validation_settings(config))
    return spec
