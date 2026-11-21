"""SkillSpec contract: what a skill declares, and all the generic engine may rely on.

A skill is one file under ``learned_judgment/skills/`` defining one :class:`SkillSpec`
subclass. The engine never branches on a skill name; everything skill-specific (schema,
physics guess, objective, episode procedure, primary loss) lives in the spec.

Array conventions are those of :mod:`learned_judgment.engine.types`:
situations (n, S), actions (n, A), outcomes (n, K), all float64, in the declared order.
A categorical value is the float index of its level; a binary outcome is 0.0 / 1.0.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

import numpy as np

from learned_judgment.engine.types import OutcomeDistribution

__all__ = [
    "ActionDim",
    "CommandField",
    "CommandSchema",
    "Outcome",
    "SkillSpec",
    "SkillSpecError",
    "UnverifiedCommandSchemaError",
    "Variable",
    "World",
    "check_command_binding",
    "command_binding_problems",
    "decode_row",
    "encode_row",
    "exposes_command_sequence",
    "validate_spec",
]

MAX_ACTION_DIMS = 3


class SkillSpecError(ValueError):
    """A SkillSpec violates the contract. The message lists every problem found."""


class UnverifiedCommandSchemaError(RuntimeError):
    """The walking policy's command schema is not verified in config.yaml.

    Raised instead of guessing command fields or ranges: a skill cannot be built until
    Person A has published the verified schema of the selected walking policy.
    """


# --------------------------------------------------------------------------------------
# Schema pieces
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Variable:
    """One observed situation feature.

    kind "float":       requires finite ``low < high`` (unit-bearing range of the feature).
    kind "categorical": requires ``levels`` (>= 1 unique strings); encoded as level index.
                        A single level is a declared feature that is constant for now.
    """

    name: str
    kind: Literal["float", "categorical"]
    unit: str
    description: str
    low: float | None = None
    high: float | None = None
    levels: tuple[str, ...] = ()

    def problems(self) -> list[str]:
        found: list[str] = []
        label = f"{type(self).__name__} {self.name!r}"
        if not self.name or not self.name.isidentifier():
            found.append(f"{label}: name must be a valid identifier")
        if not self.description:
            found.append(f"{label}: description is empty")
        if self.kind == "float":
            if self.low is None or self.high is None:
                found.append(f"{label}: float variables need low and high")
            elif not (math.isfinite(self.low) and math.isfinite(self.high) and self.low < self.high):
                found.append(f"{label}: needs finite low < high; got [{self.low}, {self.high}]")
            if self.levels:
                found.append(f"{label}: float variables must not declare levels")
        elif self.kind == "categorical":
            if len(self.levels) < 1 or len(set(self.levels)) != len(self.levels):
                found.append(f"{label}: categorical variables need >= 1 unique levels")
            if self.low is not None or self.high is not None:
                found.append(f"{label}: categorical variables must not declare low/high")
        else:
            found.append(f"{label}: unknown kind {self.kind!r}")
        return found

    @property
    def n_levels(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class ActionDim(Variable):
    """One dimension of the bounded candidate-action space.

    role "command": the value is sent to the walking policy through ``command_field``, which
                    must exist in the verified command schema, and [low, high] must lie
                    inside that field's verified range.
    role "timing":  the value says when / for how long a verified command is issued
                    (seconds); it has no command field of its own.

    A categorical command dimension must list ``command_values``: the value sent through
    ``command_field`` for each level, in level order, so that every level can be checked
    against the verified range.
    """

    role: Literal["command", "timing"] = "command"
    command_field: str | None = None
    command_values: tuple[float, ...] = ()

    def problems(self) -> list[str]:
        found = super().problems()
        label = f"ActionDim {self.name!r}"
        if self.role == "command" and not self.command_field:
            found.append(f"{label}: role 'command' needs a command_field")
        if self.role == "command" and self.kind == "categorical":
            values_ok = len(self.command_values) == len(self.levels) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in self.command_values
            )
            if not values_ok:
                found.append(f"{label}: a categorical command dimension needs one finite command value per level")
        elif self.command_values:
            found.append(f"{label}: command_values belong to categorical command dimensions only")
        if self.role == "timing":
            if self.command_field is not None:
                found.append(f"{label}: role 'timing' must not set command_field")
            if self.kind != "float" or self.unit != "s":
                found.append(f"{label}: role 'timing' must be a float in seconds")
            elif self.low is not None and self.low < 0.0:
                found.append(f"{label}: timing cannot be negative")
        if self.role not in ("command", "timing"):
            found.append(f"{label}: unknown role {self.role!r}")
        return found


@dataclass(frozen=True)
class Outcome:
    """One measured outcome, at a fixed position in the skill's outcome order.

    kind "regression": real valued; ``low`` / ``high`` are optional physical bounds used to
                       clip physics + residual (e.g. a duration cannot be negative).
    kind "binary":     measured as 0.0 / 1.0; predictions are probabilities clipped to [0, 1].
    """

    name: str
    kind: Literal["regression", "binary"]
    unit: str
    description: str
    low: float | None = None
    high: float | None = None

    def problems(self) -> list[str]:
        found: list[str] = []
        label = f"Outcome {self.name!r}"
        if not self.name or not self.name.isidentifier():
            found.append(f"{label}: name must be a valid identifier")
        if not self.description:
            found.append(f"{label}: description is empty")
        if self.kind == "binary":
            if self.low is not None or self.high is not None:
                found.append(f"{label}: binary outcomes are bounded to [0, 1] implicitly")
        elif self.kind == "regression":
            if self.low is not None and self.high is not None and not self.low < self.high:
                found.append(f"{label}: needs low < high; got [{self.low}, {self.high}]")
        else:
            found.append(f"{label}: unknown kind {self.kind!r}")
        return found

    @property
    def bounds(self) -> tuple[float, float]:
        if self.kind == "binary":
            return 0.0, 1.0
        return (
            -math.inf if self.low is None else float(self.low),
            math.inf if self.high is None else float(self.high),
        )


# --------------------------------------------------------------------------------------
# Verified command schema (published by Person A in config.yaml)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandField:
    """One high-level command field of the selected walking policy, as verified by Person A."""

    name: str
    unit: str
    low: float
    high: float
    description: str = ""


@dataclass(frozen=True)
class CommandSchema:
    """The verified high-level command interface of the selected walking policy.

    Read from ``robot.walking_policy`` in config.yaml. Expected layout (Person A owns the
    values; this is the shape Person B consumes)::

        robot:
          walking_policy:
            status: verified
            command_schema:
              update_period_s: <float>          # how often the policy accepts a new command
              fields:
                <field_name>: {unit: <str>, min: <float>, max: <float>, description: <str>}

    Nothing is defaulted: a missing or unverified schema raises
    :class:`UnverifiedCommandSchemaError`.
    """

    fields: Mapping[str, CommandField]
    update_period_s: float
    source: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CommandSchema":
        try:
            policy = config["robot"]["walking_policy"]
        except (KeyError, TypeError):
            raise UnverifiedCommandSchemaError("config has no robot.walking_policy section") from None
        status = policy.get("status")
        if status != "verified":
            raise UnverifiedCommandSchemaError(
                f"robot.walking_policy.status is {status!r}, not 'verified': Person A publishes the "
                "command schema with status 'verified' once every field and range is proven by a recorded run"
            )
        schema = policy.get("command_schema")
        if not isinstance(schema, Mapping) or not isinstance(schema.get("fields"), Mapping) or not schema["fields"]:
            raise UnverifiedCommandSchemaError("robot.walking_policy.command_schema.fields is missing or empty")
        def number(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

        period = schema.get("update_period_s")
        if not number(period) or not period > 0.0:
            raise UnverifiedCommandSchemaError("robot.walking_policy.command_schema.update_period_s must be a number > 0")
        fields: dict[str, CommandField] = {}
        for name, spec in schema["fields"].items():
            if not isinstance(name, str) or not name:
                raise UnverifiedCommandSchemaError(f"command field name {name!r} must be a non-empty string")
            if not isinstance(spec, Mapping):
                raise UnverifiedCommandSchemaError(f"command field {name!r} must be a mapping")
            missing = [key for key in ("unit", "min", "max") if spec.get(key) is None]
            if missing:
                raise UnverifiedCommandSchemaError(f"command field {name!r} is missing {missing}")
            if not isinstance(spec["unit"], str) or not spec["unit"]:
                raise UnverifiedCommandSchemaError(f"command field {name!r} needs a unit string")
            if not number(spec["min"]) or not number(spec["max"]):
                raise UnverifiedCommandSchemaError(f"command field {name!r} needs numeric, finite min and max")
            low, high = float(spec["min"]), float(spec["max"])
            if not low < high:
                raise UnverifiedCommandSchemaError(f"command field {name!r} needs min < max")
            fields[str(name)] = CommandField(
                name=str(name),
                unit=str(spec["unit"]),
                low=low,
                high=high,
                description=str(spec.get("description", "")),
            )
        return cls(fields=fields, update_period_s=float(period), source="config.yaml:robot.walking_policy")

    def field(self, name: str) -> CommandField:
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a verified command field; verified fields: {sorted(self.fields)}"
            ) from None

    def check_command(self, command: Mapping[str, float]) -> None:
        """Raise ValueError unless ``command`` uses only verified fields inside verified ranges."""
        for name, value in command.items():
            if name not in self.fields:
                raise ValueError(f"unsupported command field {name!r}; verified fields: {sorted(self.fields)}")
            spec = self.fields[name]
            if not (math.isfinite(value) and spec.low <= value <= spec.high):
                raise ValueError(
                    f"command {name}={value} {spec.unit} is outside the verified range [{spec.low}, {spec.high}]"
                )


# --------------------------------------------------------------------------------------
# Simulator interface consumed from Person A (typing only; no simulator lives here)
# --------------------------------------------------------------------------------------


class World(Protocol):
    """The public Go2World surface agreed with Person A. Person B imports the real class
    from ``learned_judgment.env``; this Protocol only types what a SkillSpec may call."""

    def reset(self, situation: Mapping[str, Any], attempt_seed: int) -> Mapping[str, Any]:
        """Start one attempt. The world's constructor seed fixes the hidden world for the whole
        run; ``attempt_seed`` seeds only this attempt's noise, so every method compared in a run
        faces the same hidden world and the same noise on the same episode."""
        ...

    def step(self, command: Mapping[str, float], duration_s: float) -> Mapping[str, Any]:
        ...

    def finish_episode(self) -> Mapping[str, Any]:
        ...


# --------------------------------------------------------------------------------------
# Row <-> vector codecs (strict: no missing keys, no extra keys)
# --------------------------------------------------------------------------------------


def _names(items: Sequence[Any]) -> tuple[str, ...]:
    return tuple(item.name for item in items)


def encode_row(variables: Sequence[Any], values: Mapping[str, Any], *, check_bounds: bool = True) -> np.ndarray:
    """Encode a name -> value mapping into a float64 vector in ``variables`` order.

    ``variables`` is a sequence of :class:`Variable` / :class:`ActionDim` / :class:`Outcome`.
    The mapping must contain exactly the declared names: an extra key is rejected, so a
    hidden simulator value cannot ride along into a feature or outcome vector.
    """
    expected = _names(variables)
    given = tuple(values.keys())
    if set(given) != set(expected) or len(given) != len(expected):
        missing = sorted(set(expected) - set(given))
        extra = sorted(set(given) - set(expected))
        raise ValueError(f"row keys do not match the schema; missing={missing} extra={extra}")
    vector = np.empty(len(expected), dtype=np.float64)
    for position, variable in enumerate(variables):
        raw = values[variable.name]
        kind = variable.kind
        if kind == "categorical":
            if raw not in variable.levels:
                raise ValueError(f"{variable.name}={raw!r} is not one of {variable.levels}")
            vector[position] = float(variable.levels.index(raw))
            continue
        if isinstance(raw, (bool, np.bool_)):
            raw = float(raw)
        if not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ValueError(f"{variable.name}={raw!r} is not a number")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{variable.name}={raw!r} is not finite")
        if kind == "binary" and number not in (0.0, 1.0):
            raise ValueError(f"{variable.name}={raw!r} must be 0 or 1")
        if check_bounds and kind == "float" and not (variable.low <= number <= variable.high):
            raise ValueError(f"{variable.name}={number} is outside [{variable.low}, {variable.high}]")
        vector[position] = number
    return vector


def decode_row(variables: Sequence[Any], vector: np.ndarray) -> dict[str, Any]:
    """Inverse of :func:`encode_row`: a plain dict of Python floats / level strings."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (len(variables),):
        raise ValueError(f"vector must have shape ({len(variables)},); got {vector.shape}")
    row: dict[str, Any] = {}
    for variable, value in zip(variables, vector):
        if variable.kind == "categorical":
            number = float(value)
            if not number.is_integer() or not 0 <= int(number) < len(variable.levels):
                raise ValueError(f"{variable.name}: {value} is not a valid level index")
            row[variable.name] = variable.levels[int(number)]
        else:
            row[variable.name] = float(value)
    return row


# --------------------------------------------------------------------------------------
# SkillSpec
# --------------------------------------------------------------------------------------


class SkillSpec(ABC):
    """Everything the generic engine needs to learn one high-level decision.

    Subclasses set the declarative attributes (usually in ``__init__`` from config and the
    verified :class:`CommandSchema`) and implement the abstract methods. All array methods
    are vectorized over rows so thousands of dreamed candidates cost one call.

    Declarative attributes:
        name, description
        situation:       observed features only -- never a hidden simulator value
        actions:         1..3 bounded dimensions bound to verified commands
        outcomes:        measured outcomes in a fixed order (the residual order)
        primary_outcome: the outcome whose predictive spread is the decision's confidence
        hidden_effects:  documentation of the hidden effects the residual has to absorb
    """

    name: str
    description: str
    situation: tuple[Variable, ...]
    actions: tuple[ActionDim, ...]
    outcomes: tuple[Outcome, ...]
    primary_outcome: str
    hidden_effects: tuple[str, ...]

    # ----- to implement per skill --------------------------------------------------

    @abstractmethod
    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict the outcomes of ``actions`` in ``situations`` from the formula alone.

        Args:    situations (n, S), actions (n, A).
        Returns: (n, K) in outcome order. A binary outcome's guess is a probability in [0, 1].
        Must use only its arguments and declared constants -- never the simulator.
        """

    @abstractmethod
    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        """Risk-aware utility of each row of ``predicted``; (n,), higher is better.

        Must account for predictive uncertainty. The engine calls it with a point mass for
        physics-only scoring and for scoring a measured outcome (regret versus the oracle).
        """

    @abstractmethod
    def episode(
        self, world: World, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int
    ) -> Mapping[str, float]:
        """Run one actual episode and return the measured outcome, keyed by outcome name.

        Uses only ``world.reset(situation, attempt_seed)`` / ``world.step`` /
        ``world.finish_episode`` with verified commands. Every returned value is measured by
        the simulator; nothing is derived from the intended action.
        """

    def reset_situation(self, situation: Mapping[str, Any]) -> dict[str, Any]:
        """The mapping this skill passes to ``world.reset`` for ``situation`` (all situation keys).

        Optional, together with :meth:`commands`. A spec that implements the pair must run
        :meth:`episode` from them, so an episode replayed outside ``episode`` (the dashboard's
        recorder, the live viewer) sends the world exactly what a measured episode sends.
        """
        raise NotImplementedError(f"skill {self.name!r} does not expose its reset request")

    def commands(
        self, situation: Mapping[str, Any], action: Mapping[str, Any]
    ) -> list[tuple[dict[str, float], float]]:
        """The command sequence of one attempt: ``[(command, duration_s), ...]`` in the order sent.

        Every command names every field of the verified command schema and stays inside the
        verified ranges; every duration is in seconds and > 0. Optional, see
        :meth:`reset_situation`. :func:`validate_spec` checks the sequence when it is implemented.
        """
        raise NotImplementedError(f"skill {self.name!r} does not expose its command sequence")

    @abstractmethod
    def loss(self, measured: np.ndarray) -> np.ndarray:
        """Preregistered primary per-episode loss of measured outcomes (n, K) -> (n,); lower is better."""

    @abstractmethod
    def success(self, measured: np.ndarray) -> np.ndarray:
        """Skill-specific hit / tolerance of measured outcomes (n, K) -> bool (n,)."""

    # ----- generic helpers -----------------------------------------------------------

    @property
    def n_situation(self) -> int:
        return len(self.situation)

    @property
    def n_action(self) -> int:
        return len(self.actions)

    @property
    def n_outcome(self) -> int:
        return len(self.outcomes)

    @property
    def situation_names(self) -> tuple[str, ...]:
        return _names(self.situation)

    @property
    def action_names(self) -> tuple[str, ...]:
        return _names(self.actions)

    @property
    def outcome_names(self) -> tuple[str, ...]:
        return _names(self.outcomes)

    @property
    def outcome_type(self) -> Literal["regression", "classification", "both"]:
        kinds = {outcome.kind for outcome in self.outcomes}
        if kinds == {"regression"}:
            return "regression"
        if kinds == {"binary"}:
            return "classification"
        return "both"

    @property
    def outcome_low(self) -> np.ndarray:
        return np.array([outcome.bounds[0] for outcome in self.outcomes], dtype=np.float64)

    @property
    def outcome_high(self) -> np.ndarray:
        return np.array([outcome.bounds[1] for outcome in self.outcomes], dtype=np.float64)

    def outcome_index(self, name: str) -> int:
        try:
            return self.outcome_names.index(name)
        except ValueError:
            raise KeyError(f"unknown outcome {name!r}; known outcomes: {self.outcome_names}") from None

    def encode_situation(self, situation: Mapping[str, Any]) -> np.ndarray:
        return encode_row(self.situation, situation)

    def encode_action(self, action: Mapping[str, Any]) -> np.ndarray:
        return encode_row(self.actions, action)

    def encode_outcome(self, outcome: Mapping[str, Any]) -> np.ndarray:
        return encode_row(self.outcomes, outcome)

    def decode_situation(self, vector: np.ndarray) -> dict[str, Any]:
        return decode_row(self.situation, vector)

    def decode_action(self, vector: np.ndarray) -> dict[str, Any]:
        return decode_row(self.actions, vector)

    def decode_outcome(self, vector: np.ndarray) -> dict[str, Any]:
        return decode_row(self.outcomes, vector)

    def sample_situations(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` situations (n, S). Default: independent uniform over each declared range.

        A skill overrides this when its situations follow a protocol published in config.yaml
        (for example the reproducible ball-throw distribution).

        The draws are taken row by row, so the first rows of a larger sample equal a smaller
        sample from the same stream: situation ``i`` does not depend on how many are drawn.
        """
        draws = rng.random((n, len(self.situation)))
        columns = []
        for index, variable in enumerate(self.situation):
            uniform = draws[:, index]
            if variable.kind == "categorical":
                n_levels = len(variable.levels)
                columns.append(np.minimum(np.floor(uniform * n_levels), n_levels - 1.0))
            else:
                columns.append(variable.low + uniform * (variable.high - variable.low))
        return np.column_stack(columns) if columns else np.empty((n, 0))

    def candidate_actions(self, situation: np.ndarray, points_per_dim: Sequence[int]) -> np.ndarray:
        """Deterministic candidate grid (C, A) inside the action bounds for one situation (S,).

        ``points_per_dim`` gives the grid resolution of each float dimension (from
        config.yaml); a categorical dimension always contributes all of its levels. A skill
        may override this to make the grid depend on the situation, but every candidate must
        stay inside the declared bounds.
        """
        if len(points_per_dim) != self.n_action:
            raise ValueError(f"points_per_dim needs {self.n_action} entries; got {list(points_per_dim)}")
        axes = []
        for dim, points in zip(self.actions, points_per_dim):
            if dim.kind == "categorical":
                axes.append(np.arange(len(dim.levels), dtype=np.float64))
            else:
                if int(points) < 2:
                    raise ValueError(f"action dim {dim.name!r} needs at least 2 grid points; got {points}")
                axes.append(np.linspace(dim.low, dim.high, int(points)))
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([axis.reshape(-1) for axis in mesh])

    def action_in_bounds(self, actions: np.ndarray) -> np.ndarray:
        """bool (n,): every dimension of each action (n, A) is inside its declared bounds.

        The comparison is exact (no tolerance): grid candidates include the bounds exactly,
        and anything that produces actions by arithmetic must clip to the bounds itself.
        """
        actions = np.atleast_2d(np.asarray(actions, dtype=np.float64))
        ok = np.ones(actions.shape[0], dtype=bool)
        for column, dim in enumerate(self.actions):
            values = actions[:, column]
            if dim.kind == "categorical":
                ok &= np.isin(values, np.arange(len(dim.levels), dtype=np.float64))
            else:
                ok &= (values >= dim.low) & (values <= dim.high)
        return ok


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def command_binding_problems(spec: SkillSpec, command_schema: CommandSchema) -> list[str]:
    """Every way ``spec``'s action space fails to stay inside the verified command schema.

    Cheap and free of side effects, so it runs in ``validate_spec`` and again immediately
    before anything steps the simulator.
    """
    problems: list[str] = []
    if not any(dim.role == "command" for dim in spec.actions):
        problems.append("at least one action dimension must drive a verified command field")
    for dim in spec.actions:
        if dim.role != "command" or not dim.command_field:
            continue
        if dim.command_field not in command_schema.fields:
            problems.append(
                f"action {dim.name!r} uses command field {dim.command_field!r}, which is not in the "
                f"verified schema {sorted(command_schema.fields)}"
            )
            continue
        verified = command_schema.fields[dim.command_field]
        if dim.unit != verified.unit:
            problems.append(f"action {dim.name!r} unit {dim.unit!r} differs from the verified unit {verified.unit!r}")
        if dim.kind == "float" and dim.low is not None and dim.high is not None:
            if dim.low < verified.low or dim.high > verified.high:
                problems.append(
                    f"action {dim.name!r} range [{dim.low}, {dim.high}] exceeds the verified range "
                    f"[{verified.low}, {verified.high}] of {dim.command_field!r}"
                )
        if dim.kind == "categorical":
            outside = [value for value in dim.command_values if not verified.low <= value <= verified.high]
            if len(dim.command_values) != len(dim.levels) or outside:
                problems.append(
                    f"action {dim.name!r} needs one command value per level inside the verified range "
                    f"[{verified.low}, {verified.high}] of {dim.command_field!r}; got {list(dim.command_values)}"
                )
    return problems


def check_command_binding(spec: SkillSpec, command_schema: CommandSchema) -> None:
    """Raise :class:`SkillSpecError` unless every command dimension of ``spec`` is bound to a
    verified command field and stays inside its verified range. Call it before any episode."""
    problems = command_binding_problems(spec, command_schema)
    if problems:
        raise SkillSpecError("; ".join(problems))


def validate_spec(
    spec: SkillSpec,
    command_schema: CommandSchema,
    *,
    hidden_value_names: Iterable[str],
    quantile_levels: Sequence[float],
    probe_seed: int,
    probe_situations: int,
    probe_points_per_dim: int,
) -> None:
    """Check ``spec`` against the SkillSpec contract; raise :class:`SkillSpecError` listing
    every violation. Passing means the engine can use the spec without skill-specific code.

    No argument has a default: the caller passes the hidden-value names, quantile levels,
    seed and probe sizes it read from config.yaml, so no tunable lives in this module.

    Structural checks: names, 1..3 bounded action dimensions, every command dimension bound
    to a verified command field inside its verified range, no hidden value among the
    situation features, a declared primary outcome, documented hidden effects.

    Behavioural probe (pure functions only, no simulator): ``physics_guess``, ``objective``,
    ``loss`` and ``success`` are called on sampled situations x a small candidate grid and
    must return finite arrays of the documented shapes.
    """
    problems: list[str] = []

    for attribute in ("name", "description"):
        if not isinstance(getattr(spec, attribute, None), str) or not getattr(spec, attribute):
            problems.append(f"spec.{attribute} must be a non-empty string")
    for attribute, kind in (("situation", Variable), ("actions", ActionDim), ("outcomes", Outcome)):
        items = getattr(spec, attribute, None)
        if not isinstance(items, tuple) or not items or not all(isinstance(item, kind) for item in items):
            problems.append(f"spec.{attribute} must be a non-empty tuple of {kind.__name__}")
    if problems:
        raise SkillSpecError("; ".join(problems))

    for item in (*spec.situation, *spec.actions, *spec.outcomes):
        problems.extend(item.problems())

    all_names = [*spec.situation_names, *spec.action_names, *spec.outcome_names]
    duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicates:
        problems.append(f"names must be unique across situation, actions and outcomes; duplicated: {duplicates}")

    if not 1 <= spec.n_action <= MAX_ACTION_DIMS:
        problems.append(f"the action space must have 1..{MAX_ACTION_DIMS} dimensions; got {spec.n_action}")
    problems.extend(command_binding_problems(spec, command_schema))

    hidden = set(hidden_value_names)
    leaked = sorted(hidden.intersection(spec.situation_names))
    if leaked:
        problems.append(f"situation features expose hidden simulator values: {leaked}")

    if getattr(spec, "primary_outcome", None) not in spec.outcome_names:
        problems.append(f"primary_outcome {getattr(spec, 'primary_outcome', None)!r} is not one of the outcomes")
    hidden_effects = getattr(spec, "hidden_effects", None)
    if not isinstance(hidden_effects, tuple) or not hidden_effects or not all(hidden_effects):
        problems.append("hidden_effects must document at least one hidden effect")

    if problems:
        raise SkillSpecError("; ".join(problems))

    # Behavioural probe on pure functions.
    rng = np.random.default_rng(probe_seed)
    situations = np.asarray(spec.sample_situations(rng, probe_situations), dtype=np.float64)
    if situations.shape != (probe_situations, spec.n_situation) or not np.all(np.isfinite(situations)):
        raise SkillSpecError(
            f"sample_situations must return finite (n, {spec.n_situation}); got {situations.shape}"
        )
    for column, variable in enumerate(spec.situation):
        values = situations[:, column]
        if variable.kind == "float" and (values.min() < variable.low or values.max() > variable.high):
            problems.append(f"sample_situations leaves the declared range of {variable.name!r}")
        if variable.kind == "categorical" and not np.all(np.isin(values, np.arange(variable.n_levels))):
            problems.append(f"sample_situations returns invalid level indices for {variable.name!r}")

    situation_rows, action_rows = [], []
    for situation in situations:
        grid = np.asarray(
            spec.candidate_actions(situation, [probe_points_per_dim] * spec.n_action), dtype=np.float64
        )
        if grid.ndim != 2 or grid.shape[1] != spec.n_action or grid.shape[0] < 1:
            raise SkillSpecError(f"candidate_actions must return (C, {spec.n_action}); got {grid.shape}")
        if not np.all(spec.action_in_bounds(grid)):
            problems.append("candidate_actions returns actions outside the declared bounds")
        situation_rows.append(np.repeat(situation[None, :], grid.shape[0], axis=0))
        action_rows.append(grid)
    batch_situations = np.concatenate(situation_rows)
    batch_actions = np.concatenate(action_rows)
    n = batch_situations.shape[0]

    physics = np.asarray(spec.physics_guess(batch_situations, batch_actions), dtype=np.float64)
    if physics.shape != (n, spec.n_outcome):
        raise SkillSpecError(f"physics_guess must return (n, {spec.n_outcome}); got {physics.shape}")
    if not np.all(np.isfinite(physics)):
        problems.append("physics_guess returns non-finite values inside the declared ranges")
    for column, outcome in enumerate(spec.outcomes):
        low, high = outcome.bounds
        if np.any(physics[:, column] < low) or np.any(physics[:, column] > high):
            problems.append(f"physics_guess for {outcome.name!r} leaves its bounds [{low}, {high}]")

    point = OutcomeDistribution.point(spec.outcome_names, physics, np.asarray(quantile_levels, dtype=np.float64))
    for label, function, dtype_kind in (
        ("objective", lambda: spec.objective(point), "f"),
        ("loss", lambda: spec.loss(physics), "f"),
        ("success", lambda: spec.success(physics), "b"),
    ):
        values = np.asarray(function())
        if values.shape != (n,):
            problems.append(f"{label} must return shape (n,)={(n,)}; got {values.shape}")
        elif dtype_kind == "b" and values.dtype != np.bool_:
            problems.append(f"{label} must return a boolean array; got {values.dtype}")
        elif dtype_kind == "f" and not np.all(np.isfinite(values.astype(np.float64))):
            problems.append(f"{label} returns non-finite values")

    problems.extend(_command_sequence_problems(spec, command_schema, batch_situations, batch_actions))

    if problems:
        raise SkillSpecError("; ".join(problems))


def exposes_command_sequence(spec: SkillSpec) -> bool:
    """True when ``spec`` implements both :meth:`SkillSpec.reset_situation` and :meth:`SkillSpec.commands`."""
    kind = type(spec)
    return kind.reset_situation is not SkillSpec.reset_situation and kind.commands is not SkillSpec.commands


def _command_sequence_problems(
    spec: SkillSpec, command_schema: CommandSchema, situations: np.ndarray, actions: np.ndarray
) -> list[str]:
    """Problems of the optional reset request and command sequence on the probe rows (no simulator).

    A spec that implements only one of the two methods is a problem: replay needs both.
    """
    kind = type(spec)
    has_reset = kind.reset_situation is not SkillSpec.reset_situation
    has_commands = kind.commands is not SkillSpec.commands
    if not has_reset and not has_commands:
        return []
    if has_reset != has_commands:
        return ["reset_situation and commands must be implemented together"]
    fields = set(command_schema.fields)
    problems: list[str] = []
    for situation_row, action_row in zip(situations, actions):
        situation, action = spec.decode_situation(situation_row), spec.decode_action(action_row)
        where = f"commands({situation}, {action})"
        request = spec.reset_situation(situation)
        if not isinstance(request, Mapping):
            problems.append(f"reset_situation must return a mapping; got {type(request).__name__}")
            break
        sequence = spec.commands(situation, action)
        if not isinstance(sequence, list) or not sequence:
            problems.append(f"{where} must return a non-empty list of (command, duration_s)")
            break
        for step in sequence:
            if not isinstance(step, tuple) or len(step) != 2 or not isinstance(step[0], Mapping):
                problems.append(f"{where}: every step must be a (command mapping, duration_s) pair; got {step!r}")
                break
            command, duration = step
            if set(command) != fields:
                problems.append(f"{where}: a command must name exactly the verified fields {sorted(fields)}; got {sorted(command)}")
                break
            try:
                command_schema.check_command(command)
            except (TypeError, ValueError) as error:
                problems.append(f"{where}: {error}")
                break
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0.0 < duration < math.inf:
                problems.append(f"{where}: duration_s must be a finite number > 0; got {duration!r}")
                break
        if problems:
            break
    return problems
