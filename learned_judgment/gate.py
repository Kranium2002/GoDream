"""Command gate between the spiking student's readout and the robot (pure logic, no I/O).

Lane contract (Person B, DODGE from the head camera)
----------------------------------------------------
For every camera frame the frozen spiking student gives a readout: a threat confidence in
[0, 1] and one score per action, in the readout order of config.yaml
skills.dodge_vision.actions (NO_OP, DODGE_LEFT, DODGE_RIGHT, RETREAT). Internal spikes never
reach this module; only the readout does. :class:`ThreatGate` turns the stream of readouts into
the action emitted for each frame and stays quiet unless the threat is clear:

    vote      a frame votes for its best active action (the highest-scored action other than
              the zero command, the first in readout order on a tie) only when
              threat confidence >= threat threshold, that action's score >= action threshold,
              and its score minus the highest other score (the zero command included)
              >= margin threshold; otherwise, or when the readout is unusable, it casts no vote
    issue     an active command is issued when the same action is voted in at least
              votes_needed of the last of_last frames and no cooldown is running
    hold      an issued command is emitted for hold_s seconds
    cooldown  no new active command is issued for cooldown_s seconds after the last one
              (cooldown_s >= hold_s); a frame that passes the gate then is a cooldown conflict
    quiet     otherwise the gate emits the zero command (no_op), or UNCERTAIN when the readout
              is unusable or the threat confidence is at or above uncertain_floor; both send the
              zero command (config.yaml inference.gate.uncertain_floor)

Thresholds are the frozen calibration saved with the exported student; votes_needed, of_last,
cooldown_s and uncertain_floor come from config.yaml inference.gate, hold_s from
skills.dodge_vision.hold_s. Scores are used as given (not renormalised). Time is only the
caller's t_s in seconds, never decreasing within a stream (no wall clock, no randomness); a new
stream starts with :meth:`ThreatGate.reset`.
"""

from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from learned_judgment.config import ConfigError, require

__all__ = ["UNCERTAIN", "GateOutput", "GateSettings", "ThreatGate"]

UNCERTAIN = "UNCERTAIN"

_THRESHOLD_KEYS = ("threat", "action", "margin")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _finite(value: Any) -> float | None:
    """``float(value)`` for a finite real number (bool excluded), else None."""
    if not _is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class GateSettings:
    """Settings of one :class:`ThreatGate`.

    action_names      readout order of the action scores (keys of skills.dodge_vision.actions)
    no_op             name of the zero command in action_names
    threat_threshold  minimum threat confidence for a vote, in [0, 1]
    action_threshold  minimum score of the best active action for a vote, in [0, 1]
    margin_threshold  minimum of that score minus the highest other score for a vote, in [0, 1]
    votes_needed      votes for the same action needed in the window, 1 <= votes_needed <= of_last
    of_last           outputs in the vote window
    cooldown_s        seconds after an issued command in which no other is issued, >= hold_s
    hold_s            seconds an issued command is emitted, finite and > 0
    uncertain_floor   threat confidence at or above which an output without a passed gate is
                      UNCERTAIN rather than no_op, in [0, 1]

    Numbers are stored as float (thresholds, floor, seconds) and int (vote counts).

    Raises:
        TypeError for a value of the wrong type (action_names not a tuple of str, a vote count
        not an int, another field not a real number); ValueError for a value out of range.
    """

    action_names: tuple[str, ...]
    no_op: str
    threat_threshold: float
    action_threshold: float
    margin_threshold: float
    votes_needed: int
    of_last: int
    cooldown_s: float
    hold_s: float
    uncertain_floor: float

    def __post_init__(self) -> None:
        names = self.action_names
        if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
            raise TypeError(f"action_names must be a tuple of str; got {names!r}")
        if len(names) < 2:
            raise ValueError(f"action_names needs at least two actions; got {names!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"action_names must be distinct; got {names!r}")
        if UNCERTAIN in names:
            raise ValueError(f"no action may be named {UNCERTAIN!r}; got {names!r}")
        if self.no_op not in names:
            raise ValueError(f"no_op {self.no_op!r} is not in action_names {names!r}")
        for field in ("threat_threshold", "action_threshold", "margin_threshold", "uncertain_floor"):
            value = self._real(field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]; got {value}")
        for field in ("votes_needed", "of_last"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{field} must be an int; got {value!r}")
            object.__setattr__(self, field, int(value))
        if not 1 <= self.votes_needed <= self.of_last:
            raise ValueError(
                f"need 1 <= votes_needed <= of_last; got votes_needed {self.votes_needed}, of_last {self.of_last}"
            )
        hold_s = self._real("hold_s")
        cooldown_s = self._real("cooldown_s")
        if not hold_s > 0.0:
            raise ValueError(f"hold_s must be > 0; got {hold_s}")
        if not cooldown_s >= hold_s:
            raise ValueError(f"cooldown_s must be >= hold_s; got cooldown_s {cooldown_s}, hold_s {hold_s}")

    def _real(self, field: str) -> float:
        value = getattr(self, field)
        if not _is_number(value):
            raise TypeError(f"{field} must be a real number; got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field} must be finite; got {number}")
        object.__setattr__(self, field, number)
        return number

    @classmethod
    def from_config(cls, config: Mapping[str, Any], thresholds: Mapping[str, float]) -> GateSettings:
        """Settings from config.yaml and the frozen threshold calibration.

        Args:
            config:     the loaded config.yaml (learned_judgment.config.load_config).
            thresholds: exactly the keys "threat", "action", "margin": the thresholds chosen on
                        simulation dev trajectories and saved with the exported student.

        action_names are the keys of skills.dodge_vision.actions in config order; no_op is the
        one action whose command fields are all 0.0; votes_needed, of_last, cooldown_s and
        uncertain_floor come from inference.gate, hold_s from skills.dodge_vision.hold_s.

        Raises:
            ConfigError for a missing or null key, or skills.dodge_vision.actions that is not a
            non-empty mapping of non-empty command mappings of numbers. TypeError when
            thresholds is not a mapping. ValueError when thresholds does not have exactly the
            three keys, or unless exactly one action has every command field 0.0. Whatever
            GateSettings validation raises.
        """
        if not isinstance(thresholds, Mapping):
            raise TypeError(f"thresholds must be a mapping; got {type(thresholds).__name__}")
        missing = [key for key in _THRESHOLD_KEYS if key not in thresholds]
        extra = [key for key in thresholds if key not in _THRESHOLD_KEYS]
        if missing or extra:
            raise ValueError(
                f"thresholds must have exactly the keys {_THRESHOLD_KEYS}; missing {missing}, extra {extra}"
            )
        actions = require(config, "skills.dodge_vision.actions")
        if not isinstance(actions, Mapping) or not actions:
            raise ConfigError(f"config.yaml key 'skills.dodge_vision.actions' must be a non-empty mapping; got {actions!r}")
        zero: list[str] = []
        for name, command in actions.items():
            if not isinstance(command, Mapping) or not command or not all(_is_number(v) for v in command.values()):
                raise ConfigError(
                    f"config.yaml key 'skills.dodge_vision.actions.{name}' must be a non-empty mapping of numbers; "
                    f"got {command!r}"
                )
            if all(float(v) == 0.0 for v in command.values()):
                zero.append(name)
        if len(zero) != 1:
            raise ValueError(
                f"exactly one action of skills.dodge_vision.actions must have every command field 0.0; got {zero!r}"
            )
        return cls(
            action_names=tuple(actions),
            no_op=zero[0],
            threat_threshold=thresholds["threat"],
            action_threshold=thresholds["action"],
            margin_threshold=thresholds["margin"],
            votes_needed=require(config, "inference.gate.votes_needed"),
            of_last=require(config, "inference.gate.of_last"),
            cooldown_s=require(config, "inference.gate.cooldown_s"),
            hold_s=require(config, "skills.dodge_vision.hold_s"),
            uncertain_floor=require(config, "inference.gate.uncertain_floor"),
        )


@dataclass(frozen=True)
class GateOutput:
    """What :meth:`ThreatGate.update` decided for one readout.

    action               emitted now: no_op, UNCERTAIN, or an active action name (while issued or held)
    issued               True only on the output that starts a new active command
    holding              True while an issued command is held (including the output that issued it)
    valid                False when the readout was not a usable set of finite numbers
    threat_confidence    the readout's threat confidence (0.0 when not valid)
    best_action          highest-scored active action (no_op when not valid)
    action_confidence    its score (0.0 when not valid)
    margin               its score minus the highest other score (0.0 when not valid)
    threat_ok, action_ok, margin_ok   the three threshold tests (all False when not valid)
    votes                entries of the vote window equal to this output's vote (0 when it casts none)
    passed               it votes and votes >= votes_needed
    cooldown_remaining_s seconds until another active command may be issued, >= 0
    hold_remaining_s     seconds the issued command is still held, >= 0
    cooldown_conflict    passed, but a new command was refused by the hold or the cooldown
    """

    action: str
    issued: bool
    holding: bool
    valid: bool
    threat_confidence: float
    best_action: str
    action_confidence: float
    margin: float
    threat_ok: bool
    action_ok: bool
    margin_ok: bool
    votes: int
    passed: bool
    cooldown_remaining_s: float
    hold_remaining_s: float
    cooldown_conflict: bool


class ThreatGate:
    """The gate over one stream of readouts (see the module docstring for the rules)."""

    def __init__(self, settings: GateSettings) -> None:
        if not isinstance(settings, GateSettings):
            raise TypeError(f"settings must be a GateSettings; got {type(settings).__name__}")
        self.settings = settings
        self._active = tuple(i for i, name in enumerate(settings.action_names) if name != settings.no_op)
        self.reset()

    def reset(self) -> None:
        """Forget the vote window, the held command, the cooldown and the last time."""
        self._history: deque[str | None] = deque(maxlen=self.settings.of_last)
        self._held: str | None = None
        self._hold_until: float | None = None
        self._cooldown_until: float | None = None
        self._last_t: float | None = None

    def _scores(self, action_scores: Any) -> list[float] | None:
        """The scores as floats, or None unless they are len(action_names) finite numbers >= 0."""
        if isinstance(action_scores, np.ndarray):
            if action_scores.ndim != 1:
                return None
        elif isinstance(action_scores, (str, bytes)) or not isinstance(action_scores, Sequence):
            return None
        if len(action_scores) != len(self.settings.action_names):
            return None
        scores: list[float] = []
        for value in action_scores:
            score = _finite(value)
            if score is None or score < 0.0:
                return None
            scores.append(score)
        return scores

    def update(self, threat_confidence: float, action_scores: Sequence[float] | np.ndarray, t_s: float) -> GateOutput:
        """Gate one readout.

        Args:
            threat_confidence: the readout's threat confidence, a finite number in [0, 1].
            action_scores:     one score per action in settings.action_names order (sequence or
                               1-D array), each finite and >= 0; used as given, not renormalised.
            t_s:               time of the readout in seconds, finite and not less than the
                               previous update's t_s.

        An unusable readout (threat or scores not as above) does not raise: the output has
        valid False, casts no vote, and emits the held action or UNCERTAIN.

        Raises:
            ValueError when t_s is not a finite number or is less than the previous update's
            t_s (a new stream must call :meth:`reset` first). The gate is unchanged then.
        """
        s = self.settings
        t = _finite(t_s)
        if t is None:
            raise ValueError(f"t_s must be a finite number; got {t_s!r}")
        if self._last_t is not None and t < self._last_t:
            raise ValueError(f"t_s {t} is less than the previous update's {self._last_t}; call reset() for a new stream")
        self._last_t = t

        threat = _finite(threat_confidence)
        scores = self._scores(action_scores)
        valid = threat is not None and 0.0 <= threat <= 1.0 and scores is not None
        if valid:
            best = max(self._active, key=lambda i: scores[i])  # first in readout order on a tie
            best_action = s.action_names[best]
            confidence = scores[best]
            margin = confidence - max(score for i, score in enumerate(scores) if i != best)
            threat_ok = threat >= s.threat_threshold
            action_ok = confidence >= s.action_threshold
            margin_ok = margin >= s.margin_threshold
            vote = best_action if threat_ok and action_ok and margin_ok else None
        else:
            threat = confidence = margin = 0.0
            best_action = s.no_op
            threat_ok = action_ok = margin_ok = False
            vote = None

        self._history.append(vote)
        votes = sum(1 for entry in self._history if entry == vote) if vote is not None else 0
        passed = vote is not None and votes >= s.votes_needed

        holding_now = self._hold_until is not None and t < self._hold_until
        in_cooldown = self._cooldown_until is not None and t < self._cooldown_until
        issued = cooldown_conflict = False
        if passed and not in_cooldown:
            self._held = vote
            self._hold_until = t + s.hold_s
            self._cooldown_until = t + s.cooldown_s
            action, issued, holding = vote, True, True
        elif passed:
            cooldown_conflict = True
            holding = holding_now
            action = self._held if holding_now else UNCERTAIN
        else:
            holding = holding_now
            if holding_now:
                action = self._held
            elif not valid or threat >= s.uncertain_floor:
                action = UNCERTAIN
            else:
                action = s.no_op

        cooldown_remaining = 0.0 if self._cooldown_until is None else max(0.0, self._cooldown_until - t)
        hold_remaining = 0.0 if self._hold_until is None else max(0.0, self._hold_until - t)
        return GateOutput(
            action=action,
            issued=issued,
            holding=holding,
            valid=valid,
            threat_confidence=threat,
            best_action=best_action,
            action_confidence=confidence,
            margin=margin,
            threat_ok=threat_ok,
            action_ok=action_ok,
            margin_ok=margin_ok,
            votes=votes,
            passed=passed,
            cooldown_remaining_s=cooldown_remaining,
            hold_remaining_s=hold_remaining,
            cooldown_conflict=cooldown_conflict,
        )
