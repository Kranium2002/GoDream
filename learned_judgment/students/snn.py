"""Per-skill spiking student, trained inside the dream.

Lane contract (Person B, per-skill spiking student)
---------------------------------------------------
The physics guess corrected by the PFN is a dream world. A :class:`~learned_judgment.engine.dream.DreamSet`
holds, for every dreamed situation, the dreamed objective score of every candidate command of
the skill's candidate grid. :meth:`SNNStudent.train_in_dream` trains one spiking network inside
that dream to choose the best candidate: one output per candidate, soft targets built from the
dreamed scores (:mod:`learned_judgment.students.targets`). The student never learns the PFN's
outcome predictions, and every number it reports about itself (agreement with the dream's
choice, dream regret) is measured inside the dream, not in the simulator.

Deciding uses only numpy (:class:`~learned_judgment.students.snn_core.SpikingNetwork`); importing
this module, loading a saved student and deciding never import torch. Training imports torch
lazily through :mod:`learned_judgment.students.snn_train`.

Shapes (float64 unless noted)::

    n situations, S situation features (spec order), D encoded inputs, C candidates, A action dims
    decide_index  (n, S) -> (n,) int64 candidate index
    decide        (n, S) -> (n, A) = spec.candidate_actions(situation, points_per_dim)[index]

On disk a student is the network's ``<stem>.npz`` + ``<stem>.json`` (snn_core.save_network,
no pickle); the JSON carries everything needed to rebuild the encoder and the candidate grid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from learned_judgment.engine.dream import DreamSet
from learned_judgment.engine.streams import stream_rng, stream_seed
from learned_judgment.skills.base import SkillSpec, Variable
from learned_judgment.students.encoding import InputEncoder
from learned_judgment.students.snn_core import (
    OperationCounts,
    SpikingNetwork,
    count_operations,
    load_network,
    save_network,
)
from learned_judgment.students.snn_train import SNNSettings
from learned_judgment.students.targets import dream_regret, score_spread, soft_targets

__all__ = ["STUDENT_KIND", "SNNStudent"]

STUDENT_KIND = "per_skill_snn"


def _points_per_dim(points_per_dim: Any, n_action: int) -> tuple[int, ...]:
    if isinstance(points_per_dim, (str, bytes)) or not isinstance(points_per_dim, Sequence):
        raise ValueError(f"points_per_dim must be a list of {n_action} integers; got {points_per_dim!r}")
    points = tuple(points_per_dim)
    if len(points) != n_action or any(
        isinstance(p, bool) or not isinstance(p, (int, np.integer)) or int(p) < 1 for p in points
    ):
        raise ValueError(f"points_per_dim must be {n_action} integers >= 1; got {list(points)}")
    return tuple(int(p) for p in points)


def _variable_record(variable: Variable) -> dict[str, Any]:
    return {
        "name": variable.name,
        "kind": variable.kind,
        "unit": variable.unit,
        "description": variable.description,
        "low": variable.low,
        "high": variable.high,
        "levels": list(variable.levels),
    }


def _variable_from_record(record: dict[str, Any]) -> Variable:
    return Variable(
        name=str(record["name"]),
        kind=record["kind"],
        unit=str(record["unit"]),
        description=str(record["description"]),
        low=None if record["low"] is None else float(record["low"]),
        high=None if record["high"] is None else float(record["high"]),
        levels=tuple(str(level) for level in record["levels"]),
    )


def _check_dreams(spec: SkillSpec, dreams: DreamSet, points: tuple[int, ...]) -> None:
    """Refuse a DreamSet that does not belong to ``spec`` and to the grid of ``points``."""
    if not isinstance(dreams, DreamSet):
        raise TypeError(f"dreams must be a DreamSet; got {type(dreams).__name__}")
    for label, have, want in (
        ("skill", dreams.skill, spec.name),
        ("situation_names", dreams.situation_names, spec.situation_names),
        ("action_names", dreams.action_names, spec.action_names),
        ("outcome_names", dreams.outcome_names, spec.outcome_names),
        ("primary_outcome", dreams.primary_outcome, spec.primary_outcome),
    ):
        if have != want:
            raise ValueError(f"DreamSet.{label} {have!r} does not match the spec's {want!r}")
    for row in range(dreams.n_rows):
        grid = np.asarray(spec.candidate_actions(dreams.situations[row], points), dtype=np.float64)
        if grid.shape != dreams.candidates.shape[1:] or not np.array_equal(grid, dreams.candidates[row]):
            raise ValueError(
                f"dream row {row}: the dream's {dreams.n_candidates} candidates are not the grid "
                f"spec.candidate_actions(situation, points_per_dim={list(points)}) gives ({grid.shape[0]} candidates); "
                "train with the points_per_dim the dream was made with"
            )


def _dream_content_sha256(dreams: DreamSet, rows: np.ndarray) -> str:
    """sha256 over the dream rows the student used (names, situations, candidates, scores, choice)."""
    hasher = hashlib.sha256(b"learned-judgment/dream-rows/v1")
    for name in (dreams.skill, dreams.method, *dreams.situation_names, *dreams.action_names):
        hasher.update(name.encode("utf-8") + b"\0")
    for array in (dreams.situations[rows], dreams.candidates[rows], dreams.scores[rows], dreams.chosen[rows]):
        array = np.ascontiguousarray(array)
        hasher.update(f"{array.dtype.str}{array.shape}".encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class SNNStudent:
    """A dream-trained spiking network that picks one candidate command per situation.

    Attributes:
        skill:          SkillSpec name the student belongs to.
        network:        the deployed numpy network; output k is candidate k of the grid.
        encoder:        situation (n, S) -> encoded inputs (n, D).
        points_per_dim: candidate grid resolution the dream and the student use.
        n_candidates:   C, number of network outputs.
        meta:           JSON-serializable provenance: settings, encoder, dream, training report,
                        parity report, and in-dream validation numbers.
    """

    skill: str
    network: SpikingNetwork
    encoder: InputEncoder
    points_per_dim: tuple[int, ...]
    n_candidates: int
    meta: dict

    def __post_init__(self) -> None:
        if not isinstance(self.skill, str) or not self.skill:
            raise ValueError(f"skill must be a non-empty string; got {self.skill!r}")
        if not isinstance(self.network, SpikingNetwork):
            raise TypeError(f"network must be a SpikingNetwork; got {type(self.network).__name__}")
        if not isinstance(self.encoder, InputEncoder):
            raise TypeError(f"encoder must be an InputEncoder; got {type(self.encoder).__name__}")
        if self.network.n_inputs != self.encoder.width:
            raise ValueError(f"the network takes {self.network.n_inputs} inputs but the encoder gives {self.encoder.width}")
        if int(self.n_candidates) != self.network.n_outputs:
            raise ValueError(f"n_candidates {self.n_candidates} differs from the network's {self.network.n_outputs} outputs")
        points = _points_per_dim(self.points_per_dim, len(self.points_per_dim))
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be a dict; got {type(self.meta).__name__}")
        object.__setattr__(self, "points_per_dim", points)
        object.__setattr__(self, "n_candidates", int(self.n_candidates))

    # ----- training ---------------------------------------------------------------------

    @classmethod
    def train_in_dream(
        cls,
        spec: SkillSpec,
        dreams: DreamSet,
        *,
        settings: SNNSettings,
        points_per_dim: Sequence[int],
        base_seed: int,
    ) -> "SNNStudent":
        """Train a spiking student for ``spec`` inside ``dreams``.

        Steps: validate the dream against ``spec`` and the grid of ``points_per_dim``; split the
        rows into train / validation with the stream ``"<skill>/snn/validation_split"``; encode
        situations with ``InputEncoder.from_variables(spec.situation)``; build soft targets with
        the score spread of the TRAINING rows; train one head named ``spec.name`` (network
        initialisation from stream ``"<skill>/snn/init"``, batch order from
        ``"<skill>/snn/train"``); export to numpy and check parity on the validation rows.

        Raises ValueError for a mismatched dream, a wrong grid, too few rows or flat scores;
        :class:`~learned_judgment.students.snn_train.TrainingError` when training does not beat
        initialisation; :class:`~learned_judgment.students.snn_train.ParityError` when the numpy
        network does not reproduce the torch model.
        """
        from learned_judgment.students import snn_train

        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if not isinstance(settings, SNNSettings):
            raise TypeError(f"settings must be SNNSettings; got {type(settings).__name__}")
        if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
            raise ValueError(f"base_seed must be a non-negative integer; got {base_seed!r}")
        base_seed = int(base_seed)
        points = _points_per_dim(points_per_dim, spec.n_action)
        _check_dreams(spec, dreams, points)

        n = dreams.n_rows
        n_validation = max(1, int(round(settings.validation_fraction * n)))
        if n_validation >= n:
            raise ValueError(
                f"{n} dream rows leave no training rows after holding out {n_validation} for validation "
                f"(validation_fraction {settings.validation_fraction})"
            )
        split_stream = f"{spec.name}/snn/validation_split"
        order = stream_rng(base_seed, split_stream).permutation(n)
        validation_rows = np.sort(order[:n_validation])
        train_rows = np.sort(order[n_validation:])

        encoder = InputEncoder.from_variables(spec.situation)
        encoded = encoder.encode(dreams.situations)
        spread = score_spread(dreams.scores[train_rows])
        targets = soft_targets(dreams.scores, temperature=settings.target_temperature, spread=spread)

        init_stream, train_stream = f"{spec.name}/snn/init", f"{spec.name}/snn/train"
        init_seed, train_seed = stream_seed(base_seed, init_stream), stream_seed(base_seed, train_stream)
        model = snn_train.build_model(encoder.width, {spec.name: dreams.n_candidates}, settings, seed=init_seed)
        report = snn_train.fit(
            model,
            {spec.name: (encoded[train_rows], targets[train_rows])},
            validation={spec.name: (encoded[validation_rows], targets[validation_rows])},
            sampling={spec.name: 1.0},
            settings=settings,
            seed=train_seed,
        )
        network = snn_train.export_network(model, spec.name)
        parity = snn_train.check_parity(model, spec.name, network, encoded[validation_rows])

        decisions = network.decide(encoded[validation_rows])
        regret = dream_regret(dreams.scores[validation_rows], decisions)
        meta = {
            "kind": STUDENT_KIND,
            "label": (
                "dream-trained spiking student: trained to choose the candidate the dream scores best; "
                "the validation numbers are measured inside the dream, not in the simulator"
            ),
            "decision_runtime": "numpy (learned_judgment.students.snn_core); torch was used for training only",
            "student": cls._student_record(spec.name, encoder, points, dreams.n_candidates, spec.action_names),
            "settings": settings.describe(),
            "encoder": encoder.describe(),
            "dream": {
                "method": dreams.method,
                "request_hashes": list(dreams.request_hashes),
                "n_rows_used": int(n),
                "n_rows_train": int(train_rows.size),
                "n_rows_validation": int(validation_rows.size),
                "n_candidates": int(dreams.n_candidates),
                "score_spread": float(spread),
                "score_spread_rows": "training rows",
                "rows_sha256": _dream_content_sha256(dreams, np.arange(n)),
            },
            "seeds": {
                "base_seed": base_seed,
                "validation_split_stream": split_stream,
                "init_stream": init_stream,
                "init_seed": int(init_seed),
                "train_stream": train_stream,
                "train_seed": int(train_seed),
            },
            "training": report.to_json(),
            "parity": parity,
            "validation_in_dream": {
                "n_rows": int(validation_rows.size),
                "agreement_with_dream_choice": float(np.mean(decisions == dreams.chosen[validation_rows])),
                "fraction_with_zero_dream_regret": float(np.mean(regret == 0.0)),
                "mean_dream_regret": float(np.mean(regret)),
                "mean_dream_regret_over_spread": float(np.mean(regret) / spread),
            },
        }
        return cls(
            skill=spec.name,
            network=network,
            encoder=encoder,
            points_per_dim=points,
            n_candidates=dreams.n_candidates,
            meta=meta,
        )

    @staticmethod
    def _student_record(
        skill: str, encoder: InputEncoder, points: tuple[int, ...], n_candidates: int, action_names: Sequence[str]
    ) -> dict[str, Any]:
        return {
            "skill": skill,
            "points_per_dim": list(points),
            "n_candidates": int(n_candidates),
            "action_names": list(action_names),
            "situation_variables": [_variable_record(variable) for variable in encoder.variables],
            "encoder": encoder.describe(),
        }

    # ----- deciding (numpy only) ----------------------------------------------------------

    @property
    def digest(self) -> str:
        """sha256 identity of the trained network (snn_core.SpikingNetwork.digest)."""
        return self.network.digest()

    def _situations(self, situations: np.ndarray) -> np.ndarray:
        array = np.asarray(situations, dtype=np.float64)
        width = len(self.encoder.variables)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"situations must have shape (n, {width}); got {array.shape}")
        if not np.all(np.isfinite(array)):
            row, column = np.argwhere(~np.isfinite(array))[0]
            raise ValueError(
                f"situations contain non-finite values (first at row {row}, feature "
                f"{self.encoder.variables[column].name!r}); the student refuses to decide on them"
            )
        return array

    def decide_index(self, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n,) int64 index of the chosen candidate (numpy only)."""
        return self.network.decide(self.encoder.encode(self._situations(situations))).astype(np.int64)

    def _check_spec(self, spec: SkillSpec) -> None:
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if spec.name != self.skill:
            raise ValueError(f"this student was trained for skill {self.skill!r}, not {spec.name!r}")
        if not self.encoder.matches(InputEncoder.from_variables(spec.situation).describe()):
            raise ValueError(
                f"spec {spec.name!r} encodes its situation differently from the encoder this student was trained "
                "with (situation variables, ranges or levels changed)"
            )
        trained_actions = tuple(self.meta.get("student", {}).get("action_names", ()))
        if spec.action_names != trained_actions:
            raise ValueError(f"spec actions {spec.action_names} differ from the trained actions {trained_actions}")

    def decide(self, spec: SkillSpec, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n, A) chosen actions: ``spec.candidate_actions(situation,
        points_per_dim)[index]`` for each row. Refuses a spec whose name, situation encoding or
        actions differ from the ones the student was trained with."""
        self._check_spec(spec)
        situations = self._situations(situations)
        index = self.decide_index(situations)
        actions = np.empty((situations.shape[0], spec.n_action), dtype=np.float64)
        for row, (situation, choice) in enumerate(zip(situations, index)):
            grid = np.asarray(spec.candidate_actions(situation, self.points_per_dim), dtype=np.float64)
            if grid.shape != (self.n_candidates, spec.n_action):
                raise ValueError(
                    f"row {row}: spec.candidate_actions gives {grid.shape}; the student was trained on "
                    f"({self.n_candidates}, {spec.n_action})"
                )
            actions[row] = grid[choice]
        if not np.all(spec.action_in_bounds(actions)):
            raise ValueError(f"skill {spec.name!r} produced a chosen action outside its declared bounds")
        return actions

    def operation_counts(self, situations: np.ndarray) -> OperationCounts:
        """Operations per decision measured on ``situations`` (n, S) (inputs to the energy estimate)."""
        return count_operations(self.network, self.encoder.encode(self._situations(situations)))

    # ----- persistence --------------------------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Write ``<stem>.npz`` + ``<stem>.json`` (no pickle); return the network digest."""
        meta = dict(self.meta)
        meta["kind"] = STUDENT_KIND
        trained_actions = meta.get("student", {}).get("action_names")
        if trained_actions is None:
            raise ValueError("meta['student']['action_names'] is missing; the student cannot be reloaded safely")
        meta["student"] = self._student_record(
            self.skill, self.encoder, self.points_per_dim, self.n_candidates, trained_actions
        )
        try:
            return save_network(path, self.network, meta)
        except (TypeError, ValueError) as error:
            raise ValueError(f"student meta is not JSON-serializable with finite numbers: {error}") from error

    @classmethod
    def load(cls, path: str | Path) -> "SNNStudent":
        """Inverse of :meth:`save`; verifies the network digest and the encoder description."""
        network, meta = load_network(path)
        if meta.get("kind") != STUDENT_KIND:
            raise ValueError(f"{path} is not a {STUDENT_KIND} file (kind {meta.get('kind')!r})")
        record = meta["student"]
        encoder = InputEncoder.from_variables([_variable_from_record(item) for item in record["situation_variables"]])
        if not encoder.matches(record["encoder"]):
            raise ValueError(f"{path}: the stored situation variables do not rebuild the stored encoder")
        return cls(
            skill=str(record["skill"]),
            network=network,
            encoder=encoder,
            points_per_dim=tuple(int(p) for p in record["points_per_dim"]),
            n_candidates=int(record["n_candidates"]),
            meta=meta,
        )
