"""Shared multi-skill spiking student, trained in the dream with dream rehearsal.

Lane contract (Person B, lane B: shared multi-skill spiking student)
--------------------------------------------------------------------
The physics guess corrected by the PFN is a dream world. A
:class:`~learned_judgment.engine.dream.DreamSet` holds, for every dreamed situation, the dreamed
objective score of every candidate command of one skill. :class:`SharedSNNStudent` is ONE spiking
network for all skills: a spiking trunk shared by every skill plus one readout head per skill,
one output per candidate of that skill's grid. Each head is trained inside its skill's dream to
choose the candidate the dream scores best (soft targets from the dreamed scores,
:mod:`learned_judgment.students.targets`). The network never learns the PFN's outcome
predictions, and every number it reports about itself is measured inside the dream, on dream
rows held out from training, not in the simulator.

Skills are added in sequence (DODGE -> BRAKE -> TURN -> RAMP in the project). Each
:meth:`SharedSNNStudent.add_skill` continues from the current network (the trunk is initialised
once, when the first skill is added) and adds the new skill's head:

    rehearsal=True   every batch mixes rows of the new skill with rows of the STORED dream sets of
                     the skills added earlier; the earlier skills together get
                     ``students.snn.rehearsal_fraction`` of each batch, split evenly among them.
                     Rehearsal rows are stored dreams: never real attempts, never a new provider call.
    rehearsal=False  only the new skill's rows (the no-rehearsal control that shows forgetting).

After training, one :class:`~learned_judgment.students.snn_core.SpikingNetwork` per skill held
(trunk + that skill's head) is exported, its parity with the torch model is checked on that
skill's held-out rows, and the decision agreement with the dream's choice and the dream regret
are measured on the held-out rows of EVERY skill held. A skill's held-out rows are chosen once,
when the skill is added, with the stream ``"<skill>/snn/validation_split"`` and the per-skill
student's recipe, so for the same base seed and dream set they are the per-skill student's
validation rows (paired comparison). :meth:`retention_matrix` is built only from these records.

Known limitation: :func:`~learned_judgment.students.snn_train.fit` early-stops on the validation
loss of every head it trains. With rehearsal the earlier skills' held-out rows are therefore
also used to pick the stopping step, which can favour the rehearsal run's retention numbers
slightly; without rehearsal they are not used. A separate retention split would need its own
config key.

Input layout (:class:`~learned_judgment.students.encoding.SharedInputEncoder`)::

    [ skill's own encoding zero-padded to students.snn.shared_feature_width | one-hot skill ID ]
    the one-hot follows students.snn.shared_skills, fixed when the student is created

Shapes (float64 unless noted)::

    n situations, S situation features of the skill (spec order), A action dims, C_k candidates of
    skill k, W = shared_feature_width + number of shared skills (network input width)
    decide_index  (skill, (n, S)) -> (n,) int64 candidate index
    decide        (spec, (n, S))  -> (n, A) = spec.candidate_actions(situation, points_per_dim)[index]

Deciding, loading and importing this module use numpy only; torch and snnTorch are imported
lazily by :mod:`learned_judgment.students.snn_train` when a skill is added.

On disk (:meth:`save`, no pickle anywhere)::

    <directory>/manifest.json            settings, skills, held-out rows, seeds, add_skill records,
                                         sha256 of the model state
    <directory>/model_state.npz          torch state dict as named arrays (weights only)
    <directory>/networks/<skill>.npz|.json   per-skill SpikingNetwork view (snn_core.save_network)
    <directory>/dreams/<skill>.npz|.json     stored dream set (engine.dream.save_dreams)
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from learned_judgment.config import require
from learned_judgment.engine.dream import DreamSet, load_dreams, save_dreams
from learned_judgment.engine.streams import stream_rng, stream_seed
from learned_judgment.fileio import atomic_write_bytes
from learned_judgment.skills.base import SkillSpec
from learned_judgment.students.encoding import InputEncoder, SharedInputEncoder
from learned_judgment.students.snn import (
    SNNStudent,
    _check_dreams,
    _dream_content_sha256,
    _points_per_dim,
    _variable_from_record,
)
from learned_judgment.students.snn_core import (
    OperationCounts,
    SpikingNetwork,
    count_operations,
    load_network,
    save_network,
)
from learned_judgment.students.snn_train import SNNSettings
from learned_judgment.students.targets import dream_regret, score_spread, soft_targets

__all__ = ["MANIFEST_FORMAT", "STUDENT_KIND", "SharedSNNStudent"]

STUDENT_KIND = "shared_snn"
MANIFEST_FORMAT = "learned-judgment/shared-spiking-student/v1"
_VIEW_KIND = "shared_snn_view"
_DREAMS_KIND = "shared_snn_stored_dreams"
_MANIFEST_FILE = "manifest.json"
_STATE_FILE = "model_state.npz"
_NETWORKS_DIR = "networks"
_DREAMS_DIR = "dreams"
_TRUNK_INIT_STREAM = "shared_snn/trunk_init"
_LABEL = (
    "dream-trained shared spiking student: one spiking trunk for all skills and one readout head per skill, "
    "each trained to choose the candidate its skill's dream scores best; every agreement and regret number is "
    "measured inside the dream on held-out dream rows, not in the simulator"
)


def _json_text(value: Any, label: str, *, sort_keys: bool) -> str:
    try:
        return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not JSON-serializable with finite numbers: {error}") from error


def _canonical_json(value: Any, label: str) -> bytes:
    """UTF-8 JSON with sorted keys and a final newline (the on-disk form)."""
    return (_json_text(value, label, sort_keys=True) + "\n").encode("utf-8")


def _json_copy(value: Any, label: str) -> Any:
    """Deep copy through JSON (key order kept); raises ValueError for non-finite numbers."""
    return json.loads(_json_text(value, label, sort_keys=False))


def _state_bytes(state: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **state)
    return buffer.getvalue()


@dataclass(frozen=True, eq=False)
class _HeldSkill:
    """One skill held by the shared network (internal).

    Attributes:
        name, position:   skill name; its place in the add sequence (0-based).
        encoder:          the skill's own situation encoder, (n, S) -> (n, D_k), D_k <= feature width.
        points_per_dim:   candidate grid resolution the dream and the head use.
        n_candidates:     C_k, outputs of the skill's head.
        action_names:     the spec's action names, checked when deciding.
        dreams:           the stored dream set (rehearsal data and held-out rows).
        validation_rows:  (n_val,) int64 sorted held-out rows, fixed when the skill was added.
        train_rows:       (n - n_val,) int64 sorted remaining rows.
        spread:           score spread of the training rows (scale of the soft targets).
        rows_sha256:      sha256 of the dream rows (names, situations, candidates, scores, choice).
        seeds:            streams and seeds used when the skill was added.
    """

    name: str
    position: int
    encoder: InputEncoder
    points_per_dim: tuple[int, ...]
    n_candidates: int
    action_names: tuple[str, ...]
    dreams: DreamSet
    validation_rows: np.ndarray
    train_rows: np.ndarray
    spread: float
    rows_sha256: str
    seeds: dict[str, Any]


class SharedSNNStudent:
    """One spiking network for several skills: a shared trunk plus one readout head per skill.

    Construct with :meth:`create` (all values from config.yaml) or explicitly with every value
    given; nothing is defaulted. The student starts empty and grows with :meth:`add_skill`.
    """

    def __init__(
        self,
        *,
        skills: Sequence[str],
        feature_width: int,
        settings: SNNSettings,
        rehearsal_fraction: float,
    ) -> None:
        """
        Args:
            skills:             every skill the network may hold, in the fixed order of the one-hot
                                skill ID (students.snn.shared_skills); identifier names.
            feature_width:      columns reserved for a skill's own encoding
                                (students.snn.shared_feature_width).
            settings:           hyperparameters of the network and of every training run.
            rehearsal_fraction: share of each batch drawn from earlier skills when rehearsing,
                                in (0, 1) (students.snn.rehearsal_fraction).
        """
        if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
            raise ValueError(f"students.snn.shared_skills must be a list of skill names; got {skills!r}")
        for index, skill in enumerate(skills):
            if not isinstance(skill, str) or not skill.isidentifier():
                raise ValueError(
                    f"students.snn.shared_skills[{index}] must be an identifier (it names a network head); got {skill!r}"
                )
        if isinstance(feature_width, bool) or not isinstance(feature_width, (int, np.integer)):
            raise ValueError(f"students.snn.shared_feature_width must be an integer; got {feature_width!r}")
        if not isinstance(settings, SNNSettings):
            raise TypeError(f"settings must be SNNSettings; got {type(settings).__name__}")
        if isinstance(rehearsal_fraction, bool) or not isinstance(rehearsal_fraction, (int, float, np.integer, np.floating)):
            raise ValueError(f"students.snn.rehearsal_fraction must be a number; got {rehearsal_fraction!r}")
        fraction = float(rehearsal_fraction)
        if not (math.isfinite(fraction) and 0.0 < fraction < 1.0):
            raise ValueError(
                f"students.snn.rehearsal_fraction must lie in (0, 1) so that the new skill and every earlier "
                f"skill get rows in each batch; got {fraction}"
            )
        self._encoder = SharedInputEncoder(skills=tuple(skills), feature_width=int(feature_width))
        self._settings = settings
        self._rehearsal_fraction = fraction
        self._held: dict[str, _HeldSkill] = {}
        self._networks: dict[str, SpikingNetwork] = {}
        self._records: list[dict[str, Any]] = []
        self._state: dict[str, np.ndarray] | None = None
        self._trunk_seed: int | None = None
        self._model: Any = None  # torch model of the committed state; rebuilt from _state when None

    @classmethod
    def create(cls, config: Mapping[str, Any]) -> "SharedSNNStudent":
        """An empty shared student from config.yaml.

        Reads ``students.kind`` (must be ``"snn"``), ``students.shared_snn`` (must be true),
        ``students.snn.shared_skills``, ``students.snn.shared_feature_width``,
        ``students.snn.rehearsal_fraction`` and every ``students.snn`` setting
        (:meth:`SNNSettings.from_config`). A missing or null key raises ``ConfigError`` naming it.
        """
        kind = require(config, "students.kind")
        if kind != "snn":
            raise ValueError(f"config.yaml students.kind is {kind!r}; the shared student is a spiking network ('snn')")
        if require(config, "students.shared_snn") is not True:
            raise ValueError("config.yaml students.shared_snn is not true: no shared spiking student is configured")
        return cls(
            skills=require(config, "students.snn.shared_skills"),
            feature_width=require(config, "students.snn.shared_feature_width"),
            settings=SNNSettings.from_config(config),
            rehearsal_fraction=require(config, "students.snn.rehearsal_fraction"),
        )

    # ----- structure ----------------------------------------------------------------------

    @property
    def skills(self) -> tuple[str, ...]:
        """Every skill the network may hold, in one-hot order (fixed at creation)."""
        return self._encoder.skills

    @property
    def feature_width(self) -> int:
        return self._encoder.feature_width

    @property
    def input_width(self) -> int:
        """W = feature_width + number of shared skills: the network's input width."""
        return self._encoder.width

    @property
    def settings(self) -> SNNSettings:
        return self._settings

    @property
    def rehearsal_fraction(self) -> float:
        return self._rehearsal_fraction

    @property
    def skills_held(self) -> tuple[str, ...]:
        """Skills added so far, in the order they were added."""
        return tuple(self._held)

    def network_view(self, skill: str) -> SpikingNetwork:
        """The deployed numpy network of ``skill``: shared trunk + that skill's head,
        layer sizes (W, H_1, ..., H_L, C_skill)."""
        return self._networks[self._held_skill(skill).name]

    def network_digests(self) -> dict[str, str]:
        """{skill: sha256 of its current network view}, in add order."""
        return {skill: network.digest() for skill, network in self._networks.items()}

    def held_out_rows(self, skill: str) -> np.ndarray:
        """(n_val,) int64 sorted indices into ``skill``'s stored dream set: its fixed held-out rows."""
        return self._held_skill(skill).validation_rows.copy()

    @property
    def records(self) -> list[dict[str, Any]]:
        """A copy of every :meth:`add_skill` record, in add order."""
        return _json_copy(self._records, "add_skill records")

    # ----- training -----------------------------------------------------------------------

    def add_skill(
        self,
        spec: SkillSpec,
        dreams: DreamSet,
        *,
        points_per_dim: Sequence[int],
        base_seed: int,
        rehearsal: bool,
    ) -> dict[str, Any]:
        """Add ``spec``'s head and train the shared network inside the dream.

        Args:
            spec:           the skill's SkillSpec; its name must be in :attr:`skills` and not held yet.
            dreams:         DreamSet of ``spec``: n rows, C candidates from the grid of ``points_per_dim``.
            points_per_dim: the grid the dream was made with (A integers).
            base_seed:      run seed. Streams under it: ``"<skill>/snn/validation_split"`` (held-out
                            rows), ``"<skill>/shared_snn/head_init"``, ``"<skill>/shared_snn/train"`` and,
                            for the first skill only, ``"shared_snn/trunk_init"``.
            rehearsal:      True mixes the earlier skills' stored dream rows into every batch;
                            False trains on the new skill's rows only (no-rehearsal control).

        Returns:
            A JSON-serializable record: settings, seeds, sampling shares, steps run, wall time,
            software versions, the training report, a parity report per skill held, the network
            digests, and ``held_out_in_dream[skill]`` = {n_rows, agreement_with_dream_choice,
            fraction_with_zero_dream_regret, mean_dream_regret, mean_dream_regret_over_spread}
            for EVERY skill held, measured on its fixed held-out rows.

        Raises:
            ValueError for a skill outside :attr:`skills`, a skill already held, an encoding wider
            than :attr:`feature_width`, a dream that does not match ``spec`` or the grid, too few
            rows or flat scores; TrainingError / ParityError from snn_train. On any error the
            student is left exactly as it was before the call.
        """
        from learned_judgment.students import snn_train

        started = time.perf_counter()
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        if not isinstance(rehearsal, (bool, np.bool_)):
            raise ValueError(f"rehearsal must be True or False; got {rehearsal!r}")
        if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
            raise ValueError(f"base_seed must be a non-negative integer; got {base_seed!r}")
        base_seed = int(base_seed)
        name = spec.name
        if name not in self.skills:
            raise ValueError(
                f"skill {name!r} is not one of the shared network's skills {list(self.skills)} "
                "(students.snn.shared_skills); the one-hot skill ID is fixed when the student is created"
            )
        if name in self._held:
            raise ValueError(
                f"skill {name!r} was already added (position {self._held[name].position} of {list(self._held)}); "
                "each skill is added once"
            )
        points = _points_per_dim(points_per_dim, spec.n_action)
        encoder = InputEncoder.from_variables(spec.situation)
        if encoder.width > self.feature_width:
            raise ValueError(
                f"skill {name!r} encodes its situation into {encoder.width} columns but "
                f"students.snn.shared_feature_width is {self.feature_width}"
            )
        _check_dreams(spec, dreams, points)

        settings = self._settings
        n = dreams.n_rows
        n_validation = max(1, int(round(settings.validation_fraction * n)))
        if n_validation >= n:
            raise ValueError(
                f"{n} dream rows leave no training rows after holding out {n_validation} "
                f"(validation_fraction {settings.validation_fraction})"
            )
        split_stream = f"{name}/snn/validation_split"
        order = stream_rng(base_seed, split_stream).permutation(n)
        validation_rows = np.sort(order[:n_validation]).astype(np.int64)
        train_rows = np.sort(order[n_validation:]).astype(np.int64)
        spread = score_spread(dreams.scores[train_rows])

        head_stream, train_stream = f"{name}/shared_snn/head_init", f"{name}/shared_snn/train"
        trunk_initialised_here = self._trunk_seed is None
        trunk_seed = stream_seed(base_seed, _TRUNK_INIT_STREAM) if trunk_initialised_here else self._trunk_seed
        seeds = {
            "base_seed": base_seed,
            "validation_split_stream": split_stream,
            "head_init_stream": head_stream,
            "head_init_seed": int(stream_seed(base_seed, head_stream)),
            "train_stream": train_stream,
            "train_seed": int(stream_seed(base_seed, train_stream)),
            "trunk_init_stream": _TRUNK_INIT_STREAM,
            "trunk_init_seed": int(trunk_seed),
            "trunk_initialised_here": trunk_initialised_here,
        }
        new = _HeldSkill(
            name=name,
            position=len(self._held),
            encoder=encoder,
            points_per_dim=points,
            n_candidates=dreams.n_candidates,
            action_names=spec.action_names,
            dreams=dreams,
            validation_rows=validation_rows,
            train_rows=train_rows,
            spread=float(spread),
            rows_sha256=_dream_content_sha256(dreams, np.arange(n)),
            seeds=seeds,
        )
        earlier = tuple(self._held.values())
        rehearse = bool(rehearsal) and bool(earlier)
        if rehearse:
            share = self._rehearsal_fraction / len(earlier)
            sampling = {name: 1.0 - self._rehearsal_fraction, **{held.name: share for held in earlier}}
        else:
            sampling = {name: 1.0}
        tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        validation: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for held in (new, *(earlier if rehearse else ())):
            tasks[held.name], validation[held.name] = self._training_pairs(held)

        try:
            model = self._torch_model(trunk_seed)
            model.add_head(name, new.n_candidates, seed=seeds["head_init_seed"])
            report = snn_train.fit(
                model, tasks, validation=validation, sampling=sampling, settings=settings, seed=seeds["train_seed"]
            )
            networks: dict[str, SpikingNetwork] = {}
            parity: dict[str, dict[str, Any]] = {}
            held_out: dict[str, dict[str, Any]] = {}
            for held in (*earlier, new):
                network = snn_train.export_network(model, held.name)
                encoded = self._encode_rows(held, held.validation_rows)
                parity[held.name] = snn_train.check_parity(model, held.name, network, encoded)
                held_out[held.name] = self._held_out_numbers(held, network, encoded)
                networks[held.name] = network
            state = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
            record = _json_copy(
                {
                    "kind": "shared_snn_add_skill",
                    "label": _LABEL,
                    "decision_runtime": "numpy (learned_judgment.students.snn_core); torch was used for training only",
                    "position": new.position,
                    "skill": name,
                    "skills_held": [held.name for held in (*earlier, new)],
                    "rehearsal_requested": bool(rehearsal),
                    "rehearsal_applied": rehearse,
                    "rehearsal_source": (
                        "stored dream sets of the earlier skills (their training rows); no real attempts, no provider calls"
                        if rehearse
                        else None
                    ),
                    "rehearsal_fraction": self._rehearsal_fraction,
                    "sampling": dict(report.sampling),
                    "settings": settings.describe(),
                    "encoder": {"shared": self._encoder.describe(), "skill": encoder.describe()},
                    "points_per_dim": list(points),
                    "dream": {
                        "method": dreams.method,
                        "request_hashes": list(dreams.request_hashes),
                        "n_rows": int(n),
                        "n_rows_train": int(train_rows.size),
                        "n_rows_held_out": int(validation_rows.size),
                        "n_candidates": int(dreams.n_candidates),
                        "score_spread": float(spread),
                        "score_spread_rows": "training rows",
                        "rows_sha256": new.rows_sha256,
                    },
                    "seeds": seeds,
                    "steps_run": report.steps_run,
                    "wall_seconds": float(time.perf_counter() - started),
                    "versions": dict(report.versions),
                    "training": report.to_json(),
                    "parity": parity,
                    "network_digests": {skill: network.digest() for skill, network in networks.items()},
                    "held_out_in_dream": held_out,
                },
                f"add_skill record of {name!r}",
            )
        except BaseException:
            self._model = None  # it may hold the new head or half-trained weights; rebuilt from the committed state
            raise

        self._held[name] = new
        self._networks = networks
        self._state = state
        self._trunk_seed = int(trunk_seed)
        self._model = model
        self._records.append(record)
        return _json_copy(record, f"add_skill record of {name!r}")

    def _training_pairs(
        self, held: _HeldSkill
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """((train inputs (n_tr, W), train targets (n_tr, C)), (held-out inputs, held-out targets))."""
        encoded = self._encoder.encode(held.name, held.encoder, held.dreams.situations)
        targets = soft_targets(held.dreams.scores, temperature=self._settings.target_temperature, spread=held.spread)
        rows_train, rows_held_out = held.train_rows, held.validation_rows
        return (encoded[rows_train], targets[rows_train]), (encoded[rows_held_out], targets[rows_held_out])

    def _encode_rows(self, held: _HeldSkill, rows: np.ndarray) -> np.ndarray:
        """Encoded inputs (len(rows), W) of the given rows of the skill's stored dream set."""
        return self._encoder.encode(held.name, held.encoder, held.dreams.situations[rows])

    def _held_out_numbers(self, held: _HeldSkill, network: SpikingNetwork, encoded: np.ndarray) -> dict[str, Any]:
        decisions = network.decide(encoded).astype(np.int64)
        rows = held.validation_rows
        regret = dream_regret(held.dreams.scores[rows], decisions)
        return {
            "n_rows": int(rows.size),
            "agreement_with_dream_choice": float(np.mean(decisions == held.dreams.chosen[rows])),
            "fraction_with_zero_dream_regret": float(np.mean(regret == 0.0)),
            "mean_dream_regret": float(np.mean(regret)),
            "mean_dream_regret_over_spread": float(np.mean(regret) / held.spread),
        }

    def _torch_model(self, trunk_seed: int) -> Any:
        """The torch model of the committed state: the cached one, or one rebuilt from the saved
        state dict (after :meth:`load` or a failed add), or a new trunk for the first skill."""
        if self._model is not None:
            return self._model
        import torch

        from learned_judgment.students import snn_train

        model = snn_train.build_model(self.input_width, {}, self._settings, seed=int(trunk_seed))
        if self._state is None:
            return model
        for held in self._held.values():
            model.add_head(held.name, held.n_candidates, seed=held.seeds["head_init_seed"])
        model.load_state_dict({key: torch.as_tensor(np.array(value)) for key, value in self._state.items()}, strict=True)
        model.eval()
        for skill, network in self._networks.items():
            if snn_train.export_network(model, skill).digest() != network.digest():
                raise ValueError(
                    f"the stored model state does not reproduce the network view of skill {skill!r}; "
                    "the state file and the networks belong to different students"
                )
        return model

    # ----- deciding (numpy only) ----------------------------------------------------------

    def _held_skill(self, skill: str) -> _HeldSkill:
        if not isinstance(skill, str):
            raise ValueError(f"skill must be a skill name; got {skill!r}")
        if skill not in self._held:
            where = "has not been added yet" if skill in self.skills else "is not one of the shared skills"
            raise ValueError(f"skill {skill!r} {where}; skills held: {list(self._held)}")
        return self._held[skill]

    def _encode_situations(self, held: _HeldSkill, situations: np.ndarray) -> np.ndarray:
        array = np.asarray(situations, dtype=np.float64)
        width = len(held.encoder.variables)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{held.name}: situations must have shape (n, {width}); got {array.shape}")
        if not np.all(np.isfinite(array)):
            row, column = np.argwhere(~np.isfinite(array))[0]
            raise ValueError(
                f"{held.name}: situations contain non-finite values (first at row {row}, feature "
                f"{held.encoder.variables[column].name!r}); the student refuses to decide on them"
            )
        return self._encoder.encode(held.name, held.encoder, array)

    def decide_index(self, skill: str, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations of ``skill`` -> (n,) int64 index into its candidate grid (numpy only)."""
        held = self._held_skill(skill)
        return self._networks[held.name].decide(self._encode_situations(held, situations)).astype(np.int64)

    def decide(self, spec: SkillSpec, situations: np.ndarray) -> np.ndarray:
        """(n, S) situations -> (n, A) chosen actions ``spec.candidate_actions(situation,
        points_per_dim)[index]``. Refuses a spec that is not held or whose situation encoding or
        action names differ from the ones its head was trained with."""
        if not isinstance(spec, SkillSpec):
            raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
        held = self._held_skill(spec.name)
        if not held.encoder.matches(InputEncoder.from_variables(spec.situation).describe()):
            raise ValueError(
                f"spec {spec.name!r} encodes its situation differently from the encoder its head was trained with "
                "(situation variables, ranges or levels changed)"
            )
        if spec.action_names != held.action_names:
            raise ValueError(f"spec actions {spec.action_names} differ from the trained actions {held.action_names}")
        array = np.asarray(situations, dtype=np.float64)
        index = self.decide_index(spec.name, array)
        actions = np.empty((array.shape[0], spec.n_action), dtype=np.float64)
        for row, (situation, choice) in enumerate(zip(array, index)):
            grid = np.asarray(spec.candidate_actions(situation, held.points_per_dim), dtype=np.float64)
            if grid.shape != (held.n_candidates, spec.n_action):
                raise ValueError(
                    f"row {row}: spec.candidate_actions gives {grid.shape}; the head of {spec.name!r} was trained on "
                    f"({held.n_candidates}, {spec.n_action})"
                )
            actions[row] = grid[choice]
        if not np.all(spec.action_in_bounds(actions)):
            raise ValueError(f"skill {spec.name!r} produced a chosen action outside its declared bounds")
        return actions

    def operation_counts(self, skill: str, situations: np.ndarray) -> OperationCounts:
        """Operations per decision of ``skill``'s view measured on ``situations`` (n, S); the
        first layer counts the full shared input width W (padding and skill ID included)."""
        held = self._held_skill(skill)
        return count_operations(self._networks[held.name], self._encode_situations(held, situations))

    # ----- retention ----------------------------------------------------------------------

    def retention_matrix(self) -> dict[str, Any]:
        """Retention of every skill as later skills were added, from the add_skill records only.

        Returns a JSON-serializable dict. With m skills added (``skills`` in add order), each
        matrix is m x m: ``R[i][j]`` is measured on skill j's fixed held-out dream rows right
        after the skill at position i was added, for j <= i, and is ``None`` for j > i. Matrices:
        ``agreement_with_dream_choice``, ``fraction_with_zero_dream_regret``,
        ``mean_dream_regret`` and ``mean_dream_regret_over_spread``. ``backward_transfer[key]`` =
        mean over j < m-1 of (R[m-1][j] - R[j][j]), ``None`` when m < 2: negative for agreement
        and positive for regret mean the earlier skills got worse.
        """
        skills = [record["skill"] for record in self._records]
        m = len(skills)
        keys = (
            "agreement_with_dream_choice",
            "fraction_with_zero_dream_regret",
            "mean_dream_regret",
            "mean_dream_regret_over_spread",
        )
        matrices = {
            key: [
                [self._records[i]["held_out_in_dream"][skills[j]][key] if j <= i else None for j in range(m)]
                for i in range(m)
            ]
            for key in keys
        }
        backward = {
            key: (None if m < 2 else float(np.mean([matrices[key][m - 1][j] - matrices[key][j][j] for j in range(m - 1)])))
            for key in keys
        }
        return _json_copy(
            {
                "label": _LABEL,
                "skills": skills,
                "rehearsal_requested": [record["rehearsal_requested"] for record in self._records],
                "rehearsal_applied": [record["rehearsal_applied"] for record in self._records],
                "n_held_out_rows": {record["skill"]: record["dream"]["n_rows_held_out"] for record in self._records},
                "definition": (
                    "R[i][j]: measured on skill j's fixed held-out dream rows right after the skill at add position i "
                    "was added (rows = add position, columns = skills in add order); null for j > i. "
                    "backward_transfer = mean over j < last of (R[last][j] - R[j][j])"
                ),
                **matrices,
                "backward_transfer": backward,
            },
            "retention matrix",
        )

    # ----- persistence --------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write the student to ``directory`` (see the module docstring for the layout); no pickle.

        Every per-skill network view (snn_core.save_network), every stored dream set
        (engine.dream.save_dreams) and the model state dict (named arrays) are written first;
        the manifest, which records their digests, is written last.
        """
        if not self._held or self._state is None:
            raise ValueError("this shared student holds no skill yet; there is nothing to save")
        directory = Path(directory)
        entries = []
        for skill, held in self._held.items():
            network = self._networks[skill]
            student_record = SNNStudent._student_record(
                skill, held.encoder, held.points_per_dim, held.n_candidates, held.action_names
            )
            digest = save_network(
                directory / _NETWORKS_DIR / skill,
                network,
                {
                    "kind": _VIEW_KIND,
                    "label": _LABEL,
                    "skill": skill,
                    "shared_encoder": self._encoder.describe(),
                    "student": student_record,
                    "settings": self._settings.describe(),
                },
            )
            save_dreams(
                directory / _DREAMS_DIR / f"{skill}.npz",
                held.dreams,
                {
                    "kind": _DREAMS_KIND,
                    "skill": skill,
                    "use": "rehearsal rows and fixed held-out rows of the shared spiking student",
                    "rows_sha256": held.rows_sha256,
                },
            )
            entries.append(
                {
                    "skill": skill,
                    "position": held.position,
                    "points_per_dim": list(held.points_per_dim),
                    "n_candidates": held.n_candidates,
                    "action_names": list(held.action_names),
                    "student": student_record,
                    "network_sha256": digest,
                    "dreams_rows_sha256": held.rows_sha256,
                    "n_dream_rows": held.dreams.n_rows,
                    "held_out_rows": [int(row) for row in held.validation_rows],
                    "score_spread": held.spread,
                    "seeds": held.seeds,
                }
            )
        state = _state_bytes(self._state)
        atomic_write_bytes(directory / _STATE_FILE, state)
        manifest = {
            "format": MANIFEST_FORMAT,
            "kind": STUDENT_KIND,
            "label": _LABEL,
            "decision_runtime": "numpy (learned_judgment.students.snn_core); torch was used for training only",
            "skills": list(self.skills),
            "feature_width": self.feature_width,
            "shared_encoder": self._encoder.describe(),
            "rehearsal_fraction": self._rehearsal_fraction,
            "settings": self._settings.describe(),
            "trunk_init_seed": self._trunk_seed,
            "held": entries,
            "state_file": _STATE_FILE,
            "state_sha256": hashlib.sha256(state).hexdigest(),
            "state_keys": list(self._state),
            "records": self._records,
            "layout": {
                "networks": f"{_NETWORKS_DIR}/<skill>.npz + .json",
                "dreams": f"{_DREAMS_DIR}/<skill>.npz + .json",
                "state": _STATE_FILE,
            },
        }
        atomic_write_bytes(directory / _MANIFEST_FILE, _canonical_json(manifest, "shared student manifest"))

    @classmethod
    def load(cls, directory: str | Path) -> "SharedSNNStudent":
        """Inverse of :meth:`save`. Verifies the manifest format, every network digest, every
        dream set's digests, the encoders and the model-state sha256. Uses numpy only; adding a
        skill to the loaded student imports torch then."""
        directory = Path(directory)
        manifest = json.loads((directory / _MANIFEST_FILE).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{directory / _MANIFEST_FILE} is not a {MANIFEST_FORMAT} manifest")
        student = cls(
            skills=manifest["skills"],
            feature_width=manifest["feature_width"],
            settings=SNNSettings(**manifest["settings"]),
            rehearsal_fraction=manifest["rehearsal_fraction"],
        )
        if student._encoder.describe() != manifest["shared_encoder"]:
            raise ValueError(f"{directory}: the stored skills and feature width do not rebuild the stored shared encoder")
        for position, entry in enumerate(manifest["held"]):
            skill = entry["skill"]
            if skill not in student.skills or skill in student._held or entry["position"] != position:
                raise ValueError(f"{directory}: manifest entry {position} ({skill!r}) is not a new shared skill in order")
            network, view_meta = load_network(directory / _NETWORKS_DIR / skill)
            if view_meta.get("kind") != _VIEW_KIND or view_meta.get("skill") != skill:
                raise ValueError(f"{directory}: networks/{skill} is not the shared network view of {skill!r}")
            if network.digest() != entry["network_sha256"]:
                raise ValueError(f"{directory}: networks/{skill} does not match the digest in the manifest")
            if network.n_inputs != student.input_width or network.n_outputs != int(entry["n_candidates"]):
                raise ValueError(
                    f"{directory}: networks/{skill} has layers {network.layer_sizes}; expected input width "
                    f"{student.input_width} and {entry['n_candidates']} outputs"
                )
            dreams, _ = load_dreams(directory / _DREAMS_DIR / f"{skill}.npz")
            n = dreams.n_rows
            if dreams.skill != skill or n != entry["n_dream_rows"]:
                raise ValueError(f"{directory}: dreams/{skill}.npz is not the stored dream set of {skill!r}")
            if _dream_content_sha256(dreams, np.arange(n)) != entry["dreams_rows_sha256"]:
                raise ValueError(f"{directory}: dreams/{skill}.npz rows do not match the digest in the manifest")
            record = entry["student"]
            encoder = InputEncoder.from_variables([_variable_from_record(item) for item in record["situation_variables"]])
            if not encoder.matches(record["encoder"]) or encoder.width > student.feature_width:
                raise ValueError(f"{directory}: the stored situation variables of {skill!r} do not rebuild its encoder")
            validation_rows = np.asarray(entry["held_out_rows"], dtype=np.int64)
            if (
                validation_rows.ndim != 1
                or validation_rows.size < 1
                or validation_rows.size >= n
                or np.any(np.diff(validation_rows) <= 0)
                or validation_rows[0] < 0
                or validation_rows[-1] >= n
            ):
                raise ValueError(f"{directory}: the held-out rows of {skill!r} are not sorted unique rows in [0, {n})")
            student._held[skill] = _HeldSkill(
                name=skill,
                position=position,
                encoder=encoder,
                points_per_dim=tuple(int(p) for p in entry["points_per_dim"]),
                n_candidates=int(entry["n_candidates"]),
                action_names=tuple(entry["action_names"]),
                dreams=dreams,
                validation_rows=validation_rows,
                train_rows=np.setdiff1d(np.arange(n, dtype=np.int64), validation_rows),
                spread=float(entry["score_spread"]),
                rows_sha256=entry["dreams_rows_sha256"],
                seeds=dict(entry["seeds"]),
            )
            student._networks[skill] = network
        if not student._held:
            raise ValueError(f"{directory}: the manifest holds no skill")
        if manifest["state_file"] != _STATE_FILE:
            raise ValueError(f"{directory}: the manifest names state file {manifest['state_file']!r}, not {_STATE_FILE!r}")
        state_bytes = (directory / _STATE_FILE).read_bytes()
        if hashlib.sha256(state_bytes).hexdigest() != manifest["state_sha256"]:
            raise ValueError(f"{directory}: {_STATE_FILE} does not match the sha256 in the manifest")
        with np.load(io.BytesIO(state_bytes), allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(manifest["state_keys"]):
                raise ValueError(f"{directory}: {_STATE_FILE} arrays differ from the manifest's state keys")
            state = {key: archive[key] for key in manifest["state_keys"]}
        records = manifest["records"]
        if [record["skill"] for record in records] != list(student._held):
            raise ValueError(f"{directory}: the add_skill records do not match the held skills")
        student._state = state
        student._records = _json_copy(records, "add_skill records")
        student._trunk_seed = int(manifest["trunk_init_seed"])
        return student
