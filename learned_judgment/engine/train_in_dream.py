"""Engine step: train a skill's spiking student inside the dream (generic, no skill branch).

Lane contract (Person B)
------------------------
The physics guess corrected by the PFN is the dream world; a
:class:`~learned_judgment.engine.dream.DreamSet` holds the dreamed objective score of every
candidate command for every dreamed situation. This step trains the skill's spiking student
inside that dream to choose the best candidate
(:meth:`learned_judgment.students.snn.SNNStudent.train_in_dream`).

Every setting comes from config.yaml: ``students.snn.*`` (via ``SNNSettings.from_config``) and
``skills.<skill>.candidate_points_per_dim``, which must be the grid the dream was made with.
Nothing is defaulted; a missing or null key raises ``ConfigError`` naming it. Which dream rows
to train on (for example after ``engine.dream.keep_confident``) is the caller's decision.
"""

from __future__ import annotations

from typing import Any, Mapping

from learned_judgment.config import require
from learned_judgment.engine.dream import DreamSet
from learned_judgment.skills.base import SkillSpec
from learned_judgment.students.snn import SNNStudent
from learned_judgment.students.snn_train import SNNSettings

__all__ = ["train_in_dream"]


def train_in_dream(spec: SkillSpec, dreams: DreamSet, config: Mapping[str, Any], *, base_seed: int) -> SNNStudent:
    """Return ``spec``'s spiking student trained inside ``dreams``.

    Args:
        spec:      the skill's SkillSpec.
        dreams:    DreamSet of ``spec`` with n rows, C candidates per row.
        config:    loaded config.yaml.
        base_seed: run seed; the validation split, network initialisation and batch order use
                   the named streams ``"<skill>/snn/validation_split"``, ``"<skill>/snn/init"``
                   and ``"<skill>/snn/train"`` under it.

    Raises:
        ValueError when config.yaml selects no per-skill spiking student (``students.kind`` is
        not ``"snn"`` or ``students.per_skill_snn`` is not true), plus every error of
        :meth:`SNNStudent.train_in_dream`.
    """
    kind = require(config, "students.kind")
    if kind != "snn":
        raise ValueError(f"config.yaml students.kind is {kind!r}; this step trains the 'snn' student only")
    if require(config, "students.per_skill_snn") is not True:
        raise ValueError("config.yaml students.per_skill_snn is not true: no per-skill spiking student is configured")
    settings = SNNSettings.from_config(config)
    points_per_dim = require(config, f"skills.{spec.name}.candidate_points_per_dim")
    return SNNStudent.train_in_dream(
        spec, dreams, settings=settings, points_per_dim=points_per_dim, base_seed=base_seed
    )
