"""PLUMBING FIXTURE for the B_pipeline_offline checks only.

Not a project skill, not the Go2, not the simulator. Every check that imports this module is a
PLUMBING CHECK: it exercises the file, seed, state and stop mechanics of
learned_judgment.pipeline. Its numbers are not evidence about the robot, the simulator, TabPFN
or any model, and none of them may enter a report or results/ (every output goes to a
temporary directory).

    FixturePipelineSpec   the B_snn_offline fixture spec (experiments/proofs/B_snn_offline/
                          fixture_snn_spec.py, FixtureSnnSpec) bound to the verified command field
                          vel_y, with an episode() that drives a world through reset / step /
                          finish_episode.
    FixturePlumbingWorld  NOT a simulator: a few lines of arithmetic with reset / step /
                          finish_episode. Its "hidden world" is a lateral gain drawn from the world
                          seed; each episode's noise comes from (world seed, attempt seed) only, so
                          an episode does not depend on what ran before (the two-level seed
                          contract of Go2World).
    make_config           a copy of config.yaml with every output directory, the teacher cache and
                          call log, and the test-access log redirected into a temporary directory.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "experiments" / "proofs" / "B_snn_offline"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fixture_snn_spec import FixtureSnnSpec  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402

BANNER = (
    "PLUMBING CHECK on a fixture world - mechanics of learned_judgment.pipeline only; not evidence about the robot, "
    "the simulator, TabPFN or any model"
)
SKILL = "fixture_pipeline"
POINTS_PER_DIM = [7, 5]  # fixture grid: 7 side speeds x 5 delays = 35 candidates
BASE_SEED = 20261121
PRESET = "default"
APPROACH_DISTANCE_M = 3.0  # fixture: the object arrives after 3 m / fx_speed seconds


class FixturePipelineSpec(FixtureSnnSpec):
    """FIXTURE: FixtureSnnSpec bound to vel_y, whose episode runs a world.

    episode: reset(situation, attempt_seed); hold all commands at 0 for the delay (when > 0);
    command the side speed on vel_y until the object arrives (at least one 0.02 s period);
    finish_episode() returns the outcome keyed by outcome name.
    """

    def __init__(self, name: str = SKILL) -> None:
        super().__init__(name=name, command_field="vel_y")

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int) -> Mapping[str, float]:
        world.reset(situation, attempt_seed)
        delay = float(action["fx_delay"])
        arrival = APPROACH_DISTANCE_M / float(situation["fx_speed"])
        if delay > 0.0:
            world.step({"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}, delay)
        world.step({"vel_x": 0.0, "vel_y": float(action["fx_side_speed"]), "yaw_rate": 0.0}, max(arrival - delay, 0.02))
        return world.finish_episode()


class FixturePlumbingWorld:
    """FIXTURE world (NOT a simulator). reset / step / finish_episode with two levels of seed."""

    description = "PLUMBING FIXTURE world (experiments/proofs/B_pipeline_offline/fixture_pipeline.py); not the Go2 simulator"

    def __init__(self, world_seed: int, world_preset: str) -> None:
        self.world_seed = int(world_seed)
        self.world_preset = world_preset
        hidden = np.random.default_rng([self.world_seed, 0])
        self.hidden_gain = float(hidden.uniform(0.7, 1.0))  # fixture hidden effect: side-step efficiency
        self.hidden_fall_bias = float(hidden.uniform(0.0, 0.05))
        self.episodes_started = 0
        self._active = False

    def reset(self, situation: Mapping[str, Any], attempt_seed: int) -> dict[str, Any]:
        self._situation = dict(situation)
        self._rng = np.random.default_rng([self.world_seed, int(attempt_seed), 1])
        self._time = 0.0
        self._displacement = 0.0
        self._effort = 0.0
        self._arrival = APPROACH_DISTANCE_M / float(situation["fx_speed"])
        self._active = True
        self.episodes_started += 1
        return {"t_s": 0.0}

    def step(self, command: Mapping[str, float], duration_s: float) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("no episode in progress")
        if set(command) != {"vel_x", "vel_y", "yaw_rate"}:
            raise ValueError(f"fixture world expects vel_x, vel_y, yaw_rate; got {sorted(command)}")
        grip = 0.6 if self._situation["fx_surface"] == "slip" else 1.0
        moving = max(0.0, min(self._time + float(duration_s), self._arrival) - self._time)
        self._displacement += float(command["vel_y"]) * moving * grip * self.hidden_gain
        self._effort += abs(float(command["vel_y"])) * moving
        self._time += float(duration_s)
        return {"t_s": self._time}

    def finish_episode(self) -> dict[str, float]:
        if not self._active:
            raise RuntimeError("no episode in progress")
        self._active = False
        lateral = float(self._situation["fx_lateral"])
        separation = abs(lateral - self._displacement + 0.02 * float(self._rng.normal()))
        slip = self._situation["fx_surface"] == "slip"
        side = self._effort / max(self._arrival, 1e-9)
        p_fall = min(1.0, 0.02 + self.hidden_fall_bias + (1.2 if slip else 0.3) * side**2)
        fall = 1.0 if float(self._rng.random()) < p_fall else 0.0
        return {"fx_separation": separation, "fx_fall": fall, "fx_effort": self._effort}


def fixture_world_factory(world_seed: int, world_preset: str) -> FixturePlumbingWorld:
    """FIXTURE factory with the signature of pipeline.go2_world_factory's result."""
    return FixturePlumbingWorld(world_seed, world_preset)


fixture_world_factory.description = FixturePlumbingWorld.description  # type: ignore[attr-defined]


def make_config(tmp: Path, *, max_steps: int = 300, budget_none: bool = False) -> dict[str, Any]:
    """config.yaml copy with every output under ``tmp`` and the fixture skill's grid.

    ``max_steps`` shortens students.snn.max_steps for a short mechanics run (printed by the
    checks). ``budget_none`` sets tabpfn.project_token_budget to null, so the teacher refuses
    every uncached request (the refusal tripwire pattern of B_teacher_offline/c5).
    """
    config = copy.deepcopy(load_config())
    tmp = Path(tmp)
    config["engine"]["attempts_dir"] = (tmp / "attempts").as_posix()
    config["engine"]["dreams_dir"] = (tmp / "dreams").as_posix()
    config["pipeline"]["manifests_dir"] = (tmp / "manifests").as_posix()
    config["pipeline"]["students_dir"] = (tmp / "students").as_posix()
    config["pipeline"]["evaluation_rows_dir"] = (tmp / "evaluation_rows").as_posix()
    config["pipeline"]["run_results_dir"] = (tmp / "run_results").as_posix()
    config["tabpfn"]["cache_dir"] = (tmp / "tabpfn_cache").as_posix()
    config["tabpfn"]["call_log"] = (tmp / "tabpfn_calls.jsonl").as_posix()
    config["evaluation"]["test_access_log"] = (tmp / "test_access_log.jsonl").as_posix()
    config["skills"][SKILL] = {"candidate_points_per_dim": list(POINTS_PER_DIM)}
    config["students"]["snn"]["max_steps"] = int(max_steps)
    if budget_none:
        config["tabpfn"]["project_token_budget"] = None
    return config


def write_config(config: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return path


def fixed_clock() -> str:
    """A fixed UTC time for fixture files (no real run time is claimed)."""
    return "2026-11-21T00:00:00+00:00"


def open_fixture_run(config: Mapping[str, Any], *, run_id: str, spec: FixturePipelineSpec | None = None,
                     world_preset: str = PRESET, base_seed: int = BASE_SEED, clock: Callable[[], str] = fixed_clock,
                     progress: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] | None = None):
    from learned_judgment.pipeline import SkillRun

    return SkillRun(
        config,
        spec if spec is not None else FixturePipelineSpec(),
        world_factory=fixture_world_factory,
        run_id=run_id,
        world_preset=world_preset,
        base_seed=base_seed,
        clock=clock,
        progress=progress,
        should_stop=should_stop,
    )


def fixture_opener(config: Mapping[str, Any], skill_name: str, **kwargs: Any):
    """pipeline.main opener for the CLI plumbing check: only the fixture skill, on the fixture world."""
    if skill_name != SKILL:
        raise KeyError(f"the plumbing opener builds only {SKILL!r}; got {skill_name!r}")
    return open_fixture_run(config, **kwargs)


class Checks:
    """Counts executed assertions of a check script and sets its exit code."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []
        self.redactions: dict[str, str] = {}  # e.g. {temporary directory: "<tmp>"} for printed labels

    def redact(self, text: str) -> str:
        for original, replacement in self.redactions.items():
            text = text.replace(original, replacement).replace(original.replace("\\", "/"), replacement)
        return text

    def expect(self, condition: bool, label: str) -> None:
        label = self.redact(label)
        print(("  [ok] " if condition else "  [FAILED] ") + label, flush=True)
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)

    def raises(self, label: str, error: type[BaseException], function: Callable[[], Any], needle: str = "") -> BaseException | None:
        try:
            function()
        except error as caught:
            message = str(caught).replace("\n", " ")
            self.expect(needle in message, f"{label} -> {type(caught).__name__}: {message[:240]}")
            return caught
        except Exception as other:  # noqa: BLE001
            self.expect(False, f"{label} raised {type(other).__name__} instead of {error.__name__}: {other}")
            return None
        self.expect(False, f"{label} did not raise {error.__name__}")
        return None

    def finish(self) -> None:
        print(f"\nCHECK RESULT (plumbing): {self.passed} assertions held, {len(self.failures)} failed {self.failures or ''}")
        raise SystemExit(0 if not self.failures else 1)
