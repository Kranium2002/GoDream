"""A plain small conv net trained on the spiking student's dream windows and targets, replayed on the same dev trace
through the same gate replay: does the 12 x 12 input with its labels limit the punch dodge, or the spiking network?
(Person B, method check R6.) A measurement only: it never becomes a student and nothing in app/ or the runtime imports it.

Inputs are built exactly as ``observed._train_command`` builds the student's (load_dream, the scores.npz checks,
_newest_frames, dream_targets, groups = dream scenario_index); the dev trace windows exactly as
``dodge_eval.offline_calibration`` builds them, and the readouts go through its ``_replay_grid``.

Command line (repository root)::

    python -m learned_judgment.conv_diagnostic --run RUN --dream DIR --scores FILE --trace FILE

writes <observed.runs_dir>/<RUN>/conv_diagnostic.json. Settings: config.yaml observed.dodge_vision.conv_diagnostic.
Verdict (rule fixed in config before any result): "network" when the chosen row passed the precision target with dev
recall >= network_limit_recall, "input_or_labels" when it passed with recall <= input_limit_recall, else "inconclusive".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from learned_judgment.config import config_digest, load_config, require

_CHUNK_ROWS = 8192  # rows per forward pass in validation and readouts


class _ConvNet(nn.Module):
    def __init__(self, frames: int, size: int, channels: Sequence[int], n_actions: int) -> None:
        super().__init__()
        c1, c2 = (int(c) for c in channels)
        self.body = nn.Sequential(nn.Conv2d(frames, c1, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.threat = nn.Linear(c2 * size * size, 2)
        self.action = nn.Linear(c2 * size * size, n_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x - 0.5)
        return self.threat(h), self.action(h)


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _forward_chunks(model: nn.Module, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        outs = [model(torch.from_numpy(np.ascontiguousarray(x[i:i + _CHUNK_ROWS], dtype=np.float32)))
                for i in range(0, x.shape[0], _CHUNK_ROWS)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def trace_windows(frames: np.ndarray, f: int, first: int) -> np.ndarray:
    """(n, T, N, N) -> (n, T - first, f, N, N): window j ends at decision frame first + j, oldest frame first
    (the layout of dodge_eval.offline_calibration)."""
    if not 1 <= f <= first + 1 or first >= frames.shape[1]:
        raise ValueError(f"a window of {f} frames from decision frame {first} does not fit {frames.shape[1]} frames")
    view = np.lib.stride_tricks.sliding_window_view(frames, f, axis=1)
    return np.moveaxis(view, -1, 2)[:, first - f + 1:]


def train_conv(windows: np.ndarray, targets: Mapping[str, np.ndarray], groups: np.ndarray, *, steps: int, seed: int,
               validation_fraction: float, channels: Sequence[int] = (16, 32), batch: int = 256,
               learning_rate: float = 1e-3) -> dict[str, Any]:
    """Train the conv net on windows (m, F, N, N) with soft targets {"threat": (m, 2), "action": (m, C)}.

    Validation rows are whole groups (scenarios): the first round(validation_fraction * n_groups) of
    np.random.default_rng(seed).permutation(unique groups), so no scenario sits on both sides.
    """
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    windows = np.asarray(windows, dtype=np.float32)
    m, f, size = windows.shape[0], windows.shape[1], windows.shape[2]
    threat_t = np.asarray(targets["threat"], dtype=np.float32)
    action_t = np.asarray(targets["action"], dtype=np.float32)
    groups = np.asarray(groups)
    if threat_t.shape != (m, 2) or action_t.shape[0] != m or groups.shape != (m,):
        raise ValueError(f"targets {threat_t.shape} {action_t.shape} and groups {groups.shape} do not fit {m} windows")
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    validation_groups = unique[:round(validation_fraction * unique.size)]
    is_validation = np.isin(groups, validation_groups)
    train_rows, validation_rows = np.flatnonzero(~is_validation), np.flatnonzero(is_validation)
    if train_rows.size == 0 or validation_rows.size == 0:
        raise ValueError(f"the split left {train_rows.size} training and {validation_rows.size} validation rows")

    model = _ConvNet(f, size, channels, action_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_all, t_all, a_all = torch.from_numpy(windows), torch.from_numpy(threat_t), torch.from_numpy(action_t)
    for _ in range(steps):
        rows = torch.from_numpy(rng.choice(train_rows, size=min(batch, train_rows.size), replace=False))
        model.train()
        threat_logits, action_logits = model(x_all[rows])
        loss = _soft_ce(threat_logits, t_all[rows]) + _soft_ce(action_logits, a_all[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    threat_logits, action_logits = _forward_chunks(model, windows[validation_rows])
    tv, av = t_all[validation_rows], a_all[validation_rows]
    validation = {
        "loss": float(_soft_ce(threat_logits, tv) + _soft_ce(action_logits, av)),
        "threat_accuracy": float((threat_logits.argmax(1) == tv.argmax(1)).float().mean()),
        "action_accuracy": float((action_logits.argmax(1) == av.argmax(1)).float().mean()),
    }
    return {
        "model": model,
        "validation": validation,
        "split": {"train_groups": np.setdiff1d(unique, validation_groups).tolist(),
                  "validation_groups": validation_groups.tolist()},
        "wall_s": time.perf_counter() - started,
    }


def conv_readouts(model: nn.Module, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, Td, F, N, N) -> threat confidence (n, Td) = softmax(threat)[1] and action scores (n, Td, C) = softmax(action),
    both float64, as observed.student_readout reads the spiking student."""
    n, td = windows.shape[:2]
    threat_logits, action_logits = _forward_chunks(model, np.reshape(windows, (n * td, *windows.shape[2:])))
    threat = torch.softmax(threat_logits.double(), dim=1)[:, 1].numpy().reshape(n, td)
    scores = torch.softmax(action_logits.double(), dim=1).numpy().reshape(n, td, -1)
    return threat, scores


def _verdict(row: Mapping[str, Any], passed: bool, settings: Mapping[str, Any]) -> str:
    recall = row["recall"]
    if not passed or recall is None:
        return "inconclusive"  # recall at a precision below the target is not comparable with the student's
    if recall >= float(settings["network_limit_recall"]):
        return "network"
    if recall <= float(settings["input_limit_recall"]):
        return "input_or_labels"
    return "inconclusive"


def diagnose(config: Mapping[str, Any], run_id: str, dream: str | Path, scores: str | Path,
             trace: str | Path) -> dict[str, Any]:
    """Train the conv net on the dream (module docstring), replay it on the dev trace, write and return the report.

    Raises ValueError for a trace that is not dev or does not fit, scores of another dream, or a dream whose action
    order differs from config skills.dodge_vision.actions (the gate's readout order).
    """
    from learned_judgment.dodge_eval import _grid, _load_trace, _replay_grid, _write_json
    from learned_judgment.observed import (_check_meta, _check_spec, _newest_frames, _predicted_mean, _run_dir,
                                           dream_targets, load_dream)
    from learned_judgment.skills.dodge_vision import build_observed_spec
    from learned_judgment.vision import VisionSettings

    settings = require(config, "observed.dodge_vision.conv_diagnostic")
    f = int(settings["frames"])
    seed = int(require(config, "runtime.seed"))
    run_dir = _run_dir(config, run_id)
    data = _load_trace(trace)  # refuse a wrong trace before any training
    if data["meta"].get("split") != "dev":
        raise ValueError(f"{data['path']} is a {data['meta'].get('split')!r} trace; this diagnostic uses dev only")

    spec = build_observed_spec(config)
    checked = _check_spec(spec)
    if list(checked.action_names) != list(require(config, "skills.dodge_vision.actions")):
        raise ValueError(f"spec actions {list(checked.action_names)} are not in the gate's readout order")
    loaded = load_dream(dream)
    _check_meta(checked, loaded["meta"], "dream")
    scores_path = Path(scores)
    with np.load(scores_path, allow_pickle=False) as archive:
        score_rows = archive["scores"]
        for name in ("scenario_index", "decision_time_s"):
            if not np.array_equal(archive[name], loaded[name]):
                raise ValueError(f"{scores_path} {name} differs from the dream's: the scores belong to another dream")
    size = VisionSettings.from_config(config, frames=f).frame_size
    raw = _newest_frames(loaded["features"], config, int(loaded["meta"]["frames"]), f)
    windows = raw.reshape(raw.shape[0], f, size, size).astype(np.float32)
    targets = dream_targets(score_rows, spec, config, predicted_mean=_predicted_mean(scores_path))
    trained = train_conv(windows, targets, loaded["scenario_index"], steps=int(settings["steps"]), seed=seed,
                         validation_fraction=float(require(config, "students.snn.validation_fraction")),
                         channels=settings["channels"], batch=int(settings["batch"]),
                         learning_rate=float(settings["learning_rate"]))

    frames, first = data["frames"], int(data["first_decision_index"])
    if frames.shape[2] != size:
        raise ValueError(f"the trace's frames are {frames.shape[2]} x {frames.shape[2]}; the dream's are {size}")
    threat, action_scores = conv_readouts(trained["model"], trace_windows(frames, f, first))
    t_s = data["t_s"][first:].tolist()
    truth = [bool(x) for x in data["threat"]]
    t_hit = [None if math.isnan(float(h)) else float(h) for h in data["t_hit"]]
    traces = [(threat[i].tolist(), action_scores[i].tolist()) for i in range(frames.shape[0])]
    _, chosen, passed, rule, _ = _replay_grid(config, _grid(config), t_s, traces, truth, t_hit)

    result_path = run_dir / "result.json"
    reference = None
    if result_path.exists():
        reference = (json.loads(result_path.read_text(encoding="utf-8")).get("sweep") or {}).get("calibration")
    report = {
        "run_id": run_id,
        "what": "plain conv net on the student's dream and dev trace: a measurement, never a student",
        "chosen": {key: chosen[key] for key in ("threat", "action", "margin", "emitted", "precision", "recall",
                                                 "false_activations")},
        "gate_passed": passed,
        "rule": rule,
        "n_scenarios": len(truth),
        "n_threats": int(sum(truth)),
        "validation": trained["validation"],
        "split": {key: len(value) for key, value in trained["split"].items()},
        "train_wall_s": trained["wall_s"],
        "reference": reference,
        "verdict": _verdict(chosen, passed, settings),
        "settings": {**dict(settings), "seed": seed, "rows": int(windows.shape[0])},
        "inputs": {"dream": str(loaded["path"]), "scores": str(scores_path.resolve()), "trace": str(data["path"])},
        "config_digest": config_digest(config),
    }
    _write_json(run_dir / "conv_diagnostic.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.conv_diagnostic", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run id: the directory under observed.runs_dir")
    parser.add_argument("--dream", required=True, help="dream file directory")
    parser.add_argument("--scores", required=True, help="scores.npz written by teach")
    parser.add_argument("--trace", required=True, help="trace_dev.npz written by dodge_eval trace")
    args = parser.parse_args(argv)
    report = diagnose(load_config(), args.run, args.dream, args.scores, args.trace)
    keys = ("chosen", "gate_passed", "validation", "train_wall_s", "verdict")
    print(json.dumps({key: report[key] for key in keys}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
