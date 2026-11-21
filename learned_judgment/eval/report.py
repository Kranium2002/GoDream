"""Run-result files and the final results table built only from them.

Lane contract (Person B, evaluation)
------------------------------------
A number enters a results table ONLY from a saved run-result file (review finding: any file
used to be accepted as the "artifact" behind a hand-typed number).

Run-result file (:func:`write_run_result` / :func:`read_run_result`): canonical JSON
(``sort_keys``, compact separators, UTF-8, ``allow_nan=False``, one trailing newline) of::

    format       "learned_judgment.run_result/v1"
    skill, method, run_id, world_preset, split     (split in splits.SPLITS)
    metrics      {metric name: finite number | "undefined"}
    provenance   the caller's JSON-serializable provenance mapping, in which "manifests" is
                 replaced by [{"path", "sha256", "attempt_file_sha256"}, ...]

The provenance must name the harness run manifests (``harness.manifest_path``) the metrics
were computed from. Every manifest must exist, say ``completed: true``, and belong to the same
skill, split and world preset; at least one must be the run of this method and run id. Each
manifest path is stored relative to the run-result file's directory (absolute when that is
impossible) with the sha256 of the manifest and of its attempt file, and reading the result
re-checks both digests, so a changed or deleted run file invalidates the result. A run-result
file is never overwritten.

:func:`build_final_table` takes run-result file paths, the skills, and the canonical method
list ``evaluation.methods`` of config.yaml, and emits one row per skill x method: RUN with
the values and their source file, or NOT RUN with a blocker for every method without a file.
A file that is not a valid run-result file, or whose skill or method is not in the lists, is
an error, not a row. Values are copied unchanged; nothing is computed, rounded or filled in.

Table columns: skill, method, status ("RUN" | "NOT RUN"), blocker, split, world_preset,
run_id, source, then every metric name (sorted). A metric absent from a RUN row's file is
"not reported"; every metric cell of a NOT RUN row is "NOT RUN".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from learned_judgment.config import require
from learned_judgment.eval.harness import read_manifest
from learned_judgment.eval.metrics import UNDEFINED
from learned_judgment.eval.splits import SPLITS
from learned_judgment.fileio import atomic_write_text

__all__ = [
    "NOT_REPORTED",
    "NOT_RUN",
    "RUN_RESULT_FORMAT",
    "build_final_table",
    "read_run_result",
    "write_final_table",
    "write_run_result",
]

NOT_RUN = "NOT RUN"
NOT_REPORTED = "not reported"
RUN_RESULT_FORMAT = "learned_judgment.run_result/v1"
_RUN = "RUN"
_NO_RESULT = "no saved run-result file for this skill and method"
_FIXED_COLUMNS = ("skill", "method", "status", "blocker", "split", "world_preset", "run_id", "source")
_RESULT_KEYS = frozenset({"format", "skill", "method", "run_id", "world_preset", "split", "metrics", "provenance"})
_MANIFEST_ENTRY_KEYS = frozenset({"path", "sha256", "attempt_file_sha256"})


# --------------------------------------------------------------------------------------
# Value checks
# --------------------------------------------------------------------------------------


def _check_text(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string; got {value!r}")
    return value


def _check_value(where: str, value: Any) -> float | int | str:
    if isinstance(value, str) and value == UNDEFINED:
        return UNDEFINED
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{where}: value must be a finite number or {UNDEFINED!r}; got {value!r}")
    return value


def _check_metrics(where: str, metrics: Any) -> dict[str, float | int | str]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"{where}: metrics must be a non-empty mapping")
    checked: dict[str, float | int | str] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name or name in _FIXED_COLUMNS:
            raise ValueError(f"{where}: invalid metric name {name!r}")
        checked[name] = _check_value(f"{where} metric {name!r}", value)
    return checked


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate key {key!r}")
        record[key] = value
    return record


def _stored_path(manifest: Path, result_dir: Path) -> str:
    """Manifest path relative to the result file's directory, or absolute across drives."""
    resolved = manifest.resolve()
    try:
        return Path(os.path.relpath(resolved, result_dir.resolve())).as_posix()
    except ValueError:
        return resolved.as_posix()


def _completed_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    """Read a manifest that must say completed; return it and its attempt file path."""
    if not path.is_file():
        raise FileNotFoundError(f"run manifest {path} does not exist")
    manifest = read_manifest(path)
    if manifest["completed"] is not True:
        raise ValueError(f"run manifest {path} is not completed: a result may only come from a completed run")
    attempt_file = path.parent / manifest["attempt_file"]
    if not attempt_file.is_file():
        raise FileNotFoundError(f"attempt file {attempt_file} of manifest {path} does not exist")
    return manifest, attempt_file


def _check_manifests(
    manifests: Sequence[tuple[dict[str, Any], Path]],
    *,
    skill: str,
    method: str,
    run_id: str,
    world_preset: str,
    split: str,
) -> None:
    for manifest, path in manifests:
        for key, expected in (("skill", skill), ("split", split), ("world_preset", world_preset)):
            if manifest[key] != expected:
                raise ValueError(f"run manifest {path} has {key} {manifest[key]!r}; the result is for {expected!r}")
    if not any(manifest["method"] == method and manifest["run_id"] == run_id for manifest, _ in manifests):
        raise ValueError(f"no listed manifest is the run of method {method!r} with run_id {run_id!r}")


# --------------------------------------------------------------------------------------
# Run-result files
# --------------------------------------------------------------------------------------


def write_run_result(
    path: str | Path,
    *,
    skill: str,
    method: str,
    run_id: str,
    world_preset: str,
    split: str,
    metrics: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> None:
    """Write one run-result file (canonical JSON with the format tag) atomically.

    Args:
        metrics:    {metric name: finite number or "undefined"} computed from the run files.
        provenance: JSON-serializable mapping with "manifests": a non-empty list of harness
                    manifest paths the metrics were computed from (other keys are kept as
                    given, e.g. the command, versions, seeds, the metric function names).

    Raises FileExistsError if ``path`` exists (results are never overwritten), ValueError for
    an invalid field, a manifest that is not completed or not of this skill / split / world
    preset, or no manifest of this method and run id, and FileNotFoundError for a missing
    manifest or attempt file. Nothing is written when any check fails.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists: a run-result file is never overwritten")
    for label, value in (("skill", skill), ("method", method), ("run_id", run_id), ("world_preset", world_preset)):
        _check_text(label, value)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    checked_metrics = _check_metrics("write_run_result", metrics)
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    listed = provenance.get("manifests")
    if not isinstance(listed, (list, tuple)) or not listed:
        raise ValueError("provenance['manifests'] must list the completed run manifests the metrics come from")
    loaded = [_completed_manifest(Path(item)) + (Path(item),) for item in listed]
    _check_manifests(
        [(manifest, manifest_file) for manifest, _, manifest_file in loaded],
        skill=skill,
        method=method,
        run_id=run_id,
        world_preset=world_preset,
        split=split,
    )
    stored = dict(provenance)
    stored["manifests"] = [
        {
            "path": _stored_path(manifest_file, path.parent),
            "sha256": _sha256(manifest_file),
            "attempt_file_sha256": _sha256(attempt_file),
        }
        for _, attempt_file, manifest_file in loaded
    ]
    record = {
        "format": RUN_RESULT_FORMAT,
        "skill": skill,
        "method": method,
        "run_id": run_id,
        "world_preset": world_preset,
        "split": split,
        "metrics": checked_metrics,
        "provenance": stored,
    }
    try:
        text = _canonical(record)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the run result is not JSON-serializable: {error}") from error
    atomic_write_text(path, text)


def read_run_result(path: str | Path) -> dict[str, Any]:
    """Read and verify a run-result file; returns its record as a dict.

    Raises ValueError when the file is not a run-result file (format tag, exact keys, types,
    metric values) or when a referenced manifest is not completed or does not match the
    result, and when the stored sha256 of a manifest or of its attempt file no longer
    matches the file on disk; FileNotFoundError when one of them is missing.
    """
    path = Path(path)
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not a run-result file: {error}") from error
    if not isinstance(record, dict) or record.get("format") != RUN_RESULT_FORMAT:
        raise ValueError(f"{path} is not a run-result file (format tag {RUN_RESULT_FORMAT!r} missing)")
    if set(record) != _RESULT_KEYS:
        raise ValueError(
            f"{path}: run-result keys do not match; missing={sorted(_RESULT_KEYS - set(record))} "
            f"extra={sorted(set(record) - _RESULT_KEYS)}"
        )
    where = str(path)
    for key in ("skill", "method", "run_id", "world_preset"):
        _check_text(f"{where}: {key}", record[key])
    if record["split"] not in SPLITS:
        raise ValueError(f"{where}: split must be one of {SPLITS}; got {record['split']!r}")
    _check_metrics(where, record["metrics"])
    provenance = record["provenance"]
    if not isinstance(provenance, dict) or not isinstance(provenance.get("manifests"), list) or not provenance["manifests"]:
        raise ValueError(f"{where}: provenance must list its run manifests")
    loaded: list[tuple[dict[str, Any], Path]] = []
    for entry in provenance["manifests"]:
        if not isinstance(entry, dict) or set(entry) != _MANIFEST_ENTRY_KEYS:
            raise ValueError(f"{where}: each provenance manifest must be {sorted(_MANIFEST_ENTRY_KEYS)}")
        manifest_file = Path(entry["path"])
        if not manifest_file.is_absolute():
            manifest_file = path.parent / manifest_file
        manifest, attempt_file = _completed_manifest(manifest_file)
        if _sha256(manifest_file) != entry["sha256"]:
            raise ValueError(f"{where}: manifest {manifest_file} changed after the result was written")
        if _sha256(attempt_file) != entry["attempt_file_sha256"]:
            raise ValueError(f"{where}: attempt file {attempt_file} changed after the result was written")
        loaded.append((manifest, manifest_file))
    _check_manifests(
        loaded,
        skill=record["skill"],
        method=record["method"],
        run_id=record["run_id"],
        world_preset=record["world_preset"],
        split=record["split"],
    )
    return record


# --------------------------------------------------------------------------------------
# Final table
# --------------------------------------------------------------------------------------


def build_final_table(
    run_result_paths: Iterable[str | Path],
    *,
    skills: Sequence[str],
    config: Mapping[str, Any],
    blockers: Mapping[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """One row per skill x method (``skills`` order x ``evaluation.methods`` order of config).

    Args:
        run_result_paths: run-result files (:func:`write_run_result`); each is read and
                          verified with :func:`read_run_result`.
        skills:           skills of the table, non-empty and without duplicates.
        config:           the loaded config.yaml; the method list is
                          ``require(config, "evaluation.methods")``, so no method can be left out.
        blockers:         optional {(skill, method): text} explaining a NOT RUN row; a pair
                          without a blocker reads "no saved run-result file for this skill and
                          method". A blocker for a pair that has a result file is an error.

    Raises ValueError for a file that is not a valid run-result file, a skill or method
    outside the lists, two files for the same skill x method, or an invalid blocker.
    """
    methods = require(config, "evaluation.methods")
    if (
        not isinstance(methods, (list, tuple))
        or not methods
        or len(set(methods)) != len(methods)
        or not all(isinstance(method, str) and method for method in methods)
    ):
        raise ValueError("config.yaml evaluation.methods must be a non-empty list of unique method names")
    skills = list(skills)
    if not skills or len(set(skills)) != len(skills) or not all(isinstance(skill, str) and skill for skill in skills):
        raise ValueError("skills must be a non-empty list of unique skill names")
    results: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    metric_names: set[str] = set()
    for item in run_result_paths:
        record = read_run_result(item)
        skill, method = record["skill"], record["method"]
        if skill not in skills or method not in methods:
            raise ValueError(f"{item}: skill {skill!r} / method {method!r} is not in the table's skills / config methods")
        if (skill, method) in results:
            raise ValueError(f"{item}: a second run-result file for ({skill}, {method}); first: {results[(skill, method)][1]}")
        results[(skill, method)] = (record, str(item))
        metric_names.update(record["metrics"])
    given_blockers = dict(blockers or {})
    for key, text in given_blockers.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] in skills and key[1] in methods):
            raise ValueError(f"blocker key {key!r} is not a (skill, method) pair of the table")
        if key in results:
            raise ValueError(f"blocker given for {key!r}, which has a run-result file")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"blocker for {key!r} must be a non-empty string")

    ordered_metrics = sorted(metric_names)
    rows: list[dict[str, Any]] = []
    for skill in skills:
        for method in methods:
            row: dict[str, Any] = {"skill": skill, "method": method}
            entry = results.get((skill, method))
            if entry is None:
                row.update(
                    status=NOT_RUN,
                    blocker=given_blockers.get((skill, method), _NO_RESULT),
                    split="",
                    world_preset="",
                    run_id="",
                    source="",
                )
                row.update({name: NOT_RUN for name in ordered_metrics})
            else:
                record, source = entry
                row.update(
                    status=_RUN,
                    blocker="",
                    split=record["split"],
                    world_preset=record["world_preset"],
                    run_id=record["run_id"],
                    source=source,
                )
                row.update({name: record["metrics"].get(name, NOT_REPORTED) for name in ordered_metrics})
            rows.append(row)
    columns = list(_FIXED_COLUMNS) + ordered_metrics
    return pd.DataFrame(rows, columns=columns, dtype=object)


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        text = repr(value)
    else:
        text = str(value)
    return text.replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def write_final_table(table: pd.DataFrame, csv_path: str | Path, md_path: str | Path) -> None:
    """Write ``table`` (from :func:`build_final_table`) as CSV and as a Markdown table.

    Floats are written at full precision (Python repr); each file is written to a temporary
    file in the same directory and moved into place with ``os.replace``.
    """
    missing = [column for column in _FIXED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"table is missing columns {missing}; build it with build_final_table")
    atomic_write_text(Path(csv_path), table.to_csv(index=False, lineterminator="\n"))
    header = "| " + " | ".join(_markdown_cell(column) for column in table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    atomic_write_text(Path(md_path), "\n".join([header, divider, *body]) + "\n")
