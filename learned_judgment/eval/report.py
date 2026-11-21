"""Final results table built only from saved run artifacts.

Lane contract (Person B, evaluation)
------------------------------------
:func:`build_final_table` turns result records into one row per skill x method. A record is
a mapping of one of two kinds:

    run record      {"skill": str, "method": str,
                     "metrics": {metric_name: {"value": float | "undefined", "artifact": path}, ...}}
    not-run record  {"skill": str, "method": str, "blocker": str}

Every numeric cell carries the path of the saved artifact it was computed from, and that
file must exist (relative paths are resolved against ``artifact_root``). A skill x method
pair without a run record is written as ``NOT RUN`` with its blocker -- the record's
blocker, or "no saved result record" when no record exists -- never as a number and never
as PASS. Values are copied from the records unchanged; nothing is computed, rounded or
filled in here.

Table columns: skill, method, status ("RUN" | "NOT RUN"), blocker, then for every metric
name (sorted) the pair ``<metric>`` and ``<metric>__artifact``. A metric absent from a run
record is written as "not reported".
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from learned_judgment.fileio import atomic_write_text

from learned_judgment.eval.metrics import UNDEFINED

__all__ = ["NOT_REPORTED", "NOT_RUN", "build_final_table", "write_final_table"]

NOT_RUN = "NOT RUN"
NOT_REPORTED = "not reported"
_RUN = "RUN"
_NO_RECORD = "no saved result record for this skill and method"
_ARTIFACT_SUFFIX = "__artifact"
_FIXED_COLUMNS = ("skill", "method", "status", "blocker")


def _check_value(where: str, value: Any) -> float | str:
    if isinstance(value, str) and value == UNDEFINED:
        return UNDEFINED
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{where}: value must be a finite number or {UNDEFINED!r}; got {value!r}")
    return value


def _check_artifact(where: str, artifact: Any, artifact_root: Path) -> str:
    if not isinstance(artifact, str) or not artifact:
        raise ValueError(f"{where}: artifact must be a non-empty path string; got {artifact!r}")
    path = Path(artifact)
    resolved = path if path.is_absolute() else artifact_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"{where}: artifact {artifact!r} does not exist (looked at {resolved})")
    return artifact


def build_final_table(
    result_records: Iterable[Mapping[str, Any]],
    *,
    skills: Sequence[str],
    methods: Sequence[str],
    artifact_root: str | Path,
) -> pd.DataFrame:
    """One row per skill x method (``skills`` x ``methods`` order).

    Raises ValueError for a malformed or duplicated record, a skill/method outside the
    given lists, a non-numeric metric value, and FileNotFoundError for a missing artifact.
    """
    if len(set(skills)) != len(skills) or len(set(methods)) != len(methods) or not skills or not methods:
        raise ValueError("skills and methods must be non-empty lists without duplicates")
    root = Path(artifact_root)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    metric_names: set[str] = set()
    for position, record in enumerate(result_records):
        where = f"record {position}"
        skill, method = record.get("skill"), record.get("method")
        if skill not in skills or method not in methods:
            raise ValueError(f"{where}: skill {skill!r} / method {method!r} not in the declared lists")
        where = f"record {position} ({skill}, {method})"
        if (skill, method) in records:
            raise ValueError(f"{where}: duplicated skill x method record")
        extra = set(record) - {"skill", "method", "metrics", "blocker"}
        if extra:
            raise ValueError(f"{where}: unknown keys {sorted(extra)}")
        has_metrics, has_blocker = "metrics" in record, "blocker" in record
        if has_metrics == has_blocker:
            raise ValueError(f"{where}: needs exactly one of 'metrics' (run) or 'blocker' (not run)")
        if has_blocker:
            blocker = record["blocker"]
            if not isinstance(blocker, str) or not blocker.strip():
                raise ValueError(f"{where}: blocker must be a non-empty string")
            records[(skill, method)] = {"blocker": blocker}
            continue
        metrics = record["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"{where}: metrics must be a non-empty mapping")
        cells: dict[str, tuple[float | str, str]] = {}
        for name, cell in metrics.items():
            if not isinstance(name, str) or not name or name in _FIXED_COLUMNS or name.endswith(_ARTIFACT_SUFFIX):
                raise ValueError(f"{where}: invalid metric name {name!r}")
            if not isinstance(cell, Mapping) or set(cell) != {"value", "artifact"}:
                raise ValueError(f"{where}: metric {name!r} must be {{'value': ..., 'artifact': ...}}")
            cells[name] = (
                _check_value(f"{where} metric {name!r}", cell["value"]),
                _check_artifact(f"{where} metric {name!r}", cell["artifact"], root),
            )
        metric_names.update(cells)
        records[(skill, method)] = {"metrics": cells}

    ordered_metrics = sorted(metric_names)
    rows: list[dict[str, Any]] = []
    for skill in skills:
        for method in methods:
            entry = records.get((skill, method))
            row: dict[str, Any] = {"skill": skill, "method": method}
            if entry is None or "blocker" in entry:
                row["status"] = NOT_RUN
                row["blocker"] = _NO_RECORD if entry is None else entry["blocker"]
                for name in ordered_metrics:
                    row[name] = NOT_RUN
                    row[name + _ARTIFACT_SUFFIX] = ""
            else:
                row["status"] = _RUN
                row["blocker"] = ""
                for name in ordered_metrics:
                    value, artifact = entry["metrics"].get(name, (NOT_REPORTED, ""))
                    row[name] = value
                    row[name + _ARTIFACT_SUFFIX] = artifact
            rows.append(row)
    columns = list(_FIXED_COLUMNS) + [column for name in ordered_metrics for column in (name, name + _ARTIFACT_SUFFIX)]
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
