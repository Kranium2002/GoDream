"""Disk persistence of the TabPFN teacher: canonical hashing, request/response cache, call log.

Lane contract (teacher lane, Person B). This module never touches the network.

Hashing
    Process-independent sha256 inputs following the recipe executed in
    experiments/proofs/B_students_eval_contract/p09_hash.py: an array is hashed as a canonical
    JSON header {tag, little-endian dtype, shape}, a NUL byte, its C-order little-endian bytes
    with -0.0 folded to +0.0 and every NaN folded to one bit pattern, and a 0x01 byte.
    Metadata is canonical JSON (sorted keys, compact separators, no NaN/inf). The builtin
    ``hash()`` is never used, so keys do not depend on PYTHONHASHSEED.

RequestCache
    One entry per request key in one directory:

        <key>.npz   the complete request AND response arrays (``np.savez``; read back with
                    ``np.load(..., allow_pickle=False)``, bitwise exact, proof p10_persist)
        <key>.json  canonical JSON metadata plus the sha256 of the .npz bytes

    Each file is written atomically: a temp file in the same directory, flush, fsync, then
    ``os.replace``, retried briefly on ``PermissionError`` because Windows refuses to replace
    a file another handle holds open (p10_persist). The .json is written last, so its presence
    marks a complete entry; :meth:`RequestCache.get` verifies the .npz digest it records.

CallLog
    Append-only JSON Lines, one canonical JSON object per line, one writer per file
    (unlocked concurrent appends lost lines on Windows in p10_persist). Every record is
    stamped ``timestamp_utc`` from an injected clock. A key that names a credential is
    rejected, so a token or header can never be written.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from learned_judgment.fileio import atomic_write_bytes

__all__ = [
    "DISPATCH_FLAG",
    "ENTRY_FORMAT",
    "QUOTE_FIELD",
    "CacheCorruptError",
    "CachedEntry",
    "CallLog",
    "RequestCache",
    "canonical_array",
    "canonical_json",
    "parse_json",
    "update_digest",
    "utc_timestamp",
]

ENTRY_FORMAT = "learned-judgment/teacher-cache-entry/v1"

# Call-log fields read back by CallLog.tokens_quoted_for_billed_calls().
DISPATCH_FLAG = "billable_dispatch"  # True only on the record written just before a billable call
QUOTE_FIELD = "quote_tokens"  # the provider's quote for that request, in tokens

_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_GROUPS = ("request", "response")

_CREDENTIAL_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "headers",
        "password",
        "prior_labs_api_key",
        "secret",
        "tabpfn_token",
        "token",
    }
)


class CacheCorruptError(RuntimeError):
    """A cache entry on disk is incomplete, altered, or does not match the request it is keyed by."""


# --------------------------------------------------------------------------------------
# Canonical JSON and hashing
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not canonical-JSON serializable")


def canonical_json(value: Any) -> str:
    """Canonical JSON text: sorted keys, compact separators, UTF-8, NaN/inf rejected (ValueError)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name!r} is not allowed")


def parse_json(text: str) -> Any:
    """Strict inverse of :func:`canonical_json`: ``NaN`` / ``Infinity`` raise ValueError."""
    return json.loads(text, parse_constant=_reject_constant)


def canonical_array(array: np.ndarray) -> np.ndarray:
    """C-contiguous little-endian copy with -0.0 -> +0.0 and one NaN bit pattern.

    Accepts bool, signed/unsigned integer and float arrays of any shape; anything else
    (object, string, datetime) raises TypeError.
    """
    array = np.asarray(array)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"cannot hash dtype {array.dtype}: only bool, integer and float arrays are allowed")
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    if array.dtype.kind == "f":
        array = array + array.dtype.type(0.0)  # -0.0 + 0.0 == +0.0 under round-to-nearest
        array = np.ascontiguousarray(np.where(np.isnan(array), array.dtype.type(np.nan), array))
    return array


def update_digest(digest: Any, array: np.ndarray, tag: str) -> None:
    """Feed one array into a ``hashlib`` digest: header {tag, dtype, shape}, NUL, bytes, 0x01."""
    array = canonical_array(array)
    header = canonical_json({"tag": tag, "dtype": array.dtype.str, "shape": list(array.shape)})
    digest.update(header.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    digest.update(b"\x01")


def utc_timestamp(clock: Callable[[], datetime]) -> str:
    """ISO 8601 text of ``clock()``, which must return a timezone-aware UTC datetime."""
    now = clock()
    if not isinstance(now, datetime) or now.utcoffset() != timedelta(0):
        raise ValueError(f"the clock must return a timezone-aware UTC datetime; got {now!r}")
    return now.isoformat()


# --------------------------------------------------------------------------------------
# Request / response cache
# --------------------------------------------------------------------------------------


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
        raise ValueError(f"a cache key must be 64 lowercase hex characters (sha256); got {key!r}")
    return key


def _check_arrays(group: str, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    checked: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{group} array name {name!r} must be a Python identifier")
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"{group} array {name!r} has dtype {array.dtype}; only numeric arrays are cached")
        checked[name] = array
    return checked


@dataclass(frozen=True)
class CachedEntry:
    """One stored provider request and its response.

    Attributes:
        key:             the request key (sha256 hex).
        request_arrays:  the arrays exactly as sent, e.g. x_train (n, F), y_train (n,), x_query (m, F).
        request_meta:    JSON metadata of the request.
        response_arrays: the validated response arrays, e.g. mean (m,), quantiles (m, Q).
        response_meta:   JSON metadata of the response (quote, usage, timings, timestamp, ...).
    """

    key: str
    request_arrays: dict[str, np.ndarray]
    request_meta: dict[str, Any]
    response_arrays: dict[str, np.ndarray]
    response_meta: dict[str, Any]


class RequestCache:
    """Persistent request/response cache, one ``<key>.npz`` + ``<key>.json`` pair per entry."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def paths(self, key: str) -> tuple[Path, Path]:
        """(``<key>.npz``, ``<key>.json``) inside the cache directory."""
        _check_key(key)
        return self.directory / f"{key}.npz", self.directory / f"{key}.json"

    def get(self, key: str) -> CachedEntry | None:
        """The stored entry, or None when ``<key>.json`` does not exist.

        Raises CacheCorruptError when the metadata names another key or format, the .npz is
        missing, its sha256 differs from the recorded one, or its members differ from the
        recorded array names.
        """
        npz_path, json_path = self.paths(key)
        try:
            text = json_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        document = parse_json(text)
        if not isinstance(document, dict) or document.get("format") != ENTRY_FORMAT or document.get("key") != key:
            raise CacheCorruptError(f"{json_path} is not a {ENTRY_FORMAT} entry for key {key}")
        try:
            payload = npz_path.read_bytes()
        except FileNotFoundError:
            raise CacheCorruptError(f"{json_path} exists but {npz_path} is missing") from None
        if hashlib.sha256(payload).hexdigest() != document.get("npz_sha256"):
            raise CacheCorruptError(f"{npz_path} does not match the sha256 recorded in {json_path}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            loaded = {name: archive[name] for name in archive.files}
        names = {group: list(document.get(f"{group}_arrays", [])) for group in _GROUPS}
        expected = {f"{group}__{name}" for group in _GROUPS for name in names[group]}
        if set(loaded) != expected:
            raise CacheCorruptError(f"{npz_path} holds {sorted(loaded)}, expected {sorted(expected)}")
        return CachedEntry(
            key=key,
            request_arrays={name: loaded[f"request__{name}"] for name in names["request"]},
            request_meta=document["request"],
            response_arrays={name: loaded[f"response__{name}"] for name in names["response"]},
            response_meta=document["response"],
        )

    def put(
        self,
        key: str,
        *,
        request_arrays: dict[str, np.ndarray],
        request_meta: dict[str, Any],
        response_arrays: dict[str, np.ndarray],
        response_meta: dict[str, Any],
    ) -> None:
        """Write one complete entry atomically (.npz first, then .json).

        Array names must be identifiers and arrays numeric. Metadata must be strict JSON
        (non-finite floats raise ValueError before any file is written). Putting an existing
        key replaces the entry with a complete new one.
        """
        _check_key(key)
        groups = {"request": _check_arrays("request", request_arrays), "response": _check_arrays("response", response_arrays)}
        members = {f"{group}__{name}": array for group, arrays in groups.items() for name, array in arrays.items()}
        buffer = io.BytesIO()
        np.savez(buffer, **members)
        payload = buffer.getvalue()
        document = {
            "format": ENTRY_FORMAT,
            "key": key,
            "npz_sha256": hashlib.sha256(payload).hexdigest(),
            "request_arrays": list(groups["request"]),
            "response_arrays": list(groups["response"]),
            "array_specs": {
                name: {"dtype": array.dtype.str, "shape": list(array.shape)} for name, array in members.items()
            },
            "request": request_meta,
            "response": response_meta,
        }
        text = canonical_json(document)
        npz_path, json_path = self.paths(key)
        atomic_write_bytes(npz_path, payload)
        atomic_write_bytes(json_path, text.encode("utf-8"))


# --------------------------------------------------------------------------------------
# Call log
# --------------------------------------------------------------------------------------


def _reject_credentials(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _CREDENTIAL_KEYS:
                raise ValueError(f"call-log field {where}.{key} names a credential; credentials are never logged")
            _reject_credentials(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_credentials(item, f"{where}[{index}]")


class CallLog:
    """Append-only JSON Lines log of provider calls, refusals and cache hits (single writer)."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None:
        self.path = Path(path)
        self._clock = clock

    def append(self, record: dict[str, Any]) -> None:
        """Append ``record`` plus ``timestamp_utc`` as one canonical JSON line, then fsync.

        Raises ValueError when the record already has ``timestamp_utc``, contains a
        credential-named key at any depth, or holds a non-finite float.
        """
        if not isinstance(record, Mapping):
            raise TypeError(f"a call-log record must be a mapping; got {type(record).__name__}")
        if "timestamp_utc" in record:
            raise ValueError("timestamp_utc is set by the call log from its clock")
        _reject_credentials(record, "record")
        line = canonical_json({**record, "timestamp_utc": utc_timestamp(self._clock)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self) -> list[dict[str, Any]]:
        """All records in order; [] when the file does not exist. A malformed line raises ValueError."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        records: list[dict[str, Any]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            try:
                record = parse_json(line)
            except ValueError as exc:
                raise ValueError(f"{self.path}:{number} is not a valid call-log line: {exc}") from None
            if not isinstance(record, dict):
                raise ValueError(f"{self.path}:{number} is not a JSON object")
            records.append(record)
        return records

    def tokens_quoted_for_billed_calls(self) -> int:
        """Sum of ``quote_tokens`` over records with ``billable_dispatch`` true.

        The teacher writes exactly one such record per request it hands to the provider, before
        the call, so a crash between the call and its result record cannot hide spent tokens.
        """
        total = 0
        for number, record in enumerate(self.read(), start=1):
            if record.get(DISPATCH_FLAG) is not True:
                continue
            quote = record.get(QUOTE_FIELD)
            if isinstance(quote, bool) or not isinstance(quote, int) or quote < 0:
                raise ValueError(f"{self.path} record {number}: a dispatch record needs a non-negative integer {QUOTE_FIELD}")
            total += quote
        return total
