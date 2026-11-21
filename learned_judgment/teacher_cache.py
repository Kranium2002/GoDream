"""Disk persistence of the TabPFN teacher: canonical hashing, request/response cache, lock, call log.

Lane contract (teacher lane, Person B). This module never touches the network.

Hashing
    Process-independent sha256 inputs following the recipe executed in
    experiments/proofs/B_students_eval_contract/p09_hash.py: an array is hashed as a canonical
    JSON header {tag, little-endian dtype, shape}, a NUL byte, its C-order little-endian bytes
    with -0.0 folded to +0.0 and every NaN folded to one bit pattern, and a 0x01 byte.
    Metadata is canonical JSON (sorted keys, compact separators, no NaN/inf). The builtin
    ``hash()`` is never used, so keys do not depend on PYTHONHASHSEED.

RequestCache
    One AUTHORITATIVE file per request key, written with ONE atomic write (temp file in the same
    directory, flush, fsync, ``os.replace`` retried on ``PermissionError``; learned_judgment.fileio),
    so an entry is either complete or absent:

        <key>.npz   np.savez archive (read back with ``allow_pickle=False``, bitwise exact) holding
                    request__<name>   the request arrays exactly as sent,
                    response__<name>  the validated response arrays,
                    meta__json        uint8 (B,) canonical-JSON metadata: format, key, array
                                      names and specs, the sha256 of every other member, and the
                                      request / response metadata.
        <key>.json  human-readable copy of the metadata, written AFTER the .npz. It is never read
                    by any code; deleting or editing it changes nothing.

    unvalidated/<key>.npz holds a provider response exactly as received (numeric values as
    arrays with their own dtype, anything else as JSON text), written BEFORE the response is
    validated, together with the request it answers and the quote/usage/provider metadata. It is
    deleted only after the validated <key>.npz has been written. An operator who abandons such a
    response moves it to unvalidated/abandoned/<key>.<UTC stamp>.npz; it is never deleted.

CacheLock
    Exclusive inter-process lock: the file <cache_dir>/.lock created with
    ``os.open(O_CREAT | O_EXCL)`` and holding the owner's pid and UTC time. A waiting process
    polls until its timeout and then raises :class:`CacheLockedError`; a lock is never broken
    automatically. The lock is re-entrant for the thread that holds it (also across CacheLock
    objects naming the same file), so the teacher can append to the call log while it holds the
    lock for a whole uncached request.

CallLog
    Append-only JSON Lines, one canonical JSON object per line written with ``ensure_ascii``, so
    no character of a record (U+2028, U+2029, U+0085, ...) can act as a line break; the text is
    split on "\\n" only. Every append and read holds the CacheLock, so concurrent processes
    neither lose nor interleave lines. Every record is stamped ``timestamp_utc`` from an
    injected clock. A key that names a credential is rejected, so a token or header can never
    be written.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.fileio import atomic_write_bytes

__all__ = [
    "DISPATCH_FLAG",
    "ENTRY_FORMAT",
    "QUOTE_FIELD",
    "RESOLVING_EVENTS",
    "UNVALIDATED_FORMAT",
    "CacheCorruptError",
    "CacheLock",
    "CacheLockedError",
    "CachedEntry",
    "CallLog",
    "RequestCache",
    "UnvalidatedResponse",
    "canonical_array",
    "canonical_json",
    "dispatch_times",
    "parse_json",
    "parse_log_text",
    "tokens_quoted",
    "unresolved_dispatches",
    "update_digest",
    "utc_timestamp",
]

ENTRY_FORMAT = "learned-judgment/teacher-cache-entry/v2"
UNVALIDATED_FORMAT = "learned-judgment/teacher-unvalidated-response/v1"

# Call-log fields read back by tokens_quoted() and unresolved_dispatches().
DISPATCH_FLAG = "billable_dispatch"  # True only on the record written just before a billable call
QUOTE_FIELD = "quote_tokens"  # the provider's quote for that request, in tokens
RESOLVING_EVENTS = ("completed", "abandoned")  # events that close an earlier dispatch of the same key

_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_ENTRY_NAME = re.compile(r"[0-9a-f]{64}\.")  # <key>.npz, <key>.json, <key>.<stamp>.npz
_GROUPS = ("request", "response")
_META_MEMBER = "meta__json"
_UNVALIDATED_DIR = "unvalidated"
_ABANDONED_DIR = "abandoned"
_LOCK_NAME = ".lock"

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
    """A cache file on disk is incomplete, altered, or does not match the request it is keyed by."""


class CacheLockedError(RuntimeError):
    """The cache lock stayed held by another process (or thread) for longer than the timeout."""


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _ascii_json(value: Any, *, indent: int | None = None) -> str:
    """Canonical JSON restricted to ASCII (every non-ASCII character escaped as \\uXXXX)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else (",", ": "),
        indent=indent,
        ensure_ascii=True,
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


def _array_sha256(array: np.ndarray) -> str:
    """sha256 of an array exactly as stored: {dtype, shape} header, NUL, logical C-order bytes."""
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Single-file archives (cache entries and unvalidated responses)
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


def _pack(members: dict[str, np.ndarray], document: dict[str, Any]) -> bytes:
    """One np.savez archive of ``members`` plus ``document`` (with every member's sha256) as meta__json.

    Raises ValueError (non-finite metadata) or TypeError before anything is written.
    """
    if _META_MEMBER in members:
        raise ValueError(f"member name {_META_MEMBER!r} is reserved for the metadata")
    document = {**document, "array_sha256": {name: _array_sha256(array) for name, array in members.items()}}
    meta = np.frombuffer(canonical_json(document).encode("utf-8"), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **members, **{_META_MEMBER: meta})
    return buffer.getvalue()


def _unpack(path: Path, expected_format: str, key: str) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """(members without meta__json, metadata document) of one archive; None when ``path`` does not exist.

    Raises CacheCorruptError when the archive cannot be read, its metadata is missing or names
    another format or key, or a member is missing, extra or differs from its recorded sha256.
    """
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            loaded = {name: archive[name] for name in archive.files}
    except Exception as exc:  # zipfile.BadZipFile, ValueError, OSError, EOFError, ...: all mean "damaged"
        raise CacheCorruptError(f"{path} is not a readable archive: {type(exc).__name__}: {exc}") from None
    meta = loaded.pop(_META_MEMBER, None)
    if meta is None or meta.dtype != np.uint8 or meta.ndim != 1:
        raise CacheCorruptError(f"{path} has no {_META_MEMBER} metadata member")
    try:
        document = parse_json(meta.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CacheCorruptError(f"{path} metadata is not strict JSON: {exc}") from None
    if not isinstance(document, dict) or document.get("format") != expected_format or document.get("key") != key:
        raise CacheCorruptError(f"{path} is not a {expected_format} archive for key {key}")
    digests = document.get("array_sha256")
    if not isinstance(digests, dict) or set(digests) != set(loaded):
        raise CacheCorruptError(f"{path} holds members {sorted(loaded)}, its metadata records {sorted(digests or {})}")
    for name, array in loaded.items():
        if _array_sha256(array) != digests[name]:
            raise CacheCorruptError(f"{path} member {name!r} differs from the sha256 recorded in its metadata")
    return loaded, document


def _jsonable(value: Any) -> Any:
    """Lossless-as-possible strict-JSON form of a raw provider value (inverse: :func:`_from_jsonable`).

    Non-finite floats become {"non_finite": repr}; numpy arrays become nested lists; mappings keep
    string keys; any other object becomes {"unrepresentable": repr, "type": name}.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"non_finite": repr(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return {"unrepresentable": repr(value), "type": type(value).__name__}


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"non_finite"}:
            return float(value["non_finite"])
        return {key: _from_jsonable(item) for key, item in value.items()}
    return value


def _encode_raw(raw: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Members raw__<i> and an index describing a raw provider response exactly as received.

    A mapping is stored item by item; any other value as one item. An item that converts to a
    numeric numpy array (bool, int, float of any dtype) is stored as that array, unchanged;
    anything else (None inside a list, ragged lists, strings, objects) as UTF-8 JSON text.
    This function never raises on the content of ``raw``.
    """
    if isinstance(raw, Mapping):
        container, items = "mapping", list(raw.items())
    else:
        container, items = "value", [(None, raw)]
    members: dict[str, np.ndarray] = {}
    index: list[dict[str, Any]] = []
    for position, (name, value) in enumerate(items):
        member = f"raw__{position}"
        try:
            array = np.asarray(value)
        except Exception:  # ragged or exotic input: stored as JSON text below
            array = None
        if array is not None and array.dtype.kind in "biuf":
            members[member] = array
            encoding = "array"
        else:
            members[member] = np.frombuffer(canonical_json(_jsonable(value)).encode("utf-8"), dtype=np.uint8)
            encoding = "json"
        index.append(
            {
                "name": name if name is None or isinstance(name, str) else repr(name),
                "name_type": type(name).__name__,
                "member": member,
                "encoding": encoding,
            }
        )
    return members, {"container": container, "items": index}


def _decode_raw(members: Mapping[str, np.ndarray], index: Mapping[str, Any], path: Path) -> Any:
    try:
        container = index["container"]
        values = []
        for item in index["items"]:
            stored = members[item["member"]]
            if item["encoding"] == "array":
                values.append((item["name"], stored))
            elif item["encoding"] == "json":
                values.append((item["name"], _from_jsonable(parse_json(stored.tobytes().decode("utf-8")))))
            else:
                raise ValueError(f"unknown encoding {item['encoding']!r}")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise CacheCorruptError(f"{path} raw-response index is damaged: {type(exc).__name__}: {exc}") from None
    if container == "mapping":
        return {name: value for name, value in values}
    if container == "value" and len(values) == 1:
        return values[0][1]
    raise CacheCorruptError(f"{path} raw-response container {container!r} is not understood")


def _probe_directory(directory: Path) -> None:
    """Write and remove a temporary file in ``directory`` with the same atomic writer the cache uses.

    This also loads persistence.* (the writer's retry settings). Raises whatever the write or the
    removal raises (OSError, ConfigError, ValueError).
    """
    probe = Path(directory) / f".write-probe-{os.getpid()}-{uuid.uuid4().hex}"
    atomic_write_bytes(probe, b"write probe\n")
    probe.unlink()


@dataclass(frozen=True)
class CachedEntry:
    """One stored provider request and its validated response.

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


@dataclass(frozen=True)
class UnvalidatedResponse:
    """A provider response stored exactly as received, before validation.

    Attributes:
        key:            the request key it answers.
        path:           the unvalidated/<key>.npz file.
        request_arrays: the arrays exactly as sent (x_train (n, F), y_train (n,), x_query (m, F)).
        request_meta:   JSON metadata of the request.
        raw_response:   the raw value: a dict of name -> numeric array (as received) or decoded
                        JSON value, or a single such value when the response was not a mapping.
        response_meta:  JSON metadata recorded on receipt (quote, usage before, provider metadata).
    """

    key: str
    path: Path
    request_arrays: dict[str, np.ndarray]
    request_meta: dict[str, Any]
    raw_response: Any
    response_meta: dict[str, Any]


# --------------------------------------------------------------------------------------
# Request / response cache
# --------------------------------------------------------------------------------------


class RequestCache:
    """Persistent request/response cache: one authoritative ``<key>.npz`` per entry (see module doc)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @property
    def unvalidated_dir(self) -> Path:
        """<cache_dir>/unvalidated: raw responses not yet validated and cached."""
        return self.directory / _UNVALIDATED_DIR

    @property
    def abandoned_dir(self) -> Path:
        """<cache_dir>/unvalidated/abandoned: raw responses an operator abandoned (kept as evidence)."""
        return self.unvalidated_dir / _ABANDONED_DIR

    @property
    def lock_path(self) -> Path:
        """<cache_dir>/.lock: the file every process using this cache locks."""
        return self.directory / _LOCK_NAME

    def paths(self, key: str) -> tuple[Path, Path]:
        """(``<key>.npz`` authoritative entry, ``<key>.json`` sidecar that is never read)."""
        _check_key(key)
        return self.directory / f"{key}.npz", self.directory / f"{key}.json"

    def unvalidated_path(self, key: str) -> Path:
        """unvalidated/<key>.npz."""
        _check_key(key)
        return self.unvalidated_dir / f"{key}.npz"

    def get(self, key: str) -> CachedEntry | None:
        """The stored entry, or None when ``<key>.npz`` does not exist (the sidecar is ignored).

        Raises CacheCorruptError when the archive is unreadable, names another key or format, or
        a member is missing, extra or altered.
        """
        npz_path, _ = self.paths(key)
        unpacked = _unpack(npz_path, ENTRY_FORMAT, key)
        if unpacked is None:
            return None
        loaded, document = unpacked
        names: dict[str, list[str]] = {}
        for group in _GROUPS:
            listed = document.get(f"{group}_arrays")
            if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
                raise CacheCorruptError(f"{npz_path} metadata does not list its {group} arrays")
            names[group] = listed
        expected = {f"{group}__{name}" for group in _GROUPS for name in names[group]}
        if set(loaded) != expected or "request" not in document or "response" not in document:
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
    ) -> str | None:
        """Write one complete entry with ONE atomic write of ``<key>.npz``, then the sidecar.

        Array names must be identifiers and arrays numeric. Metadata must be strict JSON
        (non-finite floats raise ValueError before any file is written). Putting an existing key
        replaces the entry with a complete new one.

        Returns:
            None, or the error text when the human-readable ``<key>.json`` sidecar could not be
            written. The entry is complete as soon as the .npz is in place, so a sidecar failure
            never undoes it; the caller logs the text.
        """
        _check_key(key)
        groups = {"request": _check_arrays("request", request_arrays), "response": _check_arrays("response", response_arrays)}
        members = {f"{group}__{name}": array for group, arrays in groups.items() for name, array in arrays.items()}
        document = {
            "format": ENTRY_FORMAT,
            "key": key,
            "request_arrays": list(groups["request"]),
            "response_arrays": list(groups["response"]),
            "array_specs": {name: {"dtype": array.dtype.str, "shape": list(array.shape)} for name, array in members.items()},
            "request": request_meta,
            "response": response_meta,
        }
        payload = _pack(members, document)
        npz_path, json_path = self.paths(key)
        atomic_write_bytes(npz_path, payload)
        try:
            sidecar = {**document, "note": "human-readable copy; the authoritative entry is the .npz next to it"}
            atomic_write_bytes(json_path, (_ascii_json(sidecar, indent=1) + "\n").encode("ascii"))
        except Exception as exc:  # the entry is already complete; the caller records this text
            return f"{type(exc).__name__}: {exc}"
        return None

    def put_unvalidated(
        self,
        key: str,
        *,
        request_arrays: dict[str, np.ndarray],
        request_meta: dict[str, Any],
        raw_response: Any,
        response_meta: dict[str, Any],
    ) -> Path:
        """Store a raw provider response exactly as received, with its request, in one atomic write.

        Returns the unvalidated/<key>.npz path. Replaces an earlier unvalidated file of the key.
        """
        _check_key(key)
        request = _check_arrays("request", request_arrays)
        members = {f"request__{name}": array for name, array in request.items()}
        raw_members, raw_index = _encode_raw(raw_response)
        members.update(raw_members)
        document = {
            "format": UNVALIDATED_FORMAT,
            "key": key,
            "request_arrays": list(request),
            "raw": raw_index,
            "request": request_meta,
            "response": response_meta,
        }
        path = self.unvalidated_path(key)
        atomic_write_bytes(path, _pack(members, document))
        return path

    def get_unvalidated(self, key: str) -> UnvalidatedResponse | None:
        """The stored unvalidated response of ``key``, or None when there is none."""
        path = self.unvalidated_path(key)
        unpacked = _unpack(path, UNVALIDATED_FORMAT, key)
        if unpacked is None:
            return None
        loaded, document = unpacked
        listed = document.get("request_arrays")
        if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
            raise CacheCorruptError(f"{path} metadata does not list its request arrays")
        try:
            request_arrays = {name: loaded[f"request__{name}"] for name in listed}
        except KeyError as exc:
            raise CacheCorruptError(f"{path} misses request array {exc}") from None
        raw_members = {name: array for name, array in loaded.items() if name.startswith("raw__")}
        return UnvalidatedResponse(
            key=key,
            path=path,
            request_arrays=request_arrays,
            request_meta=document.get("request", {}),
            raw_response=_decode_raw(raw_members, document.get("raw", {}), path),
            response_meta=document.get("response", {}),
        )

    def remove_unvalidated(self, key: str) -> None:
        """Delete unvalidated/<key>.npz (call only after the validated entry was written)."""
        self.unvalidated_path(key).unlink(missing_ok=True)

    def abandon_unvalidated(self, key: str, stamp: str) -> Path | None:
        """Move unvalidated/<key>.npz to unvalidated/abandoned/<key>.<stamp>.npz; None if there is none.

        ``stamp`` must be a filename-safe UTC stamp such as 20261121T120000123456Z.
        """
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6,12}Z", stamp):
            raise ValueError(f"stamp must look like 20261121T120000123456Z; got {stamp!r}")
        source = self.unvalidated_path(key)
        if not source.exists():
            return None
        target = self.abandoned_dir / f"{key}.{stamp}.npz"
        if target.exists():
            raise FileExistsError(f"{target} already exists; nothing was moved")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        return target

    def has_entries(self) -> bool:
        """True when the cache, its unvalidated folder or the abandoned folder holds any keyed file."""
        for directory in (self.directory, self.unvalidated_dir, self.abandoned_dir):
            if directory.is_dir() and any(entry.is_file() and _ENTRY_NAME.match(entry.name) for entry in directory.iterdir()):
                return True
        return False

    def probe_writable(self) -> None:
        """Write and remove a temporary file in the cache directory and in its unvalidated folder.

        Raises (OSError, ConfigError from persistence.*, ...) when either cannot be written.
        """
        for directory in (self.directory, self.unvalidated_dir):
            _probe_directory(directory)


# --------------------------------------------------------------------------------------
# Inter-process lock
# --------------------------------------------------------------------------------------

_HELD_GUARD = threading.Lock()
_HELD: dict[str, tuple[int, int]] = {}  # normalized lock path -> (owning thread id, depth)


class CacheLock:
    """Exclusive, re-entrant lock on one lock file shared by every process using a cache.

    Args:
        path:      the lock file, <cache_dir>/.lock.
        timeout_s: how long :meth:`acquire` waits for another holder (tabpfn.lock_timeout_s).
        poll_s:    pause between attempts while waiting.

    The file is created with ``os.open(O_CREAT | O_EXCL)`` and holds {"pid", "acquired_utc"}
    (wall-clock UTC, for the operator). It is removed by the holder in a ``finally`` block and
    never by anyone else: a stale lock (its process died) stays until a person deletes it.
    """

    def __init__(self, path: Path, *, timeout_s: float, poll_s: float) -> None:
        if not _is_real(timeout_s) or not timeout_s > 0:
            raise ValueError(f"timeout_s must be a positive number; got {timeout_s!r}")
        if not _is_real(poll_s) or not poll_s > 0:
            raise ValueError(f"poll_s must be a positive number; got {poll_s!r}")
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)

    @property
    def identity(self) -> str:
        """Normalized absolute path: two CacheLock objects with the same identity are one lock."""
        return os.path.normcase(os.path.abspath(self.path))

    def __enter__(self) -> "CacheLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

    def held_by_this_thread(self) -> bool:
        with _HELD_GUARD:
            held = _HELD.get(self.identity)
        return held is not None and held[0] == threading.get_ident()

    def acquire(self) -> None:
        """Take the lock (or re-enter it); raise CacheLockedError after ``timeout_s`` of waiting."""
        identity, thread = self.identity, threading.get_ident()
        deadline = time.monotonic() + self.timeout_s
        last_error: str | None = None
        while True:
            with _HELD_GUARD:
                held = _HELD.get(identity)
                if held is not None and held[0] == thread:
                    _HELD[identity] = (thread, held[1] + 1)
                    return
                if held is None:
                    created, last_error = self._create()
                    if created:
                        _HELD[identity] = (thread, 1)
                        return
            if time.monotonic() >= deadline:
                raise CacheLockedError(self._locked_message(last_error))
            time.sleep(self.poll_s)

    def _create(self) -> tuple[bool, str | None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)  # raises at once when the cache directory cannot exist
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False, None
        except PermissionError as exc:  # Windows: the previous holder's delete is still pending
            return False, f"{type(exc).__name__}: {exc}"
        try:
            content = json.dumps({"pid": os.getpid(), "acquired_utc": datetime.now(timezone.utc).isoformat()})
            os.write(handle, content.encode("ascii"))
        except BaseException:
            os.close(handle)
            self.path.unlink(missing_ok=True)
            raise
        os.close(handle)
        return True, None

    def release(self) -> None:
        """Leave the lock; the outermost release removes the lock file (retried while Windows blocks it)."""
        identity, thread = self.identity, threading.get_ident()
        with _HELD_GUARD:
            held = _HELD.get(identity)
            if held is None or held[0] != thread:
                raise RuntimeError(f"release of the cache lock {self.path}, which this thread does not hold")
            if held[1] > 1:
                _HELD[identity] = (thread, held[1] - 1)
                return
            del _HELD[identity]
        # Outside the guard: another thread of this process now waits on the file like another process.
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self.path.unlink()
                return
            except PermissionError:  # a waiting process is reading the file for its error message
                if time.monotonic() >= deadline:
                    raise
                time.sleep(self.poll_s)

    def _locked_message(self, last_error: str | None) -> str:
        try:
            content: str | None = self.path.read_text(encoding="ascii", errors="replace")
        except FileNotFoundError:
            content = None
        except OSError as exc:
            content = f"<unreadable: {type(exc).__name__}: {exc}>"
        if content is None:
            return (
                f"could not create the cache lock {self.path} within tabpfn.lock_timeout_s = {self.timeout_s} s "
                f"(last error: {last_error}); check that the cache directory is writable"
            )
        return (
            f"the cache lock {self.path} is held ({content}) and was not released within tabpfn.lock_timeout_s = "
            f"{self.timeout_s} s. Another process using the same cache is probably still working: wait and re-run. "
            f"If no process with that pid is running, the lock is stale: delete {self.path} by hand, then re-run. "
            f"The lock is never removed automatically."
        )


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


def parse_log_text(text: str, where: Any) -> list[dict[str, Any]]:
    """Records of a call-log text split on "\\n" ONLY (never on U+2028 and friends).

    The text must be empty or end with "\\n"; an unterminated last line (an interrupted append),
    a blank line or a line that is not a JSON object raises ValueError naming ``where`` and the
    line number.
    """
    if not text:
        return []
    pieces = text.split("\n")
    if pieces[-1] != "":
        raise ValueError(f"{where}:{len(pieces)} is not terminated by a newline (an interrupted append?)")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(pieces[:-1], start=1):
        try:
            record = parse_json(line)
        except ValueError as exc:
            raise ValueError(f"{where}:{number} is not a valid call-log line: {exc}") from None
        if not isinstance(record, dict):
            raise ValueError(f"{where}:{number} is not a JSON object")
        records.append(record)
    return records


def tokens_quoted(records: Sequence[Mapping[str, Any]], where: Any) -> int:
    """Sum of ``quote_tokens`` over records with ``billable_dispatch`` true.

    The teacher writes exactly one such record per request it hands to the provider, before the
    call, so a crash between the call and its result record cannot hide spent tokens.
    """
    total = 0
    for number, record in enumerate(records, start=1):
        if record.get(DISPATCH_FLAG) is not True:
            continue
        quote = record.get(QUOTE_FIELD)
        if isinstance(quote, bool) or not isinstance(quote, int) or quote < 0:
            raise ValueError(f"{where} record {number}: a dispatch record needs a non-negative integer {QUOTE_FIELD}")
        total += quote
    return total


def unresolved_dispatches(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """key -> its latest dispatch record, for every key whose latest dispatch has no LATER
    ``completed`` or ``abandoned`` record (it may have been billed without a cached response)."""
    open_dispatches: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record.get("key")
        if not isinstance(key, str):
            continue
        if record.get(DISPATCH_FLAG) is True:
            open_dispatches[key] = dict(record)
        elif record.get("event") in RESOLVING_EVENTS:
            open_dispatches.pop(key, None)
    return open_dispatches


def dispatch_times(records: Sequence[Mapping[str, Any]], where: Any) -> list[datetime]:
    """UTC times of every dispatch record; a missing, naive or unparsable stamp raises ValueError."""
    times: list[datetime] = []
    for number, record in enumerate(records, start=1):
        if record.get(DISPATCH_FLAG) is not True:
            continue
        stamp = record.get("timestamp_utc")
        try:
            when = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
        except ValueError:
            when = None
        if when is None or when.utcoffset() != timedelta(0):
            raise ValueError(f"{where} record {number}: a dispatch record needs a UTC timestamp_utc; got {stamp!r}")
        times.append(when)
    return times


class CallLog:
    """Append-only JSON Lines log of provider calls, refusals, cache hits and operator decisions.

    Args:
        path:  the log file (tabpfn.call_log).
        clock: returns a timezone-aware UTC datetime; stamps every record.
        lock:  the cache's :class:`CacheLock`; held for every append and read.
    """

    def __init__(self, path: Path, *, clock: Callable[[], datetime], lock: CacheLock) -> None:
        if not isinstance(lock, CacheLock):
            raise TypeError(f"lock must be a CacheLock; got {type(lock).__name__}")
        self.path = Path(path)
        self.lock = lock
        self._clock = clock

    def exists(self) -> bool:
        """True when the log file exists."""
        return self.path.is_file()

    def append(self, record: dict[str, Any]) -> None:
        """Append ``record`` plus ``timestamp_utc`` as one ASCII canonical JSON line, then fsync.

        Holds the lock for the whole append. Raises ValueError when the record already has
        ``timestamp_utc``, contains a credential-named key at any depth, or holds a non-finite
        float; nothing is written then.
        """
        if not isinstance(record, Mapping):
            raise TypeError(f"a call-log record must be a mapping; got {type(record).__name__}")
        if "timestamp_utc" in record:
            raise ValueError("timestamp_utc is set by the call log from its clock")
        _reject_credentials(record, "record")
        line = _ascii_json({**record, "timestamp_utc": utc_timestamp(self._clock)})
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="ascii", newline="\n") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def read(self) -> list[dict[str, Any]]:
        """All records in order (under the lock); [] when the file does not exist. See :func:`parse_log_text`."""
        with self.lock:
            try:
                text = self.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return []
        return parse_log_text(text, self.path)

    def tokens_quoted_for_billed_calls(self) -> int:
        """Sum of ``quote_tokens`` over dispatch records (see :func:`tokens_quoted`)."""
        return tokens_quoted(self.read(), self.path)

    def probe_writable(self) -> None:
        """Write and remove a temporary file in the log's directory; raises when it cannot."""
        _probe_directory(self.path.parent)
