"""Atomic file replacement shared by every Person B writer.

A file is written to a temporary file in the same directory, flushed, fsynced, and moved
into place with ``os.replace``, so a reader never sees a half-written file. On Windows
``os.replace`` raises ``PermissionError`` while another handle holds the target open
(experiments/proofs/B_students_eval_contract/p10_persist.py); the move is retried a bounded
number of times. The retry bounds are read from ``persistence.*`` in config.yaml.
"""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from learned_judgment.config import load_config, require

__all__ = ["atomic_write_bytes", "atomic_write_text"]


@lru_cache(maxsize=1)
def _replace_settings() -> tuple[int, float]:
    config = load_config()
    attempts = int(require(config, "persistence.replace_attempts"))
    pause_s = float(require(config, "persistence.replace_pause_s"))
    if attempts < 1 or pause_s < 0.0:
        raise ValueError("persistence.replace_attempts must be >= 1 and persistence.replace_pause_s >= 0")
    return attempts, pause_s


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically, creating parent directories."""
    path = Path(path)
    attempts, pause_s = _replace_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(attempts):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(pause_s)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` as UTF-8 to ``path`` atomically; line endings are written as given."""
    atomic_write_bytes(path, text.encode("utf-8"))
