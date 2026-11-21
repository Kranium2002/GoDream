"""Assertion counter shared by the B_snn_measure_offline check scripts (no fixture data here).

FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model.
Nothing computed by these scripts may enter a report or results/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

BANNER = "FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model"
REPO_ROOT = Path(__file__).resolve().parents[3]
PROOF_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Checks:
    """Counts executed assertions and sets the exit code (0 only when every assertion held)."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def expect(self, condition: bool, label: str) -> None:
        print(("  [ok] " if condition else "  [FAILED] ") + label)
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)

    def raises(self, label: str, error: type[BaseException], function: Callable[[], Any], needle: str) -> None:
        try:
            function()
        except error as caught:
            message = str(caught).replace("\n", " ")
            self.expect(needle in message, f"{label} -> {type(caught).__name__}: {message[:220]}")
            return
        except Exception as other:  # noqa: BLE001
            self.expect(False, f"{label} raised {type(other).__name__} instead of {error.__name__}: {other}")
            return
        self.expect(False, f"{label} did not raise {error.__name__}")

    def finish(self) -> None:
        print(f"\nCHECK RESULT: {self.passed} assertions held, {len(self.failures)} failed {self.failures or ''}")
        raise SystemExit(0 if not self.failures else 1)
