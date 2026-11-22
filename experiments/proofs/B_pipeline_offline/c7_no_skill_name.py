"""c7 PLUMBING CHECK: learned_judgment/pipeline.py names no skill, and skills/__init__.py stays generic.

A skill name is every registered skill (learned_judgment.skills.available_skills()) and every key
of config.yaml ``skills`` (primary, planned and stretch skills). The check searches the source text
of learned_judgment/pipeline.py and learned_judgment/skills/__init__.py for each name as a whole
word, ignoring case, and fails on any hit.

For information only (not part of this check's result): the same search over
learned_judgment/engine/, which this lane does not edit.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c7_no_skill_name.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, ROOT, Checks  # noqa: E402

print(BANNER)

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.skills import available_skills  # noqa: E402

checks = Checks()
names = sorted(set(available_skills()) | set(load_config()["skills"]))
print(f"skill names searched: {names}")
pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in names) + r")\b", re.IGNORECASE)


def hits(path: Path) -> list[str]:
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in pattern.finditer(line):
            found.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {match.group(0)!r}")
    return found


for relative in ("learned_judgment/pipeline.py", "learned_judgment/skills/__init__.py"):
    found = hits(ROOT / relative)
    checks.expect(not found, f"{relative}: no skill name {found}")

print("\nINFO (not part of this check): the same search over learned_judgment/engine/")
for path in sorted((ROOT / "learned_judgment" / "engine").glob("*.py")):
    for item in hits(path):
        print(f"  {item}")

checks.finish()
