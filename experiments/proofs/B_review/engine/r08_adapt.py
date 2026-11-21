"""R08: adapt.recovery_attempts boundary behaviour and select_context / world id."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import TS  # noqa: E402,F401

from learned_judgment.engine.adapt import recovery_attempts, world_id_column, world_id_variable  # noqa: E402

# 1. transient dip: below threshold at 5, above it again at 10, 15 and 20
curve = {0: 3.0, 5: 1.15, 10: 2.4, 15: 2.6, 20: 2.5}
print("1. curve", curve, "pre-shift 1.0, tol 0.2 -> recovery_attempts =", recovery_attempts(curve, 1.0, 0.2))

# 2. pre-shift error 0 (e.g. no collisions in the pre-shift dev set)
print("2. pre-shift 0.0, curve {0: 0.3, 5: 0.02, 10: 0.01, 20: 0.01} ->",
      recovery_attempts({0: 0.3, 5: 0.02, 10: 0.01, 20: 0.01}, 0.0, 0.2))

# 3. exact boundary in floating point: error equal to 1.2 x pre-shift as computed elsewhere
for pre in (0.1, 0.3, 0.7, 1.1, 2.3):
    error = round(pre * 1.2, 10)
    print(f"3. pre={pre} error={error!r} threshold={pre * (1 + 0.2)!r} ->", recovery_attempts({5: error}, pre, 0.2))

# 4. world id: a preset with no attempts in the context
var = world_id_variable(["default", "heavy_payload", "slippery_floor", "tired_motors", "downhill"])
print("4. world_id levels", var.levels, "column for ['downhill'] ->", world_id_column(["downhill"], var).tolist())
