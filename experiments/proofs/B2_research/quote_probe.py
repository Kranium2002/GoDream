"""B2 research probe: free TabPFN cost quotes for request shapes of today's DODGE and a 1,152-pixel DODGE.

Uses the project's TabPFNProvider.quote (tabpfn_client.estimate_cost: only row/column counts are
sent; no data upload, no quota consumed). No prediction is made. Shapes follow the teacher's
"stacked" mode: x_train (A*n, F+1) and x_query (A*m, F+1) when A >= 2 outcomes are non-constant.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B2_research\\quote_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import TabPFNProvider, TeacherSettings  # noqa: E402

settings = TeacherSettings.from_config(load_config())
provider = TabPFNProvider(settings)
provider.connect()
usage = provider.usage()
print("usage (free read):", {k: usage.get(k) for k in ("current_usage", "usage_limit")})
print(f"model_path {settings.model_path}, n_estimators {settings.n_estimators}, "
      f"max_query_rows_per_request {settings.max_query_rows_per_request}, max_tokens_per_request {settings.max_tokens_per_request}")
shapes = [
    ("dodge today: 40 attempts x 2 active outcomes, one full dream chunk", 80, 32000, 18),
    ("dodge today: 100 dev throws x 108 candidates x 2 outcomes", 80, 21600, 18),
    ("vision: 40 attempts x 2 outcomes, 2000 dreams x 4 candidates x 2 outcomes", 80, 16000, 1156),
    ("vision: same, one full chunk", 80, 32000, 1156),
    ("vision: 100 dev situations x 4 candidates x 2 outcomes", 80, 800, 1156),
    ("vision: one decision, 4 candidates x 2 outcomes", 80, 8, 1156),
    ("vision: 40 attempts x 3 outcomes, 2000 x 4 x 3", 120, 24000, 1156),
]
for label, n_train, n_query, n_features in shapes:
    quote = provider.quote(n_train, n_query, n_features)
    print(f"{label}: quote(n_train={n_train}, n_query={n_query}, n_features={n_features}) -> "
          f"{quote['estimated_cost']} tokens (pricing {quote['pricing_version']})")
print("exit 0")
