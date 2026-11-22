"""Two runs' result.json metrics side by side for one method (Person B, method checks: old setting against new).

Command line (repository root)::

    python -m learned_judgment.compare_runs RUN_A RUN_B [--method snn]

reads <observed.runs_dir>/<RUN>/result.json of both and prints a Markdown table: only metrics both runs report for
that method, difference = b - a.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

from learned_judgment.config import load_config


def side_by_side(a: Mapping[str, Any], b: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    """Rows {"name", "a", "b", "difference" (b - a), "n_a", "n_b"} in a's metric order, for the metrics present in
    both results for ``method``."""
    def by_name(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        return {m["name"]: m["values"][method] for m in result.get("metrics", []) if method in m.get("values", {})}

    left, right = by_name(a), by_name(b)
    return [{"name": name, "a": left[name]["value"], "b": right[name]["value"],
             "difference": right[name]["value"] - left[name]["value"], "n_a": left[name]["n"], "n_b": right[name]["n"]}
            for name in left if name in right]


def main(argv: Sequence[str] | None = None) -> int:
    from learned_judgment.observed import _run_dir

    parser = argparse.ArgumentParser(prog="python -m learned_judgment.compare_runs", description=__doc__.splitlines()[0])
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--method", default="snn", help="method key in result.json metrics values (default snn)")
    args = parser.parse_args(argv)
    config = load_config()
    a, b = (json.loads((_run_dir(config, run) / "result.json").read_text(encoding="utf-8")) for run in (args.run_a, args.run_b))
    print(f"| metric | {args.run_a} | {args.run_b} | difference | n {args.run_a} | n {args.run_b} |")
    print("|---|---|---|---|---|---|")
    for row in side_by_side(a, b, args.method):
        print(f"| {row['name']} | {row['a']:.4g} | {row['b']:.4g} | {row['difference']:+.4g} | {row['n_a']} | {row['n_b']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
