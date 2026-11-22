"""Near-approach threat distance for dodge_vision (run v6), from saved context and dev data only.

Rule (fixed before any table or student result was seen):
  H (comes at the head)          = incoming_center, incoming_left, incoming_right
  W (passes wide / not approaching) = off_path, static, receding, background_motion
  ambiguous is the near-miss band that the distance splits; it does not set the distance. empty has no object.
  Pool the measured NO_OP min_separation_m of the context NO_OP rows and of the dev trace's NO_OP episodes.
  a = max over H, b = min over W. If a < b: separation_m = round((a + b) / 2, 2). Else: stop, no distance.
Also tabulates the dream's predicted NO_OP separation per kind (information) and the measured share below the distance.

python experiments/proofs/B_v6_near_approach/threat_distance.py RUN_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

H = ("incoming_center", "incoming_left", "incoming_right")
W = ("off_path", "static", "receding", "background_motion")
ORDER = ("incoming_center", "incoming_left", "incoming_right", "ambiguous", "off_path", "static", "receding",
         "background_motion", "empty")


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"n": int(v.size), "min": float(q[0]), "p10": float(q[1]), "median": float(q[2]), "p90": float(q[3]),
            "max": float(q[4]), "collisions": int(np.sum(v <= 0.0))}


def main(run_dir, out_json):
    run = Path(run_dir)
    ctx = json.loads((run / "context" / "rows.json").read_text(encoding="utf-8"))
    names, outcomes = ctx["action_names"], ctx["outcome_names"]
    no_op, sep = names.index("NO_OP"), outcomes.index("min_separation_m")
    with np.load(run / "context" / "rows.npz") as rows:
        action, index, measured = rows["action"], rows["scenario_index"], rows["measured"]
    measured_by_kind = {k: [] for k in ORDER}
    source = {k: {"context": 0, "dev": 0} for k in ORDER}
    for a, i, m in zip(action, index, measured):
        if int(a) == no_op:
            kind = ctx["scenarios"][int(i)]["kind"]
            measured_by_kind[kind].append(float(m[sep]))
            source[kind]["context"] += 1
    trace = json.loads((run / "trace_dev.json").read_text(encoding="utf-8"))
    for r in trace["records"]:
        measured_by_kind[r["kind"]].append(float(r["no_op"]["min_separation_m"]))
        source[r["kind"]]["dev"] += 1
    a = max(v for k in H for v in measured_by_kind[k])
    b = min(v for k in W for v in measured_by_kind[k])
    distance = round((a + b) / 2.0, 2) if a < b else None
    dream = json.loads((run / "dream" / "dream.json").read_text(encoding="utf-8"))
    with np.load(run / "scores.npz") as s:
        predicted = np.array(s["predicted_mean"])[:, no_op, sep]
        scenario_index = np.array(s["scenario_index"])
    dream_kind = np.array([dream["scenarios"][int(i)]["kind"] for i in scenario_index])
    table = {}
    for k in ORDER:
        values = np.asarray(measured_by_kind[k])
        row = {"measured_no_op": stats(values), "sources": source[k],
               "predicted_no_op_dream_windows": stats(predicted[dream_kind == k])}
        if distance is not None and values.size:
            row["measured_below_distance"] = int(np.sum(values < distance))
            row["dream_windows_below_distance"] = int(np.sum(predicted[dream_kind == k] < distance))
        table[k] = row
    out = {"rule": __doc__.split("Rule (fixed before any table or student result was seen):")[1].split("Also tabulates")[0].strip(),
           "run_dir": str(run.resolve()), "sources": {"context": str(run / "context" / "rows.npz"),
                                                      "dev_trace": str(run / "trace_dev.json"),
                                                      "dream_scores": str(run / "scores.npz")},
           "H_max_m": a, "W_min_m": b, "separation_m": distance, "table": table}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{'kind':18s} {'n':>4s} {'min':>6s} {'p10':>6s} {'med':>6s} {'p90':>6s} {'max':>6s} {'hits':>4s} | dream n  min    med    max")
    for k in ORDER:
        m, p = table[k]["measured_no_op"], table[k]["predicted_no_op_dream_windows"]
        if m["n"]:
            print(f"{k:18s} {m['n']:4d} {m['min']:6.3f} {m['p10']:6.3f} {m['median']:6.3f} {m['p90']:6.3f} {m['max']:6.3f} "
                  f"{m['collisions']:4d} | {p.get('n', 0):6d} {p.get('min', float('nan')):6.3f} {p.get('median', float('nan')):6.3f} "
                  f"{p.get('max', float('nan')):6.3f}  below d: meas {table[k].get('measured_below_distance')} dream {table[k].get('dream_windows_below_distance')}")
    print(f"H max {a:.4f} m, W min {b:.4f} m, separation_m {distance}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
