"""p03b_aggregate: table of FIXTURE failure-mode runs (mechanics only)."""
import glob, json
rows = [json.loads(l) for f in sorted(glob.glob("p03b_runs/worker_*.jsonl")) for l in open(f, encoding="utf-8")]
order = ["base", "input_scale", "threshold", "beta", "learn_beta", "learn_beta_beta0.5", "T", "w1_init_scale", "surrogate", "surrogate_fs25_threshold5", "two_layer", "two_layer_threshold5"]
rows.sort(key=lambda r: (order.index(r["variant"]), json.dumps(r["cfg"], sort_keys=True)))
print(f"n_rows={len(rows)}")
for r in rows:
    c = r["cfg"]
    desc = f"scale={c['scale']} thr={c['threshold']} beta={c['beta']} T={c['T']} w1x={c['w1_scale']} sg={c['sg']} hidden={c['hidden']} learn_beta={c['learn_beta']}"
    print(f"{r['variant']:26s} {desc}")
    print(f"    init: fire={r['init_fire']} dead={r['init_dead_frac']} sat={r['init_sat_frac']} |gW1|={r['init_grad_norm_W1']:.2e} |gWout|={r['init_grad_norm_Wout']:.2e}")
    print(f"    end : loss {r['loss_first']:.3f}->{r['loss_last_epoch_mean']:.3f} holdout_agree={r['holdout_agree']:.3f} fire={r['fire']} dead={r['dead_frac']} sat={r['sat_frac']} beta_eff={r['learned_beta_effective']} t={r['train_seconds']}s")
