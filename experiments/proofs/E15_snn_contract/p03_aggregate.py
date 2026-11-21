"""p03_aggregate: print the p03 FIXTURE sweep as a table (mechanics numbers, not project results)."""
import glob, json
rows = [json.loads(l) for f in sorted(glob.glob("p03_runs/worker_*.jsonl")) for l in open(f, encoding="utf-8")]
rows.sort(key=lambda r: (r["loss"], r["n_train"], len(r["hidden"]), r["hidden"][0], r["T"]))
print(f"n_rows={len(rows)}")
hdr = "loss n_train hidden   T  steps epochs train_s loss_first->last  init_fire     fire_end      dead_frac   train_agree holdout_agree holdout_kl"
print(hdr)
for r in rows:
    print(f"{r['loss']:4s} {r['n_train']:5d}  {str(tuple(r['hidden'])):9s} {r['T']:2d} {r['steps']:5d} {r['epochs']:5d} {r['train_seconds']:7.1f} "
          f"{r['loss_first']:7.3f}->{r['loss_last_epoch_mean']:6.3f}  {str(r['init_fire']):13s} {str(r['fire']):13s} {str(r['dead_frac']):11s} "
          f"{r['train_agree']:.3f}       {r['holdout_agree']:.3f}         {r['holdout_kl']:.3f}")
