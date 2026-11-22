# presentation_student_a: the backup of the punch student shown on 2026-11-22

A copy of `results/runs/v6p95/student` (Person B's sweep `v6p95` winner `f4_signed_diff_h128`; network sha256 starts
`821cc5fb`), kept so the presentation can go back to it whatever a later round does. Nothing was retrained.

Values it was shown with (commit `2bcd4b5` on `person-a/go2-env`):

| Setting | Value |
|---|---|
| `app.infer.student_dir` | `results/runs/v6p95/student` (this folder is the same student) |
| Frozen thresholds (in `student.json`) | threat 0.98, action 0.4, margin 0.1 |
| `inference.gate` (calibrated, live camera) | 3 of 4 votes, cooldown 1.0 s, left and right dodges count as one decision |
| `inference.gate.precision_target` | 0.95 (lowered from 0.99 by the lead) |
| Duel gate `app.duel.dodge.votes_needed` / `of_last` | 2 of 2 |
| Duel punches `app.duel.dodge` | held-out `test` split, kinds `incoming_center`, `incoming_left`, `incoming_right`, 2 of 3 rounds centre |
| Duel `default_speed_m_s` | 0.5 |
| Turn correction | none |

What is measured for it, all simulation: dev calibration 61 commands, precision 0.967, recall 0.410 (near-approach rule;
strict collision view 0.836 / 0.423); with 2 of 2 votes on the same dev readouts 82 commands, precision 0.951, recall 0.530;
15 real duel rounds at 0.5 m/s with 2 of 2 votes: 10 dodges of 15 hits. Held-out split `test2`: NOT RUN.

To go back to it: set `app.infer.student_dir: results/runs/presentation_student_a/student` and restart the server.
