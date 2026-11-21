# Preregistered hypotheses — Learned Judgment

Date: 2026-11-21. Freeze these criteria before tuning. Any later change goes in a dated exploratory section; do not replace the original criteria.

- **H1:** Per skill, at 20 actual attempts, the TabPFN teacher beats physics-only and physics plus constant bias on the primary skill metric.
- **H2:** Per skill, at 20 or fewer attempts, the teacher is at least as good as Gaussian-process and gradient-boosting residual baselines; ties are reported.
- **H3:** Students remain within 10% of teacher error with at least 1,000× lower decision latency.
- **H4:** If an SNN is completed, it remains within 25% of teacher error with lower estimated energy per decision than the MLP.
- **H5:** After a world change, each evaluated skill recovers within 15 new actual attempts.
- **H6:** Teacher 80% predictive intervals cover 70–90% of actual outcomes.
- **H7:** After adding a new skill, the shared student keeps earlier skills within 10% of their pre-addition error.

All outcomes are simulated Go2 MuJoCo episodes. Each verdict must cite measured values, interval, sample count, seed/config, and raw artifact. A metric is never hardcoded. An unexecuted hypothesis is inconclusive or NOT RUN, not supported.

## Amendment 2026-11-21 — the student is an SNN

Recorded before any Go2 episode, TabPFN prediction, dream, or tuning existed. H1–H7 above are the original text and are unchanged.

- By decision of the project lead on 2026-11-21, the only student is a spiking neural network (SNN): one per skill, then one shared SNN with skill ID and dream rehearsal. No MLP student is built.
- H3 and H7: "students" and "the shared student" mean the per-skill SNN and the shared SNN.
- H4: its condition "if an SNN is completed" no longer applies, because the SNN is required. Its 25% criterion is reported next to H3's 10% criterion for the same SNN, and each gets its own verdict.
- H4's energy comparison "than the MLP": no MLP student is trained, so the reference is a non-spiking dense network with the same layer sizes as the SNN. Its multiply-accumulate count follows from the architecture alone. Both energies are estimates from operation counts and cited per-operation figures. Limit: this comparison cannot show that a trained MLP of that size would reach the same decision quality.
- The MLP trained on actual attempts (baseline B5 in 03_METRICS.md) is a residual baseline, not a student, and is unaffected.
