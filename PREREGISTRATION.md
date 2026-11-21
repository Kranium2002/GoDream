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
