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

## Amendment 2026-11-21 — how H4's energy criterion is judged

Recorded before any Go2 episode, TabPFN prediction, dream, or tuning existed. A review of the measurement code showed that the energy comparison can change sign with the counting convention, so the rule is fixed here.

- Two estimates are reported for the SNN, each with an interval: the convention common in the SNN literature (real-valued first layer plus one accumulate per synaptic event), and the same plus every neuron's leak, input and reset update. The reference is the non-spiking dense network of the same layer sizes.
- H4's energy criterion is **supported** only if both estimates are below the reference, **not supported** if both are above it, and **inconclusive** if they disagree.
- Every energy figure is an estimate from operation counts and cited 45 nm per-operation energies. None is a measurement.

## Amendment 2026-11-21 — what "recovers" means in H5

Recorded before any Go2 episode. A review showed that a single lucky measurement could count as recovery.

- Recovery is measured at 0, 5, 10, 15 and 20 new attempts after a world change. The threshold is the pre-shift error plus 20% (03_METRICS.md).
- H5 is judged on **sustained** recovery: the smallest number of new attempts from which the error stays at or below the threshold at every larger measured count. The first count that touches the threshold is reported next to it.
- If the pre-shift error is 0, the threshold is 0 and this is stated with the result.

## Operational definitions for DODGE, 2026-11-21

Recorded before any Go2 episode. They make "the primary skill metric" and "error" in H1 to H7 concrete for DODGE.

- **Primary loss per episode:** 1 if the ball touched the dog or the dog fell, otherwise 0. The primary metric is its mean, the failure rate. Lower is better.
- **Success (tolerance hit):** no contact and no fall.
- **Secondary:** closest separation, recovery time, commanded sideways travel, and regret against the oracle under the DODGE objective in config.yaml.
- **Objective form:** fixed in config.yaml (`skills.dodge.objective_weights`): penalties on predicted collision and fall probability, a capped reward for the low predictive quantile of separation, and costs for commanded motion and recovery time. Weights are starting values and may change from dev data only.
