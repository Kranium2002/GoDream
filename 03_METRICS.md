# Learned Judgment — preregistration and evaluation

Person B owns the evaluation implementation and reports. Read MISSION.md, 01_PLAN.md, and 02_ENVIRONMENT.md first. Evaluate the decision engine on actual Go2 MuJoCo episodes and actual provider outputs. No report may claim a test passed unless the test was executed.

## Before tuning

Create/freeze PREREGISTRATION.md before any model or threshold choice. Keep the original hypotheses unchanged; later ideas go in a dated exploratory section. Split independent seeded streams per skill:

- Dev: about 100 situations for feature, threshold, and model choices.
- Test: about 100 situations, held for one final evaluation of each chosen method.
- Shift sets: independent situations per world preset.
- Context attempts: generated separately; never reuse test situations as training attempts.

Use at least 3 seeds per configuration, 5 when budget permits. Compare methods on the same situations and noise draws; report paired bootstrap 95% confidence intervals. Parallelize independent test episodes only after seed/noise parity is verified.

## Required methods

Run these per skill where applicable:

| ID | Method | Role |
|---|---|---|
| B0 | True-world oracle | Noise floor |
| B1 | Skill physics only | Engineering baseline |
| B2 | Physics + constant mean residual | Strong simple baseline |
| B3 | Physics + Gaussian-process residual | Small-data baseline |
| B4 | Physics + gradient-boosting residual | Tabular baseline |
| B5 | MLP trained on actual attempts | Neural baseline |
| B6 | Direct TabPFN without physics feature | Measures physics contribution |
| T1 | TabPFN residual teacher | Proposed decision-maker |
| S1 | Per-skill MLP trained on teacher dreams | Fast student |
| S2 | Shared MLP trained with dream rehearsal | Multi-skill student |
| S3 | SNN | Optional, only if actually implemented and measured |

Every table row refers to an executed method. Missing results are marked NOT RUN. Never copy a result from a fixture, example, previous run, or another skill.

## Metrics

For regression: mean, median, and 90th-percentile absolute error, and skill-specific hit/tolerance rate. For classification: success, slip, collision, and fall rates with counts. For all skills: regret vs oracle and paired bootstrap intervals.

Learning curves use attempt counts 0, 5, 10, 20, and 40. Plot each skill separately. Report area under the learning curve and attempts to its preregistered tolerance; report not reached when appropriate.

Adaptation: train on one world preset with 40 attempts, switch presets, then add 0/5/10/15/20 new attempts. Compare all history, recent-only history, and all history with world ID. Report new attempts needed to recover within 20% of pre-shift error.

Teacher: 50%/80% predictive interval coverage, PIT by outcome dimension, and Spearman correlation of predicted spread with actual error. Dreams: true-world fidelity, error by distance to nearest actual attempt, and confidence-filter/dream-count sensitivity for 100/1,000/2,000/10,000 dreams as budget permits. Students: student/teacher error ratio, decision agreement, latency, and performance by situation bin.

Measure median and p95 decision latency over 1,000 decisions on the demo laptop. Include teacher network time, per-skill MLP, shared MLP, and SNN if built. Report TabPFN calls, tokens/usage, quoted cost, and actual usage where available. Count MLP MACs and SNN spikes/synaptic operations/time steps. Energy is an estimate only; verify and cite the 45 nm per-operation figures before using them.

Report results for every world preset and the DODGE ball-throw condition. Include simulator and policy versions, config ID, seeds, and raw artifact paths.

## Preregistered hypotheses

- H1: Per skill, at 20 actual attempts, the teacher beats physics-only and physics-plus-constant-bias correction on the primary metric.
- H2: Per skill, at 20 or fewer attempts, the teacher is at least as good as Gaussian-process and gradient-boosting residual baselines; ties are reported.
- H3: Students remain within 10% of teacher error at least 1,000× lower decision latency.
- H4: If SNN is completed, it remains within 25% of teacher error with lower estimated energy per decision than the MLP.
- H5: After a world change, each tested skill recovers within 15 new actual attempts.
- H6: Teacher 80% intervals cover 70–90% of actual outcomes.
- H7: After a new skill is added, the shared student keeps earlier skills within 10% of their prior error.

For each hypothesis report supported, not supported, or inconclusive with measured values, interval, sample count, and artifacts. Unsupported hypotheses remain visible.

## Real-test and anti-fabrication rules

- Run tests against actual repository code, actual MuJoCo Go2 model, selected walking policy, and actual TabPFN provider for provider claims.
- Record exact test command, exit code, software/model versions, seed/config, and raw output path. A planned or unexecuted test is NOT RUN, never PASS.
- A pure-function fixture may check serialization or math, but it cannot produce model, simulator, latency, or final-report metrics.
- Generate tables/figures from saved episode and inference outputs. Do not hardcode numbers, success labels, expected model quality, or “passed” status into report templates.
- Never tune on the test split, fabricate or cherry-pick attempts, or use a mock/fake provider as evidence.
- If API usage/credits prevent a real provider test, mark teacher results NOT RUN and leave provider acceptance OPEN. Do not substitute a local stub.

## Deliverables

- A common harness accepting any SkillSpec and returning applicable metrics.
- results/final_table.csv and results/final_table.md: one row per skill × executed method.
- Per-skill learning curves, adaptation chart, continual-learning chart, calibration plots, and dream-fidelity results.
- RESULTS.md with each hypothesis verdict, raw artifact references, honest limits, and a pitch paragraph built only from measured outcomes.

Before merging, compare the report rows against raw run artifacts and verify that every PASS corresponds to an executed command.
