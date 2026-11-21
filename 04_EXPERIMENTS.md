# Learned Judgment — research and experiment plan

Person B owns engine/model experiments; Person A owns robot/environment proof. Read MISSION.md and 01_PLAN.md first. Each experiment answers a named risk before the related production slice is built. These are gates for the single Go2 project, not options to switch projects.

## Research-agent protocol

Before coding each slice, dispatch an UltraCode research agent with one bounded question. Require primary-source citations, exact version/API/config details, a minimal proof-of-concept command, observed output, limitations, and pass/fail recommendation. The agent must not edit production files. If UltraCode is not available, use an independent research agent or primary-source inspection and record the substitution. Never say a library or policy works from a search result or README alone.

For each proof, log the hypothesis, exact source/version, command, exit code, output artifact, time/cost, decision, and limitation in EXPERIMENTS_LOG.md. An unexecuted proof is NOT RUN. If a required proof fails, stop the dependent work and resolve it; do not fall back to another robot, controller, fake, or project.

## Order and gates

### Gate 1 — Go2 walking policy compatibility

**E1: Efferent Go2 path.** Verify the current package/API, example path, ONNX deployment path, Go2 model fetch steps, license, and Python/MuJoCo compatibility. Then run the actual Go2 walk example for 30 seconds and save output.

**E2: rl_sar policy/config.** Verify repository revision, exact Go2 checkpoint path, matching robot model, observation field order, action scaling, default pose, PD gains, control rate, CPU inference, and license. Load the actual checkpoint and run the locomotion acceptance conditions in 02_ENVIRONMENT.md. Compare evidence to E1 only to choose the verified Go2 path, not to substitute another product.

**E3: Quadruped-PyMPC investigation.** Before installation, verify current MuJoCo/acados requirements, compatibility, CPU timing, and license. Use only if the preceding Go2 policy paths cannot be made to pass; all project gates still require the Go2 criteria.

The 02_ENVIRONMENT.md acceptance run is mandatory. If the selected policy falls, ignores commands, or misses the required speed/turn/stop behavior, diagnose observation/action/config mapping and repeat actual runs. Do not mark locomotion ready until it passes.

### Gate 2 — command interface and true world

**E4: Command inventory.** Extract the exact command names, units, legal ranges, update rate, and native mapping from the selected walking policy. Prove each command with an actual MuJoCo run. Publish the command schema for Person B before skill actions are implemented.

**E5: Hidden-effect validity.** In actual Go2 MuJoCo runs, switch payload, friction, slope, motor strength, and command latency individually. Record measured effects. Verify per skill that physics-only error is at least 3× oracle noise floor and that a quick residual fit on 40 actual attempts closes half the gap. Adjust world ranges only based on dev runs; document every change.

**E6: DODGE ball scene.** Verify ball model, mass/radius, throw distribution, collision masks, simulation step, logging, and reproducibility. Run actual throws that produce both safe misses and collisions under different supported Go2 commands. No scripted outcome or post-hoc ball movement.

### Gate 3 — teacher and decision engine

**E7: Hosted TabPFN contract.** Verify the pinned client’s exact regressor API, categorical handling, quantile output shape, batching, usage endpoint, and cost quote using official docs/source. First run a non-billable shape/cost check if supported. Before a billable request, inspect current usage and estimate cost; use a small actual Go2-derived dataset and record the request hash, client version, and returned usage.

**E8: Residual and action scoring.** On dev attempts, compare physics-only, constant bias, GP, gradient boosting, and TabPFN residual teacher. For DODGE, measure collision/fall rates and closest separation. Compare physics action, candidate grid, and uncertainty-aware objective. One batched provider request must score the candidates. Log actual API calls and cost.

**E9: Calibration.** Measure 50%/80% coverage and uncertainty ranking from actual teacher outputs on dev episodes. Select confidence thresholds on dev only.

### Gate 4 — dreams, students, and adaptation

**E10: Dream fidelity.** Check teacher-selected actions in the actual Go2 simulator. Measure error vs teacher uncertainty and distance to actual attempts. Try 100/1,000/2,000/10,000 dream counts as budget permits.

**E11: Per-skill MLP.** Compare widths 16/32/64, one/two layers, and actual dream counts. Measure each on fixed dev episodes; select by preregistered criteria only.

**E12: Shared-student retention.** Train DODGE, then add BRAKE, TURN, and RAMP dream sets. After each addition evaluate earlier skills on their fixed dev sets. Do not claim no forgetting without these runs.

**E13: Adaptation.** After a real simulator preset shift, compare all history, recent-only history, and all history plus world ID. Count actual new attempts to recover.

**E14: Novelty and prior work.** Research TabPFN residual physics models/control, decision-policy distillation, shared student rehearsal, and robot-dodge learning. Save sources/dates and state the supported distinction. Do not claim “first” from a quick search.

### Stretch only after all required gates pass

**E15: SNN.** Six-hour maximum. Verify snnTorch APIs, CPU support, encoding, output regression/classification, and license before coding. Measure actual quality and operations. Energy stays an estimate.

**E16: Quantization.** Only after SNN quality is stable; measure actual quality change.

## No invented results

The experiment log and final report are based only on executed runs. Every row must trace to a command, seed/config, software/policy version, and saved output file. Do not fill tables with target values, fixture outputs, hand-entered pass labels, synthetic fake-provider responses, or copied results. Unit fixtures are limited to pure utility checks and are excluded from all performance results.

No API request before live usage and estimated-cost review. If budget or access prevents a real provider call, mark that experiment NOT RUN, report the blocker, and do not represent a stub as TabPFN.
