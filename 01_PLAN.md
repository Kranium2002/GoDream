# Learned Judgment — lead plan

You are the lead engineer for the HackLab × BRISA Robotics & AI Hackathon in Berlin, Prior Labs track. This is a Unitree Go2 decision-learning project. Read MISSION.md first. Person A and Person B work from 05_PERSON_A.md and 06_PERSON_B.md.

## Product

A simulated Go2 uses a verified pre-trained walking policy. A shared PFN/TabPFN engine learns multiple high-level decision policies from 10–40 attempts per skill. Its first and main skill is dodging a thrown ball. Other planned skills are brake timing, turn timing, ramp speed, and ball kicking. The learned policy may use only commands verified in the borrowed walking brain's actual interface.

Pitch: “The legs are borrowed. The judgment is learned: one engine, many skills, each from a handful of attempts, and it never stops learning.”

The walking policy is not trained or changed. TabPFN never controls motors or runs in the fast loop. No physical robot is used. Everything is simulated and labeled.

## Architecture

Layer 1: verified Go2 walking policy at its verified control rate. Layer 2: its verified high-level command interface. Layer 3: generic decision engine and SkillSpecs.

Every skill follows the same path:

1. Propose candidate high-level commands and predict outcomes with the skill's physics_guess.
2. Fit TabPFN to residual = measured outcome − physics_guess from situation, action, and physics guess.
3. Score candidate outcomes, including uncertainty, with that SkillSpec's objective. Send all candidates in one batched TabPFN request and choose the best.
4. Dream over sampled situations; keep decisions passing a confidence threshold fixed on dev data.
5. Train a per-skill spiking neural network (SNN) inside the dream, then a shared SNN with skill ID and dream rehearsal. The SNN is the only student. It learns decisions from dreamed outcomes; it is not a copy of the PFN.
6. Add actual attempts after a world shift; add new skills through new SkillSpecs without engine-specific branches.

## Phase order and hard gates

### Phase 1 — Verify the real Go2 walking stack (H+3)

Use the exact order from 02_ENVIRONMENT.md and 05_PERSON_A.md: first Efferent's Go2 walk example, then the rl_sar Go2 policy and its exact config/observation/action setup, then Quadruped-PyMPC. Research and verify package versions, model files, API shapes, licenses, installation, and CPU runtime before implementation. Timebox each attempt as specified, but do not switch robot, controller, or simulator to hide a failure. If all verified approaches fail, stop and report the blocker; do not proceed as if the Go2 works.

Acceptance: actual Go2 trots for 30 seconds at commanded 0.3, 0.6, and 0.9 m/s; responds to turn and stop commands; stays standing. Save the actual MuJoCo run, exact commands/config, policy/model identity, and headless real-time factor.

### Phase 2 — World, command API, and environment report

Person A owns Go2/MuJoCo integration, command mapping, hidden world effects, ball scene, config.yaml, dashboard hooks, and ENV_REPORT.md. Lock a public command/observation contract before Person B integrates. Verify hidden-effect difficulty per skill with actual simulator runs.

### Phase 3 — Decision engine and skills

Person B owns the generic engine and SkillSpecs. Build DODGE first after Phase 1 passes. The dog must perceive the simulated ball state, select a supported dodge command and timing, and have outcomes measured by the true simulator. Then implement BRAKE, TURN, and RAMP in that order. KICK is stretch.

### Phase 4 — Dreams, students, adaptation

Start with 2,000 dreams per skill, selecting confidence thresholds on dev sets. Train one SNN per skill, then a shared SNN with dream rehearsal. Test continual learning in DODGE → BRAKE → TURN → RAMP sequence. After a world change, measure actual attempts required for recovery. The SNN is the only student; no MLP student is built.

### Phase 5 — Course demonstration

Course: walk → dodge a thrown ball → turn through a gate → climb a ramp → brake on a mark. Show per-skill attempt tables, physics guess versus TabPFN-corrected decisions, real MuJoCo render, and measured results. Change payload/floor between runs and show measured adaptation across skills.

### Phase 6 — Final evaluation

Freeze PREREGISTRATION.md before tuning. Follow 03_METRICS.md. Use separate dev, test, and shift sets per skill; at least 3 seeds (5 if budget allows), paired comparisons, and bootstrap 95% intervals. Test data is final-only.

## Shared data and interfaces

Attempt row: run_id, seed, skill, world_preset, source (real | dream), situation, action, physics_guess, measured_outcome, residual, timestamp_utc. Use documented structured encoding for nested objects. “Real” means an actual Go2 MuJoCo episode.

Person A owns the stable environment API. Minimum contract:

    World(config, seed)
    World.reset(situation) -> observation
    World.step(command, duration_s) -> observation
    World.finish_episode() -> measured_outcome

Observations include the verified dog state, ball state when used, and contacts/fall flags. Commands are restricted to the exact high-level fields/ranges exposed by the selected walking policy. Person A publishes that command schema and maps it to the native policy interface. Person B consumes this public API only.

Each SkillSpec contains name, description, situation features/types/ranges, 1–3D action space, outcome type, physics_guess(situation, action), objective(predicted distribution), episode(world, situation, action), and hidden-effect notes. The generic engine contains no skill-specific conditionals.

## API, evidence, and honesty

- CPU-only laptops. Use hosted tabpfn-client unless a small verified CPU test establishes local inference is fast and licensed.
- Check current usage and quote cost before a billable call. Batch candidate actions and dreams. Cache complete requests and responses keyed by all data/settings; never pay twice for identical inputs. Log usage/cost without credentials.
- Use UltraCode research agents before each production slice to verify exact current docs, library APIs, policy files, versions, licenses, and a minimal executable proof. If unavailable, use an independent research agent or primary-source research and record the substitution.
- Tests are executed against actual project code. System claims require the actual Go2 MuJoCo model and selected policy. Record command, exit code, versions, seed/config, and artifacts. No fake policy/provider, hardcoded metric, fabricated pass, cherry-picking, or test tuning.
- Results tables are calculated from saved outputs. An unexecuted check is NOT RUN, never PASS. A fixture may test pure utilities but cannot become a model-performance result.
- If a required capability fails, stop at that gate and fix it. No alternate robot, controller, simulator, hand-coded behavior, animation, or project.
- Label simulation, execution location, and estimated energy honestly. Do not claim novelty, hardware readiness, or energy savings without evidence.
- Keep LOG.md current and record decisions in DECISIONS.md. Tag working commits demo-safe-N.

## Acceptance before calling the project complete

- The selected Go2 policy passes Phase 1 with recorded evidence.
- DODGE uses the real simulated ball and Go2, and actual tests show the complete state → decision → supported command → measured outcome path.
- TabPFN teacher and request cache pass a real provider check within the reviewed API budget.
- Physics-only, constant-bias, teacher, per-skill SNN, and shared SNN results are computed from the same preregistered dev situations; all hypotheses and failures are reported.
- No critical acceptance item is OPEN. No report contains invented or placeholder pass results.

Submission is Sunday 22 November 2026 at 17:45 (confirm on site); pitches at 19:00. Gates are relative to kickoff; keep T−8h evaluation, T−5h feature freeze, and T−3h backup/rehearsal milestones.
