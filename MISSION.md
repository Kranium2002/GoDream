# Learned Judgment

## Mission

Build a simulated Unitree Go2 robot dog that learns useful high-level decisions from a small number of attempts. Dodging a thrown ball is the primary behavior. The same engine must also support skills such as braking, turning through a gate, ramp approach, and a ball kick without skill-specific branches inside the engine.

Use a verified pre-trained Go2 walking policy for locomotion. PFN/TabPFN learns when and which supported high-level command to issue. It does not control the 12 motors directly, and its learning does not change the borrowed walking-policy weights. Every candidate action must use commands and ranges verified for the exact selected walking policy.

Everything runs in simulation on CPU-only laptops. There is no physical robot deployment. Reports must say when results are simulated and where inference ran.

## How each skill learns

1. **Physics proposes:** a SkillSpec formula predicts what a command will do.
2. **TabPFN learns the residual:** measured outcome minus physics prediction, from observed situation, candidate action, and physics prediction.
3. **The engine decides:** score bounded supported command candidates in one batched TabPFN call, combine physics and predicted residual, account for uncertainty through the SkillSpec objective, and choose the best command.
4. **TabPFN dreams:** generate decisions for many sampled situations and keep decisions passing a confidence threshold selected on dev data.
5. **The SNN learns inside the dream:** the physics guess corrected by the PFN is the dream world. For each dreamed situation it predicts the outcome of every candidate command, and a spiking neural network (SNN) is trained there to choose the best one. Train one SNN per skill first, then a shared SNN with skill ID and dream rehearsal. The SNN is the only student; no MLP student is built. The SNN is not a copy of the PFN: it never learns the PFN's predictions, only decisions.
6. **The engine adapts:** add actual simulator attempts after payload, floor, slope, motor strength, or latency changes. Add skills as SkillSpecs, not engine branches.

Use about 10–40 attempts per skill. The primary demo shows the dog initially colliding with or failing to evade a simulated ball, then improving its dodge decision from attempts.

## Scope

- Primary skill: dodge a thrown ball.
- Planned skills: brake timing, turn timing/yaw rate, ramp approach speed, and ball-kick speed/angle.
- The only robot is Unitree Go2. The only walking policy is the exact borrowed policy selected and verified for this project.
- The borrowed walking brain remains frozen. PFN learns decision policies over the verified high-level commands available to it.
- Do not substitute another robot, controller, simulator, kinematic animation, recorded success, hand-coded dodge policy, or fabricated model result. If a critical Go2 gate fails, stop and resolve it before proceeding.

## Three layers

    Layer 3  DECISIONS   SkillSpecs and learned command choices       BUILD
    Layer 2  COMMANDS    verified high-level command interface        USE
    Layer 1  WALKING      verified pre-trained Go2 policy, about 50 Hz BORROW

The exact Layer 2 fields, ranges, units, signs, and update rate must come from the selected walking policy's source/config and a verified run. Do not invent command names or ranges.

## SkillSpec and shared engine

One file per skill under skills/. A SkillSpec contains: name and description; observed situation features, types, and ranges; bounded 1–3 dimensional candidate action space using only supported commands; outcome type (regression, classification, or both); physics_guess(situation, action); risk-aware objective(predicted_outcome_distribution); episode(world, situation, action) returning measured outcome; and documentation of relevant hidden effects.

The generic engine provides collect, teacher, decide, dream, train-in-dream, and adapt. Skill-specific logic lives in SkillSpecs. Score a candidate set in one batched TabPFN request and include the physics action among candidates.

Shared attempt row:

    run_id, seed, skill, world_preset, source, situation, action,
    physics_guess, measured_outcome, residual, timestamp_utc

Source is real for an actual Go2 MuJoCo episode and dream for a teacher-generated decision. Store structured values in a documented format. Residual is measured outcome minus physics guess, using a fixed outcome order per skill. Never expose hidden world values as teacher features.

## Go2 world

True world: MuJoCo with the verified Go2 model and walking policy. Hidden effects include payload mass 0–3 kg and off-center position (declared mass can be wrong by up to 20%), floor friction by surface with per-run variation, slope up to ±5°, motor strength 80–100%, command latency 0.05–0.3 seconds plus jitter, and a simulated ball with recorded initial position/velocity and collision outcomes for DODGE.

Presets in config.yaml: default, heavy_payload, slippery_floor, tired_motors, downhill. Every tunable is in config. Keep fixed world-level variation separate from per-attempt noise and use seeded random streams.

The evaluation-only oracle searches supported actions in the true world with repeated noise draws. Its remaining error is the noise floor. Keep oracle sets small (about 100 situations) and parallelize independent episodes only after deterministic behavior is established.

Per skill, ENV_REPORT.md records measured physics-only error and oracle noise floor. Target difficulty: physics-only error at least 3× the noise floor, and a residual fit on 40 actual attempts closes at least half the gap. If calibration misses, investigate the simulator; never change or fabricate report values.

## Skills and order

1. Verify the Go2 walking policy and simulator contract before building learned decisions.
2. DODGE (primary): throw a simulated ball toward the dog. Situation contains observed dog/ball state and relevant declared world values. Candidate actions select a supported dodge side and command timing/parameters. Measure closest ball/body separation, collision, fall, recovery time, and command cost. The objective strongly penalizes collision and falling, then rewards separation and low unnecessary motion.
3. BRAKE: trot toward a mark; choose stopping distance/timing. Baseline stopping distance is speed × latency + speed²/(2 × deceleration). Measure signed stop error and fall.
4. TURN: pass a laterally offset gate; choose turn start and supported yaw command. Measure lateral miss and collision.
5. RAMP: climb a ramp; choose approach speed. Measure success/slip/fall and elapsed time.
6. KICK (stretch): move a ball to a target; choose speed and approach angle. Measure target error and success.

For every skill, validate its spec, measure the physics-only baseline, compare the teacher on dev data, then train and measure the SNN student. Do not start a later skill while a required Go2 or DODGE gate is unresolved.

## Students and continual learning

- The only student is a spiking neural network (SNN). No MLP student is built.
- Start with one SNN per skill.
- Then train one shared SNN with skill ID and dream rehearsal across skills.
- Measure old skills after adding each new skill in sequence: DODGE → BRAKE → TURN → RAMP.
- Test whether world ID or a payload/surface table helps a new skill learn faster; report either result.
- The SNN student does not replace or bypass the Go2, DODGE, or evaluation gates. It trains on actual teacher dreams and is measured on actual Go2 MuJoCo episodes, with spike counts, synaptic operations, and time steps reported. Energy is an estimate.

## Evaluation

Freeze PREREGISTRATION.md before tuning. Use independent dev, final test, and shift sets per skill. Use paired situations/noise, at least 3 seeds (5 if budget allows), and paired bootstrap 95% intervals. Keep test sets untouched until final evaluation.

Per skill report regression error or classification success/fall rate, skill-specific tolerance rate, regret versus oracle, learning curves at 0/5/10/20/40 attempts, recovery after world changes, interval coverage, dream fidelity in the true simulator, latency, API usage, and limitations.

Baselines: oracle, physics only, constant residual bias, Gaussian-process and gradient-boosting residuals, MLP on actual attempts, direct TabPFN without physics features, residual teacher, per-skill SNN, and shared SNN.

results/final_table.md has one row per skill × method. Save learning curves per skill, adaptation and continual-learning results, and RESULTS.md with each preregistered hypothesis marked supported, not supported, or inconclusive.

## Evidence rules

- No hardcoded observed metrics, prewritten pass output, fake episodes, fake teacher, fake policy, cherry-picking, or test-set tuning.
- Tests execute against the actual Go2 simulator/policy and actual production components. Record command, exit code, versions, seed/config, and output artifact.
- A fixture may test a pure function, but its output cannot enter performance tables or be described as Go2/TabPFN evidence.
- Reports are generated from saved run outputs. A planned or unexecuted check is NOT RUN, never PASS.
- A TabPFN claim requires an actual request to the configured provider after current usage and estimated cost are reviewed. If it cannot be run, mark NOT RUN; do not replace it with a stub.
- Cache every request and response by a hash of all inputs and settings. Log calls, usage, and cost without credentials.
- Research library, policy, model, and robot-interface assumptions before coding each slice. Cite primary sources and save proof-of-concept commands and outputs. If unsupported, stop and resolve the gate; do not switch robot or project.
- Do not claim novelty, hardware readiness, energy efficiency, or improvement unless measurements support it. Label estimated energy.

## Course demo and gates

Course: walk → dodge a thrown ball → turn through a gate → climb a ramp → brake on a mark. Each segment is decided by its SkillSpec and logged to that skill's table. Between runs a judge changes payload or floor; the dog initially struggles, then adapts.

| Gate | Required evidence |
|---|---|
| H+3 | Verified Go2 policy walks 30 seconds, moves at 0.3/0.6/0.9 m/s, turns, stops, and remains standing; measured headless real-time factor |
| H+6 | Generic engine integrated with primary DODGE SkillSpec; actual simulator attempts and first teacher comparison |
| H+9 | BRAKE and TURN run through verified commands |
| H+12 | RAMP and dashboard showing actual MuJoCo episodes and per-skill results |
| T−14h | SNN student (per skill, then shared) trained on actual teacher dreams and measured |
| T−8h | Final evaluation |
| T−5h | Feature freeze |
| T−3h | Backup video, slides, and two rehearsals |

If a gate fails, stop and fix that gate. No substitute robot/controller, fake success, or scope change. The project is not complete until Go2 DODGE and its measured report pass their acceptance criteria.
