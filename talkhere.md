# Punch-dodge demo plan — revised after webcam smoke test

Date: 2026-11-21 (Europe/Berlin)
Owner split: Dev A = MuJoCo world and product UI; Dev B = PFN/SNN learning and evaluation.

## Product goal

The flagship demo teaches a simulated Unitree Go2 to react to an approaching punch in MuJoCo. The dog should stay still when no threat is present, then choose a safe dodge when a threat arrives. The user sees real simulation attempts, PFN-guided learning, the SNN training curve, held-out metrics, and finally a live laptop-camera view driving the trained local SNN while the simulated dog responds.

The laptop camera is an inference sensor only. PFN training and dream generation use simulated Go2 episodes and rendered MuJoCo camera pixels. Live camera frames remain local and go directly through deterministic preprocessing to the SNN. PFN never runs per live frame. The slope lesson remains a secondary skill after the punch-dodge vertical slice passes its gates.

## What the webcam smoke test established

Three five-second clips were recorded locally. Each sampled image used 12×12 grayscale pixels plus one frame-difference channel, flattened to 288 pixel values. The hosted TabPFN regressor received those values as a numeric table alongside three candidate actions and a fixed prior-utility column (292 features total). A real hosted request completed; the project's dense snnTorch student then trained on PFN-generated action targets.

Observed smoke results, from `.cache/camera_punch_dataset/results.json`:

- PFN context: 96 candidate rows; query: 18 frames; hosted held-out score: 3/5 correct (60%).
- SNN: 13 PFN dream rows for training and 5 for validation; action accuracy on the five-frame held-out slice: 1/5 (20%); agreement with PFN: 2/5 (40%).
- SNN training ran for 8.5 seconds. Offline forward latency was about 4.9 ms for a batch of five rows (p95 5.9 ms).
- The request used the real hosted client (`tabpfn-client` 0.10.1, v3.5 model). The call was sent at the user's direction without a usage or cost preflight. Exact numeric request/response data and videos are in the ignored local `.cache` directory.

This confirms the basic plumbing: tiny flattened image features can be submitted to hosted tabular TabPFN, and PFN outputs can train the current dense SNN. The five-frame result is too small and too weak to support a dodge-accuracy claim. The test did not use MuJoCo, simulation-derived examples, a no-threat class, or Go2 control. “More data will improve accuracy” remains a hypothesis to test in simulation.

## Input and action contract

### Image sequence

The two-channel smoke-test input loses too much motion information. Start the simulation experiment with a rolling window of **8 consecutive 12×12 grayscale frames** sampled at 30 Hz. Flatten the ordered frames to 1,152 pixel features; the window covers about 233 ms from first frame to last. Compare this with a 4-frame and a 6-frame variant on development seeds before freezing the stack. Measure the extra observation delay, TabPFN request dimensions, SNN accuracy, and local CPU latency for each variant. Keep size, cadence, frame count, normalization, and channel order in configuration.

The MuJoCo head camera and laptop camera must use the same crop, grayscale conversion, resize, normalization, frame order, and cadence. No learned detector or external image model is in the live path. The current project student is a dense SNN that accepts flattened pixels; a convolutional SNN has not been tested and is not an assumed dependency.

### Motion representation experiments

Use the ordered grayscale stack as the first baseline. Compare three deterministic/model-side alternatives on the same seeded simulation split:

- Add signed or absolute differences between adjacent frames, optionally pooled over small image tiles. This highlights where image content moved while keeping preprocessing simple.
- Feed the SNN frames sequentially and preserve its membrane state across the window. The current student resets state at each `forward` call, so this requires an explicit sequence-training/runtime change and a parity test.
- Benchmark classical CPU optical flow (for example, OpenCV Farnebäck) as an optional feature source for direction and speed. It is not a learned model, but it adds compute and can be sensitive to background motion; retain it only if measured accuracy gains justify its latency.

Keep each feature variant matched across MuJoCo and laptop preprocessing. Split evaluation by complete trajectories and seeds. Choose the representation using dev results for action accuracy, no-threat false activations, misses, and end-to-end latency; do not choose from adjacent-frame random splits.

### Four actions

The student and teacher compare four candidates on every situation:

1. `NO_OP`: send `vel_x=0`, `vel_y=0`, and `yaw_rate=0` using the verified schema; measure any residual drift and only use it when the scene is safe.
2. `DODGE_LEFT`: use the verified lateral command in the dog's frame.
3. `DODGE_RIGHT`: use the opposite verified lateral command.
4. `RETREAT`: use the verified backward command.

Build command values from the current `config.yaml` schema and validate them through the existing command contract. No motor-level actions or invented command ranges. The physics outcome scorer can see true simulated object pose and collision state; the teacher and SNN receive only permitted image/action/prior features.

### Temporal smoothing and action cooldown

The SNN may evaluate each camera update, while a command adapter controls when a new action reaches MuJoCo. Smooth action scores across a short rolling window (begin with 3–5 consecutive SNN outputs; select settings on development runs). Require a calibrated threat/action confidence threshold before issuing an active dodge. `NO_OP` is the learned class for safe/no-threat situations. Show uncertainty in the UI and count uncertain threats as misses during evaluation.

After any issued left, right, or retreat action, enforce a **minimum 1.0-second active-action cooldown** before another active command can be issued. Continue observing during cooldown; an immediate `NO_OP`/zero command is always allowed to stop motion safely. Test the cooldown against two threats close together, since a lockout can also block a necessary second dodge. Measure the combined delay from the 8-frame history, score smoothing, command cooldown, and Go2 response. Tune only on development runs, never on the held-out set.

## Simulation-first learning design

1. Add a calibrated Go2 head camera in the actual MuJoCo scene and a physically colliding incoming punch proxy. Generate both threatening and harmless scenes. Log true trajectories, contact, time-to-contact, minimum body clearance, fall, recovery, and command travel as scoring-only outcomes.
2. Generate deterministic, seeded sequences spanning no threat, distant/slow approach, left/right/center approach, varied speed and starting distance, late/early warning, and visual distractors. Include enough idle windows to measure false activations. Split by seed and trajectory so adjacent frames from one clip or episode cannot leak across training and test.
3. For actual Go2 MuJoCo attempts, score all four supported candidates with the declared physics model and measured outcomes. Train hosted TabPFN on simulation-derived feature rows and residuals. Preserve the production teacher's exact request/response cache and call log.
4. Generate PFN dream rows over held-out development situations and four candidate actions. Use the SkillSpec objective to turn predicted outcomes and uncertainty into soft action targets. Train the local SNN on those PFN dream targets and image sequences.
5. Evaluate physics-only, PFN teacher, and SNN on the same paired MuJoCo situations and independent seeds. Report no-threat false-positive rate, threat miss rate, action accuracy, collision rate, falls, minimum separation, recovery, cooldown conflicts, action count, calibration, and end-to-end latency.
6. After simulation gates pass, run the webcam locally through the identical preprocessing and SNN. Display the chosen action and confidence, cooldown state, and the simulated Go2 response. Measure camera age and camera-to-command delay under rendering load.

Do not train the production model from the three human clips. They were a one-off wiring test. The upcoming learning claims must come from actual simulated attempts and PFN dreams. Keep the clips and their cached pixel values local; they are not part of the simulation training set.

## Evidence and gates

**G0 — World and sensors:** real Go2 MuJoCo model, verified walking policy, physical punch proxy, calibrated head camera, and matching simulation/live preprocessing. Hidden ground truth stays out of policy features.

**G1 — Four-action dataset:** simulation rows contain image sequence, candidate, prior, measured outcomes, and residuals. Include `NO_OP` examples from truly safe scenes. Reproduce counts and splits from saved run data.

**G2 — PFN and SNN learning:** real hosted teacher call through the production teacher/cache; PFN dream targets; successful SNN training; improvement over the fixed no-op/physics baselines on development data. The current five-frame webcam result does not satisfy this gate.

**G3 — Held-out simulation:** independent seeds and trajectories; report all safety and action metrics with uncertainty. Compare the teacher and SNN. Require a measured improvement over no-op and physics-only, and SNN performance close enough to the PFN teacher under a preregistered tolerance. If the cooldown causes missed threats, surface that failure.

**G4 — Live camera loop:** local SNN only, 8-frame stack (or a measured alternative), smoothing, confidence gate, and ≥1.0 s active-action cooldown. Run at least 20 controlled approaches plus no-threat windows. Measure frame age, preprocessing, SNN, command, and total response latency; compare with the shortest dodgeable time measured in simulation. Do not claim a successful live dodge before these checks pass.

**G5 — Product story:** chat instruction → real simulation attempts → visible PFN/SNN learning → held-out results → local live camera preview driving the simulated dog. Every animation and metric comes from the actual run. The slope lesson follows after this primary vertical slice, time permitting.

## Revised 12-hour parallel plan

| Window | Dev A — world/UI | Dev B — learning/backend | Exit evidence |
|---|---|---|---|
| 0–1 h | Confirm the primary punch-dodge story and typed backend/UI events. | Preserve current WIP; define the four-action spec and image-sequence schema with A. | Shared action names, units, feature order, errors, and events agreed. |
| 1–3 h | Build the actual head camera and colliding punch proxy; include safe/no-threat scenes. | Implement configurable 4/6/8-frame preprocessing, four candidate actions, and cooldown/smoothing adapter. | Sensor tensors match; command contract validates every action. |
| 3–5 h | Add seeded, visible simulation scenarios and event streaming. | Collect real MuJoCo attempts and validate teacher request shape/cache on simulation data. | Reproducible simulation rows; no webcam pixels sent to PFN. |
| 5–8 h | Build teach, training, and result screens from live run events. | PFN residual/dream pipeline; train SNN and compare against baselines on development data. | Actual learning curve and measured dev comparison. |
| 8–10 h | Wire local webcam preview beside the simulated dog's head view. | Run SNN-only live inference with temporal smoothing and ≥1 s cooldown; measure latency. | G3 passes before any success claim; misses remain visible. |
| 10–12 h | Polish the synchronized demo and rehearsal flow. | Freeze settings, run held-out evaluation, publish artifacts and limitations. | One repeatable end-to-end demo and honest final metrics. |

The 12-hour plan is a prioritization, not a waiver of gates. If simulation or timing gates fail, show the measured result and identify the blocker; do not substitute a staged dodge.

## Dev A prompt — copy-ready assignment

You own `app/` and the Go2 MuJoCo world/camera files in your lane. Read `CLAUDE.md`, `MISSION.md`, and the numbered prompts. Coordinate the four-action and image-window contract with Dev B before wiring UI.

Build the product side of the primary punch-dodge demo:

1. Create a real Go2 head camera and a physically colliding incoming punch proxy in MuJoCo. Provide threatening trajectories and safe/no-threat scenes with deterministic scenario IDs/seeds. Keep object truth out of policy inputs.
2. Create the chat flow for “teach the dog to dodge an approaching punch, and stay still when safe.” Show the validated action contract and useful corrections/errors.
3. Show real simulation attempts, PFN teacher/dream progress, SNN training, and held-out results. Drive all progress and numbers from actual backend events/artifacts.
4. Add a live laptop-camera panel that sends local, deterministically preprocessed frame windows to Dev B's SNN runtime. PFN is never called per frame; do not upload live camera frames or write them to disk.
5. Render the actual simulated Go2 response beside the live camera view. Display `NO_OP`/left/right/retreat, confidence, smoothing, cooldown, frame age, command latency, and measured outcomes. Make all gate failures visible.
6. Keep the original slope chat as a secondary follow-on after punch-dodge gates. Do not stage an animation or present the webcam smoke score as a product metric.

Do not edit B-owned `skills/`, `engine/`, `teacher.py`, `students/`, `eval/`, pipeline, or B experiment results. Before handoff, list changed files, exact run/artifact evidence, and any failed/open gate.

## Dev B prompt — copy-ready assignment

You own SkillSpecs, the shared engine, hosted TabPFN teacher/cache, SNN student, experiments, and evaluation. Read `CLAUDE.md`, `MISSION.md`, and the numbered prompts. Preserve existing WIP and coordinate the public contract with Dev A.

Build the simulation-trained four-action visual dodge backend:

1. Define `NO_OP`, `DODGE_LEFT`, `DODGE_RIGHT`, and `RETREAT` using only verified high-level Go2 commands. Validate each candidate against `config.yaml` and the command schema.
2. Add the configurable ordered frame stack (test 4/6/8 grayscale frames at 12×12 initially, around 30 Hz). Measure the time window, feature count, PFN request behavior, SNN accuracy, and local CPU latency. Do not assume a larger stack is automatically better.
3. Use actual MuJoCo attempts: image sequence + candidate action + physics guess as teacher inputs; measured collision, clearance, fall/recovery and command cost as outcomes. Train the hosted residual teacher and generate PFN dreams for the SNN. Use the production teacher/cache path for this work.
4. Include balanced no-threat scenes; the safe target is `NO_OP`. Split by trajectory and seed, not adjacent frames. Build dev/held-out reports from saved data only.
5. Implement a temporal score smoother and a command adapter with at least 1.0 s cooldown between active dodge commands. Evaluate repeated threats during lockout, confidence thresholds, false positives, and misses; expose state for Dev A.
6. Compare physics-only, PFN teacher, and dream-trained SNN on paired MuJoCo seeds. Report no-threat false activations, threat misses, collisions, falls, minimum clearance, cooldown conflicts, latency, and uncertainty. Treat the current webcam smoke as wiring evidence only: 96 PFN rows, 18 query frames, PFN 3/5, SNN 1/5; it did not use simulation.

Do not claim accuracy improvement from the three webcam clips. Do not add another learned vision model, create a Conv-SNN assumption, bypass the verified robot interface, or tune on held-out data. For handoff, provide contracts, exact commands, versions, seeds/config, artifacts, and every open gate.

## References and client note

- [PriorLabs TabPFN Client](https://github.com/PriorLabs/TabPFN-client) — hosted TabPFN is a tabular service; image tensors enter it as numeric feature columns.
