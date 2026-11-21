# Simulation-trained PFN→SNN punch-dodge demo

Date: 2026-11-21 (Europe/Berlin)
Owner split: Dev A = MuJoCo world and product UI; Dev B = PFN/SNN learning and evaluation.

## Product goal

The flagship demo teaches a simulated Unitree Go2 to react to a fast incoming object in MuJoCo. The dog should stay still when no threat is present, then choose a safe dodge when a threat arrives. The user sees real simulation attempts, PFN-guided learning, the SNN training curve, held-out metrics, and finally a webapp camera view driving the frozen local SNN while the simulated dog responds.

The laptop camera is for final inference testing only. Every train, dev, validation, dream, and offline held-out example used to fit or tune PFN/SNN comes from simulation. During the live test, browser-captured frames stay in memory on the same computer and go through fixed preprocessing to the frozen SNN. Do not fit, fine-tune, adapt, tune thresholds, send real-camera pixels to PFN, or store frames/features. Store prediction/timing summaries only. PFN never runs per live frame. The slope lesson remains a secondary skill after the punch-dodge vertical slice passes its gates.

## Starting point and excluded prototype

The current `app/` contains an episode viewer and hooks, not the requested camera webapp. The current SNN can export a checked NumPy deployment model, but its `SkillSpec`/dream path must be extended for fixed-size image-window features.

Ignore the earlier webcam prototype for product learning and claims. Its human clips, derived pixel values, cache entry, checkpoint, and metrics are excluded from every production training, tuning, and evaluation run. The only production training source is MuJoCo-generated simulation data; the live webapp camera is used only after the simulation-trained model is frozen.

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

1. Add a calibrated Go2 head camera in the actual MuJoCo scene and a physically colliding incoming punch/ball. Generate threatening and harmless scenes. Randomize simulation-only object geometry, color, backgrounds, lighting, blur, and camera exposure to reduce the sim-to-live gap. Log true trajectories, contact, time-to-contact, minimum body clearance, fall, recovery, and command travel as scoring-only outcomes.
2. Generate deterministic, seeded image sequences for no threat, harmless/off-path motion, and left/right/center threats at varied speed, distance, and warning time. The policy sees rendered pixels only. Raycasts, exact ball pose/velocity, and collision truth may inform physics guesses and scored outcomes, but stay out of policy features unless an equivalent deterministic browser-camera conversion is designed and validated.
3. Collect a small, varied context of actual MuJoCo action outcomes across all four supported commands. For each executed situation/action, record the image window, action candidate, physics-predicted outcomes, measured outcomes, and residuals. Split complete trajectories and seeds; never split neighboring frames across train and test.
4. Fit the hosted TabPFN residual teacher on those simulation-only rows through the production teacher, request cache, and call log. Candidate features are the flattened ordered image window, verified action fields, and physics guess; targets are measured-minus-physics outcomes. Verify the selected hosted model accepts the feature width and batch sizes.
5. Sample many additional physically valid simulated situations and render their image windows without running a complete action-outcome episode for each candidate. Query PFN in batches for all four actions. Combine physics guesses and PFN residual distributions, then apply the SkillSpec objective and uncertainty penalty to produce soft action scores for `NO_OP`, left, right, and retreat. These are PFN dreams, not measured ground truth.
6. Train the SNN only on those simulation-rendered image windows and PFN dream targets. The SNN loss teaches its action readout to match the PFN's soft action distribution; spike activity is produced by learned weights and the snnTorch dynamics. Export the trained student to the checked local NumPy inference format and run the existing parity checks.
7. Validate dream predictions against a held-out subset of actual MuJoCo action rollouts. Then evaluate the frozen PFN teacher and SNN on paired held-out MuJoCo seeds. Report no-threat false activations, threat misses, action accuracy, collisions, falls, minimum separation, recovery, cooldown conflicts, calibration, and latency. Do not use the held-out episodes to tune the teacher, SNN, smoother, or thresholds.
8. After simulation gates pass, launch the local webapp. The browser opens the camera with an explicit user start action, keeps a rolling frame window in memory, and sends only the preprocessed window to same-computer SNN inference. The frozen model returns a prediction; the validated command adapter drives the MuJoCo Go2. Keep the PFN offline. Log action/timing summaries only and never use the live session to train or tune.

### What PFN contributes

TabPFN is a hosted tabular predictor, not an image/video generator. MuJoCo creates all image windows. PFN uses a smaller set of measured simulation outcomes plus physics features to predict candidate-action outcomes for more sampled simulation situations. This avoids a full outcome rollout for every dream candidate, while actual MuJoCo rollouts remain the source of teacher context and validation truth. Dream-label errors must be measured, not hidden.

**Production data rule:** no real-world frame, label, pixel feature, or camera-derived measurement may enter PFN fitting, SNN training/distillation, validation, threshold selection, or offline evaluation. All of those use simulation-generated data. Real webcam frames are used only for the final frozen-model inference test; its aggregate predictions and timing may be logged, but never recycled into training or tuning. Do not upload or persist live frames/features.

The three human clips and the cached features/checkpoint from the one-off feasibility smoke are excluded from all future training and reports. The upcoming learning claims must come from actual simulated attempts and PFN dreams.

## Evidence and gates

**G0 — World and sensors:** real Go2 MuJoCo model, verified walking policy, physical punch proxy, calibrated head camera, and matching simulation/live preprocessing. Hidden ground truth stays out of policy features.

**G1 — Four-action dataset:** simulation rows contain image sequence, candidate, prior, measured outcomes, and residuals. Include `NO_OP` examples from truly safe scenes. Reproduce counts and splits from saved run data.

**G2 — PFN teaches the SNN:** real hosted teacher call through the production teacher/cache using simulation rows; PFN dream scores for simulation-rendered situations; SNN trains successfully on those dream targets; export parity passes. Validate a sample of dream predictions against actual MuJoCo outcomes. The prior webcam prototype is excluded and does not satisfy this gate.

**G3 — Held-out simulation:** independent seeds and trajectories; report all safety and action metrics with uncertainty. Compare the teacher and SNN. Require a measured improvement over no-op and physics-only, and SNN performance close enough to the PFN teacher under a preregistered tolerance. If the cooldown causes missed threats, surface that failure.

**G4 — Webapp live inference:** freeze the simulation-trained SNN first. The local webapp opens the browser camera and runs the same frame preprocessing and 8-frame stack (or a measured alternative). Only preprocessed features reach the same-computer inference adapter; raw frames stay in browser memory and never go to hosted PFN or disk. The validated command adapter applies smoothing, a confidence gate, and ≥1.0 s active-action cooldown before driving MuJoCo. Run at least 20 controlled approaches plus no-threat windows. Measure frame age, preprocessing, SNN, command, and total response latency; compare with the shortest dodgeable time measured in simulation. Do not fit/adapt/tune on live frames or outcomes.

**G5 — Product story:** chat instruction → real simulation attempts → visible PFN/SNN learning → held-out results → local live camera preview driving the simulated dog. Every animation and metric comes from the actual run. The slope lesson follows after this primary vertical slice, time permitting.

## Revised 12-hour parallel plan

| Window | Dev A — world/UI | Dev B — learning/backend | Exit evidence |
|---|---|---|---|
| 0–1 h | Confirm the primary punch-dodge story and typed backend/UI events. | Preserve current WIP; define the four-action spec and image-sequence schema with A. | Shared action names, units, feature order, errors, and events agreed. |
| 1–3 h | Build the actual head camera and colliding punch proxy; include safe/no-threat scenes. | Implement configurable 4/6/8-frame preprocessing, four candidate actions, and cooldown/smoothing adapter. | Simulation and browser produce the same tensor contract; command contract validates every action. |
| 3–5 h | Add seeded, visible simulation scenarios and event streaming. | Collect simulation-only outcomes; fit/cache PFN teacher and generate rendered dream queries. | Reproducible simulation rows; PFN request uses no real camera data. |
| 5–8 h | Build teach, training, and result screens from actual run events. | Train/export PFN-distilled SNN; validate dream fidelity and held-out MuJoCo episodes. | Actual learning curve, checked export, and measured held-out results. |
| 8–10 h | Wire browser camera preview beside the simulated dog's head view. | Run frozen SNN through local inference adapter with smoothing and ≥1 s cooldown; measure latency. | Live loop uses local camera only; misses and false activations remain visible. |
| 10–12 h | Polish the synchronized demo and rehearsal flow. | Freeze settings, run held-out evaluation, publish artifacts and limitations. | One repeatable end-to-end demo and honest final metrics. |

The 12-hour plan is a prioritization, not a waiver of gates. If simulation or timing gates fail, show the measured result and identify the blocker; do not substitute a staged dodge.

## Dev A prompt — copy-ready assignment

You own `app/` and the Go2 MuJoCo world/camera files in your lane. Read `CLAUDE.md`, `MISSION.md`, and the numbered prompts. Coordinate the four-action and image-window contract with Dev B before wiring UI.

Build the product side of the primary punch-dodge demo:

1. Create a real Go2 head camera and a physically colliding incoming punch proxy in MuJoCo. Provide threatening trajectories and safe/no-threat scenes with deterministic scenario IDs/seeds. Keep object truth out of policy inputs.
2. Create the chat flow for “teach the dog to dodge an approaching punch, and stay still when safe.” Show the validated action contract and useful corrections/errors.
3. Show real simulation attempts, PFN teacher/dream progress, SNN training, and held-out results. Drive all progress and numbers from actual backend events/artifacts.
4. Build the webapp camera start flow with `getUserMedia`. Keep the rolling buffer in browser memory, preprocess to the agreed image-window contract, and call Dev B's same-computer inference adapter. Do not upload or persist raw frames/features; do not train or tune from the live test.
5. Render the actual simulated Go2 response beside the camera view. Display `NO_OP`/left/right/retreat, confidence, smoothing, cooldown, frame age, command latency, and measured outcomes. Make all gate failures visible.
6. Keep the original slope chat as a secondary follow-on after punch-dodge gates. Do not stage an animation or present the webcam smoke score as a product metric.

Do not edit B-owned `skills/`, `engine/`, `teacher.py`, `students/`, `eval/`, pipeline, or B experiment results. Before handoff, list changed files, exact run/artifact evidence, and any failed/open gate.

## Dev B prompt — copy-ready assignment

You own SkillSpecs, the shared engine, hosted TabPFN teacher/cache, SNN student, experiments, and evaluation. Read `CLAUDE.md`, `MISSION.md`, and the numbered prompts. Preserve existing WIP and coordinate the public contract with Dev A.

Build the simulation-trained PFN→SNN four-action visual dodge backend:

1. Define `NO_OP`, `DODGE_LEFT`, `DODGE_RIGHT`, and `RETREAT` using only verified high-level Go2 commands. Validate each candidate against `config.yaml` and the command schema.
2. Add the configurable ordered frame stack (start with 8 grayscale frames at 12×12 and 30 Hz; compare 4/6/8 only on simulation dev seeds). Extend the existing SkillSpec, teacher layout, dream, and SNN input encoder for this fixed image window. Measure PFN feature width/limits, SNN accuracy, and local CPU latency.
3. Collect actual MuJoCo outcomes across varied situations and all four candidate actions. Fit the hosted residual teacher on image window + candidate action + physics guess → measured outcome residual. Use the production teacher/cache path; all rows and images are simulation-derived.
4. Render many valid simulated dream situations, batch PFN predictions over all candidate actions, and convert residual distributions through the SkillSpec objective into soft action targets. Train the local SNN on those image windows and targets; validate predicted dream quality against a held-out set of actual MuJoCo rollouts. PFN targets for unrun actions are pseudo-labels.
5. Include balanced no-threat scenes; the safe target is `NO_OP`. All teacher fit, SNN distillation, validation, threshold selection, and offline evaluation data must come from simulation. Split by full trajectory and seed, not adjacent frames.
6. Implement temporal score smoothing and a command adapter with at least 1.0 s cooldown between active dodge commands. Evaluate repeated threats during lockout, confidence thresholds, false positives, and misses; expose state to Dev A.
7. Export the frozen SNN to the checked local NumPy runtime and provide Dev A a typed same-computer inference API: image window in, action scores/action/confidence/cooldown state out. Verify feature order and preprocessing parity against MuJoCo.

Use the webapp camera only after freezing the simulation-trained model. Keep frames in browser memory and send only the fixed feature window to the local inference adapter; never send camera data to hosted services or persist it. Never use live frames, labels, outputs, or outcomes for training, fine-tuning, or threshold selection. Ignore the previous webcam prototype and its metrics. Do not add another learned vision model, create a Conv-SNN assumption, bypass the verified robot interface, or tune on held-out simulation data. For handoff, provide contracts, exact commands, versions, seeds/config, artifacts, and every open gate.

## References and client note

- [PriorLabs TabPFN Client](https://github.com/PriorLabs/TabPFN-client) — hosted TabPFN is a tabular service; image tensors enter it as numeric feature columns.
