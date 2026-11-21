# Chat-taught, simulation-trained PFN→SNN Go2 demo

Date: 2026-11-21 (Europe/Berlin)
Owner split: Dev A = MuJoCo world and product UI; Dev B = PFN/SNN learning and evaluation.

## Product goal

The demo starts with a chat request such as “teach the dog to cross this slope safely,” then shows the complete learning process: measured pre-training baseline, real MuJoCo attempts, PFN fit and dreams, SNN training, and post-training results on the same held-out scenarios. The finale asks the dog to dodge a fast incoming object; the webapp opens the laptop camera and the frozen local SNN drives the simulated Go2. Show measured metrics before and after learning for both skills.

The laptop camera is for final inference testing only. Every train, dev, validation, dream, and offline held-out example used to fit or tune PFN/SNN comes from simulation. During the live test, browser-captured frames stay in memory on the same computer and go through fixed preprocessing to the frozen SNN. Do not fit, fine-tune, adapt, tune thresholds, send real-camera pixels to PFN, or store frames/features. Store prediction/timing summaries only. PFN never runs per live frame.

## Chat-to-skill workflow

1. The user describes a goal in chat, for example “learn to cross this 5° slope without slipping” or “dodge an incoming punch or ball only when it is actually coming at the dog.”
2. The chat layer converts that request into a typed `DeclaredSpec` referencing a registered skill (`RAMP` or `DODGE`) and supported parameters. It emits data, never executable code. A backend validator checks parameter ranges, available simulator features, outcome definitions, and verified Go2 commands.
3. If valid, start the learning run and show the resolved skill and parameters. If a requested behavior lacks a real SkillSpec or simulator capability, explain the exact missing piece and do not invent a result.
4. Reuse the same generic collect → physics → PFN teacher → PFN dream → SNN train → evaluate pipeline for each registered skill. Add BRAKE, TURN, and KICK through new SkillSpecs and simulator support, without adding skill-specific branches to the shared engine.

“Learn to walk on a slope” means learn high-level choices such as approach speed and timing using the existing verified walking policy. On normal flat ground, RAMP must pass through the nominal walking command unchanged. It applies the learned correction only after observable, permitted input indicates slope entry, then returns to the nominal command after slope exit. Use image/sensor cues available to the policy; never feed the hidden MuJoCo slope angle or terrain label into it. Keep the locomotion policy frozen; this demo does not train new motor gaits. Measure false slope activations on flat ground and missed slope entries. Choose a real incline where the measured baseline has room to improve; if it already succeeds, report that instead of faking a before/after failure.

## Starting point and excluded prototype

The current `app/` contains an episode viewer and hooks, not the requested chat/camera webapp. The current skills package contains DODGE only. `config.yaml` has a slope variation range, while named preset values are still null; there is no completed RAMP SkillSpec or visible traversable incline. The current SNN can export a checked NumPy deployment model, but its `SkillSpec`/dream path must be extended for fixed-size image-window features.

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

### Threat gate: stay quiet unless the threat is clear

The SNN must not emit an active command just because it sees motion or an object. Treat an empty scene, static object, receding object, off-path object, harmless background motion, and ambiguous/low-confidence input as `NO_OP` (no active command). Sim training must include incoming punches and balls as positive cases, and all those safe/ambiguous cases as negatives. “Threat” means the image sequence indicates an object is approaching on a collision course within the validated response horizon; seeing a punch or ball that is not coming toward the dog is not enough. Only the simulation scorer may use exact trajectory/contact truth to label these cases; the SNN sees the image sequence and permitted inputs only.

Use the same SNN, with a no-threat/threat score and an action readout. The command adapter may emit LEFT/RIGHT/RETREAT only when all configured checks pass: threat score exceeds a high-confidence threshold; the best active action clears its threshold and margin over alternatives; and the decision is consistent in at least 3 of the last 4 outputs. An `UNCERTAIN`/invalid result abstains to `NO_OP`. Calibrate thresholds on simulation development trajectories only, preregister a high-precision operating target (initial target: at least 99% precision among emitted active commands on dev data), and report the resulting coverage/recall and misses. Freeze thresholds before held-out evaluation and live inference. If the target is not met, mark the gate failed; do not lower the bar or claim certainty. Internal SNN neuron spikes are not commands: only the gated action readout can cause movement.

### Temporal smoothing and action cooldown

The SNN may evaluate each camera update, while a command adapter controls when a new action reaches MuJoCo. Smooth action scores across a short rolling window (begin with 3–5 consecutive SNN outputs; select settings on development runs) and use the threat gate above. `NO_OP` is the learned class for safe/no-threat situations; uncertainty never becomes an active command. Show `NO_OP`, `UNCERTAIN`, threat confidence, action confidence/margin, and gate status in the UI. Count uncertain true threats as misses, and count every active command during a safe scene as a false activation.

After any issued left, right, or retreat action, enforce a **minimum 1.0-second active-action cooldown** before another active command can be issued. Continue observing during cooldown; an immediate `NO_OP`/zero command is always allowed to stop motion safely. Test the cooldown against two threats close together, since a lockout can also block a necessary second dodge. Measure the combined delay from the 8-frame history, score smoothing, command cooldown, and Go2 response. Tune only on development runs, never on the held-out set.

## Simulation-first learning design

1. Add a calibrated Go2 head camera in the actual MuJoCo scene and physically colliding incoming punch and ball proxies. Generate threatening and harmless scenes. Randomize simulation-only object geometry, color, backgrounds, lighting, blur, and camera exposure to reduce the sim-to-live gap. Log true trajectories, contact, time-to-contact, minimum body clearance, fall, recovery, and command travel as scoring-only outcomes.
2. Generate deterministic, seeded image sequences for empty scenes, static/receding/off-path punch and ball motion, unrelated background movement, ambiguous views, and left/right/center inbound threats at varied speed, distance, and warning time. The policy sees rendered pixels only. Raycasts, exact object pose/velocity, collision truth, and slope angle may inform physics guesses and scored outcomes, but stay out of policy features unless an equivalent observable input is designed and validated.
3. Collect a small, varied context of actual MuJoCo action outcomes across all four supported commands. For each executed situation/action, record the image window, action candidate, physics-predicted outcomes, measured outcomes, and residuals. Split complete trajectories and seeds; never split neighboring frames across train and test.
4. Fit the hosted TabPFN residual teacher on those simulation-only rows through the production teacher, request cache, and call log. Candidate features are the flattened ordered image window, verified action fields, and physics guess; targets are measured-minus-physics outcomes. Verify the selected hosted model accepts the feature width and batch sizes.
5. Sample many additional physically valid simulated situations and render their image windows without running a complete action-outcome episode for each candidate. Query PFN in batches for all four actions. Combine physics guesses and PFN residual distributions, then apply the SkillSpec objective and uncertainty penalty to produce soft action scores for `NO_OP`, left, right, and retreat. Safe scenes must make `NO_OP` the target; active actions must be supported by a clear incoming-threat situation and worthwhile predicted outcome. These are PFN dreams, not measured ground truth.
6. Train the SNN only on those simulation-rendered image windows and PFN dream targets. The SNN loss teaches its action readout to match the PFN's soft action distribution; spike activity is produced by learned weights and the snnTorch dynamics. Export the trained student to the checked local NumPy inference format and run the existing parity checks.
7. Validate dream predictions against a held-out subset of actual MuJoCo action rollouts. Then evaluate the frozen PFN teacher and SNN on paired held-out MuJoCo seeds. Report precision among emitted active commands, no-threat false activations (also per safe-scene minute), threat misses/recall, abstentions, action accuracy, collisions, falls, minimum separation, recovery, cooldown conflicts, calibration, and latency. Include flat→slope→flat RAMP runs and report flat-ground command changes, slope-entry latency, missed entries, ascent success/slip/fall, progress, time, and recovery. Do not use the held-out episodes to tune the teacher, SNN, smoother, or thresholds.
8. After simulation gates pass, launch the local webapp. The browser opens the camera with an explicit user start action, keeps a rolling frame window in memory, and sends only the preprocessed window to same-computer SNN inference. The frozen model returns a prediction; the validated command adapter drives the MuJoCo Go2. Keep the PFN offline. Log action/timing summaries only and never use the live session to train or tune.

### What PFN contributes

TabPFN is a hosted tabular predictor, not an image/video generator. MuJoCo creates all image windows. PFN uses a smaller set of measured simulation outcomes plus physics features to predict candidate-action outcomes for more sampled simulation situations. This avoids a full outcome rollout for every dream candidate, while actual MuJoCo rollouts remain the source of teacher context and validation truth. Dream-label errors must be measured, not hidden.

**Production data rule:** no real-world frame, label, pixel feature, or camera-derived measurement may enter PFN fitting, SNN training/distillation, validation, threshold selection, or offline evaluation. All of those use simulation-generated data. Real webcam frames are used only for the final frozen-model inference test; its aggregate predictions and timing may be logged, but never recycled into training or tuning. Do not upload or persist live frames/features.

The three human clips and the cached features/checkpoint from the one-off feasibility smoke are excluded from all future training and reports. The upcoming learning claims must come from actual simulated attempts and PFN dreams.

## Evidence and gates

## What the webapp shows before, during, and after learning

- **Before:** run the actual MuJoCo dev suite before fitting the SNN. Show the slope and dodge baselines, scenario count/seeds, and skill-specific outcomes. Include a no-op/physics baseline; label every number as measured simulation data.
- **During:** stream actual episode events and measured outcomes, PFN context-fit state, dream situation/action counts, SNN optimizer progress, validation loss, and spike activity. Show the skill, world setup, and current action. Never animate invented progress.
- **After:** run the same frozen held-out simulation situations for the PFN teacher and trained SNN, then show paired before/after metrics side by side with sample counts and uncertainty. These held-out runs do not feed back into training or tuning.
- **Slope/RAMP metrics:** completed ascents, slip/fall rate, uphill progress, elapsed time, recovery, and command travel.
- **Punch/DODGE metrics:** active-command precision, collision and threat-miss rates, recall/abstention, safe-scene false activations per minute, minimum separation, falls/recovery, action/cooldown counts, confidence calibration, and response latency. Show that empty/static/receding/off-path scenes stay at `NO_OP` and that only clear incoming punch/ball cases can pass the gate.
- **RAMP conditionality:** show a flat→slope→flat episode with a `NORMAL`/`SLOPE` mode indicator. On flat sections, show the nominal walking command passing through unchanged; show the learned correction and its timing only on the slope, then the return to nominal command after exit.
- Show the local camera preview and live SNN prediction only after the simulation-trained model is frozen. Clearly label live inference as a separate test; never use it to update the model.

**G0 — World and sensors:** real Go2 MuJoCo model, verified walking policy, physical punch proxy, calibrated head camera, and matching simulation/live preprocessing. Hidden ground truth stays out of policy features.

**G1 — Simulation data and quiet-safe policy:** simulation rows contain image sequence, candidate, prior, measured outcomes, and residuals. Include `NO_OP` examples from empty, static, receding, off-path, background-motion, and ambiguous scenes, plus incoming punch and ball threats. Include complete flat→slope→flat episodes. Reproduce counts and seed/trajectory splits from saved run data; object/slope truth remains scoring-only.

**G2 — PFN teaches the SNN:** real hosted teacher call through the production teacher/cache using simulation rows; PFN dream scores for simulation-rendered situations; SNN trains successfully on those dream targets; export parity passes. Validate a sample of dream predictions against actual MuJoCo outcomes. The prior webcam prototype is excluded and does not satisfy this gate.

**G3 — Held-out simulation:** independent seeds and trajectories; report all safety and action metrics with uncertainty. Compare the teacher and SNN. Require the preregistered high-precision active-command target, a measured improvement over no-op and physics-only, and SNN performance close enough to the PFN teacher under a preregistered tolerance. Report recall/coverage alongside precision; a quiet model that misses every threat does not pass. Measure RAMP false activations on flat and adaptation only on the slope. If the cooldown causes missed threats, surface that failure.

**G4 — Webapp live inference:** freeze the simulation-trained SNN and thresholds first. The local webapp opens the browser camera and runs the same frame preprocessing and 8-frame stack (or a measured alternative). Only preprocessed features reach the same-computer inference adapter; raw frames stay in browser memory and never go to hosted PFN or disk. The validated command adapter applies smoothing, the high-confidence/persistence gate, and ≥1.0 s active-action cooldown before driving MuJoCo. Run controlled incoming-punch approaches, incoming-ball approaches, and safe empty/static/receding/off-path/background-motion windows. Measure safe-scene false activations, active-command precision/recall, abstentions, frame age, preprocessing, SNN, command, and total response latency; compare with the shortest dodgeable time measured in simulation. Do not fit/adapt/tune on live frames or outcomes. The live camera run is a domain-shift demonstration, not proof of a universal punch detector.

**G5 — Product story:** chat “learn to cross this slope” → actual RAMP baseline and PFN/SNN learning → paired held-out result → chat “dodge an incoming punch” → simulation learning and metrics → local live-camera inference driving the simulated dog. Every animation and metric comes from actual runs. Additional registered skills use the same chat and pipeline after RAMP and DODGE pass.

## Revised 12-hour parallel plan

| Window | Dev A — world/UI | Dev B — learning/backend | Exit evidence |
|---|---|---|---|
| 0–1 h | Agree chat-to-skill UX and typed events for RAMP and DODGE. | Preserve WIP; define `DeclaredSpec`, registry, conditional RAMP contract, threat-gate outputs, and pixel-window contract. | Supported intents, abstention semantics, metrics, and events agreed. |
| 1–3 h | Build the real traversable slope, head camera, punch/ball proxies, and chat/result surfaces. | Implement RAMP pass-through, DODGE image windows and four actions, and confidence/persistence/cooldown adapter. | Real worlds and typed contracts; safe scenes default to NO_OP. |
| 3–5 h | Add seeded flat→slope→flat episodes and incoming/harmless object scenarios with event streaming. | Collect simulation-only outcomes; fit/cache PFN teacher and generate rendered dream queries for both skills. | Reproducible rows and actual pre-fit baselines; safe negatives are represented. |
| 5–8 h | Show baseline → actual PFN/SNN progress → paired post-training metrics. | Train/export per-skill PFN-distilled SNNs; validate dream fidelity, threat precision/recall, and held-out MuJoCo episodes. | Checked exports and measured before/after results for RAMP and DODGE. |
| 8–10 h | Wire browser camera preview beside the simulated dog's head view; display quiet/uncertain/gated states. | Run frozen DODGE SNN through local inference adapter with confidence gate, smoothing, and cooldown; measure latency. | Live camera is inference-only; false activations, misses, abstentions, and latency remain visible. |
| 10–12 h | Polish the two-skill chat story and rehearsal flow. | Freeze settings, run held-out evaluations, publish artifacts and limitations. | Repeatable chat→learn→compare slope run and punch finale. |

The 12-hour plan is a prioritization, not a waiver of gates. If simulation or timing gates fail, show the measured result and identify the blocker; do not substitute a staged dodge.

## Dev A prompt — copy-ready assignment

You own `app/` and the Go2 MuJoCo world/camera files in your lane. Read `CLAUDE.md`, `MISSION.md`, and the numbered prompts. Coordinate the four-action, threat-gate, and image-window contracts with Dev B before wiring UI.

Build the chat-driven product flow for a reusable Go2 skill learner:

1. Build chat input for “teach it to cross this slope safely.” Show the parsed `DeclaredSpec`, supported parameters, validation state, and start the real RAMP learning run when valid. Add a real traversable MuJoCo incline and show measured baseline performance; handle unsupported requests by naming the missing capability.
2. Add the chat follow-up “teach it to dodge an incoming punch or ball, but stay quiet when safe.” Add a calibrated Go2 head camera and physically colliding punch/ball proxies. Include incoming and safe static/receding/off-path/background-motion scenarios; keep object truth out of policy inputs.
3. For both skills, show measured pre-training baseline, actual PFN/SNN progress, and post-training results on the same held-out simulation situations. Drive progress and metrics from backend events/artifacts; no staged animation or placeholder numbers.
4. Build the webapp camera start flow with `getUserMedia`. Keep the rolling buffer in browser memory, preprocess to the agreed image-window contract, and call Dev B's same-computer inference adapter. Do not upload or persist raw frames/features; do not train or tune from the live test.
5. Render the actual simulated Go2 response beside the camera view. Display `NO_OP`/`UNCERTAIN`/left/right/retreat, threat and action confidence, confidence margin, persistence gate, smoothing, cooldown, frame age, command latency, and outcomes. Show that safe scenes remain `NO_OP`. For RAMP, show `NORMAL`/`SLOPE` mode and unchanged nominal walking commands on flat ground; show adaptation only on the slope, plus ascent/slip/fall/time/recovery metrics. Make gate failures visible.
6. Keep the UI generic over the registered skill catalog; do not imply arbitrary natural-language skills work until the backend has a validated SkillSpec and simulator support.

Do not edit B-owned `skills/`, `engine/`, `teacher.py`, `students/`, `eval/`, pipeline, or B experiment results. Before handoff, list changed files, exact run/artifact evidence, and any failed/open gate.

## Dev B prompt — copy-ready assignment

You own SkillSpecs, the shared engine, hosted TabPFN teacher/cache, SNN student, experiments, and evaluation. Read `CLAUDE.md`, `MISSION.md`, and the numbered prompts. Preserve existing WIP and coordinate the public contract with Dev A.

Build the generic simulation-trained PFN→SNN backend for chat-launched skills:

1. Implement a constrained `DeclaredSpec` parser/validator and registry of supported SkillSpecs. Map “walk/cross a slope safely” to `RAMP`; map punch avoidance to `DODGE`. Reject unsupported skills/fields instead of generating executable code or inventing commands.
2. Implement RAMP so `NO_OP`/normal mode passes the verified nominal walking command through unchanged on flat terrain; use only observable permitted inputs to gate the learned approach-speed/timing correction on slope entry, then restore nominal command on exit. Evaluate full flat→slope→flat episodes, including flat false activations and missed slope entries. Use the frozen walking policy and do not claim motor-gait learning.
3. Define DODGE `NO_OP`, `DODGE_LEFT`, `DODGE_RIGHT`, and `RETREAT` using verified commands. Add the configurable ordered image window (start with 8 grayscale frames at 12×12 and 30 Hz; compare 4/6/8 on sim dev data). Extend SkillSpec, teacher layout, dream, and SNN input encoding as required. Include an SNN no-threat/threat confidence readout in addition to action choice; if showing punch/ball type, it must be an auxiliary head of this same SNN, not another model.
4. Collect actual simulation outcomes for each supported skill. Fit the hosted residual teacher on situation/image sequence + candidate action + physics guess → measured residual, using the production teacher/cache path. Training, validation, dreams, and offline test rows are simulation-only.
5. Sample valid simulated dream situations, render their observations, batch PFN predictions for candidates, and convert residual distributions through each SkillSpec objective into soft action targets. Train one SNN per skill; validate dream fidelity against actual held-out MuJoCo rollouts.
6. Build metrics artifacts for baseline-before-fit, PFN teacher, and post-SNN runs on paired held-out scenarios. Record counts/seeds and skill-specific outcomes; no test-set tuning.
7. Implement a conservative DODGE gate: safe/ambiguous inputs abstain to `NO_OP`; active commands require high threat confidence, action confidence and margin, and consistency in at least 3 of 4 recent outputs. Calibrate on simulation dev data only, target at least 99% precision among emitted active commands while reporting coverage/recall, then freeze and evaluate on held-out trajectories. If the target fails, report the gate failure rather than weakening it. Distinguish internal neuron spikes from actionable output events. Add temporal smoothing and at least 1.0 s cooldown between active commands; measure cooldown misses.
8. Export checked SNNs to the local NumPy runtime and provide Dev A a typed same-computer inference API returning `NO_OP`/`UNCERTAIN`/action, threat/action confidence, margin, and gate/cooldown state. Verify preprocessing parity against MuJoCo.

Use the webapp camera only after freezing the simulation-trained model. Keep frames in browser memory and send only the fixed feature window to the local inference adapter; never send camera data to hosted services or persist it. Never use live frames, labels, outputs, or outcomes for training, fine-tuning, or threshold selection. Ignore the previous webcam prototype and its metrics. Do not add another learned vision model, create a Conv-SNN assumption, bypass the verified robot interface, or tune on held-out simulation data. For handoff, provide contracts, exact commands, versions, seeds/config, artifacts, and every open gate.

## References and client note

- [PriorLabs TabPFN Client](https://github.com/PriorLabs/TabPFN-client) — hosted TabPFN is a tabular service; image tensors enter it as numeric feature columns.
