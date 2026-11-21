# 12-hour Go2 demo plan and Dev A handoff

Date: 2026-11-21 (Europe/Berlin)

## Demo story

1. The user tells the dog in chat: “Learn to cross this incline safely.” The system validates a declarative skill, runs real Go2 MuJoCo attempts, shows the PFN teacher and SNN learning stages as they happen, then compares measured performance on held-out situations.
2. The finale uses the laptop camera as the dog's visual sensor. A person moves a fist toward the camera; the system estimates image direction and time-to-contact, then the already-trained SNN chooses a supported dodge for the simulated Go2 shown on screen. The camera view and the dog-head view make the sensor-to-action link visible. This is a simulation demo, not a claim about a physical robot.

Keep the two lessons connected by the same chat, validation, learning pipeline, evidence panel, and learned SNN deployment. Keep the incline lesson as the deliverable floor; camera dodge is the flagship only if its measurement gates pass.

## Current repository state checked

- Local branch: `person-b` at `f479b67`, merging Person A's `b583c79`. Do not reset or overwrite this tree.
- The current working tree has WIP: `config.yaml`, `learned_judgment/skills/__init__.py`, untracked `learned_judgment/pipeline.py`, and B proof/probe files. Pipeline review and proof runs are still open. Preserve them.
- Latest fetched `origin/messages` tip is `77c1bd0` (2026-11-21 17:49 UTC). It says chat-driven `DeclaredSpec` is the priority and camera dodge is parked. This brief records the user's newer demo direction.
- No `DeclaredSpec`/`check_declaration` implementation or chat UI exists yet. `app/` currently has the episode viewer and hooks only.
- The slope work is not implemented yet: `Go2World._hidden_effects()` returns `{}`; the named preset values in `config.yaml` are `null`; there is no verified visible incline scene. Do not present a null preset as a working slope.
- The simulator renders a third-person tracking camera. The Go2 model has no head camera. There is no laptop-camera path. OpenCV and MediaPipe are not installed.
- Existing exact-state DODGE development results were physics-only 1/100 collisions, teacher 1/100, SNN 5/100. Those results do not establish a learned advantage and do not evaluate camera observations. The current in-progress pipeline is not yet evidence.
- The repo pins Python 3.13 and `tabpfn-client==0.10.1`. The current teacher is a hosted tabular TabPFN path with usage/cost gates; do not send camera frames to it or call it per frame.

## Architecture and honest product claim

- Simulation must create the camera observations the dog could sense. The policy receives image-derived observations only (for example object bearing, apparent angular size and its growth/TTC), plus any allowed IMU/joint signals. Ground-truth object position, world slope, pose/velocity unavailable to the real dog, and hidden preset values are scoring-only.
- Match the simulation sensor and laptop sensor schemas, units, timing, detector delay, noise, and out-of-range behavior. Show the head-mounted MuJoCo view beside the live laptop view and the simulated dog's decision.
- PFN trains/evaluates on the camera-observation rows and candidate actions in simulation; the SNN learns in the PFN dream and is the only live decision brain. The live path must not call the hosted PFN API on every frame.
- A raw-image route is a research gate, not an assumption. PriorLabs' `tabpfn-extensions` documents `TabPFNWithImages`, which uses frozen DINOv3 image features before TabPFN, and labels its extensions experimental. Check Python 3.13/Windows dependency and real-time latency before including it. The current hosted teacher is tabular. A robust image detector that emits the shared numeric observation schema is the reliable first path; that still makes the learner image-conditioned. Keep all laptop frames in memory; never save, log, upload, or commit them.
- A fast punch from arm's length may arrive before the dog can move. Measure the shortest dodgeable TTC. The UI must decline/out-of-range a threat below that measured limit and say so plainly. Start camera validation with a known-size soft ball as a calibration target; do not market fist detection until it is measured.

## Twelve-hour build sequence

| Window | Dev A: UI / world / camera | Dev B: backend / learning | Exit evidence |
|---|---|---|---|
| 0–1 h | Freeze the 4-screen story and data contract with B. Keep actual simulator playback in the viewer; no mock progress. | Inspect and preserve WIP; finish/review pipeline acceptance; publish the exact declaration JSON and `check_declaration` contract to A. | Same named fields, units, episode lifecycle and error states agreed. |
| 1–3 h | Add a physically simulated, visible incline (or stop and report if it cannot be made real/verified). Add a calibrated dog-head MuJoCo camera; expose its rendered frame without saving. | Implement `DeclaredSpec` as validated data, safe formula templates plus explicit neutral-guess option, serialization, and two-phase observe/decide/act contract. Implement the incline skill through generic engine interfaces. | A real incline episode and a declaration round-trip; hidden truth never reaches policy input. |
| 2–4 h, parallel | Prototype laptop capture and fist/ball detector in memory; display FPS and end-to-end age/latency. | Define camera features, bounds, simulated delay/noise, and refusal rules in config. Check image-extension Python 3.13/Windows feasibility without changing dependencies until proved. | G1 camera measurements on at least 20 controlled approaches/tosses. |
| 3–7 h | Build chat → validated declaration → live learning progress → result screen with the actual head-camera render. | Train/evaluate incline through actual Go2 episodes; then run camera-observation DODGE through PFN teacher → dream → SNN, with paired seeds and dev data only. | G2 simulation gate; saved run records drive the UI metrics. |
| 7–10 h | Add laptop camera → shared observation schema → SNN decision → actual simulated Go2 episode. Overlay bearing/TTC, action, episode result, and refusal reason. | Integrate the SNN-only live inference endpoint and strict range/latency guard. Compare physics, teacher and SNN on the same held-out dev situations. | G3 live-path latency and supported-range check; no live PFN calls. |
| 10–12 h | Polish transitions, typography, synchronized camera/render panels, and rehearsal reset. | Freeze configs and report only measured metrics; record unrun checks as NOT RUN. | One reproducible end-to-end run for each lesson; honest demo script and fallback. |

The 12-hour sequence is a priority order, not permission to skip a gate. If time forces a choice, finish the measured incline learning story and show camera dodge as an explicitly gated prototype; never imply it passed.

## Go/no-go gates

- **G0, simulator reality:** incline geometry/preset and head-camera pixels are produced by the real Go2 MuJoCo model; commands use the verified walking policy; observations exclude hidden world state. Any missing piece blocks the associated claim.
- **G1, live vision:** measure frame rate, detector age/latency, bearing error, apparent-size/TTC error, and supported distance on at least 20 measured approaches. Pass only when observed error and end-to-end delay leave a measured margin over the Go2's required movement time. Report failures and shortest dodgeable TTC.
- **G2, learning is useful:** on paired held-out simulation situations with camera-model noise and detection delay, compare physics-only, PFN teacher, and dream-trained SNN. Require measured headroom over physics-only and SNN performance that is not materially worse than teacher; report collisions, falls, minimum separation, command travel and uncertainty. If not, the model has not learned a camera dodge.
- **G3, live inference:** live frames map to the same feature schema used for simulation dreams; the SNN alone decides; out-of-range or too-late approaches are refused. Report measured end-to-end latency and decision count.
- **G4, incline learning:** actual MuJoCo incline attempts show the pre-learning baseline, training curve, and held-out outcome (success/fall, speed, elapsed time/recovery). Report attempt count and seeds. Do not use a frozen animation or a prewritten metric.

No test-set tuning. Keep the existing preregistration/ownership rules: treat demo runs as development unless the lead explicitly amends the protocol. Every figure is generated from saved run outputs, never typed into the UI as a result.

## Dev A prompt — copy-ready assignment

You are Dev A on the Go2 demo. Own the UI, `app/`, and the environment/world/camera files that belong to your lane. The current user direction is a chat-taught incline lesson followed by a live laptop-camera punch-to-simulation dodge, with the PFN teacher and dream-trained SNN visibly explained. Read the repo rules and this plan first; coordinate the exact declaration and decision contracts with Dev B before wiring UI calls.

Build the polished front end as a real vertical slice:

1. **Teach screen:** chat for “learn to cross this incline safely”; show the validated declaration in understandable language and let the user correct it. The model may produce data only, never executable code. Use Dev B's `check_declaration` result as the source of validation errors.
2. **Learning screen:** stream actual MuJoCo attempts and actual pipeline progress (physics prior → attempts → PFN fit/decisions → PFN dream → SNN training). Make the dog and sensor view large and readable. Every label/count comes from a real run event; show errors/blocked states when a stage is unavailable.
3. **Result screen:** compare the real baseline and learned SNN on held-out dev situations, with sample counts, seeds, outcome metrics, and limitations. Never write placeholder numbers.
4. **Live camera finale:** capture from the laptop in memory only; show camera FPS/latency, detected bearing/TTC and the SNN action; trigger the actual simulated Go2 episode and show the head-camera and third-person views. Add an explicit “too late/out of range” state. Do not claim that a real dog dodged a punch.
5. **World deliverables:** make the incline visible and physical in MuJoCo, and add a real dog-head camera/view if the model supports it. Keep hidden slope, object ground truth and other hidden effects out of the policy input. Align simulated camera timing/noise/features with the laptop feature contract from B. If a capability cannot be measured in this 12-hour window, show it as unavailable rather than simulating a pass.

Do not edit B-owned `learned_judgment/skills/`, `engine/`, `teacher.py`, `students/`, `eval/`, `pipeline.py`, or B experiment results. Do not replace MuJoCo episodes with a staged animation, fabricate progress/metrics, store camera frames, or silently alter preregistration. Keep the interface contract small and typed so B can implement it in parallel. Before handoff, list files changed, the real run/artifact that proves each UI state, any gate that did not pass, and the exact integration contract you used.

## Primary-source note on image support

- [PriorLabs TabPFN Extensions README](https://github.com/PriorLabs/tabpfn-extensions/blob/main/README.md)
- [PriorLabs TabPFN Extensions 0.6.3 changelog](https://github.com/PriorLabs/tabpfn-extensions/blob/main/CHANGELOG.md)
- [PriorLabs TabPFN Client README](https://github.com/PriorLabs/TabPFN-client)

These document the image extension and the separate hosted tabular client. They do not establish that the current project's pinned stack can run real-time camera inference; prove that on this Windows/Python 3.13 machine before depending on it.
