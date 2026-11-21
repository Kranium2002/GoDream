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

- The simulated world renders a physically simulated incoming fist through the Go2 head camera. The policy input is a compact grayscale image tensor (current frame plus a frame-difference channel), flattened into numeric pixel features. Image size, frame stack, resize/normalization, camera cadence and all other tunables live in config. The SNN learns the visual/action mapping directly; no MediaPipe, DINOv3, or other learned vision model sits in the live path. Object pose, contact time and hidden world values are scoring-only.
- Use the same pixel formatting for MuJoCo and laptop frames: dimensions, grayscale conversion, resize, normalization, temporal stack and cadence. Measure camera age and end-to-end processing delay. Show the dog-head MuJoCo view beside the laptop view.
- PFN stays hosted and table-based. Flatten each small simulated camera tensor into numeric pixel columns, append candidate action and physics guess, and fit the hosted teacher on simulator-derived rows. PFN estimates outcome residuals and scores candidate actions for dreamed image vectors. The SNN is trained in that dream on the same pixel vectors and teacher-selected action targets; its live input is the same flattened camera tensor. The hosted PFN never runs per live frame.
- **No secondary vision model:** local code captures and deterministically downsamples grayscale frames. Send only synthetic MuJoCo pixel columns to hosted PFN; never upload real laptop frames or pixels. Choose the image dimensions/channel stack only after hosted cost quotes and held-out fidelity checks, since pixel columns affect request cost and model behavior. If those gates fail, reduce resolution or stop; do not swap in another model. Measure camera-to-SNN-to-MuJoCo command latency on the demo laptop.
- A fast punch from arm's length may arrive before the dog can move. Measure the shortest dodgeable TTC. The UI must decline/out-of-range a threat below that measured limit and say so plainly. Start camera validation with a known-size soft ball as a calibration target; do not market fist detection until it is measured.

## Twelve-hour build sequence

| Window | Dev A: UI / world / camera | Dev B: backend / learning | Exit evidence |
|---|---|---|---|
| 0–1 h | Freeze the 4-screen story and data contract with B. Keep actual simulator playback in the viewer; no mock progress. | Inspect and preserve WIP; finish/review pipeline acceptance; publish the exact declaration JSON and `check_declaration` contract to A. | Same named fields, units, episode lifecycle and error states agreed. |
| 1–3 h | Add a physically simulated, visible incline and incoming fist/hand prop. Add a calibrated dog-head MuJoCo camera; expose its frames without saving. | Implement `DeclaredSpec`, safe formula templates and the neutral-guess option. Add direct image-tensor input/metadata to the SNN decision path and the two-phase observe/decide/act contract. | Real incline and fist episodes; image tensor shape is identical in sim and live; hidden truth never enters policy input. |
| 2–4 h, parallel | Capture laptop frames in memory; measure capture FPS and frame age during controlled fist approaches. | Put grayscale size, frame-difference stack, normalization and cadence in config. Quote the hosted PFN pixel-table request before any billable call; keep the hosted backend mandatory. | G1 camera-to-action timing and held-out image-input shape proof. |
| 3–7 h | Build chat → validated declaration → actual learning progress → result screen, using real head-camera renders. | Train/evaluate incline and pixel-input DODGE on actual Go2 episodes; PFN teacher → pixel-vector dream → direct-image SNN, paired seeds and dev data only. | G2 simulation gate; saved run records drive every displayed metric. |
| 7–10 h | Wire laptop frames directly to the SNN, then trigger the actual simulated Go2 episode. Show camera FPS, decision/action, command latency and outcome; no detector overlay. | Enforce SNN-only live inference, image-shape checks and refusal on low confidence/out-of-range inputs. Measure full camera-to-actuation latency under rendering load. | G3 live path meets the measured dodge-time margin on CPU; no live PFN calls. |
| 10–12 h | Polish transitions, typography, synchronized camera/render panels, and rehearsal reset. | Freeze configs and report only measured metrics; record unrun checks as NOT RUN. | One reproducible end-to-end run for each lesson; honest demo script and fallback. |

The 12-hour sequence is a priority order, not permission to skip a gate. If time forces a choice, finish the measured incline learning story and show camera dodge as an explicitly gated prototype; never imply it passed.

## Go/no-go gates

- **G0, simulator reality:** incline geometry/preset and head-camera pixels are produced by the real Go2 MuJoCo model; commands use the verified walking policy; observations exclude hidden world state. Any missing piece blocks the associated claim.
- **G1, live vision:** on at least 20 controlled approaches, measure camera FPS, frame age, deterministic image-preprocessing time and end-to-end camera→SNN→MuJoCo command latency. Score action success and shortest dodgeable approach time from measured trajectories. Pass only when latency leaves measured movement margin; report misses and too-late cases.
- **G2, learning is useful:** on paired held-out simulation situations with camera-model noise and detection delay, compare physics-only, PFN teacher, and dream-trained SNN. Require measured headroom over physics-only and SNN performance that is not materially worse than teacher; report collisions, falls, minimum separation, command travel and uncertainty. If not, the model has not learned a camera dodge.
- **G3, live inference:** laptop frames become the exact pixel tensor used for simulated dreams; the SNN alone decides on CPU. Refuse unsupported image shapes, out-of-range operating conditions and too-late threats. Report end-to-end latency under the rendered simulator load.
- **G4, incline learning:** actual MuJoCo incline attempts show the pre-learning baseline, training curve, and held-out outcome (success/fall, speed, elapsed time/recovery). Report attempt count and seeds. Do not use a frozen animation or a prewritten metric.

No test-set tuning. Keep the existing preregistration/ownership rules: treat demo runs as development unless the lead explicitly amends the protocol. Every figure is generated from saved run outputs, never typed into the UI as a result.

## Dev A prompt — copy-ready assignment

You are Dev A on the Go2 demo. Own the UI, `app/`, and the environment/world/camera files that belong to your lane. The current user direction is a chat-taught incline lesson followed by a live laptop-camera punch-to-simulation dodge, with the PFN teacher and dream-trained SNN visibly explained. Read the repo rules and this plan first; coordinate the exact declaration and decision contracts with Dev B before wiring UI calls.

Build the polished front end as a real vertical slice:

1. **Teach screen:** chat for “learn to cross this incline safely”; show the validated declaration in understandable language and let the user correct it. The model may produce data only, never executable code. Use Dev B's `check_declaration` result as the source of validation errors.
2. **Learning screen:** stream actual MuJoCo attempts and actual pipeline progress (physics prior → attempts → PFN fit/decisions → PFN dream → SNN training). Make the dog and sensor view large and readable. Every label/count comes from a real run event; show errors/blocked states when a stage is unavailable.
3. **Result screen:** compare the real baseline and learned SNN on held-out dev situations, with sample counts, seeds, outcome metrics, and limitations. Never write placeholder numbers.
4. **Live camera finale:** capture frames in memory, deterministically resize/grayscale/stack them, and pass the pixel tensor straight to the SNN. Show camera FPS, the SNN action/refusal, elapsed camera-to-command time, and the actual simulated Go2 episode from head and third-person views. Do not add a learned detector or claim a real dog dodged a punch.
5. **World deliverables:** make the incline and incoming fist/hand prop physical and visible in MuJoCo; add a real dog-head camera. Keep object ground truth and hidden effects out of policy inputs. Match camera framing, image shape, frame stack and pixel preprocessing between simulation and laptop. If a capability cannot be measured in this window, show it as unavailable.

Do not edit B-owned `learned_judgment/skills/`, `engine/`, `teacher.py`, `students/`, `eval/`, `pipeline.py`, or B experiment results. Do not replace MuJoCo episodes with a staged animation, fabricate progress/metrics, store camera frames, or silently alter preregistration. Keep the interface contract small and typed so B can implement it in parallel. Before handoff, list files changed, the real run/artifact that proves each UI state, any gate that did not pass, and the exact integration contract you used.

## Primary-source note on image support

- [PriorLabs TabPFN Extensions README](https://github.com/PriorLabs/tabpfn-extensions/blob/main/README.md)
- [PriorLabs TabPFN Extensions 0.6.3 changelog](https://github.com/PriorLabs/tabpfn-extensions/blob/main/CHANGELOG.md)
- [PriorLabs TabPFN Client README](https://github.com/PriorLabs/TabPFN-client)

These document the image extension and the separate hosted tabular client. They do not establish that the current project's pinned stack can run real-time camera inference; prove that on this Windows/Python 3.13 machine before depending on it.