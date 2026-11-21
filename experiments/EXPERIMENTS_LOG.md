# Learned Judgment experiment log

Entries are appended below the template. An experiment with no entry has not been run.

For each entry, record:

- experiment ID, date, owner, and bounded question
- hypothesis and pass/fail criteria
- primary sources and exact package/policy/simulator versions
- seed, config, and actual dataset/episode artifact paths
- exact command, exit code, measured result, and limitations
- wall time, TabPFN calls, current usage check, quoted cost, and actual usage
- decision and next step

Do not enter planned outcomes as results. Mark an unexecuted item NOT RUN. Synthetic fixtures and test doubles may only be used in pure utility tests; exclude them from all system or model metrics.

---

## E1 (research half) — Efferent Go2 path — 2026-11-21 — Person A

**Question.** Can Efferent drive a Unitree Go2 in MuJoCo on this machine (Windows 10, Python 3.13.18, CPU only, no cmake)?

**Pass criterion (set before the run).** The sources show a real, licensed, installable Go2 walking example that could run headless here, and a second agent confirms every claim the walking stack depends on.

**Method.** Research workflow W1, run ID `wf_90dc7685-68e`: one read-only researcher, then one adversarial verifier that re-cloned every source, both Opus 5.5. No repository code was run and no checkpoint was loaded. **The proof run is NOT RUN**, so whether the dog walks on this machine is not established.

**Result: PASS on the criterion.** 25 of 28 claims confirmed. The 3 refuted claims each contain a wrong side statement that the walking stack does not depend on (other GitHub projects share the name; estop is pure damping; rl_sar does not apply its clamped torque).

- **Package.** `efferent` 0.5.0 on PyPI, Apache-2.0, repository `Eximius-Labs/efferent` at commit `f51e1af953afe1016dae9ceeff2b5328fa51781d`. The wheel's source is identical to the repository at that commit. Pure Python; `efferent[mujoco]` resolves from wheels only for CPython 3.13 on win_amd64.
- **Policy.** `examples/go2_walk/go2_walk.app` (705,514 B, sha256 `89a5bb7e…4147`) is a zip holding `manifest.yaml` and `policy.onnx` (758,424 B). The ONNX is a conversion of rl_sar's `policy/go2/robot_lab/policy.pt` (rl_sar commit `376d42c9`, Apache-2.0); the `.pt` kept beside it has the same sha256 as rl_sar's file. That the conversion is numerically equivalent is stated by the README only and is not verified.
- **Model.** Unitree `unitree_mujoco` at commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`, `unitree_robots/go2/` (`go2.xml`, `scene.xml`, 16 OBJ meshes, 28.4 MB, BSD-3-Clause).
- **Observation (45).** Base angular velocity ×0.25 (3), projected gravity (3), command `[vel_x, vel_y, yaw_rate]` ×1 (3), joint position minus default (12), joint velocity ×0.05 (12), previous action (12). Clipped to ±100. Policy joint order FR, FL, RR, RL × (hip, thigh, calf).
- **Action (12).** Joint target = default + action × `[0.125, 0.25, 0.25]` per leg; default pose `[0, 0.8, −1.5]` per leg. PD gains kp 20, kd 0.5. Torque limit 23.5 N·m, enforced only by scaling kp.
- **Timing.** Policy at 50 Hz; physics timestep 0.002 s, 10 substeps; torques written to `qfrc_applied`.
- **Commands.** `vel_x`, `vel_y` in m/s and `yaw_rate` in rad/s. Efferent does not clamp them. robot_lab's current training ranges are ±1 for each, but the commit this policy was trained at is unknown.

**Limitations.**
- Unitree's `scene.xml` puts 8 cm bars and a staircase on the +x path. The README's 1.9 m result is consistent with the dog stopping at the first stair. Acceptance needs a flat scene.
- Upstream documents only forward walking at 0.4 m/s. Lateral, yaw and other speeds are untested anywhere.
- Efferent's `Runner.run()` blocks for the whole run, has no single-step API and no fall detection.
- In training (robot_lab HEAD) planar commands with norm ≤ 0.2 m/s were zeroed and yaw commands came from heading control.
- The Go2 example is not in Efferent's CI.

**Raw artifact.** `artifacts/env/research/W1-walking-stack.json`, id `e1`: every claim with its quoted source, the file list with sizes and hashes, the proof command and the verifier's checks.

**Cost.** W1 as a whole: about 54 minutes wall time, 6 agents. TabPFN calls: 0.

**Decision and next step.** Efferent is first in the mandated order and passes on paper, so its proof runs first.

## E1 (proof run) — Efferent Go2 walks headless on this machine — 2026-11-21 — Person A

**Question.** Does the E1 proof load the real Efferent Go2 checkpoint and keep the Go2 upright, headless, on this machine?

**Pass criterion (set before the run, in the research command).** Exit code 0 and `E1 PROOF: PASS`, which the script prints only if the run is not estopped, takes 400 policy steps, ends with base x > 1.6 m and z > 0.15 m, and never tilts 45° or more.

**Command.** The research proof command, saved as `artifacts/env/research/e1_proof.sh` verbatim except that the work directory is the first argument instead of `mktemp -d`. It downloads the pinned files, checks their sha256, then runs the proof with `python -I` in a throwaway `uv run --no-project --isolated` environment (efferent 0.5.0, mujoco 3.11.0). Recorded with the command below. After the run, the home directory in the recorded command was replaced by `~` in `run.json` (same function the recorder now applies, `"home_redacted": true`); no other value changed:

    uv run python -m learned_judgment.env.evidence e1-proof -- "C:/Program Files/Git/bin/bash.exe" artifacts/env/research/e1_proof.sh ~/AppData/Local/Temp/claude/b5a20353bdb366d1/69c7996f-bc6a-4bc2-bb52-6077272ed315/scratchpad/e1-proof

**Result: PASS.** Exit code 0. The four lines of `stdout.txt`:

    versions efferent=0.5.0 mujoco=3.11.0 onnxruntime=1.32.0
    timestep=0.002 substeps=10 rate_hz=50.0 obs_dim=45 act_dim=12 steps=400
    estopped=False x=1.88m z=0.399m max_tilt=14.7deg wall=1.0s for 12s sim
    E1 PROOF: PASS

**Versions.** efferent 0.5.0, mujoco 3.11.0, onnxruntime 1.32.0 (printed by the proof). Python 3.13 by `--python 3.13`; the recorder ran 3.13.18, the proof's own patch version was not printed. Windows-10-10.0.19045-SP0, CPU only.

**Artifact.** `artifacts/env/20261121T144656-e1-proof/` (`run.json`, `stdout.txt`, `stderr.txt`). Started 2026-11-21T14:46:56Z at git `5b0cc67` (dirty: the script was not yet committed).

**Wall time.** 40.7 s for the whole script (`run.json` `wall_s`), including downloads and installing 21 packages; the proof's own timer gives 1.0 s for the walking run.

**Limitations.**
- One command only: forward 0.4 m/s for 400 steps on Unitree's `scene.xml` (bars and a staircase on the +x path). Speeds, lateral, yaw, stopping, 30 s duration and the real-time factor are not measured: the H+3 locomotion gate is NOT RUN. The flat-scene run was not needed and is NOT RUN.
- "for 12s sim" is fixed wording in the proof script, not a measurement. The 400 steps at 50 Hz are 8 s of policy time, after 1 s of settling (`settle_s=1.0` in the script) and 3 s of soft start (`startup.interp_to_default_s: 3.0` in `go2.robot.yaml`).
- Upright is judged from final z, final x and maximum tilt only.
- `run.json` `packages` lists the recorder's project environment, not the proof's isolated one; the proof's versions are its first stdout line.
- Plain `bash` started from Python here resolves to `C:\Windows\System32\bash.exe` (WSL), so the recorder was given Git Bash's full path.
- That the ONNX policy is numerically equivalent to rl_sar's `policy.pt` is still unverified.

**Cost.** TabPFN calls: 0.

**Decision and next step.** Efferent is selected (DECISIONS.md); rl_sar (E2) stays the fallback and its proof is NOT RUN. The checked Efferent files are vendored under `learned_judgment/env/assets/`; the Unitree model comes from the integrator's submodule `third_party/unitree_mujoco` at the same commit `1eb6642e`. Both are pinned in `config.yaml`.

## E2 (research half) — rl_sar Go2 policies — 2026-11-21 — Person A

**Question.** Can a pre-trained Go2 policy from rl_sar run with CPU inference in the Python `mujoco` package on this machine?

**Method.** Same workflow and rules as E1. **The proof run is NOT RUN.**

**Result: PASS on the criterion.** 29 of 31 claims confirmed; the 2 refuted are side details (extra launch files; which gains rl_sar uses on its first get-up). Both Go2 checkpoints exist at rl_sar commit `376d42c` as TorchScript archives (`himloco.pt`, `robot_lab/policy.pt`), and every config value for the adapter recipe was read from the source config. CPU inference needs `torch` 2.15.0+cpu (126.6 MB).

**Limitations.**
- The hip action scale for the himloco policy is not settled.
- `torch.jit.load` on these archives was not run.
- rl_sar has no Python or MuJoCo-Python runner to mirror.

**Raw artifact.** `artifacts/env/research/W1-walking-stack.json`, id `e2`.

**Decision and next step.** Held as the fallback if the E1 proof fails. E1's policy is the same robot_lab policy in ONNX form.

## M — MuJoCo on Windows with Python 3.13 — 2026-11-21 — Person A

**Question.** Which facts about the Python `mujoco` package on this machine must the environment code rely on?

**Method.** Same workflow. The researcher wrote and ran small scripts in a scratch environment (mujoco 4.0.1, numpy 2.5.5, cross-checked on 3.17.0) against `google-deepmind/mujoco_menagerie` at commit `370c43965323c0944b433b202aac5058b5f907a9`; the verifier re-ran them in its own environment.

**Result: PASS.** All 8 items established by executed scripts; 32 of 33 claims confirmed (the other: contact forces differ in the last digits between versions).

- **Versions.** Since 4.0.0 `mujoco` on PyPI is a metapackage and the native code is in `mujoco-core`; the newest single-package wheel for cp313 Windows is 3.17.0.
- **Scene building.** `MjSpec` adds a free body with geom and mass before compiling; its 7 qpos are appended after the robot's 19 and the keyframe is padded.
- **Distance.** `mj_geomDistance(m, d, geom1, geom2, distmax, fromto)` returns a float, negative on overlap.
- **Rendering.** `mujoco.Renderer` gives an offscreen RGB frame through a hidden GLFW window on WGL (Intel HD Graphics 620, OpenGL 4.5). `MUJOCO_GL=egl` or `osmesa` makes `import mujoco` fail.
- **Determinism.** qpos after 2000 steps is byte-identical across fresh model, fresh data and separate processes, but not across MuJoCo versions.
- **Friction.** The menagerie Go2 foot geoms have priority 1, so changing only the floor friction has no effect on foot contacts.
- **Velocities.** Free-joint linear velocity is in the world frame, angular velocity in the body frame.
- **Throughput, zero control, single thread.** Medians between about 9,000 and 11,500 steps/s across runs (real-time factor about 18–23 at 0.002 s), without policy inference.

**Limitations.**
- Measured on mujoco 4.0.1 and 3.17.0 with the menagerie model. The E1 path pins mujoco 3.11.0 and uses Unitree's own model, so each fact must be re-checked there before code depends on it.
- Rendering was tested only in an interactive desktop session.
- `onnxruntime` and `torch` were only dry-run resolved, not imported.

**Raw artifacts.** `artifacts/env/research/W1-walking-stack.json`, id `m`; the scripts and their outputs in `artifacts/env/research/W1-mujoco/` (home directory replaced by `~`).

---

## E6 (research half) — ball specification — 2026-11-21 — Person A

**Question.** Which standardised balls could be the thrown ball, and what are their mass and size according to the governing body's own rules?

**Pass criterion (set before the run).** At least three balls with mass and size read from a governing-body document that was actually opened, each confirmed by a second agent that re-opened the source.

**Method.** Research workflow W3, run ID `wf_c662a377-def`: one read-only researcher, then one adversarial verifier, both Opus 5.5. Document research only. No simulator command was executed, so the simulator half of E6 (ball model, collision masks, throws that hit and miss, reproducibility) is **NOT RUN**.

**Result: PASS on the criterion.** 25 of 25 claims confirmed by the verifier; no numeric value changed. Mid-range mass and the radius derived from the mid-range circumference or diameter:

| Ball | Rule (mass; size) | Mid mass kg | Radius m | Source |
|---|---|---|---|---|
| Football | 410–450 g; circumference 68–70 cm | 0.430 | 0.1098 | IFAB Laws of the Game 2026/27, Law 2.1, p.55 |
| Handball size 3 (resin) | 425–475 g; 58–60 cm | 0.450 | 0.0939 | IHF Rules of the Game, 1 July 2025, Rule 3:2 |
| Handball size 2 (resin) | 325–375 g; 54–56 cm | 0.350 | 0.0875 | same |
| Volleyball | 260–280 g; 65–67 cm | 0.270 | 0.1050 | FIVB Official Volleyball Rules 2025–2028, Rule 3.1 |
| Basketball size 7 | 580–620 g; 750–770 mm | 0.600 | 0.1210 | FIBA equipment rules 2026 |
| Foam dodgeball | 134–146 g; diameter 17.8 cm (nominal, no tolerance) | 0.140 | 0.0890 | WDBF foam rules 2026 |
| Futsal | 400–440 g; 62–64 cm | 0.420 | 0.1003 | FIFA Futsal Laws of the Game 2025-26, Law 2.1, p.27 |
| Netball | 400–450 g; 690–710 mm | 0.425 | 0.1114 | World Netball Rules of Netball 2024, Rule 1.15 |

Rebound tests stated by the rules, with the implied coefficient of restitution `sqrt(rebound/drop)`: FIBA 1800 mm → 1035–1085 mm (0.758–0.776); IHF sizes 1–3, 200 cm → 90–110 cm on steel (0.671–0.742); FIFA futsal 2 m → 50–65 cm (0.500–0.570). Unitree's own page gives the Go2 weight with battery as "About 15kg".

**Raw artifact.** `artifacts/env/research/W3-ball-spec.json` holds every claim with its quoted rule text, URL and the verifier's check (the home directory in local paths is replaced by `~`).

**Limitations.**
- Tennis (ITF) is not established: the ITF documents could not be opened by the agents.
- The rules give no rebound value for football, volleyball, netball or dodgeball.
- The restitution values come from ball-on-floor tests. How they map to MuJoCo contact parameters is not established, nor are ball friction or rotational inertia.
- The mass of the Go2 model that will be simulated is not checked; no model is selected yet.
- The 4–10 m/s throw-speed band used for ranking is an assumption, not a sourced value.

**Cost.** About 16 minutes wall time, 2 agents. TabPFN calls: 0.

**Decision and next step.** The integrator chose the IHF size 3 handball on 2026-11-21 (0.450 kg, radius 0.0939 m: a ball made to be thrown, in the middle of the momentum band). `config.yaml` carries those values. If dev runs show that every hit topples the dog, the fallback is the volleyball, and that change is logged in DECISIONS.md before it is made.

## Gate 1 — locomotion acceptance — 2026-11-21 — Person A

**Question.** Does the borrowed Efferent Go2 policy, run by our adapter on a flat floor, trot 30 s at 0.3, 0.6 and 0.9 m/s, turn both ways, stop and stand, within the tolerances fixed in the plan?

**Pass criteria (fixed in the plan before any run; `config.yaml` `robot.acceptance`).** Every condition: not fallen (no trunk geom touches the floor and trunk tilt ≤ 60° at every physics step of the commanded run). Trots: |mean forward speed along the dog's heading − vx| ≤ 0.2·vx. Turns (vx 0.3, yaw ±0.5 rad/s = half of `yaw_rates_supported_by_policy` [−1, 1], 10 s): |mean yaw rate − yaw| ≤ 0.3·|yaw|. Stop (5 s at 0.6 m/s, then 5 s at zero, one episode): mean planar speed over the last 2 s < 0.05 m/s. Stand (zero command, 30 s): not fallen. Means exclude the first 5 s of the commanded run.

**Setup.** `learned_judgment/env/scene.py` builds Unitree's `go2.xml` (submodule `1eb6642e`) on a flat plane with `MjSpec`; Unitree's `scene.xml` is not used. `learned_judgment/env/policy.py` drives Efferent 0.5.0's own components one 50 Hz step at a time (spawn at [0, 0, 0.35] with auto base height, 1 s settle, 3 s soft start, then the policy). Each episode is a fresh model and policy; seed 20261121 is recorded, and nothing in an episode draws random numbers.

**Adapter parity: PASS.** `tests/env/test_policy_parity.py` runs Efferent's `Runner.run(2.0)` and the adapter on the same flat model at the constant command [0.4, 0, 0] and asserts the final `qpos` and the last observation are equal (`np.array_equal`). Recorded: `artifacts/env/20261121T152519-policy-parity/`, exit code 0, `1 passed`.

**Command.**

    uv run python -m learned_judgment.env.evidence locomotion -- uv run python -m learned_judgment.env.acceptance --out artifacts/env/locomotion

**Result: FAIL.** Exit code 1, `LOCOMOTION GATE: FAIL ['trot_0.3']`. Six of seven conditions pass. Values from `artifacts/env/locomotion/locomotion.json`, rounded to 4 decimals:

| Condition | Commanded | Measured | Tolerance | Result |
|---|---|---|---|---|
| trot_0.3 | vx 0.3 m/s, 30 s | mean speed 0.2256 m/s; not fallen, max tilt 3.0119° | \|mean − 0.3\| ≤ 0.06 | **FAIL** (off by 0.0744) |
| trot_0.6 | vx 0.6 m/s, 30 s | mean speed 0.5430 m/s; not fallen, max tilt 4.3325° | \|mean − 0.6\| ≤ 0.12 | PASS |
| trot_0.9 | vx 0.9 m/s, 30 s | mean speed 0.8596 m/s; not fallen, max tilt 4.7539° | \|mean − 0.9\| ≤ 0.18 | PASS |
| turn_left | vx 0.3 m/s, yaw +0.5 rad/s, 10 s | mean yaw rate 0.4976 rad/s; not fallen, max tilt 4.4356° | \|mean − 0.5\| ≤ 0.15 | PASS |
| turn_right | vx 0.3 m/s, yaw −0.5 rad/s, 10 s | mean yaw rate −0.5066 rad/s; not fallen, max tilt 5.0779° | \|mean + 0.5\| ≤ 0.15 | PASS |
| stop | vx 0.6 m/s 5 s, then 0 for 5 s | final speed 0.0130 m/s; not fallen, max tilt 4.3325° | final < 0.05 | PASS |
| stand | 0, 30 s | not fallen, max tilt 2.1584° | not fallen | PASS |

**Real-time factor and CPU.** Per condition `rtf` 6.7794 (turn_left) to 9.5524 (stop); all seven episodes together 178.0 s simulated (settle, soft start and commanded run) in 22.83 s wall, 7.80 (sums of `sim_s` and `wall_s` in the seven `.npz` traces). `cpu_fraction` (CPU seconds of the process, all threads, per wall second) 1.3922 to 1.8028 on 2 cores. Measured with the fall checks running after every physics step. The recorder's `wall_s` for the whole command, including `uv` start-up and model building, is 29.77 s. Nothing else was run by this agent while it ran; other agents of the session may have been active.

**Observed beyond the criteria.** At zero yaw command the dog turns counter-clockwise: mean yaw rate 0.1209 (trot_0.3), 0.0734 (trot_0.6), 0.0522 (trot_0.9), 0.1088 (stop) and 0.1369 rad/s (stand), from the same file. In `trot_0.3.npz` the dog ends at x −1.158 m, y 3.411 m (it walks a circle); in `stand.npz` it turns in place and ends at x −0.097 m, y 0.017 m. Mean body-frame forward velocity over t > 5 s in `trot_0.3.npz`, computed with Efferent's `quat_rotate_inverse`, is 0.2252 m/s, which agrees with the heading-projected 0.2256 m/s.

**Versions.** efferent 0.5.0, mujoco 3.11.0, onnxruntime 1.32.0, numpy 2.5.5, Python 3.13.18, Windows-10-10.0.19045-SP0, CPU only (recorder `run.json`). `config_sha256` f0029728…ccfb4. Git `ca9b0cc`, dirty (the Task 2 code was not yet committed).

**Artifacts.** `artifacts/env/20261121T152237-locomotion/` (recorder: `run.json`, `stdout.txt`, `stderr.txt`); `artifacts/env/locomotion/locomotion.json` and one trace per condition (`trot_0.3.npz` … `stand.npz`: t, qpos, qvel, command at 50 Hz, seed, sim_s, wall_s). The same seven conditions as tests: `artifacts/env/20261121T152324-locomotion-tests/`, exit code 1, `1 failed, 6 passed` (`test_trots_thirty_seconds[0.3]`, |0.2256 − 0.3| = 0.0744 > 0.06).

**Limitations.**
- One run per condition. The episodes are deterministic on this machine (the parity test reproduces `qpos` bit for bit), so repeating them gives the same values; other machines or MuJoCo versions may not.
- Flat floor only, no payload, latency or other hidden effects.
- Why 0.3 m/s tracks low and why zero yaw commands drift was not investigated: the policy's training commit and its command sampling are unknown (E1 research entry: robot_lab HEAD zeroed planar commands of norm ≤ 0.2 m/s and derived yaw commands from heading control).

**Cost.** TabPFN calls: 0.

**Decision and next step.** Gate 1 fails on trot_0.3. Parity with Efferent's own Runner passes, so by the rule fixed in the plan this is a measured limitation of the borrowed policy, not an adapter fault, and nothing was tuned, loosened or retried. The integrator decides how to proceed. `robot.walking_policy.status` is unchanged.
