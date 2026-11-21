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

## E7 (non-billable part) — hosted TabPFN client contract

- Date and owner: 2026-11-21, Person B. Research agent: UltraCode workflow `person-b-slice0-research`, agent `research:tabpfn-client-contract`.
- Question: what is the exact contract of tabpfn-client 0.10.1 for a residual teacher that fits on 5–40 rows, scores a candidate batch in one prediction request with median and quantiles, quotes cost, reads usage, and sits behind a persistent cache?
- Pass criteria: contract established from installed source, official docs, and non-billable calls well enough to write the teacher wrapper without guessing.
- Versions and sources: tabpfn-client 0.10.1, tabpfn-common-utils 0.2.23, Python 3.13.16. Installed source matches GitHub PriorLabs/tabpfn-client tag v0.10.1, commit eb3fa878ed1e3662086b250a54a4785bc1fdcd80, for 10 core files. Docs: docs.priorlabs.ai (metering, rate limits, preprocessing, FAQ).
- Commands (all exit code 0), run from the repository root with `.venv\Scripts\python.exe`: `experiments\proofs\E7_tabpfn_contract\p1_signatures_offline.py`, `p2_auth_usage.py`, `p3_cost_estimates.py`, `p3b_cost_scaling.py`; plus `p4_github_source_match.txt` and `p5_pypi_versions.txt` from `gh api` and `curl`. Outputs are the `.txt` files next to each script.
- Measured:
  - Authentication through `prior_labs.connect()` succeeded. Usage read: 40,000 of 35,000,000 tokens used (daily and monthly limits both 35,000,000; monthly reset 2026-12-01 00:00 UTC).
  - `estimate_cost` sends shapes only. About 30 quotes left usage unchanged at 40,000.
  - Every quoted shape from (5 train, 8 query, 6 features) to (40, 10,000, 10) costs 10,000 tokens, the minimum per prediction request (pricing version quota_v3, model v3.5, 8 estimators). (40, 32,000, 10) is still 10,000; (40, 100,000, 10) is 26,170.
  - `predict(X, output_type="main", quantiles=[...])` sends one request and returns mean, median, mode, and quantiles (source read).
  - `y` must be 1-D: a 2-D target raises `ValueError` (executed, offline).
  - Categorical columns are not auto-detected below 100 rows; they must be declared with `categorical_features_indices` (docs and executed payload dump).
  - Neither client nor server deduplicates predictions; every predict is a new billed job (source read).
- TabPFN calls: 0 prediction requests, 0 fits. Quoted cost of work done: 0. Actual usage change: 0.
- Wall time: about 15 minutes for the workflow, three agents in parallel.
- NOT RUN: a real fit and predict. Still open from that: the actual shape and orientation of the quantile array, quantile monotonicity, response metadata, charge versus quote, latency, and repeat-request determinism. The smallest settling request is quoted at 10,000 tokens, 20,000 with one uncached repeat. 04_EXPERIMENTS.md requires a Go2-derived dataset for it, and none exists yet.
- Limitation: Prior Labs' General Terms list "knowledge distillation" among violations that allow blocking access. The wording was paraphrased by the fetch tool; exact text is being retrieved in `experiments\proofs\E7_provider_terms\`. Whether training a student on teacher decisions falls under it is a decision for the project lead.
- Decision: PARTIAL. The teacher wrapper and cache can be written. Provider acceptance stays OPEN until the real request runs.

## B-contract — libraries for baselines, evaluation statistics, seeded streams, hashing, persistence

- Date and owner: 2026-11-21, Person B. Agent `research:students-eval-contract`, same workflow.
- Question: with the installed packages, what are the executed APIs and small-sample behaviours for residual baselines, evaluation statistics, seeded streams, cache-key hashing, and persistence?
- Versions: scikit-learn 1.9.0, scipy 1.18.1, numpy 2.5.5, pandas 2.3.3, Python 3.13.16, Windows 11, 22 logical CPUs.
- Commands (all exit code 0): scripts `p00_env.py` to `p10b_jsonl_pandas.py` in `experiments\proofs\B_students_eval_contract\`, each with an `.out.txt` beside it.
- Data: synthetic fixtures for API, shape, and failure-mode checks only. None of the numbers is evidence about the robot, the simulator, TabPFN, or any student.
- Measured (API facts):
  - `HistGradientBoostingRegressor` defaults give a constant model below 40 rows; `min_samples_leaf=2, early_stopping=False` makes it split from 5 rows.
  - `GaussianProcessRegressor` raises at 0 rows and is overconfident at 1 row.
  - `scipy.stats.bootstrap` takes `rng=`; BCa returns NaN on constant differences.
  - A KS test on PIT values interpolated from 5 quantiles rejects a perfectly calibrated fixture; bin counts with chi-square and randomized PIT do not.
  - Named streams from `SeedSequence(base_seed, spawn_key=(sha256(name) words, index))` reproduce across processes.
  - A canonical sha256 key over arrays and settings is stable across processes and changes with any cell, dtype, column order, quantile list, or setting.
  - JSON float repr and `.npz` round-trip bitwise; `pd.read_json` with default settings does not.
  - Unlocked concurrent appends to one file lose lines on Windows.
- TabPFN calls: 0.
- Limitation: the MLP-student parts of this entry (sklearn `MLPRegressor`) are no longer used for students after the 2026-11-21 decision that the only student is an SNN. They still apply to the MLP-on-attempts baseline.
- Decision: PASS for baselines, statistics, streams, hashing, and persistence.

## B-lookahead — command interface of the candidate Go2 walking paths (unverified)

- Date and owner: 2026-11-21, Person B. Agent `research:go2-command-lookahead`, same workflow.
- Question: what high-level command interface does each candidate walking path expose, so the SkillSpec binding layer has the right shape?
- Status: UNVERIFIED LOOK-AHEAD. No policy, controller, or episode was run. Person A's verified schema is authoritative; nothing here is a verified command.
- Sources at pinned commits, saved under `experiments\proofs\B_lookahead_command_interface\fetched\`: Eximius-Labs/efferent f51e1af; fan-ziqi/rl_sar 376d42c; robot_lab 8b447f2 and 26acb22; vendored unitree_mujoco 1eb6642.
- Commands (all exit code 0): `inspect_unitree_mujoco_go2.py`, `inspect_contact_priority.py`, `verify_lookahead_sources.sh`, and a repository-metadata fetch, each with an `.out.txt`.
- Found (source read unless marked):
  - The Efferent go2_walk checkpoint and the rl_sar robot_lab checkpoint are the same file (identical git blob, executed comparison).
  - Both RL candidates take a base-frame command of forward velocity, lateral velocity, and yaw rate at 50 Hz. Native names and observation scaling differ between them.
  - Trained command ranges are not established for either checkpoint.
  - Upstream HIMLoco is CC BY-NC-SA 4.0 and has no Go2 config.
  - Vendored Go2 model (executed read-only load, MuJoCo 3.3.6): timestep 0.002, base body `base_link`, 12 actuators, unnamed trunk collision geoms, no ball in the scene, foot geoms with contact priority 1 over the floor.
- TabPFN calls: 0.
- Decision: PASS as a look-ahead. The SkillSpec binds to canonical command fields read from config.yaml and refuses to build until Person A publishes a verified schema.

## E7-terms — Prior Labs published terms on distillation and service results

- Date and owner: 2026-11-21, Person B. UltraCode workflow `person-b-snn-research`, agent `research:provider-terms-distillation`.
- Question: what do Prior Labs' published terms say about knowledge distillation, use of API outputs to train other models, rights in service results, and exceptions?
- Sources retrieved 2026-11-21 13:51–14:01 UTC with `curl` (HTTP 200): General Terms and Conditions "Status: June 23, 2026" (https://priorlabs.ai/general-terms-and-conditions), Acceptable Use Policy "Status: January 2025" (https://www.priorlabs.ai/aup), privacy policy, API docs, model licence pages, public pages for the event. URLs and timestamps: `experiments/proofs/E7_provider_terms/fetch_log.tsv`. The page copies stay on the local machine and are not committed.
- Command: the agent's quote-check script confirmed that all 61 quotes it reported appear in the saved files (missing 0, exit code 0).
- Found (verbatim in the saved text):
  - Section 10.1.1 allows blocking access where the customer is "engaging in reverse engineering or knowledge distillation of the Contract Software, or by exceeding the agreed usage parameters". "Distillation" appears once in the terms and is not defined.
  - Section 8.2: "The Customer may freely use the Service Results."
  - No research, evaluation, non-commercial, or hackathon exception is written in the terms. Individual agreements take precedence over them.
  - Prior Labs' docs describe its own "distillation engine", which turns TabPFN into a dataset-specific MLP or tree ensemble under a commercial enterprise licence.
  - No public brief or participant terms for the Prior Labs track were found.
- TabPFN calls: 0.
- Limitation: fact-finding only, not legal advice.
- Decision: recorded in DECISIONS.md (2026-11-21). The project lead decided to proceed; the SNN is trained inside the PFN dream and does not reproduce the PFN's predictions.

## E15-literature — sources for the energy estimate and the SNN design

- Date and owner: 2026-11-21, Person B. Same workflow, agent `research:snn-energy-literature`.
- Question: which primary sources support the per-operation energy figures, the estimate formula, its limits, and the encoding and readout choices for a small SNN that picks one of C commands?
- Artifacts: `experiments/proofs/E15_snn_literature/` (decode script and output for the Horowitz figure, notes, bibliography).
- Found:
  - Horowitz, ISSCC 2014, Fig. 1.1.9, "45nm 0.9V" (doi:10.1109/ISSCC.2014.6757323), read in the primary PDF: 32-bit float add 0.9 pJ, 32-bit float multiply 3.7 pJ, 32-bit integer add 0.1 pJ, 32-bit integer multiply 3.1 pJ.
  - SNN papers use E_MAC = 4.6 pJ and E_AC = 0.9 pJ, with E_ANN = E_MAC × operations and the real-valued first layer of an SNN costed with E_MAC (DIET-SNN; Spikformer; Spike-driven Transformer). They differ on whether the first layer is counted once or at every step; none counts leak, reset, or memory.
  - Published critiques (Dampfhoffer et al. 2023; Davidson and Furber 2021; Lemaire et al. 2022) find that memory access dominates and give spike-sparsity limits below which an SNN beats an ANN.
  - Direct (constant-current) encoding with a membrane-potential readout and surrogate-gradient training is the best-supported choice at few time steps.
  - Prior work exists for each ingredient (ANN-to-SNN training, SNN control policies, compact students of tabular foundation models). No "first" claim is supportable.
- TabPFN calls: 0. No SNN was trained in this entry.
- Limitations: one copy of the Horowitz PDF was decoded; several papers were read as preprints; the prior-work search was limited to about ten searches.
- Decision: PASS. Constants and citation are in config.yaml (students.snn.energy); the formula and the required caveats are in learned_judgment/students/energy.py.

## E15 — SNN student contract: PyTorch (CPU) and snnTorch, executed

- Date and owner: 2026-11-21, Person B. UltraCode workflow `person-b-snn-research`, agent `research:snn-install-api-poc`.
- Question: can a spiking student be built, trained, and deployed on this machine with PyTorch (CPU) and snnTorch under Python 3.13 on Windows next to the pinned packages, and what are the exact APIs, neuron dynamics, small-data behaviour, latency, and operation counts?
- Pass criteria: both packages install and import next to the pins, and a proof-of-concept network trains, exports, and matches a numpy forward pass.
- Versions: Python 3.13.16, Windows 11, Intel Core Ultra 7 155H; torch 2.15.0+cpu, snntorch 1.0.0 (MIT), numpy 2.5.5, pandas 2.3.3, scikit-learn 1.9.0, scipy 1.18.1. Installs ran in throwaway environments outside the repository; the project venv was only dry-run and read by this agent.
- Commands and outputs: `experiments/proofs/E15_snn_contract/`, scripts `p01*` to `p10*` with their `.out.txt` files; the full list with exit codes is `p00_commands_and_exit_codes.txt`. Verified model, training loop, and export: `snn_common.py`; numpy forward pass: `np_forward.py`.
- Data: an analytic fixture task defined in `snn_common.py`. Every accuracy, loss, firing-rate, operation, and latency number below is a mechanics number on that fixture. None is evidence about the Go2, the simulator, TabPFN, or the project's student.
- Measured:
  - Install: adding the two packages added eleven packages and left the pinned ones unchanged. The portable pyproject setup uses the explicit PyTorch CPU index; it resolved and synced in a scratch copy.
  - Dynamics of `snntorch.Leaky`: the reset uses the previous step's membrane; a neuron fires on a strict "greater than"; the default surrogate is atan with alpha 2.
  - Pitfalls found by execution: membrane state carries over unless passed explicitly; `utils.reset` misses nested modules; `.double()` leaves spikes in float32 and widens float32(0.9); `deepcopy` and `torch.save(module)` fail after a forward.
  - Training on soft targets (KL to a softmax of the scores) generalised far better than hard labels on the fixture.
  - numpy forward pass against the trained torch model: float64 bit-exact (zero logit difference, zero spike mismatches in 1.28 million); float32 logit difference 1.1e-5 with zero spike mismatches.
  - Determinism: bitwise identical across processes at 1 and 4 threads, different at 16 threads.
  - Single-decision latency: numpy 0.06–0.43 ms median; torch 1.1–9.3 ms median. Importing torch takes about 10 s warm on this machine.
  - Operation counts for a (64, 64) network at 10 steps: mean 32,619 synaptic operations per decision at firing rates of 0.31 and 0.49, against 8,704 multiply-accumulates for the dense reference. At such rates an energy advantage is not given; it needs lower firing rates or fewer steps.
  - Shared trunk with one head per skill: works; without rehearsal the earlier fixture skills were lost, with rehearsal they were mostly kept. A new head must be added to the optimizer.
- TabPFN calls: 0.
- NOT RUN: Linux and macOS installs; early stopping and spike-rate regularisation; any training on real dreams.
- Decision: PASS. PyTorch 2.15.0 (CPU index) and snnTorch 1.0.0 are declared in pyproject.toml and installed. The student is a classifier over the candidate grid, trained in float64 at one thread, and deployed through the numpy network in learned_judgment/students/snn_core.py.

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

## E4 — command inventory — 2026-11-21 — Person A

**Question.** Which commands does the borrowed policy take, in what units and order, and does each one move the simulated dog the commanded way at both ends of its candidate range without a fall?

**Sources (no new research run).** The plan's research workflow W2 for E4 was cancelled by the integrator: the E1 research already established the names, units and candidate range. Names and native order: `manifest.yaml` inside the vendored `go2_walk.app` (efferent@f51e1af), observation term `command`, names `[vel_x, vel_y, yaw_rate]`, scale 1.0 (lines 95–104), `control.rate_hz: 50.0` (line 11), read here from the vendored file. Units (m/s, m/s, rad/s), "Efferent does not clamp" and the candidate range −1 to 1 for each field (robot_lab's current Go2 training range; the checkpoint's training commit is unknown): entry "E1 (research half)". Frame and signs: the policy's command is in its base heading frame (x forward, y left, yaw anticlockwise positive), and this run measures each sign.

**Pass criterion (fixed by the integrator before the run).** A range end is proven if the dog does not fall (as in Gate 1: no trunk geom on the floor, tilt ≤ 60° at every physics step) and the measured response has the commanded sign. Response, as the mean over the 10 s commanded run after its first 5 s: forward speed along the heading for `vel_x`, leftward speed along the heading frame's y for `vel_y`, yaw rate for `yaw_rate`. This is not a tracking tolerance. An end not proven would have been narrowed to 0.75, then 0.5 of the candidate.

**Setup.** `prove_commands` in `learned_judgment/env/acceptance.py`: for each field and each end, one fresh episode on the flat scene through `WalkingPolicy` (same spawn, 1 s settle and 3 s soft start as Gate 1), 10 s at that value with the other two fields 0. The leftward response is a new key `mean_lateral_speed_m_s` of the Gate 1 episode measurement; its other keys are computed as before. Seed 20261121 recorded; nothing draws random numbers.

**Command.**

    uv run python -m learned_judgment.env.evidence commands-proof -- uv run python -c "from learned_judgment.env import load_config; from learned_judgment.env.acceptance import prove_commands; prove_commands(load_config(), 'artifacts/env/commands')"

**Result: PASS, all six ends proven at the candidate range; nothing narrowed.** Exit code 0. Values from `artifacts/env/commands/commands.json`, rounded to 4 decimals:

| Field | End | Commanded | Measured (signed) | Fallen | Proven |
|---|---|---|---|---|---|
| vel_x | min | −1.0 m/s | −0.8880 m/s | no | yes |
| vel_x | max | 1.0 m/s | 0.9243 m/s | no | yes |
| vel_y | min | −1.0 m/s | −1.0349 m/s | no | yes |
| vel_y | max | 1.0 m/s | 0.8939 m/s | no | yes |
| yaw_rate | min | −1.0 rad/s | −0.9555 rad/s | no | yes |
| yaw_rate | max | 1.0 rad/s | 0.9687 rad/s | no | yes |

**Published.** `config.yaml` `robot.walking_policy.command_schema`: `update_period_s` 0.02 (one command per 50 Hz control step), and per field `unit`, `min` −1.0, `max` 1.0, `native_index`, `proven_by: artifacts/env/commands/commands.json`, `measured_at_min`, `measured_at_max` (the table's values) and a plain-words `description` (frame, sign, Gate 1 tracking). `robot.walking_policy.status` is `verified`, which means only that these fields and ranges were each proven by this run; it says nothing about speed-tracking accuracy (Gate 1, trot_0.3, still fails).

**Tests.** `tests/env/test_commands.py`: two pure-function tests of `validate_command` on the published schema (edges accepted unchanged; unknown, missing, NaN and above-max fields rejected, not clipped) and one system test that re-runs `prove_commands` into a temporary directory and asserts every published end has the commanded sign and no fall. Recorded with `uv run python -m learned_judgment.env.evidence commands -- uv run pytest tests/env/test_commands.py -v`: exit code 0, `6 passed`.

**Versions.** efferent 0.5.0, mujoco 3.11.0, onnxruntime 1.32.0, numpy 2.5.5, Python 3.13.18, Windows-10-10.0.19045-SP0, CPU only (recorder `run.json`). Git `e4ddadd`, dirty (the Task 3 code was not yet committed). The proof ran with the candidate schema (status `not_verified`, no measured values; `config_sha256` e65e53b4…dec6b); the tests ran with the published one (3502d5c8…633d9a1).

**Artifacts.** `artifacts/env/20261121T153523-commands-proof/` (recorder), `artifacts/env/commands/commands.json` and one trace per run (`vel_x_min.npz` … `yaw_rate_max.npz`: t, qpos, qvel, command at 50 Hz, seed, sim_s, wall_s); test run `artifacts/env/20261121T153641-commands/`.

**Wall time.** Proof command 16.17 s (`run.json` `wall_s`); the six episodes 84.0 s simulated in 11.02 s wall, real-time factor 7.62 (sums of `sim_s` and `wall_s` in the six traces). Test command 19.94 s.

**Limitations.**
- Sign and staying upright only, one 10 s run per end, one field at a time, flat floor, no payload, latency or other hidden effects. Tracking accuracy, values between the ends, combined commands, changing commands mid-run and the response delay to a new command are not measured here.
- Values beyond ±1 were not tried; the published range is the candidate, not a measured limit.
- Cross-axis motion during each run (for example the anticlockwise drift seen in Gate 1) is in the traces but was not analysed.
- Whether small planar commands (norm ≤ 0.2 m/s, zeroed in robot_lab training at HEAD) are tracked is not tested.
- No second agent re-read the sources for this entry (W2 cancelled). E1's verifier confirmed the names and order (check `c14`) and the units and candidate range (`c23`) in `artifacts/env/research/W1-walking-stack.json`.

**Cost.** TabPFN calls: 0.

**Decision and next step.** The command schema is published with `status: verified` (Person B's loader requirement). Task 4 builds `Go2World` on it.

## M2 — MuJoCo facts for Go2World on the pinned stack — 2026-11-21 — Person A

**Question.** Entry M measured the free-joint velocity frames on mujoco 4.0.1 and 3.17.0 with the menagerie model, and did not test bad states. On the pinned stack (mujoco 3.11.0, Unitree `go2.xml` on the flat scene), in which frames are the free joint's linear and angular velocities, and what does MuJoCo do with a NaN or huge value in qpos, qvel or qacc?

**Method.** `artifacts/env/research/m2_world_facts.py`. (1) Gravity off, trunk rolled +90° about world x and lifted 2 m so nothing touches, free-joint qvel set to linear [1, 0, 0] and angular [0, 0, 1], one step; compare the change of position and of orientation. (2) Four fresh resets with one value corrupted (qpos[0] NaN, qvel[0] NaN, qvel[0] 1e11, qfrc_applied[6] inf so qacc is bad), two steps each, reading `MjData.warning` counts, time and finiteness; then `mj_resetData`.

**Command.**

    uv run python -m learned_judgment.env.evidence m2-world-facts -- uv run python -X utf8 artifacts/env/research/m2_world_facts.py

**Result.** Exit code 0, `artifacts/env/20261121T155604-m2-world-facts/stdout.txt`:
- Linear velocity is in the world frame: position change / dt = [1.0009, 0.001, 0] for qvel [1, 0, 0] with the trunk rolled.
- Angular velocity is in the body frame: with the body z axis along world −y, qvel [0, 0, 1] turned the trunk at [−0.0002, −1.0, 0.1149] rad/s in the world frame ([−0.0002, 0.1149, 1.0] in the body frame).
- Each bad state raised its warning (`mjWARN_BADQPOS`, `mjWARN_BADQVEL` twice, `mjWARN_BADQACC`), MuJoCo printed it and reset the state itself (finite again, base z 0.4451 m, near the model's initial 0.445 m; `data.time` 0.002 s after that step, so it restarted from 0). The warning count stayed 1 after the reset and after a further step; only `mj_resetData` cleared it.

**Consequences.** `Go2World` reports angular velocity rotated into the world frame, counts its own physics steps instead of reading `data.time`, and sets `sim_unstable` from the three warning counts (cleared by its own spawn) plus a finiteness check of the final state. MuJoCo also appends each warning to `MUJOCO_LOG.TXT` in the working directory.

**Limitations.** One step per frame check. The 0.1149 rad/s component is the rolled trunk's own dynamics over that step, not a frame effect. Warning behaviour was checked with auto-reset enabled (the default) only.

**Cost.** TabPFN calls: 0.

## Single-threaded policy inference — Gate 1 re-run — 2026-11-21 — Person A

**Question.** Task 4 runs the walking policy's ONNX Runtime session on one intra-op and one inter-op thread (`WalkingPolicy`, which replaces Efferent's session by the same model with those options). Does that change any Gate 1 result?

**Command.** Re-runs the gate into a temporary directory and compares it with the committed run:

    uv run python -m learned_judgment.env.evidence single-thread-gate1 -- uv run python -X utf8 -c "<runs acceptance.main(['--out', tmp]) and compares tmp with artifacts/env/locomotion>"

(the full one-line program is in `run.json` `command`).

**Result: no change.** `artifacts/env/20261121T161152-single-thread-gate1`, exit code 0 (the comparison's; the gate itself printed `LOCOMOTION GATE: FAIL ['trot_0.3']` as before). For all seven conditions every measured value except `rtf` and `cpu_fraction` is equal and the `t`, `qpos`, `qvel` and `command` traces are bit-identical to `artifacts/env/locomotion/`. `cpu_fraction` fell from 1.3922–1.8028 to 0.9403–0.9865; per-condition `rtf` 5.67–8.58 (was 6.78–9.55).

**Limitations.** This machine only (2 cores). Equal results here do not show that multi-threaded inference would be bit-identical on a machine with another core count; single-threading removes that dependence.

**Cost.** TabPFN calls: 0.

## E5 — hidden effects and world presets — 2026-11-21 — Person A

**Question.** Do the five hidden effects of MISSION.md (payload, floor friction, slope, motor strength, command latency) each change the simulated Go2 on their own, switch off without a trace and stay out of what Person B sees, and do the five presets load and stand?

**Pass criteria (fixed in the plan, Task 6, before any run).** With all five effects switched off, every measurement equals the baseline exactly. Each effect switched on alone changes its own measurement: steady speed at 0.6 m/s commanded (payload 3 kg, slope +5°, motor strength 0.8), stopping distance from 0.9 m/s (floor `slippery`), time from the command to half the commanded speed (latency 0.3 s). Each preset stands 1 s at zero command without a fall. No hidden key in an observation or outcome. Motors at strength 0 make the dog fall and the fall is reported, not raised. Added for the integrator's decisions: the seed fixes the hidden world and the declared payload; the latency jitter is drawn per attempt; `command_active` lags `commands_received` by the latency; the latency is a pure delay line (fifty one-period steps equal one fifty-period step, bit for bit); the declared values are fixed attributes equal to every observation and the outcome; a non-integer hidden value reaches `<trace>.hidden.json` intact; on a slope the policy senses true gravity.

**Simulator facts (proof).** Entry M saw the foot geoms' contact priority on the menagerie model only. Re-checked on the pinned stack:

    uv run python -m learned_judgment.env.evidence e5-effect-facts -- uv run python -X utf8 artifacts/env/research/e5_effect_facts.py

Exit code 0, `artifacts/env/20261121T165353-e5-effect-facts/stdout.txt`. Unitree's foot geoms `FL`, `FR`, `RL`, `RR` have priority 1 and friction [0.4, 0.02, 0.01]; every foot-floor contact uses sliding friction 0.4, and still 0.4 with the floor's friction set to 0.05; with the foot geoms set to 0.15 the contacts use 0.15. `MjSpec` adds a 3 kg child body to `base_link` (model mass 15.2064 → 18.2064 kg), edits a foot geom's friction and the gravity option. `mju_quatZ2Vec(−g)` then `mju_negQuat` gives the rotation under which a trunk level on a +5° floor reads 5.0° nose up. numpy's `Generator.spawn` leaves the parent stream's own draws unchanged.

**Implementation.** `build_model(config, hidden)`: a 15 × 9 × 8 cm box body of the payload's mass on the top face of the trunk's collision box at the preset's x, y offset, colliding with nothing; the floor type's sliding friction on the four foot geoms; the slope as gravity tilted about y (`# shortcut: exact only for an infinite flat floor`). `WalkingPolicy(config, model, motor_strength)`: the PD gains and feed-forward torque are scaled after Efferent's torque clamp, so every torque and the torque limit scale, from the spawn on; with tilted gravity the trunk orientation the policy reads is turned to be relative to true gravity. `Go2World`: the hidden values and the declared payload drawn at construction from the `world` stream; latency a delay line of whole 0.02 s control periods, jitter −1, 0 or +1 period drawn at each reset from a child of the attempt's `noise` stream. Each effect is switched by `world.hidden_effects.enabled`. `measure_effects` in `learned_judgment/env/effects.py` measures through `Go2World` only.

**Friction sweep (dev, for the `slippery` type).** Only friction on, no per-run variation, seed 5: a 20 s trot at 0.6 m/s (mean speed along the heading over its last 4 s) and a stop from 0.9 m/s, at 13 sliding frictions from 0.4 to 0.02.

    uv run python -m learned_judgment.env.evidence friction-sweep -- uv run python -X utf8 artifacts/env/research/friction_sweep.py

Exit code 0, `artifacts/env/20261121T164432-friction-sweep`, values in `artifacts/env/friction-sweep/sweep.json`. No fall at any friction. Trot speed 0.5448 (0.4), 0.5947 (0.2), 0.5405 (0.1), 0.5291 (0.08), 0.4995 (0.06), 0.4726 (0.05), 0.2612 m/s (0.02); stop from 0.9 m/s 0.1428 m (0.4) to 0.2957 m (0.1). With no fall to bound it, the bound is the trot: 0.06 is the lowest friction within Gate 1's 20% speed tolerance at 0.6 m/s; `slippery` is 0.1 (DECISIONS.md).

**Payload offset check (dev, for the `heavy_payload` offset).** Only payload on, 3 kg, seed 5, 10 s trots at 0.6 and 0.9 m/s:

    uv run python -m learned_judgment.env.evidence payload-offset-check -- uv run python -X utf8 artifacts/env/research/payload_offset_check.py

Exit code 0, `artifacts/env/20261121T170552-payload-offset-check/stdout.txt`. Speeds along the heading (mean of the last 5 s) at 0.6 / 0.9 m/s: centred 0.5612 / 0.8696, rear end 0.5450 / 0.8477, front end 0.6138 / 0.8823, rear-left corner 0.5457 / 0.8666. No fall.

**Tests.**

    uv run python -m learned_judgment.env.evidence effects -- uv run pytest tests/env/test_effects.py -v

Exit code 0, `artifacts/env/20261121T171133-effects`, `20 passed`. Each test written for a decision was first seen failing for the expected reason, or, where it passed on the first implementation (delay line, slope sensing), was run against a version of the code with the bug it guards and failed. Full `tests/env`: 87 passed, 1 failed (`test_trots_thirty_seconds[0.3]`, the accepted Gate 1 result); `tests/env/test_determinism.py` and `tests/env/test_policy_parity.py` pass unmodified.

**Measurements.**

    uv run python -m learned_judgment.env.evidence effects-data -- uv run python -c "from learned_judgment.env import load_config; from learned_judgment.env.effects import measure_effects; measure_effects(load_config(), 5, 'artifacts/env/effects-data')"

(`effects`, not the plan's `calibrate`: the integrator moved `measure_effects` there.) Exit code 0, `artifacts/env/20261121T171250-effects-data`, values in `artifacts/env/effects-data/effects.json`, seed 5, one episode per value, 6 s after the decision instant. Baseline: the default preset with all effects off. "All off": a preset with every effect at its demanding end, all switched off; equal to the baseline in every value.

| Effect on alone | Value | Quantity | Baseline | Measured | Fall |
|---|---|---|---|---|---|
| payload | 3 kg, centred | steady speed at 0.6 m/s | 0.5480 m/s | 0.5645 m/s | no |
| slope | +5° (uphill) | steady speed at 0.6 m/s | 0.5480 m/s | 0.5214 m/s | no |
| motor_strength | 0.8 | steady speed at 0.6 m/s | 0.5480 m/s | 0.5552 m/s | no |
| friction | `slippery`, 0.0951 this world | stopping distance from 0.9 m/s | 0.1428 m | 0.3061 m | no |
| latency | 0.3 s nominal, 0.30 s this attempt | time to half of 0.6 m/s | 0.20 s | 0.42 s | no |

**Result: PASS on the criteria.** All off equals the baseline exactly; each effect alone changes its quantity; every preset stands; no hidden key reaches an observation or outcome; with motors at 0 the dog falls and the fall is reported with its time.

**Versions.** efferent 0.5.0, mujoco 3.11.0, onnxruntime 1.32.0, numpy 2.5.5, Python 3.13.18, Windows-10-10.0.19045, CPU only (recorder `run.json`). Git `8084642`, dirty (the Task 6 code was not yet committed). `config_sha256` 4552b401… for the sweep (before `slippery` was set), 3cd6778e… for the facts proof and the offset check, 988aa823… for the tests and the measurements (the committed `config.yaml`; it differs from 3cd6778e… only in the comment citing the offset check). Wall time: tests 70.06 s, measurements 39.95 s, sweep 175.63 s, offset check 58.15 s, facts 24.28 s (`run.json` `wall_s`).

**Limitations.**
- One seed and one episode per value. The world is deterministic, so a difference is the effect's, but its size at other seeds, speeds and durations is not measured, and the criterion asks only for a change.
- The 3 kg payload and the 0.8 motor strength made the dog slightly faster at 0.6 m/s, not slower. Not investigated.
- The latency of 0.30 s moved the time to half speed by 0.22 s, not 0.30 s. The dog steps in place while the command is delayed, so it may meet the command in another gait phase than the baseline; not investigated. The quantity is the first control period whose instantaneous trunk speed along the heading reaches half the command.
- Slope is tilted gravity: no visible incline, and the dog's anticlockwise drift (Gate 1) means "along the heading" holds at the start only. The policy senses the tilt, so `policy_obs[3:6]` shows it.
- Friction changes only the foot geoms' sliding coefficient. Other robot contacts with the floor (trunk, legs) keep MuJoCo's pair rule, the larger value, the floor's 1.0.
- The payload box collides with nothing.
- The latency jitter is drawn once per attempt, not per command.
- The preset values are a proposal; the integrator confirms them with the project owner.

**Cost.** TabPFN calls: 0.

**Decision and next step.** Presets proposed in `config.yaml` `world.presets`; floor types for `skills.dodge.declared_surfaces`: `default`, `slippery`. The integrator confirms the values at the merge.
