# Learned Judgment experiment log

Entries are below the recording rules. No Go2 episode and no TabPFN prediction request has been run yet.

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
