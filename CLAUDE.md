# Project agent rules

@MISSION.md

## Operating rules

- Read MISSION.md and the relevant numbered work prompt before changing code.
- The product is Learned Judgment for a Unitree Go2. The primary behavior is dodging a thrown ball. Use only the verified Go2 stack; if a gate fails, stop and resolve it without substituting another robot, controller, or project.
- The walking policy is borrowed and stays frozen. PFN/TabPFN learns high-level decisions only through verified commands supported by that policy. Never invent commands or controller behavior.
- Person A owns Go2/MuJoCo integration, environment, robot/world configuration, and dashboard hooks. Person B owns SkillSpecs, shared engine, TabPFN teacher/cache, students, experiments, and evaluation. Keep to the ownership in 05_PERSON_A.md and 06_PERSON_B.md.
- Before each implementation slice, use an UltraCode research agent to verify exact APIs, versions, policy files, configs, licenses, and runtime assumptions from primary sources. Ask for a minimal proof-of-concept command and observed output. If UltraCode is unavailable, use an independent agent or primary-source research and document that substitution.
- Stop at an unsupported or failed gate and resolve it. Do not replace a real component with a mock, fixture, animation, another policy, or alternate project to claim completion.
- Tests must run against actual project code and, for system claims, the actual Go2 simulator and selected walking policy. Record exact commands, exit codes, software/model versions, seeds, configs, and artifact paths.
- Never invent, hardcode, cherry-pick, or tune reported results on test data. Generate metrics from saved run data. Label unexecuted checks NOT RUN; never present them as PASS.
- Use deterministic seeds for reproduction, not to force favorable outputs. Fixtures are permitted only for pure-function unit tests and cannot enter evaluation results.
- Before a real TabPFN request, check current usage and estimated cost. Cache complete requests/responses, prevent duplicate prediction calls, and keep secrets out of source, logs, commits, and browser code.
- Keep every tunable in config.yaml. Record meaningful work in LOG.md and decisions in DECISIONS.md.
- Do not edit vendored third-party documentation as project documentation. Keep upstream content intact.
