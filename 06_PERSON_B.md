# Person B — SkillSpecs, PFN engine, students, and evaluation

You own the learned high-level decisions. Read MISSION.md and all four numbered prompts. The primary skill is DODGE: the Go2 chooses when and which verified command to use to evade a physically simulated thrown ball. The same engine must support additional skills through SkillSpecs.

## Research before each slice

Before implementation, send an UltraCode research agent one bounded verification task. Prioritize the pinned tabpfn-client APIs, categorical inputs, residual/quantile outputs, batch behavior, cost/usage calls, cache semantics, MLP behavior at small sample sizes, and evaluation calculations. Require primary source links, exact package/version/API names, minimal proof command/output, limitations, and pass/fail recommendation. Research agents do not edit production files. If UltraCode is unavailable, use an independent research agent or primary-source verification and record the substitution.

Never treat a package name, README, type hint, or mock call as proof that the actual operation works. Test signatures and output shapes before making production code depend on them. If a real provider request is required, first inspect current usage and estimated cost. A billable request is not the first test of a guessed signature.

## Owned files

- skills/** — one SkillSpec per skill; DODGE first.
- learned_judgment/engine/** — generic collect, teacher, decide, dream, distill, and adapt pipeline.
- learned_judgment/teacher.py — real TabPFN provider, batching, uncertainty, disk request/response cache, and call/cost log.
- learned_judgment/students/** — per-skill MLP first, then shared MLP with skill ID/dream rehearsal; SNN only after required work passes.
- learned_judgment/eval/** — fixed splits, real-run harness, metrics, bootstrap, and evaluation reports.
- experiments/** and results/** — actual experiment logs, result tables, and figures.
- PREREGISTRATION.md — freeze criteria and record dated exploratory amendments separately.

Person A owns learned_judgment/env/**, config.yaml, app/**, and ENV_REPORT.md. Do not create another simulator or robot adapter. Do not change shared setup files or the four numbered prompts without agreement from the integrator.

## Contract to consume

Import the actual simulator only through the public interface agreed with Person A:

    Go2World(config, seed)
    world.reset(situation) -> observation
    world.step(command, duration_s) -> observation
    world.finish_episode() -> measured_outcome

Use only observations and high-level commands that Person A proved are exposed by the selected walking policy. Never train on hidden payload, friction, motor strength, latency, or other simulator-only values. Do not assume command fields or ranges; consume Person A's verified command schema.

Each SkillSpec contains situation schema, bounded 1–3 dimensional actions, outcome type, physics_guess, objective over predicted outcome distributions, episode(world, situation, action), and hidden-effect notes. The engine must not contain skill-specific conditionals.

Teacher features are observed situation + candidate action + physics_guess. Residual target is measured_outcome − physics_guess, in the fixed per-skill outcome order. The teacher API is:

    teacher.fit(features, residuals)
    median_residual, predictive_spread = teacher.predict_residuals(query_features)

Return arrays in a documented, stable shape. Candidate actions, including the physics action, are evaluated in one batched TabPFN request. The objective belongs to the SkillSpec and must explicitly handle uncertainty, collision, and fall risk for DODGE.

## Development without Person A's merge

Work on pure interfaces, SkillSpec validation, cache serialization, and engine structure while Person A works. Do not generate performance metrics until actual Go2 MuJoCo rows are available. Do not build a synthetic performance dataset or deterministic fake teacher to stand in for the simulator/provider. A pure-function fixture may be used only to check that utility and is excluded from all reports.

When Person A hands off, connect the engine to the actual public environment and run the actual policy. Start with DODGE. The episode must use the physical ball scene and return measured separation/collision/fall outcomes, not values inferred from the selected action.

## Provider, cache, and cost

Use the configured hosted TabPFN client unless a documented local CPU test proves a licensed local path. Cache the complete request and response on disk, keyed by all training features/targets, query rows, output type/quantiles, and model settings. A repeated identical request must return the stored response without another prediction request. Log actual calls and quoted/actual usage without credentials or raw secrets.

Before every uncached hosted batch, inspect current account usage and estimate cost. The request contains simulated project data only. Never print or commit credentials. If the budget or key is unavailable, stop and mark provider acceptance NOT RUN / OPEN; never substitute a fake response.

## Acceptance criteria

Mark each criterion PASS only with executed evidence; otherwise mark OPEN / NOT RUN and name the blocker:

- The engine imports and uses the public Go2 interface; actual episode rows match the shared schema and residual sign.
- DODGE has a SkillSpec whose candidate actions are all verified supported commands and whose objective penalizes collision/fall while rewarding separation and controlling unnecessary motion.
- Actual TabPFN request/response behavior, categorical handling, quantile shapes, cost quote, and usage check match verified documentation and a real small provider request.
- An identical real provider request is served from the persistent cache without a second prediction request; changed inputs/settings produce a new cache key.
- Teacher candidate scores use one batched request and are computed from physics guess plus actual predicted residual distribution.
- Per-skill and shared MLPs train on actual teacher dreams and are evaluated on actual Go2 MuJoCo episodes. Small-sample failures are recorded, not hidden.
- Fixed dev/test/shift splits and seeds follow 03_METRICS.md. Test data is untouched until final evaluation.
- The DODGE end-to-end run computes physics-only, constant-bias, teacher, and MLP metrics from the same actual dev situations and ball throws.
- A fresh Python 3.13 environment installs the declared packages and runs the documented module commands.

## No fake tests, reports, or performance

Tests must execute against actual repository code. System and provider claims require the actual MuJoCo Go2/policy and actual TabPFN provider. Unit fixtures may test pure math/serialization only and cannot create report rows. Record command, exit code, dependency/policy/model versions, seed/config, request hash where relevant, and output artifact. Generate every metric/table/figure from saved run output; no hardcoded values, copied runs, synthetic provider responses, fabricated PASS labels, cherry-picking, or test tuning. Unrun means NOT RUN, never PASS.

## Handoff and integration

Return public method signatures, exact feature/action/outcome order, cache proof from an actual request, API cost log sample with no secrets, evaluation commands and artifacts, dependency requests, and limitations. The integration is incomplete until the same actual Go2 DODGE episodes pass through the true simulator, physics baseline, TabPFN teacher, dream generation, and measured student. Failed hypotheses may be reported honestly; missing or fabricated evidence cannot be called success.
