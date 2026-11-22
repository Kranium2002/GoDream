"""Check 0: the sideways-travel formula of SidestepReachSpec equals the one inline in DodgeSpec,
plus pure-function checks of the spec (objective, loss, success, schedule, sampling).

PLUMBING CHECK. No simulator and no model run here. DodgeSpec cannot be built from the real
config.yaml today (its ball and body keys are null until Person A's ball work), so it is built
from a FIXTURE copy of config.yaml in which only these keys are set:

    skills.dodge.episode_duration_s, skills.dodge.situation_ranges.* (the seven null ones),
    skills.dodge.declared_surfaces, skills.dodge.physics.body_half_length_m / body_radius_m /
    base_height_m (null today), and skills.dodge.physics.gravity_m_s2 (9.81 -> 1e-200).

How DodgeSpec's sideways travel is read out of its physics guess: DodgeSpec returns only the
closest ball-to-trunk clearance, not the travel. The fixture puts a ball at y = +/-3 m beside a
trunk of zero length that does not move forward, with almost no gravity (the ball stays at the
base height for the whole episode) and a speed of 1000 m/s along x timed so that the ball
crosses x = 0 exactly at grid time t_k. At every other grid time it is at least 20 m away, so
the clearance is |3 - travel(t_k)| - reach, and travel(t_k) = Y - sign(Y) (clearance + reach).
The same arithmetic is replayed with this spec's formula, so equal formulas give bit-equal
clearances; the recovered travels are also compared, with the largest difference printed and
held against dev_checks.sidestep_reach.formula_match_tolerance_m.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_sidestep_dev\\c0_formula_match.py
"""

from __future__ import annotations

import copy

import numpy as np

from _common import PLUMBING_LABEL, Checks

print(PLUMBING_LABEL)

from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.skills.base import CommandSchema  # noqa: E402
from learned_judgment.skills.dodge import _TIME_TOLERANCE_S, BALL_DOG_KEYS, DodgeSpec  # noqa: E402
from sidestep_reach_spec import ROOT, SidestepReachSpec, sideways_travel  # noqa: E402

checks = Checks()
config = load_config()
schema = CommandSchema.from_config(config)
tolerance = float(require(config, f"{ROOT}.formula_match_tolerance_m"))

# Fixture declared values for a spec built without a world (plumbing only).
FIXTURE_DECLARED = {"declared_payload_kg": 0.0, "declared_surface": "default"}
spec = SidestepReachSpec(config, schema, **FIXTURE_DECLARED)
print(f"\nspec {spec.name}: situation {spec.situation_names}, actions {spec.action_names}, outcomes {spec.outcome_names}")
print(f"FIXTURE declared values for this plumbing spec: {FIXTURE_DECLARED}")

# ----- fixture DODGE config ----------------------------------------------------------------------
BALL_Y_M = 3.0  # |ball y|: further than any travel of the grid (< 1 m) plus the reach, so the clearance stays > 0
BALL_SPEED_M_S = 1000.0  # x speed: one 0.02 s grid step away from t_k the ball is 20 m off, far beyond 3 m + 1 m
side_field = schema.field(require(config, "skills.dodge.action.side_velocity_field"))
forward_field = schema.field(require(config, "skills.dodge.action.forward_velocity_field"))
FIXTURE_VALUES = {
    "skills.dodge.episode_duration_s": spec.episode_duration_s,
    "skills.dodge.situation_ranges.ball_rel_x_m": [-1.0e4, 1.0e4],
    "skills.dodge.situation_ranges.ball_rel_y_m": [-1.0e4, 1.0e4],
    "skills.dodge.situation_ranges.ball_rel_z_m": [-1.0e4, 1.0e4],
    "skills.dodge.situation_ranges.ball_vel_x_m_s": [-1.0e4, 1.0e4],
    "skills.dodge.situation_ranges.ball_vel_y_m_s": [-1.0e4, 1.0e4],
    "skills.dodge.situation_ranges.ball_vel_z_m_s": [-1.0e4, 1.0e4],
    "skills.dodge.situation_ranges.dog_forward_speed_m_s": [forward_field.low, forward_field.high],
    "skills.dodge.declared_surfaces": ["default", "fixture_level"],
    "skills.dodge.physics.body_half_length_m": 0.0,
    "skills.dodge.physics.body_radius_m": 0.001,
    "skills.dodge.physics.base_height_m": 1.0,
    "skills.dodge.physics.gravity_m_s2": 1.0e-200,
}
fixture = copy.deepcopy(config)
for key, value in FIXTURE_VALUES.items():
    node = fixture
    *parents, leaf = key.split(".")
    for part in parents:
        node = node[part]
    print(f"  FIXTURE {key}: {node[leaf]!r} -> {value!r}")
    node[leaf] = value
dodge = DodgeSpec(fixture, schema)
times = dodge._time_grid()
checks.check(
    "DodgeSpec builds from the fixture config; its time grid is every control period to the episode end",
    times.shape == (spec.episode_periods + 1,) and times[-1] == spec.episode_duration_s,
    f"{times.shape[0]} grid times, last {times[-1]} s, step {dodge.physics.time_step_s} s",
)
checks.check(
    "both specs use the same formula constants",
    (dodge.physics.assumed_latency_s, dodge.physics.response_time_s) == (spec.assumed_latency_s, spec.response_time_s),
    f"assumed_latency_s {spec.assumed_latency_s}, response_time_s {spec.response_time_s}",
)

# ----- the comparison grid: every candidate of this spec x every control period -------------------
any_situation = spec.sample_situations(stream_rng(0, "sidestep_reach/c0/plumbing"), 1)[0]
candidates = spec.candidate_actions(any_situation, spec.candidate_points_per_dim)
n_c, n_t = candidates.shape[0], times.shape[0]
time_index = np.repeat(np.arange(n_t), n_c)
side = np.tile(candidates[:, 0], n_t)
hold = np.tile(candidates[:, 1], n_t)
t_k = times[time_index]
ball_y = np.where(side >= 0.0, BALL_Y_M, -BALL_Y_M)
crossing = BALL_SPEED_M_S * t_k
situations = np.zeros((n_c * n_t, len(BALL_DOG_KEYS) + 2))
situations[:, BALL_DOG_KEYS.index("ball_rel_x_m")] = -crossing
situations[:, BALL_DOG_KEYS.index("ball_rel_y_m")] = ball_y
situations[:, BALL_DOG_KEYS.index("ball_vel_x_m_s")] = BALL_SPEED_M_S
actions = np.column_stack([side, np.zeros_like(side), hold])  # start delay 0
clearance_dodge = dodge.physics_guess(situations, actions)[:, dodge.outcome_index("min_separation_m")]
reach = dodge.physics.body_radius_m + dodge.physics.ball_radius_m
travel_dodge = ball_y - np.sign(ball_y) * (clearance_dodge + reach)

latency, tau = spec.assumed_latency_s, spec.response_time_s
travel_mine = sideways_travel(side, hold, t_k, start_delay_s=0.0, assumed_latency_s=latency, response_time_s=tau)

# Replay DodgeSpec's clearance arithmetic over the whole grid with this spec's formula.
g = dodge.physics.gravity_m_s2
closest = np.full(side.shape[0], np.inf)
closest_at = np.zeros(side.shape[0], dtype=np.int64)
p_z = situations[:, BALL_DOG_KEYS.index("ball_rel_z_m")]
height = p_z - (dodge.physics.ball_radius_m - dodge.physics.base_height_m)
landing = np.where(height > 0.0, (0.0 + np.sqrt(0.0 * 0.0 + 2.0 * g * np.maximum(height, 0.0))) / g, 0.0)
for index, t in enumerate(times):
    flight = np.minimum(t, landing)
    ball_x = -crossing + BALL_SPEED_M_S * flight
    ball_z = p_z + 0.0 * flight - 0.5 * g * flight * flight
    y = sideways_travel(side, hold, t, start_delay_s=np.zeros_like(side), assumed_latency_s=latency, response_time_s=tau)
    along = ball_x - 0.0 * t
    off_axis = along - np.clip(along, -0.0, 0.0)
    distance = np.sqrt(off_axis * off_axis + (ball_y - y) ** 2 + ball_z * ball_z)
    smaller = distance - reach < closest
    closest_at = np.where(smaller, index, closest_at)
    np.minimum(closest, distance - reach, out=closest)
clearance_mine = np.maximum(closest, 0.0)

checks.check(
    "fixture geometry: the closest approach falls exactly at the queried grid time for every row",
    np.array_equal(closest_at, time_index),
    f"{side.shape[0]} rows = {n_c} candidates x {n_t} times",
)
checks.check(
    "fixture geometry: every clearance is positive, so it is not clipped at 0",
    bool(np.all(clearance_dodge > 0.0)),
    f"smallest clearance {clearance_dodge.min():.3f} m",
)
bit_equal = int(np.count_nonzero(clearance_mine == clearance_dodge))
print(f"\nclearances equal to the last bit: {bit_equal} of {side.shape[0]}")
difference = np.abs(travel_mine - travel_dodge)
print(f"largest |travel(this spec) - travel(DodgeSpec, recovered)|: {difference.max():.3e} m (tolerance {tolerance:.1e} m)")
print(f"travel range on the grid: {travel_mine.min():.4f} to {travel_mine.max():.4f} m")
checks.check(
    "this spec's sideways travel equals DodgeSpec's on every candidate x control period",
    bit_equal == side.shape[0] or float(difference.max()) <= tolerance,
    f"bit-equal clearances {bit_equal}/{side.shape[0]}, largest travel difference {difference.max():.3e} m",
)

# ----- pure-function checks of the spec ------------------------------------------------------------
print("\n== spec functions (plumbing) ==")
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
mass = SidestepReachSpec._quantile_weights(levels)
EPS = np.finfo(np.float64).eps  # float64 rounding unit: the comparisons below allow rounding only, nothing tuned
checks.check("quantile weights are positive and sum to 1", bool(np.all(mass > 0)) and abs(mass.sum() - 1.0) <= mass.size * EPS, f"{mass.tolist()}")
situation_rows = np.repeat(any_situation[None, :], n_c, axis=0)
physics = spec.physics_guess(situation_rows, candidates)
point = OutcomeDistribution.point(spec.outcome_names, physics, levels)
w = spec.weights
expected = -w.abs_error * np.abs(physics[:, 0]) - w.motion * physics[:, 2]
checks.check(
    "objective of a point mass is -abs_error |error| - motion x commanded travel (fall 0, spread 0)",
    np.allclose(spec.objective(point), expected, rtol=2 * mass.size * EPS, atol=0.0),
    f"{n_c} candidates",
)
checks.check(
    "physics guess: error = formula travel at the episode end - target; lateral = |side| x hold; fall 0",
    np.array_equal(
        physics[:, 0],
        sideways_travel(candidates[:, 0], candidates[:, 1], spec.episode_duration_s, start_delay_s=0.0,
                        assumed_latency_s=latency, response_time_s=tau) - any_situation[1],
    )
    and np.array_equal(physics[:, 2], np.abs(candidates[:, 0]) * candidates[:, 1])
    and not physics[:, 1].any(),
    "",
)
measured = np.array([[0.04, 0.0, 0.3], [-0.06, 0.0, 0.3], [0.0, 1.0, 0.3]])
checks.check("loss is |sideways error|", np.array_equal(spec.loss(measured), np.array([0.04, 0.06, 0.0])), "")
checks.check(
    "success needs |error| <= success_tolerance_m and no fall",
    spec.success(measured).tolist() == [True, False, False],
    f"tolerance {spec.success_tolerance_m} m",
)
situation_map = spec.decode_situation(any_situation)
schedule = spec.commands(situation_map, {"side_velocity_m_s": 0.5, "hold_duration_s": 0.3})
total = sum(duration for _, duration in schedule)
checks.check(
    "schedule: hold then tail, whole control periods summing to the episode duration",
    len(schedule) == 2 and abs(total - spec.episode_duration_s) <= _TIME_TOLERANCE_S
    and all(abs(d - round(d / spec.control_period_s) * spec.control_period_s) <= _TIME_TOLERANCE_S for _, d in schedule),
    f"{[(c, round(d, 6)) for c, d in schedule]}",
)
try:
    spec.commands(situation_map, {"side_velocity_m_s": 0.5, "hold_duration_s": 0.31})
    refused = False
except ValueError:
    refused = True
checks.check("a hold that is not a whole number of control periods is refused before any world call", refused, "0.31 s")
rng_a = stream_rng(0, "sidestep_reach/c0/plumbing/sample")
rng_b = stream_rng(0, "sidestep_reach/c0/plumbing/sample")
checks.check(
    "sample_situations: row i does not depend on n; declared values appended",
    np.array_equal(spec.sample_situations(rng_a, 5), spec.sample_situations(rng_b, 50)[:5]),
    "",
)

checks.finish()
