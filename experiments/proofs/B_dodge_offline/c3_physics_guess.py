"""Check 3: DODGE physics_guess against hand-computed cases and an independent scalar reference.

FIXTURE CHECK - mechanics only. Uses the fixture config (null skills.dodge keys filled with
labelled fixture values; body sizes and base height are NOT the Go2's) and hand-made throws.
It checks that the formula is implemented as documented, not that it predicts the robot.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c3_physics_guess.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import math  # noqa: E402

import numpy as np  # noqa: E402

from fixture_dodge import FIXTURE_DECLARED, Checks, fixture_config, fixture_throw_sampler  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.skills import load_skill  # noqa: E402

checks = Checks()
config = fixture_config(load_config())
spec = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)
c = spec.physics
print(f"\nconstants (config skills.dodge.physics + ball.radius_m; body/base values are FIXTURE): {c}")
print(f"episode duration (FIXTURE) {spec.episode_duration_s} s")
SEP, COL, FALL, REC, LAT = (spec.outcome_index(n) for n in spec.outcome_names)
REACH = c.body_radius_m + c.ball_radius_m
REST_Z = c.ball_radius_m - c.base_height_m  # ball centre resting on the floor, relative to the base
DECLARED = [FIXTURE_DECLARED["declared_payload_kg"], 0.0]


def situation(x, y, z, vx, vy, vz, forward):  # noqa: ANN001, ANN201
    return np.array([x, y, z, vx, vy, vz, forward, *DECLARED], dtype=np.float64)


def guess(rows, actions):  # noqa: ANN001, ANN201
    return spec.physics_guess(np.atleast_2d(rows), np.atleast_2d(np.asarray(actions, dtype=np.float64)))


def logistic_of_clearance(clearance: float) -> float:
    return 1.0 / (1.0 + math.exp(clearance / c.collision_softness_m))


# ---------------------------------------------------------------------------------------
# Independent scalar reference: plain Python floats, one row at a time, no numpy.
# Lateral position is integrated step by step with the exact first-order update
# (v <- target + (v - target) * exp(-dt / tau)), a different route from the closed form.
# ---------------------------------------------------------------------------------------


def reference(row, action):  # noqa: ANN001, ANN201
    x0, y0, z0, vx, vy, vz, forward = (float(v) for v in row[:7])
    side, delay, hold = (float(v) for v in action)
    dt, duration, g, tau = c.time_step_s, spec.episode_duration_s, c.gravity_m_s2, c.response_time_s
    height = z0 - REST_Z
    landing = (vz + math.sqrt(vz * vz + 2.0 * g * height)) / g if height > 0 else 0.0
    on, off = delay + c.assumed_latency_s, delay + c.assumed_latency_s + hold
    times = [k * dt for k in range(int(math.floor(duration / dt + 1e-9)) + 1)]
    times = [min(t, duration) for t in times]
    if duration - times[-1] > 1e-9:
        times.append(duration)
    # substeps so the edges of the command fall exactly on integration points
    y, v, prev_t, best = 0.0, 0.0, 0.0, math.inf
    for t in times:
        for edge in sorted(e for e in (on, off) if prev_t < e < t) + [t]:
            span = edge - prev_t
            target = side if on <= prev_t < off else 0.0
            if span > 0:
                decay = math.exp(-span / tau)
                y += target * span + (v - target) * tau * (1.0 - decay)
                v = target + (v - target) * decay
            prev_t = edge
        f = min(t, landing)
        bx, by, bz = x0 + vx * f, y0 + vy * f, z0 + vz * f - 0.5 * g * f * f
        along = bx - forward * t
        off_axis = along - max(-c.body_half_length_m, min(c.body_half_length_m, along))
        best = min(best, math.sqrt(off_axis**2 + (by - y) ** 2 + bz**2) - REACH)
    return [max(best, 0.0), logistic_of_clearance(best), 0.0, 0.0, abs(side) * hold]


print("\n== hand-computed case F: a ball that stays far to the side ==")
# Ball at rest 0.5 m ahead and 1.0 m to the left, dog standing, no dodge: it drops straight
# down; the closest point is at t = 0: sqrt((0.5 - half)^2 + 1^2) - (body radius + ball radius).
far = situation(0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
expected = math.sqrt((0.5 - c.body_half_length_m) ** 2 + 1.0) - REACH
out = guess(far, [0.0, 0.0, 0.2])[0]
print(f"  min_separation {out[SEP]:.9f} (hand {expected:.9f}); collision {out[COL]:.3e}")
checks.check("far ball: separation equals the hand value", abs(out[SEP] - expected) < 1e-12)
checks.check("far ball: large separation, collision guess near 0", out[SEP] > 0.8 and out[COL] < 1e-6)
checks.check("fall and recovery guesses are 0", out[FALL] == 0.0 and out[REC] == 0.0)

print("\n== hand-computed case R: ball resting on the floor ahead, dog trotting toward it ==")
# Ball resting at x = 1.0, y = 0.5 on the floor; dog at 0.5 m/s, no dodge. The nose of the
# capsule reaches x = 0.5 * 1.6 + half = 0.99 at the end: distance sqrt(0.01^2 + 0.5^2 + rest_z^2).
resting = situation(1.0, 0.5, REST_Z, 0.0, 0.0, 0.0, 0.5)
end_nose = 0.5 * spec.episode_duration_s + c.body_half_length_m
expected = math.sqrt((1.0 - end_nose) ** 2 + 0.25 + REST_Z**2) - REACH
out = guess(resting, [0.0, 0.0, 0.2])[0]
print(f"  min_separation {out[SEP]:.9f} (hand {expected:.9f})")
checks.check("resting ball: separation equals the hand value at the last time step", abs(out[SEP] - expected) < 1e-9)

print("\n== hand-computed case A: a ball aimed at the trunk ==")
# Thrown from 3 m ahead at 5 m/s with vz = g * 0.6 / 2: it is back at z = 0 at x = 0 at t = 0.6 s.
aimed = situation(3.0, 0.0, 0.0, -5.0, 0.0, 0.5 * c.gravity_m_s2 * 0.6, 0.0)
no_dodge = [0.0, 0.0, 0.2]
out = guess(aimed, no_dodge)[0]
hand_collision = logistic_of_clearance(-REACH)
print(f"  no dodge: min_separation {out[SEP]}, collision {out[COL]:.6f} (hand, centre through the axis: {hand_collision:.6f})")
checks.check("aimed ball, no dodge: separation 0", out[SEP] == 0.0)
checks.check("aimed ball, no dodge: collision guess >= 0.5", out[COL] >= 0.5)
checks.check("aimed ball, no dodge: collision equals the hand value", abs(out[COL] - hand_collision) < 1e-9)

full_left, full_right = [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]
left, right = guess(np.stack([aimed, aimed]), [full_left, full_right])
# hand value of the sideways shift at t = 0.6 s (on the crossing): v * (s - tau * (1 - exp(-s / tau))), s = 0.6 - latency
s = 0.6 - c.assumed_latency_s
shift = s - c.response_time_s * (1.0 - math.exp(-s / c.response_time_s))
print(f"  full early dodge left: {left[SEP]:.4f} m, right: {right[SEP]:.4f} m (hand shift at the crossing {shift:.4f} m)")
checks.check("full early dodge to the left separates more than no dodge", left[SEP] > out[SEP])
checks.check("full early dodge to the right separates more than no dodge", right[SEP] > out[SEP])
checks.check("full early dodge lowers the collision guess below 0.5", left[COL] < 0.5 and right[COL] < 0.5)
checks.check("full dodge clearance is at most the hand shift minus the reach", left[SEP] <= shift - REACH + 1e-12)

print("\n== start delay ==")
slow = situation(5.0, 0.0, 0.0, -5.0, 0.0, 0.5 * c.gravity_m_s2 * 1.0, 0.0)  # crosses x = 0 at t = 1.0 s
delays = [0.0, 0.2, 0.4, 0.6]
by_delay = guess(np.stack([slow] * 4), [[1.0, d, 1.0] for d in delays])[:, SEP]
print(f"  separation for start delays {delays}: {np.round(by_delay, 4).tolist()}")
checks.check("a later start never separates more", bool(np.all(np.diff(by_delay) <= 0.0)))
checks.check("a late start (0.6 s) separates less than an early one (0 s)", by_delay[-1] < by_delay[0])
checks.check("0.2 s separates less than 0 s, 0.4 s less than 0.2 s", by_delay[1] < by_delay[0] and by_delay[2] < by_delay[1])

print("\n== mirror symmetry ==")
checks.check("left and right full dodges of a centred throw are equal", np.array_equal(left, right))
rng = stream_rng(11, "fixture/dodge/physics/mirror")
rows = spec.sample_situations(rng, 300)
grid = spec.candidate_actions(rows[0], spec.candidate_points_per_dim)
actions = grid[rng.integers(0, grid.shape[0], size=300)]
mirrored_rows, mirrored_actions = rows.copy(), actions.copy()
mirrored_rows[:, [1, 4]] *= -1.0  # ball_rel_y_m, ball_vel_y_m_s
mirrored_actions[:, 0] *= -1.0  # side velocity
in_range = np.all((mirrored_rows[:, [1, 4]] >= [-2.0, -3.0]) & (mirrored_rows[:, [1, 4]] <= [2.0, 3.0]))
checks.check("mirrored rows stay inside the fixture ranges", bool(in_range))
checks.check(
    "300 sampled throws: mirrored situation + mirrored side give identical guesses",
    np.array_equal(spec.physics_guess(rows, actions), spec.physics_guess(mirrored_rows, mirrored_actions)),
)

print("\n== lateral_command_m ==")
batch = spec.physics_guess(rows, actions)
checks.check("lateral_command_m == |side velocity| x hold exactly", np.array_equal(batch[:, LAT], np.abs(actions[:, 0]) * actions[:, 2]))

print("\n== batch equals row by row, and the scalar reference ==")
single = np.concatenate([spec.physics_guess(rows[i : i + 1], actions[i : i + 1]) for i in range(rows.shape[0])])
checks.check("300 rows: batch call equals row-by-row calls exactly", np.array_equal(batch, single))
ref = np.array([reference(rows[i], actions[i]) for i in range(rows.shape[0])])
worst = float(np.max(np.abs(ref - batch)))
print(f"  largest |vectorised - scalar reference| over 300 x 5 values: {worst:.3e}")
checks.check("vectorised guess matches the independent scalar reference within 1e-9", worst < 1e-9)
hits = int(np.sum(batch[:, SEP] == 0.0))
print(f"  fixture throws with a zero separation guess: {hits} of 300 (fixture sampler aims near the dog)")
checks.check("the fixture throws include both hits and misses", 0 < hits < 300)
checks.check("guess inside the outcome bounds", bool(np.all(batch >= spec.outcome_low) and np.all(batch <= spec.outcome_high)))

print("\n== refusals ==")
checks.raises("action outside the bounds", ValueError, lambda: guess(aimed, [1.5, 0.0, 0.2]), "outside the declared action bounds")
checks.raises("wrong situation width", ValueError, lambda: spec.physics_guess(np.zeros((2, 8)), np.zeros((2, 3))), "situations must have shape (n, 9)")
checks.raises("row count mismatch", ValueError, lambda: spec.physics_guess(np.stack([aimed] * 2), [no_dodge]), "must have the same rows")
nan_row = aimed.copy()
nan_row[0] = np.nan
checks.raises("non-finite situation", ValueError, lambda: guess(nan_row, no_dodge), "non-finite")

checks.finish()
