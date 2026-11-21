"""The DODGE throw protocol: a thrower releases the ball on a ballistic path aimed at where the trotting dog will be.

A situation's ball keys are in the frame Go2World fixes at the decision instant (t_s 0): origin at the trunk origin, x
along the dog's heading (yaw only), y to its left, z up. ball_rel_* is the ball's position in it, ball_vel_* the ball's
velocity over the ground on those axes (not relative to the dog). The ball is released at that instant.
"""
import numpy as np

BALL_KEYS = ("ball_rel_x_m", "ball_rel_y_m", "ball_rel_z_m", "ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s")
_DRAWS = ("dog_forward_speed_range_m_s", "throw_origin_range_m", "throw_direction_range_deg",
          "throw_release_height_range_m", "throw_speed_range_m_s", "throw_aim_error_range_m")


def sample_dodge_situations(config: dict, rng: np.random.Generator, n: int) -> list[dict]:
    """n throws, each a dict that Go2World.reset accepts as is, its keys always in this order: the six BALL_KEYS as
    listed, then dog_forward_speed_m_s; values are Python floats. Needs no world and no physics.

    Per throw, uniform draws from the config.yaml `ball` ranges, six per throw in this order (so n at once equals n
    calls of one): the dog's forward speed s; the release point's horizontal distance d from the trunk origin, its
    bearing b (anticlockwise from the heading) and its height h above the trunk origin; the ball's horizontal speed u;
    the aim error e. The thrower aims at trunk-origin height at (s*T, e): where a dog keeping speed s along its heading
    will be after the flight time T, missed sideways by e. T solves |(s*T, e) - (d cos b, d sin b)| = u*T."""
    ball = config["ball"]
    lo, hi = np.array([ball[k] for k in _DRAWS], dtype=float).T
    s, d, b, h, u, e = rng.uniform(lo, hi, size=(n, len(_DRAWS))).T
    x0, y0 = d * np.cos(np.radians(b)), d * np.sin(np.radians(b))
    a = u * u - s * s  # (u^2 - s^2) T^2 + 2 s x0 T - (x0^2 + (e - y0)^2) = 0; u > s, so one positive root
    flight = (np.sqrt((s * x0) ** 2 + a * (x0 * x0 + (e - y0) ** 2)) - s * x0) / a
    vz = ball["gravity_m_s2"] * flight / 2 - h / flight  # back at the trunk origin's height after T
    rows = np.column_stack((x0, y0, h, (s * flight - x0) / flight, (e - y0) / flight, vz, s))
    return [dict(zip((*BALL_KEYS, "dog_forward_speed_m_s"), map(float, row))) for row in rows]
