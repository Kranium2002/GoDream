"""Pure functions of the DODGE work: the throw sampler, its published ranges and the recovery measure."""
import math

import numpy as np
import pytest

from learned_judgment.env import BALL_KEYS, sample_dodge_situations
from learned_judgment.env.world import recovery

SITUATION = (*BALL_KEYS, "dog_forward_speed_m_s")


def test_sampler_returns_situations_reset_accepts(config):
    throws = sample_dodge_situations(config, np.random.default_rng(0), 3)
    assert len(throws) == 3 and all(list(t) == list(SITUATION) for t in throws)
    assert all(type(v) is float for t in throws for v in t.values())


def test_sampling_n_at_once_equals_n_single_draws(config):
    batch = sample_dodge_situations(config, np.random.default_rng(1), 4)
    rng = np.random.default_rng(1)
    assert batch == [sample_dodge_situations(config, rng, 1)[0] for _ in range(4)]


def test_throws_come_down_to_trunk_height_where_the_dog_will_be(config):
    # The thrower aims at the trunk origin's height where a dog keeping its forward speed will be, missing sideways by
    # at most the aim error. Flight times stay in the agreed window, 0.6 to 1.6 s (decision 4).
    g, (lo, hi) = config["ball"]["gravity_m_s2"], config["ball"]["throw_aim_error_range_m"]
    for t in sample_dodge_situations(config, np.random.default_rng(2), 1000):
        vz, h = t["ball_vel_z_m_s"], t["ball_rel_z_m"]
        flight = (vz + math.sqrt(vz * vz + 2 * g * h)) / g  # the later root of h + vz t - g t^2 / 2 = 0
        assert 0.6 <= flight <= 1.6
        assert t["ball_rel_x_m"] + t["ball_vel_x_m_s"] * flight == pytest.approx(t["dog_forward_speed_m_s"] * flight)
        assert lo - 1e-9 <= t["ball_rel_y_m"] + t["ball_vel_y_m_s"] * flight <= hi + 1e-9


def test_large_sample_stays_inside_the_published_situation_ranges(config):
    # The ranges were derived from another seed's sample (calibrate.situation_ranges); this is an independent one.
    ranges = config["skills"]["dodge"]["situation_ranges"]
    throws = sample_dodge_situations(config, np.random.default_rng(12345), 20000)
    for key in SITUATION:
        values = [t[key] for t in throws]
        assert ranges[key][0] <= min(values) and max(values) <= ranges[key][1], key


def test_recovery_time_and_flag():
    t = np.arange(150) * 0.02  # 3 s at 50 Hz
    calm = np.ones(150, dtype=bool)
    assert recovery(t, calm, hold_s=0.5) == (0.0, 1)
    knocked = calm.copy()
    knocked[50:65] = False  # out of the limits from 1.00 s to 1.28 s, then within them to the end
    assert recovery(t, knocked, hold_s=0.5) == (pytest.approx(0.30), 1)
    knocked[80:82] = False  # out again at 1.60 s, so the calm from 1.30 s did not hold; it holds from 1.64 s
    assert recovery(t, knocked, hold_s=0.5) == (pytest.approx(0.64), 1)
    knocked[100:] = False  # out from 2.00 s to the end: never held, so the time runs to the end of the episode
    assert recovery(t, knocked, hold_s=0.5) == (pytest.approx(t[-1] - 1.0), 0)
