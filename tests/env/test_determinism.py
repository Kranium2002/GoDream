"""Seeded random streams and bit-identical episodes."""
import numpy as np
import pytest

from learned_judgment.env import Go2World, make_streams


def test_streams_are_independent_and_repeatable():
    a, b = make_streams(7), make_streams(7)
    a["ball"].random(1000)
    assert a["noise"].random() == b["noise"].random()


def test_attempt_streams_differ_from_the_world_streams():
    world, attempt_0 = make_streams(7), make_streams(7, attempt_seed=0)
    assert all(world[k].random() != attempt_0[k].random() for k in world)


@pytest.mark.system
def test_fixed_seed_reproduces_the_episode(config, forward):
    def run():
        w = Go2World(config, seed=20261121); w.reset(attempt_seed=3); w.step(forward, 3.0)
        return dict(np.load(w.finish_episode()["trace_path"]))
    first, second = run(), run()
    assert first.keys() == second.keys() and all(np.array_equal(first[k], second[k]) for k in first)


@pytest.mark.system
def test_reused_world_equals_fresh_world(config, forward):
    # A world that ran other episodes before, switching between the scenes with and without the ball, gives a trace
    # bit-identical to a fresh world's for the same seed, situation and attempt seed.
    throw = Go2World(config, seed=6).sample_dodge_situation()

    def run(world, situation, attempt_seed):
        world.reset(situation, attempt_seed=attempt_seed); world.step(forward, 1.0)
        return dict(np.load(world.finish_episode()["trace_path"]))
    reused = Go2World(config, seed=6)
    run(reused, throw, 0)
    for situation, attempt_seed in (({}, 3), (throw, 4)):
        again, fresh = run(reused, situation, attempt_seed), run(Go2World(config, seed=6), situation, attempt_seed)
        assert again.keys() == fresh.keys() and all(np.array_equal(again[k], fresh[k]) for k in again)


@pytest.mark.system
def test_attempt_seed_fixes_the_noise_whatever_ran_before(config, zero, period):
    a, b = Go2World(config, seed=5), Go2World(config, seed=5)
    a.reset(attempt_seed=1)  # an unrelated episode that draws from its own streams
    a.attempt_streams["noise"].random(100)
    a.step(zero, period)
    a.finish_episode()
    a.reset(attempt_seed=9)
    b.reset(attempt_seed=9)
    assert all(a.attempt_streams[k].random() == b.attempt_streams[k].random() for k in a.attempt_streams)
    a.reset()  # third episode of this world: attempt_seed None means the episode counter, 2
    assert a.attempt_streams["noise"].random() == make_streams(5, attempt_seed=2)["noise"].random()
