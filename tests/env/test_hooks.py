"""record_episode, the dashboard's recorder, on the real simulator and walking policy."""
import numpy as np
import pytest

from app.hooks import record_episode
from learned_judgment.env import Go2World


@pytest.mark.system
def test_records_real_frames(config, forward):
    r = record_episode(config, seed=5, situation=None, commands=[(forward, 2.0)])
    assert r["frames"].shape[0] == 50 and r["frames"].shape[-1] == 3
    assert not np.array_equal(r["frames"][0], r["frames"][-1])
    assert r["outcome"]["fall"] == 0


@pytest.mark.system
def test_recording_runs_the_same_episode_as_plain_steps(config, forward):
    turn = forward | {"yaw_rate": 0.5}
    recorded = record_episode(config, 5, None, [(forward, 0.5), (turn, 0.51)])
    world = Go2World(config, 5)
    world.reset(); world.step(forward, 0.5); world.step(turn, 0.51)
    plain = world.finish_episode()
    keys = ("final_base_pos_m", "elapsed_s", "fall")
    assert {k: recorded["outcome"][k] for k in keys} == {k: plain[k] for k in keys}
    # 0.51 s is 25.5 control periods, held for 26 as step() rounds it: one observation per command, at its end
    assert [o["t_s"] for o in recorded["observations"]] == pytest.approx([0.5, 1.02])


@pytest.mark.system
def test_fps_must_divide_the_control_rate(config, forward):
    with pytest.raises(ValueError, match="fps"):
        record_episode(config, seed=5, situation=None, commands=[(forward, 0.1)], fps=30)
