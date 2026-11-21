"""record_episode, the dashboard's recorder, on the real simulator and walking policy."""
import numpy as np
import pytest

from app.hooks import record_episode


@pytest.mark.system
def test_records_real_frames(config, forward):
    r = record_episode(config, seed=5, situation=None, commands=[(forward, 2.0)])
    assert r["frames"].shape[0] == 50 and r["frames"].shape[-1] == 3
    assert not np.array_equal(r["frames"][0], r["frames"][-1])
    assert r["outcome"]["fall"] == 0


@pytest.mark.system
def test_fps_must_divide_the_control_rate(config, forward):
    with pytest.raises(ValueError, match="fps"):
        record_episode(config, seed=5, situation=None, commands=[(forward, 0.1)], fps=30)
