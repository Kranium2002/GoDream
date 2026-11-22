"""WalkingPolicy is Efferent's own control loop, one step at a time: same model, start and command, same result."""
import numpy as np
import pytest
from efferent import Command, Runner

from learned_judgment.env.policy import WalkingPolicy
from learned_judgment.env.scene import build_model

pytestmark = pytest.mark.system


def test_adapter_reproduces_efferent_runner_bit_for_bit(config):
    model = build_model(config)
    command = np.array([0.4, 0.0, 0.0], dtype=np.float32)

    # Reference: Efferent's Runner end to end (connect, settle, soft start, 2 s of policy) on its own components.
    ref = WalkingPolicy(config, model)
    result = Runner(ref.binding, ref.session, ref.backend, keep_history=True,
                    command_source=lambda t: Command(values=command.copy())).run(2.0, soft_start=True)
    assert not result.estopped and result.steps == 100

    ours = WalkingPolicy(config, model)
    ours.reset()
    substeps = []
    for _ in range(result.steps):
        ours.step(command, on_substep=lambda: substeps.append(ours.data.time))
    assert len(substeps) == result.steps * round(ours.control_dt / model.opt.timestep)
    assert np.array_equal(ours.last_obs, result.history[-1].obs)
    assert np.array_equal(ours.data.qpos, ref.data.qpos)
    assert ours.data.qpos[0] > 0.3  # it walked forward, so the equality above is not two robots standing still
