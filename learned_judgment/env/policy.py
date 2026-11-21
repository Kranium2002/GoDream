"""The borrowed Go2 walking policy, driven one 50 Hz control step at a time.

A thin adapter over Efferent 0.5.0. Efferent's Runner.run() blocks for a whole run, so this class owns the loop
and calls Efferent's own pieces once per step(): the backend (spawn, settle, PD, physics), the Runner's soft
start, ObsBuilder, PolicySession, ActionRateLimiter, the Runner's action mapping and CommandClamp.
tests/env/test_policy_parity.py checks the result is bit-identical to Runner.run().
"""
import shutil
from collections.abc import Callable

import mujoco
import numpy as np
import onnxruntime as ort
from efferent import Binding, Command, PolicySession, Runner, load_descriptor, load_policy
from efferent.backends.mujoco_backend import MujocoBackend


class _Backend(MujocoBackend):
    """Efferent's MujocoBackend on a given MjModel, with an optional hook after every physics step."""

    def __init__(self, model: mujoco.MjModel, control_dt: float, start_pose: dict, base_pos: list,
                 auto_base_height: bool, settle_s: float):
        # MujocoBackend.__init__ (mujoco_backend.py:23-52) only loads its model from an XML path and stores its
        # arguments; these are the same attributes, set on our model instead.
        self._mj, self.model, self.data = mujoco, model, mujoco.MjData(model)
        self.control_dt, self.substeps = control_dt, max(1, round(control_dt / model.opt.timestep))
        self.render, self._viewer, self.estopped = False, None, False
        self.start_pose, self.base_pos, self.base_quat = start_pose, base_pos, None
        self.auto_base_height, self.settle_s = auto_base_height, settle_s
        self.on_substep: Callable[[], None] | None = None

    def write(self, cmd) -> None:
        if self.on_substep is None:
            return super().write(cmd)
        substeps, self.substeps = self.substeps, 1  # Efferent's own PD and mj_step, one physics step per call
        try:
            for _ in range(substeps):
                super().write(cmd)
                self.on_substep()
        finally:
            self.substeps = substeps


class WalkingPolicy:
    def __init__(self, config: dict, model: mujoco.MjModel):
        wp = config["robot"]["walking_policy"]
        manifest, onnx_path, workdir = load_policy(wp["checkpoint"])
        self.session = PolicySession(manifest, onnx_path)
        # PolicySession takes no session options, so its ONNX Runtime session is replaced by the same model on one
        # intra-op and one inter-op thread: inference then does not depend on the machine's thread count.
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        shutil.rmtree(workdir, ignore_errors=True)  # load_policy unzips to a fresh temp dir; the session holds the model
        self.binding = Binding(manifest, load_descriptor(wp["matching_config"][0]))  # go2.robot.yaml
        self.control_dt = 1.0 / manifest.control.rate_hz
        self.model = model
        spawn = wp["spawn"]
        self.backend = _Backend(model, self.control_dt, start_pose={j.name: j.default for j in manifest.joints},
                                base_pos=spawn["base_pos"], auto_base_height=spawn["auto_base_height"],
                                settle_s=spawn["settle_s"])
        self.data = self.backend.data
        self._runner = Runner(self.binding, self.session, self.backend)  # for its components, never run()
        self.last_obs = np.zeros(0, dtype=np.float32)

    def reset(self, on_substep: Callable[[], None] | None = None) -> None:
        """Spawn, settle and soft start, as Runner.run() does before its first policy step. on_substep runs after
        every physics step of the settle and the soft start."""
        r = self._runner
        self.backend.on_substep = on_substep
        try:
            self.backend.connect(self.binding.descriptor)
            self.session.reset()
            r.obs_builder.reset()
            r.rate_limiter.reset()
            r._soft_start()
        finally:
            self.backend.on_substep = None
        assert self.binding.manifest.envelope is None  # so Runner.run() would not check a start envelope here
        self._last_action = np.zeros(self.binding.manifest.num_joints(), dtype=np.float32)
        self._steps = 0
        self.last_obs = np.full(self.session.expected_obs_dim, np.nan, dtype=np.float32)  # no policy step yet

    def step(self, command: np.ndarray, on_substep: Callable[[], None] | None = None) -> None:
        """One control step: command is [vel_x, vel_y, yaw_rate] in m/s, m/s, rad/s. on_substep runs after
        every physics step. The body is Runner._run_policy's loop body (loop.py:172-184)."""
        r = self._runner
        state = self.backend.read()
        obs = r.obs_builder.build(state, self._last_action, Command(values=np.asarray(command, dtype=np.float32)),
                                  self._steps * self.control_dt)
        action = r.rate_limiter(self.session.step(obs))
        cmd = r.clamp(r._map_action(action, state), np.asarray(state.q, dtype=np.float32))
        self.backend.on_substep = on_substep
        try:
            self.backend.write(cmd)
        finally:
            self.backend.on_substep = None
        self._last_action, self.last_obs = action, obs
        self._steps += 1
