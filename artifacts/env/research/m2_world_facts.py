"""Two simulator facts Go2World relies on, re-checked on the pinned stack (mujoco 3.11.0, Unitree go2.xml on the flat
scene of learned_judgment/env/scene.py), because entry M measured them on other versions and another model:
1. the frames of the free joint's linear and angular velocity;
2. what MuJoCo does with a bad state (NaN or beyond 1e10 in qpos, qvel, qacc) and whether MjData.warning counts it.
Run from the repository root: uv run python -X utf8 artifacts/env/research/m2_world_facts.py
"""
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from learned_judgment.env import load_config  # noqa: E402
from learned_judgment.env.scene import build_model  # noqa: E402

print("mujoco", mujoco.__version__)
m = build_model(load_config())
d = mujoco.MjData(m)
dt = m.opt.timestep

# 1. Frames. Trunk rolled +90 deg about world x and lifted 2 m, gravity off, so nothing touches and the body z axis
# points along world -y. Set the free joint's qvel to linear [1, 0, 0] and angular [0, 0, 1], take one step.
m.opt.gravity[:] = 0.0
mujoco.mj_resetData(m, d)
d.qpos[2] = 2.0
d.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
d.qvel[0:3] = [1.0, 0.0, 0.0]
d.qvel[3:6] = [0.0, 0.0, 1.0]
p0, q0 = d.qpos[0:3].copy(), d.qpos[3:7].copy()
mujoco.mj_step(m, d)
q0inv, dq_world, dq_body = np.empty(4), np.empty(4), np.empty(4)
mujoco.mju_negQuat(q0inv, q0)
mujoco.mju_mulQuat(dq_world, d.qpos[3:7], q0inv)  # q1 = dq_world * q0
mujoco.mju_mulQuat(dq_body, q0inv, d.qpos[3:7])  # q1 = q0 * dq_body
w_world, w_body, body_z = np.empty(3), np.empty(3), np.empty(3)
mujoco.mju_quat2Vel(w_world, dq_world, dt)
mujoco.mju_quat2Vel(w_body, dq_body, dt)
mujoco.mju_rotVecQuat(body_z, np.array([0.0, 0.0, 1.0]), q0)
print("qvel[0:3] = [1, 0, 0]: base position change / dt in world =", np.round((d.qpos[0:3] - p0) / dt, 4))
print("qvel[3:6] = [0, 0, 1]: rotation rate from the quaternion change, world frame =", np.round(w_world, 4),
      "body frame =", np.round(w_body, 4), "; body z axis in world =", np.round(body_z, 4))

# 2. Bad states. Each case starts from a fresh reset with gravity back on, corrupts one value, then steps twice.
m.opt.gravity[:] = [0.0, 0.0, -9.81]
names = {w: mujoco.mjtWarning(w).name for w in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                                                mujoco.mjtWarning.mjWARN_BADQACC)}


def counts():
    return {names[w]: d.warning[w].number for w in names}


for label, corrupt in [("qpos[0] = nan", lambda: d.qpos.__setitem__(0, np.nan)),
                       ("qvel[0] = nan", lambda: d.qvel.__setitem__(0, np.nan)),
                       ("qvel[0] = 1e11", lambda: d.qvel.__setitem__(0, 1e11)),
                       ("qfrc_applied[6] = inf", lambda: d.qfrc_applied.__setitem__(6, np.inf))]:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.5
    for _ in range(10):
        mujoco.mj_step(m, d)
    corrupt()
    mujoco.mj_step(m, d)
    after1 = (counts(), round(float(d.time), 4), bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
              round(float(d.qpos[2]), 4))
    d.qfrc_applied[:] = 0.0
    mujoco.mj_step(m, d)
    print(f"{label}: after the next step warnings={after1[0]} time={after1[1]} finite={after1[2]} base_z={after1[3]};"
          f" one more step warnings={counts()} time={round(float(d.time), 4)}")
mujoco.mj_resetData(m, d)
print("after mj_resetData: warnings =", counts())
