"""E6, simulator half: the facts the ball work relies on, on the pinned stack (mujoco 3.11.0, Unitree go2.xml on the flat
scene of learned_judgment/env/scene.py), and the values read from the model and the saved traces.
1. mujoco.mj_geomDistance exists and gives a signed distance (negative on overlap).
2. The model's geoms by (group, contype, conaffinity): which ones a ball with contype 1, conaffinity 1 collides with.
3. The trunk's collision geoms and the capsule around them (skills.dodge.physics.body_half_length_m, body_radius_m).
4. The trunk origin's mean height while trotting (skills.dodge.physics.base_height_m).
5. The tilt and tilt rate of normal commanded walking, and the trot's gait period (robot.recovery).
6. A drop test of the ball from 2 m for a sweep of solref damping ratios (ball.solref).
Run from the repository root: uv run python -X utf8 artifacts/env/research/e6_sim_facts.py
"""
import glob
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from learned_judgment.env import load_config  # noqa: E402
from learned_judgment.env.scene import build_model  # noqa: E402

config = load_config()
print("mujoco", mujoco.__version__)

# 1. Distance.
print("1. hasattr(mujoco, 'mj_geomDistance'):", hasattr(mujoco, "mj_geomDistance"))
config["ball"]["solref"] = [0.02, 1.0]  # any value: only the geometry matters here
m = build_model(config, ball=True)
d = mujoco.MjData(m)
ball_q = m.jnt_qposadr[m.body("ball").jntadr[0]]
trunk_box = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("base_link").id
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
for x in (1.0, 0.2):
    d.qpos[ball_q:ball_q + 3] = [x, 0.0, d.qpos[2]]  # ball centre on the trunk's x axis, x m ahead of its origin
    mujoco.mj_forward(m, d)
    print(f"   ball centre {x} m ahead of the trunk origin: mj_geomDistance(ball, trunk box) =",
          round(mujoco.mj_geomDistance(m, d, m.geom("ball").id, trunk_box, 10.0, None), 6),
          "| expected", round(x - 0.1881 - config["ball"]["radius_m"], 6))

# 2. Collision masks.
groups = {}
for g in range(m.ngeom):
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    groups.setdefault((int(m.geom_group[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g])), []).append(body)
print("2. geoms by (group, contype, conaffinity): count and bodies")
for key, bodies in sorted(groups.items()):
    print("  ", key, len(bodies), sorted(set(bodies)))

# 3. Trunk capsule: the trunk's collision geoms (base_link, contype or conaffinity nonzero), surface points in the
# trunk frame. Radius: the largest distance of a point from the trunk's x axis (no smaller capsule on that axis can hold
# them); half length: the shortest segment [-L, L] on that axis whose capsule of that radius holds every point.
plain = build_model(config)
base = plain.body("base_link").id
trunk = [g for g in range(plain.ngeom) if plain.geom_bodyid[g] == base and (plain.geom_contype[g] or plain.geom_conaffinity[g])]
rng = np.random.default_rng(0)
points = []
for g in trunk:
    kind, size, pos = plain.geom_type[g], plain.geom_size[g], plain.geom_pos[g]
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, plain.geom_quat[g])
    u = rng.normal(size=(200000, 3))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        local = size[0] * u / np.linalg.norm(u, axis=1, keepdims=True)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:  # corners and points on the faces
        local = np.clip(u, -1, 1) * size
        face = rng.integers(0, 3, len(local))
        local[np.arange(len(local)), face] = np.sign(u[np.arange(len(local)), face]) * size[face]
    else:  # cylinder: axis z, radius size[0], half height size[1]; rim, sides and caps
        ang = rng.uniform(0, 2 * np.pi, len(u))
        rad = size[0] * np.sqrt(rng.uniform(0, 1, len(u)))
        rad[: len(u) // 2] = size[0]
        local = np.column_stack((rad * np.cos(ang), rad * np.sin(ang), rng.uniform(-1, 1, len(u)) * size[1]))
        local[len(u) // 2:, 2] = np.sign(local[len(u) // 2:, 2]) * size[1]
    points.append(local @ rot.reshape(3, 3).T + pos)
    print("3. trunk collision geom", g, mujoco.mjtGeom(kind).name, "size", np.round(size, 5).tolist(),
          "pos", np.round(pos, 5).tolist())
pts = np.vstack(points)
radial = np.hypot(pts[:, 1], pts[:, 2])
radius = radial.max()
half_length = (np.abs(pts[:, 0]) - np.sqrt(np.maximum(radius ** 2 - radial ** 2, 0))).max()
print(f"   x extent of the trunk geoms: {pts[:, 0].min():.4f} to {pts[:, 0].max():.4f} m")
print(f"   capsule on the trunk x axis: radius {radius:.4f} m, segment half length {half_length:.4f} m "
      f"(reaches x = +-{half_length + radius:.4f} m)")

# 4. Base height while trotting: Gate 1 trot traces, after robot.acceptance.exclude_first_s.
late = config["robot"]["acceptance"]["exclude_first_s"]
heights = {}
for path in sorted(glob.glob("artifacts/env/locomotion/trot_*.npz")):
    trace = np.load(path)
    heights[path] = trace["qpos"][trace["t"] > late, 2].mean()
    print(f"4. {path}: mean trunk origin height after {late} s = {heights[path]:.4f} m")
print(f"   mean of the three: {np.mean(list(heights.values())):.4f} m")

# 5. Tilt and tilt rate (horizontal part of the world-frame angular velocity) over every saved commanded walking run:
# Gate 1 (artifacts/env/locomotion) and the command proof (artifacts/env/commands), whole commanded run.
worst_tilt, worst_rate = 0.0, 0.0
for path in sorted(glob.glob("artifacts/env/locomotion/*.npz")) + sorted(glob.glob("artifacts/env/commands/*.npz")):
    trace = np.load(path)
    q, v = trace["qpos"], trace["qvel"]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2), -1, 1)))
    world = np.empty((len(q), 3))
    for i in range(len(q)):
        mujoco.mju_rotVecQuat(world[i], v[i, 3:6], q[i, 3:7])
    rate = np.hypot(world[:, 0], world[:, 1])
    worst_tilt, worst_rate = max(worst_tilt, tilt.max()), max(worst_rate, rate.max())
    print(f"5. {path}: max tilt {tilt.max():.3f} deg, max tilt rate {rate.max():.4f} rad/s")
print(f"   largest: tilt {worst_tilt:.3f} deg, tilt rate {worst_rate:.4f} rad/s")
# Gait period (robot.recovery.hold_s): main frequency of the FL knee angle (qpos 9) in each trot, after exclude_first_s.
for path in sorted(glob.glob("artifacts/env/locomotion/trot_*.npz")):
    trace = np.load(path)
    knee = trace["qpos"][trace["t"] > late, 9]
    amp = np.abs(np.fft.rfft(knee - knee.mean()))
    freq = np.fft.rfftfreq(len(knee), trace["t"][1] - trace["t"][0])[1:][amp[1:].argmax()]
    print(f"5. {path}: gait frequency {freq:.3f} Hz, period {1 / freq:.4f} s")

# 6. Drop test: ball bottom at `drop` m, far from the dog; height of its bottom at the top of the first rebound.
r = config["ball"]["radius_m"]


def rebound(solref, drop=2.0):
    config["ball"]["solref"] = solref
    m = build_model(config, ball=True)
    d = mujoco.MjData(m)
    q, dof = m.jnt_qposadr[m.body("ball").jntadr[0]], m.body("ball").dofadr[0]
    d.qpos[q:q + 3] = [10.0, 10.0, drop + r]
    bounced, top = False, 0.0
    for _ in range(4000):
        mujoco.mj_step(m, d)
        z, vz = d.qpos[q + 2], d.qvel[dof + 2]
        bounced |= z < r + 0.005
        if bounced and vz > 0:
            top = max(top, z)
        if bounced and vz < 0 and top > 0:
            break
    return top - r


ratios = np.round(np.arange(0.05, 0.21, 0.01), 2)
for tc in (0.02, 0.03, 0.04, 0.05):
    print(f"6. solref [{tc}, ratio], rebound from 2.0 m:", [(float(k), round(rebound([tc, k]), 3)) for k in ratios])
for solref in ([0.04, 0.1], [0.04, 0.11]):
    print(f"   solref {solref}, coefficient of restitution sqrt(rebound / drop) for drops of 0.5, 1, 2, 4 m:",
          [round(float(np.sqrt(rebound(solref, h) / h)), 3) for h in (0.5, 1.0, 2.0, 4.0)])
