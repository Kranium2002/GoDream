# Item 2: load the menagerie Go2 scene as an MjSpec, add a free body with a box geom and mass, compile.
import sys, inspect
import mujoco, numpy as np

print("mujoco", mujoco.__version__, "numpy", np.__version__)
scene = sys.argv[1]

base = mujoco.MjModel.from_xml_path(scene)
print("before: nbody", base.nbody, "ngeom", base.ngeom, "nq", base.nq, "nv", base.nv)

spec = mujoco.MjSpec.from_file(scene)
body = spec.worldbody.add_body(name="obstacle", pos=[1.0, 0.0, 0.1])
body.add_freejoint(name="obstacle_free")
geom = body.add_geom(name="obstacle_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.1, 0.1, 0.1], mass=2.5, rgba=[1, 0, 0, 1])
# second variant: static body (no joint), mass set explicitly on the body
wall = spec.worldbody.add_body(name="wall", pos=[2.0, 0.0, 0.25], mass=7.0,
                               inertia=[1, 1, 1], explicitinertial=True)
wall.add_geom(name="wall_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 1.0, 0.25])

m = spec.compile()
d = mujoco.MjData(m)
print("after: nbody", m.nbody, "ngeom", m.ngeom, "nq", m.nq, "nv", m.nv)
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wall")
print("obstacle body id", bid, "body_mass", m.body_mass[bid], "geom id", gid, "geom_size", m.geom_size[gid])
print("wall body id", wid, "body_mass", m.body_mass[wid], "(static, no joint -> welded to world: body_weldid=%d)" % m.body_weldid[wid])
print("obstacle qpos adr", m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_free")])
print("keyframe count", m.nkey, "key_qpos len", m.key_qpos.shape)
print("key 'home' qpos[0:19] unchanged:", np.array_equal(m.key_qpos[0][:19], base.key_qpos[0]), "padded qpos[19:26]:", m.key_qpos[0][19:])
# robot qpos is first 19 entries; new freejoint appended after (body order = tree order)
mujoco.mj_resetDataKeyframe(m, d, 0)
for _ in range(500):
    mujoco.mj_step(m, d)
print("obstacle z after 1 s (should rest at 0.1 on floor):", round(float(d.xpos[bid][2]), 4))
print("signature MjSpec.compile:", mujoco.MjSpec.compile.__doc__.strip().splitlines()[0])
print("signature MjsBody.add_body doc head:", (mujoco.MjsBody.add_body.__doc__ or "").strip().splitlines()[0])
