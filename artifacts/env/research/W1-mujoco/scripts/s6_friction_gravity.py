# Item 6: change floor friction and gravity direction on a compiled model; show the change takes effect.
import sys, math
import mujoco, numpy as np
print("mujoco", mujoco.__version__)

# Part A: box on a floor under tilted gravity (box friction tiny, so floor friction decides via element-wise max).
XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="box" pos="0 0 0.05"><freejoint/>
      <geom name="box" type="box" size="0.05 0.05 0.05" mass="1" friction="0.01 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
print("default gravity", m.opt.gravity, "default floor friction", m.geom_friction[floor])

def slide(friction0, gravity):
    m.geom_friction[floor, 0] = friction0
    m.opt.gravity[:] = gravity
    d = mujoco.MjData(m)
    for _ in range(500):  # 1 s at dt=0.002
        mujoco.mj_step(m, d)
    con = d.contact[0] if d.ncon else None
    return d.qpos[0], (con.friction[0] if con is not None else None)

th = math.radians(20)
tilt = [9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)]  # gravity rotated 20 deg towards +x
print("upright gravity, mu=1.0   -> x after 1 s: %.5f" % slide(1.0, [0, 0, -9.81])[0])
x, mu = slide(1.0, tilt)
print("tilted 20 deg,  mu=1.0   -> x after 1 s: %.5f  (contact.friction[0]=%s, tan20=%.3f so it sticks)" % (x, mu, math.tan(th)))
x, mu = slide(0.1, tilt)
expected = 0.5 * 9.81 * (math.sin(th) - 0.1 * math.cos(th)) * 1.0 ** 2
print("tilted 20 deg,  mu=0.1   -> x after 1 s: %.5f  (contact.friction[0]=%s, Coulomb estimate %.3f)" % (x, mu, expected))

# Part B: the Go2 scene. Foot geoms have priority=1, so their friction wins over the floor's.
scene = sys.argv[1]
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
print("go2 floor: friction", m.geom_friction[floor], "priority", m.geom_priority[floor], "condim", m.geom_condim[floor])
print("go2 FL foot: friction", m.geom_friction[fl], "priority", m.geom_priority[fl], "condim", m.geom_condim[fl])

def foot_contact_friction():
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(100):
        mujoco.mj_step(m, d)
    for c in d.contact[:d.ncon]:
        if {c.geom1, c.geom2} == {floor, fl}:
            return c.friction.copy(), c.dim
    return None

print("FL-floor contact friction (as loaded):      ", foot_contact_friction())
m.geom_friction[floor] = [0.2, 0.005, 0.0001]
print("after floor friction -> 0.2 (foot priority): ", foot_contact_friction())
m.geom_priority[floor] = 2
print("after floor priority -> 2:                   ", foot_contact_friction())
m.geom_priority[floor] = 0
m.geom_friction[m.geom_priority == 1, 0] = 0.3  # all four feet
print("floor priority back 0, foot friction -> 0.3: ", foot_contact_friction())
