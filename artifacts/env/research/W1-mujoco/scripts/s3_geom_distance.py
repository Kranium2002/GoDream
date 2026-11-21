# Item 3: mj_geomDistance signature and a two-sphere example.
import mujoco, numpy as np

print("mujoco", mujoco.__version__)
print("DOC:", mujoco.mj_geomDistance.__doc__.strip())
XML = """
<mujoco>
  <worldbody>
    <geom name="a" type="sphere" size="0.1" pos="0 0 1"/>
    <body name="b" pos="0.5 0 1"><freejoint/><geom name="b" type="sphere" size="0.2"/></body>
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)  # geom_xpos must be current before calling
ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "a")
gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "b")
print("nativeccd enabled:", not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
fromto = np.zeros(6)
r = mujoco.mj_geomDistance(m, d, ga, gb, 10.0, fromto)
print("centres 0.5 apart, radii 0.1+0.2 -> expected 0.2; got", repr(r), type(r).__name__, "fromto", fromto)
r2 = mujoco.mj_geomDistance(m, d, ga, gb, 0.1, fromto)
print("distmax=0.1 (< true distance) ->", repr(r2), "fromto", fromto)
r3 = mujoco.mj_geomDistance(m, d, ga, gb, 10.0, None)
print("fromto=None ->", repr(r3))
d.qpos[0] = 0.25  # freejoint qpos is world x of body b: centres now 0.25 apart -> penetration 0.05
mujoco.mj_forward(m, d)
r4 = mujoco.mj_geomDistance(m, d, ga, gb, 10.0, fromto)
print("centres 0.25 apart -> expected -0.05 (signed, penetration); got", repr(r4), "fromto", fromto)
