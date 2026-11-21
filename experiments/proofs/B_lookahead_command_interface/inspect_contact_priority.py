"""Read-only: print contact priority/friction/solmix of floor and Go2 foot/trunk geoms in the vendored scene.xml.
No mj_step. UNVERIFIED LOOK-AHEAD.
Usage: python inspect_contact_priority.py <path-to-unitree_robots/go2/scene.xml>
"""
import sys

import mujoco

m = mujoco.MjModel.from_xml_path(sys.argv[1])
base_id = m.body("base_link").id
print(f"mujoco {mujoco.__version__}; opt.timestep={m.opt.timestep}")
for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
    if name in ("floor", "FL", "FR", "RL", "RR") or (m.geom_bodyid[i] == base_id and m.geom_contype[i]):
        print(
            f"geom {i:2d} name={name!r:8} body={body!r:11} priority={m.geom_priority[i]} "
            f"friction={[round(float(x), 4) for x in m.geom_friction[i]]} condim={m.geom_condim[i]} solmix={m.geom_solmix[i]}"
        )
print(f"subtree mass of base_link (whole robot) = {m.body_subtreemass[base_id]:.6f} kg")
