"""Read-only inspection of the vendored unitree_mujoco Go2 model (UNVERIFIED LOOK-AHEAD).

Loads go2.xml and scene.xml with mujoco.MjModel.from_xml_path and lists
opt.timestep, bodies, joints, geoms (with type/contype/conaffinity/group/size),
actuators (order, joint, ctrlrange), sensors (order, dim, adr), keyframes, and
whether any free joint other than the robot base exists.  No mj_step is called.

Usage: python inspect_unitree_mujoco_go2.py <path-to-unitree_robots/go2>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

GEOM_TYPES = {int(v): k for k, v in mujoco.mjtGeom.__members__.items()}
JNT_TYPES = {int(v): k for k, v in mujoco.mjtJoint.__members__.items()}


def name(m, objtype, i):
    n = mujoco.mj_id2name(m, objtype, i)
    return n if n is not None else ""


def dump(path: Path) -> None:
    m = mujoco.MjModel.from_xml_path(str(path))
    print(f"=== {path.name} ===")
    print(f"mujoco.__version__ = {mujoco.__version__}")
    print(f"opt.timestep = {m.opt.timestep!r}")
    print(f"opt.integrator = {int(m.opt.integrator)} ({mujoco.mjtIntegrator(m.opt.integrator).name})")
    print(f"opt.cone = {int(m.opt.cone)} ({mujoco.mjtCone(m.opt.cone).name}), opt.impratio = {m.opt.impratio}")
    print(f"opt.gravity = {m.opt.gravity.tolist()}")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} nsensor={m.nsensor} nsensordata={m.nsensordata} nkey={m.nkey}")
    print("-- bodies (id, name, parentid, mass) --")
    for i in range(m.nbody):
        print(f"  body {i:2d} {name(m, mujoco.mjtObj.mjOBJ_BODY, i)!r:14} parent={m.body_parentid[i]} mass={m.body_mass[i]:.6g}")
    print("-- joints (id, name, type, qposadr, range) --")
    free_joints = []
    for i in range(m.njnt):
        jt = JNT_TYPES[int(m.jnt_type[i])]
        if jt == "mjJNT_FREE":
            free_joints.append((i, name(m, mujoco.mjtObj.mjOBJ_JOINT, i), name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[i])))
        print(f"  jnt {i:2d} {name(m, mujoco.mjtObj.mjOBJ_JOINT, i)!r:18} {jt:12} qposadr={m.jnt_qposadr[i]} range={m.jnt_range[i].tolist()} limited={bool(m.jnt_limited[i])}")
    print(f"-- free joints: {free_joints}")
    print("-- geoms (id, name, body, type, contype, conaffinity, group, size) [visual group-2 meshes omitted] --")
    n_vis = 0
    for i in range(m.ngeom):
        if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
            n_vis += 1
            continue
        print(
            f"  geom {i:2d} name={name(m, mujoco.mjtObj.mjOBJ_GEOM, i)!r:8} body={name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])!r:10} "
            f"type={GEOM_TYPES[int(m.geom_type[i])]:14} contype={m.geom_contype[i]} conaff={m.geom_conaffinity[i]} group={m.geom_group[i]} "
            f"size={np.round(m.geom_size[i], 5).tolist()} pos={np.round(m.geom_pos[i], 5).tolist()} friction={np.round(m.geom_friction[i], 4).tolist()} condim={m.geom_condim[i]}"
        )
    print(f"  (omitted {n_vis} non-colliding visual geoms)")
    print("-- actuators (index, name, joint, ctrlrange, ctrllimited, gear0) --")
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        print(
            f"  act {i:2d} {name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)!r:11} joint={name(m, mujoco.mjtObj.mjOBJ_JOINT, j)!r:16} "
            f"ctrlrange={m.actuator_ctrlrange[i].tolist()} ctrllimited={bool(m.actuator_ctrllimited[i])} gear0={m.actuator_gear[i, 0]}"
        )
    print("-- sensors (index, name, dim, adr) --")
    for i in range(m.nsensor):
        print(f"  sens {i:2d} {name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)!r:16} dim={m.sensor_dim[i]} adr={m.sensor_adr[i]}")
    print("-- sites --")
    for i in range(m.nsite):
        print(f"  site {i} {name(m, mujoco.mjtObj.mjOBJ_SITE, i)!r} body={name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[i])!r}")
    print("-- keyframes --")
    for i in range(m.nkey):
        print(f"  key {i} {name(m, mujoco.mjtObj.mjOBJ_KEY, i)!r} qpos={m.key_qpos[i].tolist()} ctrl={m.key_ctrl[i].tolist()}")
    print()


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for fname in ("go2.xml", "scene.xml"):
        dump(root / fname)
