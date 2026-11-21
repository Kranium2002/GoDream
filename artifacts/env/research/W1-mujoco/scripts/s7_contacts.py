# Item 7: detect contacts between two named geoms from data.contact.
import sys
import mujoco, numpy as np
print("mujoco", mujoco.__version__)
m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
floor = gid("floor")
feet = {n: gid(n) for n in ("FL", "FR", "RL", "RR")}
print("geom ids: floor", floor, "feet", feet, "unknown name ->", gid("nope"))

def touching(d, g1, g2):
    """Return (in_contact, total normal force) for the geom pair, either order."""
    if d.ncon == 0:
        return False, 0.0
    pair = d.contact.geom[:d.ncon]            # (ncon, 2) int array of geom ids
    hits = np.flatnonzero(((pair[:, 0] == g1) & (pair[:, 1] == g2)) | ((pair[:, 0] == g2) & (pair[:, 1] == g1)))
    f6 = np.zeros(6); fn = 0.0
    for i in hits:
        mujoco.mj_contactForce(m, d, int(i), f6)   # contact frame; f6[0] is the normal force
        fn += f6[0]
    return hits.size > 0, fn

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
print("t=0 (home keyframe): ncon", d.ncon, {n: touching(d, g, floor) for n, g in feet.items()})
for _ in range(25):
    mujoco.mj_step(m, d)
print("t=%.3f ncon %d" % (d.time, d.ncon), {n: (t, round(f, 2)) for n, (t, f) in ((n, touching(d, g, floor)) for n, g in feet.items())})
c = d.contact[0]
print("contact[0] fields: geom1", c.geom1, "geom2", c.geom2, "geom", c.geom, "dist", round(float(c.dist), 6), "pos", c.pos.round(4), "dim", c.dim, "exclude", c.exclude, "efc_address", c.efc_address)
print("names of contact[0]:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), "/", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
print("all contacts by body:", [(m.body(m.geom_bodyid[a]).name, m.body(m.geom_bodyid[b]).name) for a, b in d.contact.geom[:d.ncon]])
for _ in range(1500):
    mujoco.mj_step(m, d)
print("t=%.3f after collapse (zero ctrl): ncon %d, base-body geoms touching floor: %s" % (
    d.time, d.ncon, sorted({int(g) for a, b in d.contact.geom[:d.ncon] for g in (a, b) if m.geom_bodyid[g] == m.body("base").id})))
