"""Simulator and library facts the hidden effects rely on, on the pinned stack (mujoco 3.11.0, Unitree go2.xml on the
flat scene of learned_judgment/env/scene.py). Entry M saw the foot geoms' contact priority on the menagerie model only.
1. Which friction a foot-floor contact uses: the foot geom's (priority 1) or the floor's.
2. MjSpec edits used by build_model: a child body on base_link, the foot geoms' friction, the gravity option.
3. mju_quatZ2Vec: the rotation it builds, for the orientation the policy senses on a tilted-gravity slope.
4. numpy Generator.spawn: a child stream leaves the parent's own draws unchanged.
Run from the repository root: uv run python -X utf8 artifacts/env/research/e5_effect_facts.py
"""
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from learned_judgment.env import load_config  # noqa: E402
from learned_judgment.env.scene import build_model  # noqa: E402

print("mujoco", mujoco.__version__, "numpy", np.__version__)
config = load_config()
FEET = ("FL", "FR", "RL", "RR")


def foot_contact_friction(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_step(model, d)  # the model's initial pose (legs straight) has its four feet on the floor
    feet = {model.geom(n).id for n in FEET}
    return sorted({tuple(np.round(d.contact.friction[i][:3], 4)) for i in range(d.ncon)
                   if feet & set(d.contact.geom[i])})


# 1. Friction of foot-floor contacts: as compiled, with the floor's friction changed, with the feet's changed.
spec = mujoco.MjSpec.from_file(config["robot"]["model_asset"]["path"])
print("foot geom priority, friction in go2.xml:",
      [(n, spec.geom(n).priority, [float(f) for f in spec.geom(n).friction]) for n in FEET])
base = build_model(config)
print("floor geom friction (MuJoCo default):", base.geom("floor").friction)
print("1a. foot-floor contact friction [slide, slide, torsion]:", foot_contact_friction(base))
floor = build_model(config)
floor.geom_friction[floor.geom("floor").id] = [0.05, 0.005, 0.0001]
print("1b. floor friction set to 0.05:", foot_contact_friction(floor))
feet = build_model(config)
for n in FEET:
    feet.geom_friction[feet.geom(n).id][0] = 0.15
print("1c. foot sliding friction set to 0.15:", foot_contact_friction(feet))

# 2. MjSpec edits.
spec = mujoco.MjSpec.from_file(config["robot"]["model_asset"]["path"])
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
trunk_box = [g for g in spec.body("base_link").geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX]
print("2a. base_link box geoms (pos, size):", [(list(g.pos), list(g.size)) for g in trunk_box])
before = spec.compile()
body = spec.body("base_link").add_body(name="payload", pos=[0.0, 0.0, 0.1])
body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.075, 0.045, 0.04], mass=3.0, contype=0, conaffinity=0)
spec.geom("FL").friction[0] = 0.2
print("2b. spec.option.gravity before:", list(spec.option.gravity))
spec.option.gravity = [-9.81 * np.sin(np.radians(5)), 0.0, -9.81 * np.cos(np.radians(5))]
after = spec.compile()
print("2c. total mass before / after adding a 3 kg payload body:", round(sum(before.body_mass), 4),
      round(sum(after.body_mass), 4), "payload body mass", after.body("payload").mass,
      "parent", after.body(int(after.body("payload").parentid[0])).name)
print("2d. FL friction after edit:", after.geom("FL").friction, "gravity:", np.round(after.opt.gravity, 4))

# 3. mju_quatZ2Vec(q, up) maps z to up; its inverse maps the floor frame's up (slope +5 deg: gravity tilted toward -x)
# back to z. A level trunk in the floor frame then reads as pitched nose up, as on a real uphill slope.
g = after.opt.gravity
q = np.empty(4)
mujoco.mju_quatZ2Vec(q, -g)
z = np.empty(3)
mujoco.mju_rotVecQuat(z, np.array([0.0, 0.0, 1.0]), q)
print("3a. quatZ2Vec(-g) =", np.round(q, 6), "rotates z to", np.round(z, 6),
      "; -g/|g| =", np.round(-g / np.linalg.norm(g), 6))
level = np.empty(4)
mujoco.mju_negQuat(level, q)
nose = np.empty(3)
mujoco.mju_rotVecQuat(nose, np.array([1.0, 0.0, 0.0]), level)
print("3b. level trunk's nose (floor-frame +x) in the gravity frame:", np.round(nose, 6),
      "pitch up deg:", round(float(np.degrees(np.arcsin(nose[2]))), 4))

# 4. Generator.spawn.
a, b = (np.random.default_rng(np.random.SeedSequence(5, spawn_key=(4, 2))) for _ in range(2))
child = a.spawn(1)[0]
print("4. child draw", child.integers(-1, 2),
      "| parent's next draw equals an untouched twin's:", a.random() == b.random())
