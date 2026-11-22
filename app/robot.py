"""What the browser's 3D view draws, from the compiled MuJoCo model of the stage world: GET /api/robot and SSE pose.

    GET /api/robot  {bodies: [{id, name, parent}], geoms: [{id, body, group, type, size, pos, quat, rgba, mesh}],
                     meshes: [{id, vertices, faces}], up: "z", units: "m", version}
    SSE pose        {t_s, version, pos: [[x, y, z] per body], quat: [[w, x, y, z] per body]}, world frame, body id
                    order, at app.pose_hz, newest only
    SSE robot       {reason: "world_rebuilt", version} when the stage's model changes: fetch /api/robot again

Drawn geoms: of the robot (the bodies under base_link's root) only its visual geoms, geom group 2, which go2.xml's
class "visual" gives (contype 0, conaffinity 0; its collision geoms are group 3); every geom of the other bodies (the
floor, a thrown object, a ramp). pos and quat are the geom's pose in its body's frame, size is MuJoCo's geom_size
(sphere [r], capsule and cylinder [r, half length], box and ellipsoid half sizes, plane [half x, half y, grid step],
0 meaning infinite), rgba the material's colour when the geom has one. A mesh is base64 little-endian float32
vertices (n x 3, in the mesh frame the geom pose places) and uint32 faces (m x 3).
"""
import base64

import mujoco
import numpy as np

from app.routes import Refusal, route

TYPES = {mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
         mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
         mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_BOX: "box",
         mujoco.mjtGeom.mjGEOM_MESH: "mesh"}
VISUAL_GROUP = 2  # go2.xml <default class="visual"><geom ... group="2"/>


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def describe(model, trained_rgba: dict | None = None) -> dict:
    """GET /api/robot's answer for a compiled MuJoCo model (without version). trained_rgba (config
    robot.trained_material_rgba, material name -> rgba) adds trained_rgba to each geom: the colour the trained dog's view
    draws it in, its own rgba when its material is not named."""
    trained_rgba = trained_rgba or {}
    robot_root = model.body_rootid[model.body("base_link").id]
    geoms = []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if model.body_rootid[body] == robot_root and model.geom_group[g] != VISUAL_GROUP:
            continue  # a robot collision geom
        kind = TYPES.get(mujoco.mjtGeom(model.geom_type[g]))
        if kind is None:
            continue  # heightfields and SDFs: none in this stage
        matid = model.geom_matid[g]
        rgba = (model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[g]).tolist()
        material = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid) if matid >= 0 else None
        geoms.append({"id": g, "body": body, "group": int(model.geom_group[g]), "type": kind,
                      "size": model.geom_size[g].tolist(),
                      "pos": model.geom_pos[g].tolist(), "quat": model.geom_quat[g].tolist(),
                      "rgba": rgba, "trained_rgba": [float(v) for v in trained_rgba.get(material, rgba)],
                      "mesh": int(model.geom_dataid[g]) if kind == "mesh" else None})
    meshes = []
    for i in sorted({g["mesh"] for g in geoms if g["mesh"] is not None}):
        v0, nv, f0, nf = model.mesh_vertadr[i], model.mesh_vertnum[i], model.mesh_faceadr[i], model.mesh_facenum[i]
        meshes.append({"id": i, "vertices": _b64(model.mesh_vert[v0:v0 + nv], "<f4"),
                       "faces": _b64(model.mesh_face[f0:f0 + nf], "<u4")})
    bodies = [{"id": b, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "",
               "parent": int(model.body_parentid[b])} for b in range(model.nbody)]
    return {"bodies": bodies, "geoms": geoms, "meshes": meshes, "up": "z", "units": "m"}


def pose(data, t_s: float) -> dict:
    """Every body's world pose now, in body id order."""
    return {"t_s": t_s, "pos": data.xpos.tolist(), "quat": data.xquat.tolist()}


@route("GET", "/api/robot")
def get_robot(h) -> None:
    text = h.server.sim.robot_json()
    if text is None:
        raise Refusal(503, "robot_unavailable", "The stage world is still being built; try again in a moment.")
    h.send_json_text(200, text)
