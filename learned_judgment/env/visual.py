"""The `visual` situation key: how one episode looks (config.yaml world.visual).

Fields, each optional (absent: the scene's own value):
    object_rgb       [r, g, b], each 0-1: the thrown object's colour.
    object_radius_m  the thrown object's radius, within world.visual.object_radius_range_m. Physical: the sphere is
                     resized with its bounding data and inertia and MuJoCo's constants recomputed, which gives the
                     same episode as a sphere compiled at that radius (proof: scratchpad poc-w3/resize.py).
    floor_grey       0-1: the floor's grey level.
    backdrop_grey    0-1: the backdrop's grey level (the skybox and the horizon haze).
    light_dir        [x, y, z], finite, z < 0 (from above): the light's direction, normalised here.
    light_strength   the light's diffuse level, within world.visual.light_strength_range.
"""
import math
import numbers

import mujoco
import numpy as np

FIELDS = ("object_rgb", "object_radius_m", "floor_grey", "backdrop_grey", "light_dir", "light_strength")


def _unit(field: str, value) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)) or not math.isfinite(value):
        raise ValueError(f"visual.{field} must be a finite number, got {value!r}")
    return float(value)


def _within(field: str, value, lo: float, hi: float) -> float:
    v = _unit(field, value)
    if not lo <= v <= hi:
        raise ValueError(f"visual.{field} must be from {lo} to {hi}, got {v}")
    return v


def _vector(field: str, value) -> list[float]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError(f"visual.{field} must be three numbers, got {value!r}")
    return [_unit(field, v) for v in value]


def check_visual(value, config: dict) -> dict:
    """The validated settings (floats and lists of floats); ValueError naming the field otherwise."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"visual must be a dict of {list(FIELDS)}, got {value!r}")
    unknown = sorted(set(value) - set(FIELDS), key=repr)
    if unknown:
        raise ValueError(f"unknown visual field(s) {unknown}; visual accepts {list(FIELDS)}")
    v, out = config["world"]["visual"], {}
    if "object_rgb" in value:
        out["object_rgb"] = [_within("object_rgb", c, 0.0, 1.0) for c in _vector("object_rgb", value["object_rgb"])]
    if "object_radius_m" in value:
        out["object_radius_m"] = _within("object_radius_m", value["object_radius_m"], *v["object_radius_range_m"])
    for field in ("floor_grey", "backdrop_grey"):
        if field in value:
            out[field] = _within(field, value[field], 0.0, 1.0)
    if "light_dir" in value:
        d = _vector("light_dir", value["light_dir"])
        if not d[2] < 0.0:
            raise ValueError(f"visual.light_dir must point down (z < 0), got {d}")
        out["light_dir"] = d
    if "light_strength" in value:
        out["light_strength"] = _within("light_strength", value["light_strength"], *v["light_strength_range"])
    return out


def defaults(model: mujoco.MjModel) -> dict:
    """What check_visual's fields are in a freshly built scene, to restore before each episode's settings."""
    out = {"floor_rgba": model.geom_rgba[model.geom("floor").id].copy(), "light_dir": model.light_dir[0].copy(),
           "light_diffuse": model.light_diffuse[0].copy(), "haze": model.vis.rgba.haze.copy(),
           "backdrop": _backdrop(model).copy()}
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball") >= 0:
        g, b = model.geom("ball").id, model.body("ball").id
        k = model.body_inertia[b, 0] / (model.body_mass[b] * model.geom_size[g, 0] ** 2)  # 2/3 shell, 2/5 solid
        out |= {"object_rgba": model.geom_rgba[g].copy(), "object_radius_m": float(model.geom_size[g, 0]),
                "inertia": mujoco.mjtGeomInertia.mjINERTIA_SHELL if abs(k - 2 / 3) < 1e-6 else
                mujoco.mjtGeomInertia.mjINERTIA_VOLUME}
    return out


def _sphere_inertia(radius: float, mass: float, kind) -> np.ndarray:
    """MuJoCo's own inertia of such a sphere (a one-body compile, about a millisecond), so a resized object matches a
    compiled one to the last bit; k m r^2 computed here differed by 4e-19."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body()
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], mass=mass, typeinertia=kind)
    return spec.compile().body_inertia[1].copy()


def _backdrop(model: mujoco.MjModel) -> np.ndarray:
    t = model.texture("backdrop").id
    adr, size = model.tex_adr[t], model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
    return model.tex_data[adr:adr + size]


def apply(model: mujoco.MjModel, scene_defaults: dict, settings: dict, znear_m: float) -> bool:
    """Set the scene to its defaults overlaid with settings. Returns True if the backdrop texture changed (renderers
    must then be rebuilt to upload it)."""
    d, s = scene_defaults, settings
    model.geom_rgba[model.geom("floor").id] = [s["floor_grey"]] * 3 + [1.0] if "floor_grey" in s else d["floor_rgba"]
    model.light_dir[0] = np.array(s["light_dir"]) / np.linalg.norm(s["light_dir"]) if "light_dir" in s \
        else d["light_dir"]
    model.light_diffuse[0] = [s["light_strength"]] * 3 if "light_strength" in s else d["light_diffuse"]
    grey = round(s["backdrop_grey"] * 255) if "backdrop_grey" in s else None
    backdrop = _backdrop(model)
    old = backdrop.copy()
    backdrop[:] = d["backdrop"] if grey is None else grey
    model.vis.rgba.haze = d["haze"] if grey is None else [grey / 255] * 3 + [1.0]
    if "object_radius_m" in d:
        g = model.geom("ball").id
        model.geom_rgba[g] = s["object_rgb"] + [1.0] if "object_rgb" in s else d["object_rgba"]
        r = s.get("object_radius_m", d["object_radius_m"])
        if model.geom_size[g, 0] != r:
            b = model.body("ball").id
            assert model.body_bvhnum[b] == 1  # one sphere: its body's bounding-volume tree is one box
            model.geom_size[g, 0] = model.geom_rbound[g] = r
            model.geom_aabb[g] = model.bvh_aabb[model.body_bvhadr[b]] = [0.0, 0.0, 0.0, r, r, r]
            model.body_inertia[b] = _sphere_inertia(r, float(model.body_mass[b]), d["inertia"])
            mujoco.mj_setConst(model, mujoco.MjData(model))  # the solver's constants (invweight0) follow the size
            model.vis.map.znear = znear_m / model.stat.extent  # in case the extent moved with them
    return not np.array_equal(old, backdrop)
