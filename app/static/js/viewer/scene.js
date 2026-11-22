// /api/robot -> what the viewer draws. Geom pos and quat are in their body's frame; bodies are placed by pose events.
import { decodeFloat32, decodeUint32, primitive, vertexNormals } from "./geometry.js";
import { fromQuatPos } from "./mat.js";

const IDENTITY = fromQuatPos([1, 0, 0, 0], [0, 0, 0]);

// {drawables: [{geom, body, meshKey, local, rgba}], meshes: Map(key -> geometry), floorZ, rootBody, bodyCount}.
// Which geoms: with a group field, groups 0 to 2 (MuJoCo's visual groups; collision geoms are group 3); without it,
// only the meshes of a body that has meshes (its primitives are collision shapes), and every other body's primitives.
// The plane is the floor, drawn by the viewer: only its height is kept. Throws Error with the reason if unreadable.
export function buildScene(robot) {
  const bodies = Array.isArray(robot?.bodies) ? robot.bodies : [];
  const geoms = Array.isArray(robot?.geoms) ? robot.geoms : [];
  if (!bodies.length) throw new Error("The robot description has no bodies.");
  const meshById = new Map((robot.meshes ?? []).map((m) => [m.id, m]));
  const grouped = geoms.some((g) => typeof g.group === "number");
  const meshBodies = new Set(geoms.filter((g) => g.type === "mesh").map((g) => g.body));
  const meshes = new Map();
  const drawables = [];
  let floorZ = 0;
  for (const g of geoms) {
    if (g.type === "plane") { if (g.body === 0) floorZ = g.pos?.[2] ?? 0; continue; }
    if (grouped ? !(g.group <= 2) : meshBodies.has(g.body) && g.type !== "mesh") continue;
    let key;
    if (g.type === "mesh") {
      const m = meshById.get(g.mesh);
      if (!m) throw new Error(`Geom ${g.id} names mesh ${g.mesh}, which the description does not hold.`);
      key = `m${m.id}`;
      if (!meshes.has(key)) {
        const positions = decodeFloat32(m.vertices), indices = decodeUint32(m.faces);
        meshes.set(key, { positions, indices, normals: vertexNormals(positions, indices) });
      }
    } else {
      key = `${g.type}:${(g.size ?? []).join(",")}`;
      if (!meshes.has(key)) {
        const p = primitive(g.type, g.size ?? []);
        if (!p) continue; // a type the viewer cannot draw
        meshes.set(key, p);
      }
    }
    drawables.push({ geom: g.id, body: g.body, meshKey: key, local: fromQuatPos(g.quat ?? [1, 0, 0, 0], g.pos ?? [0, 0, 0]), rgba: g.rgba ?? [0.6, 0.6, 0.6, 1] });
  }
  const drawn = new Set(drawables.map((d) => d.body));
  const root = bodies.find((b) => b.id !== 0 && b.parent === 0 && drawn.has(b.id));
  return { drawables, meshes, floorZ, rootBody: root ? root.id : null, bodyCount: bodies.length };
}

// The contract's "scene" event -> drawables in the world frame: [{id, type, size, model, rgba, transparent}].
// Object sizes follow MuJoCo geom sizes in metres, as /api/robot sends them: sphere [radius], box [half x, half y,
// half z], cylinder [radius, half length along its z]. Markers carry no size; they are drawn at fixed display sizes:
// a disc 0.10 m in radius and 4 mm thick lying on its point, a line as a rod 8 mm in radius from pos to `to`. An entry
// the viewer cannot read is left out, never guessed.
const MARKER = { discRadius: 0.1, discHalf: 0.002, lineRadius: 0.008 };
const NEED = { sphere: 1, box: 3, cylinder: 2 };
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const vec = (v, n) => Array.isArray(v) && v.length >= n && v.slice(0, n).every(finite);
const colour = (c) => (vec(c, 4) ? c.slice(0, 4) : vec(c, 3) ? [...c.slice(0, 3), 1] : [0.85, 0.85, 0.85, 1]);

export function sceneItems(scene) {
  const out = [];
  for (const o of Array.isArray(scene?.objects) ? scene.objects : []) {
    const n = NEED[o?.shape];
    if (!n || !vec(o.size, n) || !vec(o.pos, 3) || o.size.slice(0, n).some((s) => s <= 0)) continue;
    const rgba = colour(o.rgba);
    out.push({ id: String(o.id), type: o.shape, size: o.size.slice(0, n), model: fromQuatPos(vec(o.quat, 4) ? o.quat : [1, 0, 0, 0], o.pos), rgba, transparent: rgba[3] < 1 });
  }
  for (const m of Array.isArray(scene?.markers) ? scene.markers : []) {
    if (!vec(m?.pos, 3)) continue;
    const rgba = colour(m.rgba);
    if (m.shape === "disc") {
      out.push({ id: String(m.id), type: "cylinder", size: [MARKER.discRadius, MARKER.discHalf], model: fromQuatPos([1, 0, 0, 0], [m.pos[0], m.pos[1], m.pos[2] + MARKER.discHalf]), rgba, transparent: rgba[3] < 1 });
    } else if (m.shape === "line" && vec(m.to, 3)) {
      const d = [0, 1, 2].map((k) => m.to[k] - m.pos[k]), len = Math.hypot(...d);
      if (len < 1e-6) continue;
      const [bx, by, bz] = d.map((v) => v / len);
      const q = bz < -0.999999 ? [0, 1, 0, 0] : [1 + bz, -by, bx, 0]; // turns +z onto the line's direction
      out.push({ id: String(m.id), type: "cylinder", size: [MARKER.lineRadius, len / 2], model: fromQuatPos(q, [0, 1, 2].map((k) => (m.pos[k] + m.to[k]) / 2)), rgba, transparent: rgba[3] < 1 });
    }
  }
  return out;
}

// The world matrix of body `id` in a pose {pos: [[x, y, z]...], quat: [[w, x, y, z]...]}; a body the pose does not
// hold (the world) stays at the origin.
export function bodyMatrix(pose, id) {
  const p = pose?.pos?.[id], q = pose?.quat?.[id];
  return p && q ? fromQuatPos(q, p) : IDENTITY;
}
