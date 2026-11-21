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

// The world matrix of body `id` in a pose {pos: [[x, y, z]...], quat: [[w, x, y, z]...]}; a body the pose does not
// hold (the world) stays at the origin.
export function bodyMatrix(pose, id) {
  const p = pose?.pos?.[id], q = pose?.quat?.[id];
  return p && q ? fromQuatPos(q, p) : IDENTITY;
}
