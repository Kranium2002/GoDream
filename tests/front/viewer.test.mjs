import { test } from "node:test";
import assert from "node:assert/strict";
import { lookAt, multiply, normalMatrix, perspective, fromQuatPos, scaling, transformPoint } from "../../app/static/js/viewer/mat.js";
import { createOrbit } from "../../app/static/js/viewer/orbit.js";
import { decodeFloat32, decodeUint32, primitive, vertexNormals } from "../../app/static/js/viewer/geometry.js";
import { buildScene, bodyMatrix } from "../../app/static/js/viewer/scene.js";
import { createPoseBuffer } from "../../app/static/js/viewer/pose.js";

const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;
const vecNear = (a, b, eps = 1e-6) => a.every((v, i) => near(v, b[i], eps)) || assert.fail(`${[...a]} vs ${b}`);
const b64 = (typed) => Buffer.from(typed.buffer).toString("base64");

// ---- matrices (column-major Float32Array(16), as WebGL takes them) ---------------------------------------------
test("a quaternion and position make a rigid transform", () => {
  const s = Math.SQRT1_2;
  const m = fromQuatPos([s, 0, 0, s], [1, 2, 3]); // 90 degrees about z, then moved
  vecNear(transformPoint(m, [1, 0, 0]), [1, 3, 3]); // x axis turns into y
  vecNear(transformPoint(fromQuatPos([1, 0, 0, 0], [0, 0, 0]), [4, 5, 6]), [4, 5, 6]);
});

test("multiply applies the right matrix first, scaling scales", () => {
  const m = multiply(fromQuatPos([1, 0, 0, 0], [10, 0, 0]), scaling(2, 3, 4));
  vecNear(transformPoint(m, [1, 1, 1]), [12, 3, 4]);
});

test("normalMatrix is the inverse transpose of the upper 3 x 3", () => {
  const n = normalMatrix(scaling(2, 1, 1)); // a normal along x shrinks under a stretch along x
  vecNear([n[0], n[4], n[8]], [0.5, 1, 1]);
});

test("lookAt puts the target straight ahead on the negative z axis of view space", () => {
  const v = lookAt([0, -5, 0], [0, 0, 0], [0, 0, 1]);
  vecNear(transformPoint(v, [0, 0, 0]), [0, 0, -5]);
  vecNear(transformPoint(v, [0, 0, 1]), [0, 1, -5]); // world up is screen up
});

test("perspective maps the near and far planes to -1 and 1", () => {
  const p = perspective(Math.PI / 2, 1, 1, 10);
  const clip = (z) => { const x = p[10] * z + p[14], w = p[11] * z + p[15]; return x / w; };
  assert.ok(near(clip(-1), -1));
  assert.ok(near(clip(-10), 1));
});

// ---- orbit camera (z up) --------------------------------------------------------------------------------------
test("the orbit camera circles its target and clamps pitch and distance", () => {
  const o = createOrbit({ target: [1, 1, 0], distance: 2, yaw: 0, pitch: 0, minDistance: 0.5, maxDistance: 5 });
  vecNear(o.eye(), [3, 1, 0]);
  o.rotate(Math.PI / 2, 0);
  vecNear(o.eye(), [1, 3, 0]);
  o.rotate(0, 10); // far past straight down or up
  assert.ok(o.pitch < Math.PI / 2 && o.pitch > 0);
  o.zoom(100);
  assert.equal(o.distance, 5);
  o.zoom(0.0001);
  assert.equal(o.distance, 0.5);
  o.setTarget([0, 0, 0.3]);
  vecNear(o.target, [0, 0, 0.3]);
});

// ---- geometry -------------------------------------------------------------------------------------------------
test("base64 decodes to little-endian float32 and uint32 arrays", () => {
  vecNear(decodeFloat32(b64(Float32Array.of(1.5, -2, 0.25))), [1.5, -2, 0.25]);
  assert.deepEqual([...decodeUint32(b64(Uint32Array.of(0, 1, 70000)))], [0, 1, 70000]);
});

test("vertex normals of one triangle point out of its face, unit length", () => {
  const n = vertexNormals(Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0), Uint32Array.of(0, 1, 2));
  for (let v = 0; v < 3; v++) vecNear([n[3 * v], n[3 * v + 1], n[3 * v + 2]], [0, 0, 1]);
});

test("primitives follow MuJoCo sizes: box half extents, sphere radius, cylinder and capsule radius and half length", () => {
  const extent = (p) => {
    const max = [0, 1, 2].map((a) => Math.max(...p.positions.filter((_, i) => i % 3 === a)));
    const min = [0, 1, 2].map((a) => Math.min(...p.positions.filter((_, i) => i % 3 === a)));
    return max.map((v, a) => v - min[a]);
  };
  vecNear(extent(primitive("box", [0.1, 0.2, 0.3])), [0.2, 0.4, 0.6], 1e-6);
  vecNear(extent(primitive("sphere", [0.05])), [0.1, 0.1, 0.1], 1e-3);
  vecNear(extent(primitive("cylinder", [0.02, 0.1])), [0.04, 0.04, 0.2], 1e-3);
  vecNear(extent(primitive("capsule", [0.02, 0.1])), [0.04, 0.04, 0.24], 1e-3);
  vecNear(extent(primitive("ellipsoid", [0.1, 0.2, 0.3])), [0.2, 0.4, 0.6], 1e-3);
  assert.equal(primitive("plane", [1, 1, 0]), null);
  const box = primitive("box", [1, 1, 1]);
  assert.equal(box.normals.length, box.positions.length);
  assert.ok(Math.max(...box.indices) < box.positions.length / 3);
});

// ---- scene ----------------------------------------------------------------------------------------------------
const s = Math.SQRT1_2;
const mesh = { id: 0, vertices: b64(Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0)), faces: b64(Uint32Array.of(0, 1, 2)) };
const robot = {
  bodies: [{ id: 0, name: "world", parent: 0 }, { id: 1, name: "base_link", parent: 0 }, { id: 2, name: "FL_calf", parent: 1 }, { id: 3, name: "ball", parent: 0 }],
  geoms: [
    { id: 0, body: 0, type: "plane", size: [0, 0, 0.05], pos: [0, 0, 0.01], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], mesh: null },
    { id: 1, body: 1, type: "mesh", size: [1, 1, 1], pos: [0, 0, 0], quat: [1, 0, 0, 0], rgba: [0.2, 0.2, 0.2, 1], mesh: 0 },
    { id: 2, body: 1, type: "box", size: [0.1, 0.1, 0.1], pos: [0, 0, 0], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], mesh: null },
    { id: 3, body: 2, type: "sphere", size: [0.02], pos: [0, 0, -0.2], quat: [1, 0, 0, 0], rgba: [0.5, 0.5, 0.5, 1], mesh: null },
    { id: 4, body: 3, type: "sphere", size: [0.1], pos: [0, 0, 0], quat: [s, 0, 0, s], rgba: [1, 0.3, 0.1, 1], mesh: null },
  ],
  meshes: [mesh], up: "z", units: "m",
};

test("buildScene draws only the meshes of a body that has meshes, other bodies' primitives, and no plane", () => {
  const sc = buildScene(robot);
  assert.deepEqual(sc.drawables.map((d) => d.geom), [1, 3, 4]); // body 2 has no mesh, so its sphere stays
  assert.equal(sc.floorZ, 0.01);
  assert.equal(sc.rootBody, 1); // the first child of the world body that carries geometry
  assert.equal(sc.bodyCount, 4);
  assert.deepEqual(sc.drawables[2].rgba, [1, 0.3, 0.1, 1]);
  assert.ok(sc.meshes.get("m0").normals.length === 9);
});

test("buildScene uses the group field when the server sends it: groups 0 to 2 only", () => {
  const grouped = { ...robot, geoms: robot.geoms.map((g) => ({ ...g, group: g.id === 2 ? 3 : g.id === 1 ? 2 : 0 })) };
  assert.deepEqual(buildScene(grouped).drawables.map((d) => d.geom), [1, 3, 4]);
  const allVisual = { ...robot, geoms: robot.geoms.map((g) => ({ ...g, group: 2 })) };
  assert.deepEqual(buildScene(allVisual).drawables.map((d) => d.geom), [1, 2, 3, 4]);
});

test("buildScene refuses a robot it cannot read, with the reason", () => {
  assert.throws(() => buildScene({ bodies: [], geoms: [] }), /no bodies/);
  assert.throws(() => buildScene({ ...robot, geoms: [{ ...robot.geoms[1], mesh: 9 }] }), /mesh 9/);
});

test("a body is placed where the pose says; a missing body (the world) stays at the origin", () => {
  const pose = { pos: [[0, 0, 0], [1, 0, 0.3]], quat: [[1, 0, 0, 0], [s, 0, 0, s]] };
  vecNear(transformPoint(bodyMatrix(pose, 1), [1, 0, 0]), [1, 1, 0.3]);
  vecNear(transformPoint(bodyMatrix(pose, 3), [1, 2, 3]), [1, 2, 3]);
});

// ---- poses: interpolate between the last two, never beyond the newest -----------------------------------------
const P = (x, deg = 0) => {
  const h = (deg * Math.PI) / 360;
  return { t_s: x, pos: [[0, 0, 0], [x, 0, 0]], quat: [[1, 0, 0, 0], [Math.cos(h), 0, 0, Math.sin(h)]] };
};

test("the pose buffer is empty until the first pose, then shows it", () => {
  const b = createPoseBuffer();
  assert.equal(b.sample(0), null);
  b.push(P(1), 1000);
  vecNear(b.sample(1000).pos[1], [1, 0, 0]);
  vecNear(b.sample(5000).pos[1], [1, 0, 0]);
});

test("between two poses it moves from the older to the newer over one arrival interval, then holds", () => {
  const b = createPoseBuffer();
  b.push(P(0, 0), 1000);
  b.push(P(1, 90), 1040);
  vecNear(b.sample(1040).pos[1], [0, 0, 0]);
  vecNear(b.sample(1060).pos[1], [0.5, 0, 0]);
  const q = b.sample(1060).quat[1];
  vecNear([q[0], q[3]], [Math.cos(Math.PI / 8), Math.sin(Math.PI / 8)], 1e-6); // halfway, 45 degrees
  vecNear(b.sample(2000).pos[1], [1, 0, 0]); // never past the newest pose
});

test("the pose buffer interpolates the short way round and survives a body count change by restarting", () => {
  const b = createPoseBuffer();
  b.push({ pos: [[0, 0, 0]], quat: [[1, 0, 0, 0]] }, 0);
  b.push({ pos: [[0, 0, 0]], quat: [[-1, 0, 0, 0]] }, 40); // the same rotation, opposite sign
  vecNear(b.sample(20).quat[0].map(Math.abs), [1, 0, 0, 0]);
  b.push({ pos: [[0, 0, 0], [1, 1, 1]], quat: [[1, 0, 0, 0], [1, 0, 0, 0]] }, 80);
  vecNear(b.sample(80).pos[1], [1, 1, 1]);
  b.clear();
  assert.equal(b.sample(100), null);
});
