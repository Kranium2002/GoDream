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
  const o = createOrbit({ target: [1, 1, 0], distance: 2, yaw: 0, pitch: 0.5, minDistance: 0.5, maxDistance: 5 });
  vecNear(o.eye(), [1 + 2 * Math.cos(0.5), 1, 2 * Math.sin(0.5)]);
  o.rotate(Math.PI / 2, 0);
  vecNear(o.eye(), [1, 1 + 2 * Math.cos(0.5), 2 * Math.sin(0.5)]);
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
  assert.deepEqual(sc.drawables[2].trainedRgba, [1, 0.3, 0.1, 1]); // no trained_rgba: the geom's own colour
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

// Synthesised pose streams (test input only): the dog at x = t_s, one pose every 40 ms of simulated time.
const stream = (b, arrivals) => arrivals.forEach(([t, at]) => b.push({ t_s: t, pos: [[0, 0, 0], [t, 0, 0]], quat: [[1, 0, 0, 0], [1, 0, 0, 0]] }, at));

test("playback runs on the simulator's clock, between two poses, and never past the newest", () => {
  const b = createPoseBuffer();
  stream(b, [[0, 1000], [0.04, 1040], [0.08, 1080], [0.12, 1120]]); // real time: 40 ms of simulation per 40 ms
  const first = b.sample(1120).pos[1][0];
  assert.ok(first >= 0 && first < 0.12, `starts a delay behind the newest pose, at ${first}`);
  let last = first;
  for (let now = 1136; now <= 1600; now += 16) {
    const x = b.sample(now).pos[1][0];
    assert.ok(x >= last - 1e-9 && x <= 0.12 + 1e-9, `at ${now}: ${x} after ${last}`);
    last = x;
  }
  assert.ok(Math.abs(last - 0.12) < 1e-6, `settles on the newest pose, at ${last}`);
  assert.equal(b.moving(), false);
});

test("bursty arrivals from a slow simulator still move the dog on almost every frame", () => {
  // The laptop's measured pattern: the simulator at about 0.4 of real time, poses (40 ms of simulated time apart)
  // arriving in pairs about 200 ms apart.
  const b = createPoseBuffer(), arrivals = [];
  for (let k = 0; k < 60; k++) arrivals.push([k * 0.04, 1000 + Math.floor(k / 2) * 200 + (k % 2) * 5]);
  const frames = [];
  let next = 0;
  for (let now = 1000; now <= 7000; now += 16) {
    while (next < arrivals.length && arrivals[next][1] <= now) { stream(b, [arrivals[next]]); next++; }
    const s = b.sample(now);
    if (s && now > 2500 && now < 6500) frames.push(s.pos[1][0]);
  }
  let still = 0, longest = 0, run = 0;
  for (let k = 1; k < frames.length; k++) {
    const moved = frames[k] > frames[k - 1] + 1e-9;
    assert.ok(frames[k] >= frames[k - 1] - 1e-9, "never goes backwards");
    if (!moved) { still++; run++; longest = Math.max(longest, run); } else run = 0;
  }
  assert.ok(still / frames.length < 0.05, `${still} of ${frames.length} frames held still`);
  assert.ok(longest <= 2, `longest hold ${longest} frames`);
});

test("the pose buffer interpolates the short way round and restarts on a body count change or a clock reset", () => {
  const b = createPoseBuffer();
  b.push({ t_s: 0, pos: [[0, 0, 0]], quat: [[1, 0, 0, 0]] }, 0);
  b.push({ t_s: 0.04, pos: [[0, 0, 0]], quat: [[-1, 0, 0, 0]] }, 40); // the same rotation, opposite sign
  for (let now = 40; now < 200; now += 16) vecNear(b.sample(now).quat[0].map(Math.abs), [1, 0, 0, 0]);
  b.push({ t_s: 0.08, pos: [[0, 0, 0], [1, 1, 1]], quat: [[1, 0, 0, 0], [1, 0, 0, 0]] }, 80);
  vecNear(b.sample(80).pos[1], [1, 1, 1]);
  b.push({ t_s: 0.12, pos: [[0, 0, 0], [2, 2, 2]], quat: [[1, 0, 0, 0], [1, 0, 0, 0]] }, 120);
  b.push({ t_s: 0, pos: [[0, 0, 0], [5, 5, 5]], quat: [[1, 0, 0, 0], [1, 0, 0, 0]] }, 160); // a new episode
  vecNear(b.sample(160).pos[1], [5, 5, 5]);
  b.clear();
  assert.equal(b.sample(200), null);
});

test("the pose buffer says when it is still moving, and when it has settled", () => {
  const b = createPoseBuffer();
  assert.equal(b.moving(), false);
  b.push(P(0), 1000);
  assert.equal(b.moving(), false); // one pose: nothing to interpolate
  b.push(P(0.04), 1040);
  assert.equal(b.moving(), true);
  for (let now = 1040; now < 2000; now += 16) b.sample(now);
  assert.equal(b.moving(), false);
});

// ---- the contract's "scene" event: synthesised test input only, never shown as a result -----------------------
test("scene objects and markers become world-frame drawables; unreadable entries are left out", async () => {
  const { sceneItems } = await import("../../app/static/js/viewer/scene.js");
  const items = sceneItems({
    objects: [
      { id: "ball", shape: "sphere", size: [0.1], rgba: [1, 0.4, 0.1, 1], pos: [1, 2, 0.3], quat: [1, 0, 0, 0] },
      { id: "ramp", shape: "box", size: [1, 0.5, 0.05], rgba: [0.5, 0.5, 0.5, 0.5], pos: [2, 0, 0.1], quat: [s, 0, s, 0] },
      { id: "post", shape: "cylinder", size: [0.05, 0.4], rgba: [0, 1, 0, 1], pos: [0, 0, 0.4] },
      { id: "bad", shape: "cone", size: [1], rgba: [1, 1, 1, 1], pos: [0, 0, 0] },
      { id: "bad2", shape: "box", size: [1, 1], rgba: [1, 1, 1, 1], pos: [0, 0, 0] },
    ],
    markers: [
      { id: "target", shape: "disc", pos: [3, 0, 0], to: null, rgba: [0, 0.5, 1, 0.6] },
      { id: "path", shape: "line", pos: [0, 0, 0], to: [0, 2, 0], rgba: [1, 1, 0, 1] },
      { id: "zero", shape: "line", pos: [1, 1, 1], to: [1, 1, 1], rgba: [1, 1, 0, 1] },
    ],
  });
  assert.deepEqual(items.map((i) => [i.id, i.type, i.transparent]), [["ball", "sphere", false], ["ramp", "box", true], ["post", "cylinder", false], ["target", "cylinder", true], ["path", "cylinder", false]]);
  vecNear(transformPoint(items[0].model, [0, 0, 0]), [1, 2, 0.3]);
  vecNear(transformPoint(items[1].model, [0, 0, 1]), [3, 0, 0.1], 1e-6); // the quat turns +z onto +x
  vecNear(transformPoint(items[3].model, [0, 0, 0]), [3, 0, 0.002]); // the disc lies on its point
  vecNear(transformPoint(items[4].model, [0, 0, 0]), [0, 1, 0]); // the rod's centre
  vecNear(transformPoint(items[4].model, [0, 0, 1]), [0, 2, 0], 1e-6); // its half length along +z reaches `to`
  assert.deepEqual(items[4].size, [0.008, 1]);
  assert.deepEqual(sceneItems(null), []);
});

test("scene objects ride on the poses: interpolated by id, dropped with the scene that drops them", async () => {
  const { blendScene } = await import("../../app/static/js/viewer/pose.js");
  const ball = (x) => ({ objects: [{ id: "ball", shape: "sphere", size: [0.1], rgba: [1, 0, 0, 1], pos: [x, 0, 0.5], quat: [1, 0, 0, 0] }], markers: [{ id: "m", shape: "disc", pos: [x, 0, 0], rgba: [1, 1, 1, 1] }] });
  vecNear(blendScene(ball(0), ball(1), 0.25).objects[0].pos, [0.25, 0, 0.5]);
  vecNear(blendScene(ball(0), ball(1), 0.75).objects[0].pos, [0.75, 0, 0.5]);
  assert.deepEqual(blendScene(ball(0), { objects: [], markers: [] }, 0.6).objects, []); // the player stopped
  assert.equal(blendScene(ball(0), { objects: [], markers: [] }, 0.4).objects.length, 1);
  const b = createPoseBuffer();
  stream(b, [[0, 0]]);
  b.pushScene(ball(0));
  stream(b, [[0.04, 40]]);
  b.pushScene(ball(1));
  stream(b, [[0.08, 80]]);
  b.pushScene({ objects: [], markers: [] });
  const seen = [];
  for (let now = 80; now < 1000; now += 16) seen.push(b.sample(now).scene.objects.length);
  assert.equal(seen.at(-1), 0); // the empty scene the server sends when the player stops removes the ball
  assert.ok(seen.includes(1));
});

test("the quality controller lowers the resolution when frames are slow and raises it when they are fast", async () => {
  const { createQuality } = await import("../../app/static/js/viewer/quality.js");
  const q = createQuality({ min: 0.5, max: 1, slowMs: 40, fastMs: 22, settleMs: 500 });
  assert.equal(q.scale, 1);
  let t = 0;
  for (let k = 0; k < 20; k++) q.frame((t += 66)); // 15 frames per second for over a second
  assert.ok(q.scale < 1 && q.scale >= 0.5, `scale ${q.scale}`);
  const low = q.scale;
  for (let k = 0; k < 200; k++) q.frame((t += 66));
  assert.equal(q.scale, 0.5); // never below the floor
  for (let k = 0; k < 300; k++) q.frame((t += 12)); // fast again
  assert.ok(q.scale > low);
  for (let k = 0; k < 2000; k++) q.frame((t += 12));
  assert.equal(q.scale, 1); // never above the ceiling
});

test("the quality controller ignores the gap when drawing pauses (nothing changed, tab hidden)", async () => {
  const { createQuality } = await import("../../app/static/js/viewer/quality.js");
  const q = createQuality({ min: 0.5, max: 1, slowMs: 40, fastMs: 22, settleMs: 500 });
  let t = 0;
  for (let k = 0; k < 50; k++) q.frame((t += 16));
  q.frame((t += 5000)); // a pause, not a slow frame
  for (let k = 0; k < 3; k++) q.frame((t += 16));
  assert.equal(q.scale, 1);
});

test("headingOf reads the yaw of a quaternion about z", async () => {
  const { headingOf } = await import("../../app/static/js/viewer/orbit.js");
  assert.ok(near(headingOf([1, 0, 0, 0]), 0));
  assert.ok(near(headingOf([Math.cos(Math.PI / 4), 0, 0, Math.sin(Math.PI / 4)]), Math.PI / 2));
  assert.ok(near(Math.abs(headingOf([0, 0, 0, 1])), Math.PI));
});

test("smoothing moves part of the way per frame, the short way round for angles, and settles", async () => {
  const { smooth, smoothAngle } = await import("../../app/static/js/viewer/orbit.js");
  const half = smooth(0, 10, 0.4 * Math.LN2 * 1000, 400); // after one half-life: halfway
  assert.ok(near(half, 5, 1e-6));
  assert.equal(smooth(3, 10, 0, 400), 3);
  assert.ok(near(smooth(0, 10, 1e6, 400), 10));
  const a = smoothAngle(3.0, -3.0, 0.4 * Math.LN2 * 1000, 400); // across the +-pi seam, not through zero
  assert.ok(Math.abs(a) > 3.0, `went the long way: ${a}`);
});

test("the orbit never puts the camera below the target's height", () => {
  const o = createOrbit({ target: [0, 0, 0.22], distance: 2, yaw: 0, pitch: -1 });
  assert.ok(o.pitch > 0);
  assert.ok(o.eye()[2] > 0.22);
});

test("the trained dog's lettering is one quad on each side of the trunk, read left to right from outside", async () => {
  const { labelQuads } = await import("../../app/static/js/viewer/scene.js");
  const q = labelQuads({ text: "tiramisu", centre_m: [0.1, 0.099, 0.034], size_m: [0.2, 0.05] });
  assert.equal(q.length, 2 * 6 * 5); // two sides, two triangles each, (x, y, z, u, v) per vertex
  const verts = Array.from({ length: 12 }, (_, k) => Array.from(q.slice(5 * k, 5 * k + 5)));
  const near = (a, b) => Math.abs(a - b) < 1e-6;
  for (const [side, firstLetterX] of [[1, 0.2], [-1, 0.0]]) { // left side: the text starts at the head; right side: at the tail
    const mine = verts.filter((v) => Math.sign(v[1]) === side);
    assert.equal(mine.length, 6);
    assert.ok(mine.every((v) => near(Math.abs(v[1]), 0.099)));
    for (const [x, , z, u, v] of mine) {
      assert.ok(near(x, u === 0 ? firstLetterX : 0.2 - firstLetterX)); // u = 0 is the first letter's edge
      assert.ok(near(z, v === 1 ? 0.059 : 0.009)); // v = 1 is the top of the letters
    }
  }
  assert.equal(labelQuads(null), null);
  assert.equal(labelQuads({ text: "", centre_m: [0, 0.1, 0], size_m: [0.2, 0.05] }), null);
  assert.equal(labelQuads({ text: "x", centre_m: [0, 0.1], size_m: [0.2, 0.05] }), null);
});
