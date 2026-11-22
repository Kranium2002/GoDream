import { test } from "node:test";
import assert from "node:assert/strict";
import { createFrameWindow, createRateMeter, createSampler, createSender } from "../../app/static/js/live.js";

const f = (v) => Float32Array.of(v, v + 0.5);

test("the rolling window keeps the newest frames, oldest first", () => {
  const w = createFrameWindow(3, 2);
  w.push(f(1), 10);
  w.push(f(2), 20);
  assert.equal(w.full(), false);
  assert.equal(w.snapshot(), null);
  w.push(f(3), 30);
  w.push(f(4), 40);
  assert.equal(w.full(), true);
  const snap = w.snapshot();
  assert.deepEqual(Array.from(snap.window), [2, 2.5, 3, 3.5, 4, 4.5]);
  assert.ok(snap.window instanceof Float32Array);
  assert.equal(snap.capturedAt, 40); // the newest frame's capture time
  w.clear();
  assert.equal(w.full(), false);
});

test("the rolling window refuses a frame of the wrong size", () => {
  assert.throws(() => createFrameWindow(2, 2).push(Float32Array.of(1), 0), /2 values/);
});

function deferredSend() {
  const calls = [];
  const send = (item) => new Promise((resolve) => calls.push({ item, resolve }));
  return { calls, send };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

test("the sender sends at once when idle and never has two requests in flight", async () => {
  const { calls, send } = deferredSend();
  const answers = [];
  const s = createSender(send, (item, result) => answers.push([item, result]));
  s.offer("a");
  assert.deepEqual(calls.map((c) => c.item), ["a"]);
  s.offer("b");
  s.offer("c"); // b is replaced before it was sent
  assert.equal(calls.length, 1);
  assert.equal(s.dropped(), 1);
  calls[0].resolve("A");
  await tick();
  assert.deepEqual(answers, [["a", "A"]]);
  assert.deepEqual(calls.map((c) => c.item), ["a", "c"]); // the newest waiting window goes next
  calls[1].resolve("C");
  await tick();
  assert.deepEqual(answers, [["a", "A"], ["c", "C"]]);
  assert.equal(s.busy(), false);
});

test("reset drops the waiting window and ignores the answer in flight", async () => {
  const { calls, send } = deferredSend();
  const answers = [];
  const s = createSender(send, (item, result) => answers.push(item));
  s.offer("a");
  s.offer("b");
  s.reset();
  assert.equal(s.dropped(), 0);
  calls[0].resolve("A");
  await tick();
  assert.deepEqual(answers, []);
  assert.equal(calls.length, 1);
  s.offer("c");
  assert.deepEqual(calls.map((c) => c.item), ["a", "c"]);
});

test("the sampler takes every frame of a camera at its own rate", () => {
  const s = createSampler(30);
  const taken = [0, 33.4, 66.7, 100.1, 133.4].filter((t) => s.take(t));
  assert.equal(taken.length, 5);
});

test("the sampler takes every other frame of a 60 Hz camera at 30 Hz", () => {
  const s = createSampler(30);
  const times = Array.from({ length: 12 }, (_, k) => k * 16.67);
  assert.equal(times.filter((t) => s.take(t)).length, 6);
});

test("the sampler restarts its schedule after a long pause", () => {
  const s = createSampler(30);
  assert.equal(s.take(0), true);
  assert.equal(s.take(1000), true);
  assert.equal(s.take(1010), false);
  assert.equal(s.take(1034), true);
});

test("the sampler never takes the same camera frame twice", () => {
  const s = createSampler(30);
  assert.equal(s.take(0, 1), true);
  assert.equal(s.take(40, 1), false); // a callback for a frame already taken
  assert.equal(s.take(41, 2), true);
  assert.equal(s.take(80, 3), true);
});

test("recent statistics give the median and the 95th percentile of the last span", async () => {
  const { createRecent } = await import("../../app/static/js/live.js");
  const r = createRecent(1000);
  assert.equal(r.summary(0), null);
  [10, 20, 30, 40, 100].forEach((v, k) => r.add(k * 100, v));
  r.add(450, NaN); // not a number: ignored
  assert.deepEqual(r.summary(500), { n: 5, p50: 30, p95: 100 });
  assert.deepEqual(r.summary(1250), { n: 2, p50: 40, p95: 100 }); // older than the span: gone
});

test("a capture time on the monotonic clock becomes epoch ms anchored at now", async () => {
  const { toEpochMs } = await import("../../app/static/js/live.js");
  assert.equal(toEpochMs(1000, 1030, 1_700_000_000_030), 1_700_000_000_000);
});

test("the camera is slow only after it stayed below the model's rate for more than a second", async () => {
  const { createPace } = await import("../../app/static/js/live.js");
  const p = createPace(30);
  assert.deepEqual(p.update(0, 20), { slow: true, warn: false });
  assert.deepEqual(p.update(900, 20), { slow: true, warn: false });
  assert.deepEqual(p.update(1100, 20), { slow: true, warn: true });
  assert.deepEqual(p.update(1200, 29), { slow: false, warn: true }); // stays until it kept up for a second
  assert.deepEqual(p.update(2300, 30), { slow: false, warn: false });
  assert.deepEqual(p.update(2400, null), { slow: true, warn: false });
});

test("the event picture follows the server's event rule; other encodings show the signed change", async () => {
  const { changePicture } = await import("../../app/static/js/live.js");
  const older = Float32Array.of(0.5, 0.5, 0.5, 0.2);
  const newer = Float32Array.of(0.6, 0.4, 0.52, 0.2);
  assert.deepEqual(Array.from(changePicture(older, newer, "events", 0.04)), [1, 0, 0.5, 0.5]);
  assert.throws(() => changePicture(older, newer, "events", undefined), /threshold/);
  const signed = Array.from(changePicture(older, newer, "signed_diff", 0.04)).map((v) => Math.round(v * 100) / 100);
  assert.deepEqual(signed, [1, 0, 0.6, 0.5]);
  assert.throws(() => changePicture(older, Float32Array.of(1), "events", 0.04), /differ in size/);
});

test("the rate meter counts frames in the last second", () => {
  const m = createRateMeter();
  assert.equal(m.rate(0), null);
  for (let t = 0; t <= 1000; t += 40) m.add(t); // 26 frames, 25 intervals over 1000 ms
  assert.equal(m.rate(1000), 25);
  assert.equal(m.rate(5000), null); // nothing in the last second
});

test("camera errors read as plain messages", async () => {
  const { cameraProblem } = await import("../../app/static/js/live.js");
  assert.match(cameraProblem({ secure: false, hasApi: true }), /secure/);
  assert.match(cameraProblem({ secure: true, hasApi: false }), /does not offer camera access/);
  assert.equal(cameraProblem({ secure: true, hasApi: true }), null);
  assert.match(cameraProblem({ error: { name: "NotAllowedError" } }), /permission was denied/);
  assert.match(cameraProblem({ error: { name: "SecurityError" } }), /permission was denied/);
  assert.match(cameraProblem({ error: { name: "NotFoundError" } }), /No camera/);
  assert.match(cameraProblem({ error: { name: "OverconstrainedError" } }), /No camera/);
  assert.match(cameraProblem({ error: { name: "NotReadableError" } }), /another program/);
  assert.match(cameraProblem({ error: { name: "WeirdError", message: "boom" } }), /WeirdError: boom/);
});

test("the rolling window lists its frames oldest first, for the strip the page draws", () => {
  const w = createFrameWindow(3, 2);
  w.push(f(1), 10); w.push(f(2), 20); w.push(f(3), 30); w.push(f(4), 40);
  assert.deepEqual(w.list().map((v) => v[0]), [2, 3, 4]);
  w.clear();
  assert.deepEqual(w.list(), []);
});
