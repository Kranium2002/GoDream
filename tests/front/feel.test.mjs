import { test } from "node:test";
import assert from "node:assert/strict";
import { contactRuns, createTrace, readEye, readFeel, sharedScale, tracePath } from "../../app/static/js/feel.js";
import { changePicture } from "../../app/static/js/live.js";

const feel = (t, nose, feet = [1, 0, 0, 1], side = "before") => ({ side, t_s: t, tilt_deg: [nose, 0], turn_rate: [0, 0, 0.1], feet });

test("feel events are read as the contract writes them; anything else is refused", () => {
  assert.deepEqual(readFeel(feel(1, 4)), { side: "before", t: 1, tilt: [4, 0], turn: [0, 0, 0.1], feet: [true, false, false, true] });
  assert.equal(readFeel({ ...feel(1, 4), side: undefined }).side, null);
  assert.equal(readFeel({ ...feel(1, 4), feet: [1, 0, 1] }), null);
  assert.equal(readFeel({ ...feel(1, 4), tilt_deg: [1] }), null);
  assert.equal(readFeel({ t_s: "x" }), null);
});

test("eye events need cells in 0..1 of the declared count", () => {
  const cells = Array.from({ length: 144 }, (_, k) => (k % 2 ? 1 : 0));
  assert.equal(readEye({ side: "after", cells }, 144).cells.length, 144);
  assert.equal(readEye({ cells }, 100), null);
  assert.equal(readEye({ cells: [...cells.slice(1), 2] }, 144), null);
  assert.equal(readEye({ cells: cells.slice(0, 143) }), null); // not a square
});

test("a trace keeps the last span on the simulator's clock and restarts when time runs back", () => {
  const tr = createTrace(2);
  for (const t of [0, 0.5, 1, 1.5, 2, 2.5, 3]) tr.push(readFeel(feel(t, t)));
  assert.deepEqual(tr.samples().map((s) => s.t), [1, 1.5, 2, 2.5, 3]);
  tr.push(readFeel(feel(0.2, 0)));
  assert.deepEqual(tr.samples().map((s) => s.t), [0.2]);
});

test("both lanes share one scale: the largest magnitude seen in either", () => {
  const a = [1, -6, 2].map((v, k) => readFeel(feel(k, v)));
  const b = [3, 4].map((v, k) => readFeel(feel(k, v, undefined, "after")));
  assert.equal(sharedScale([a, b], (s) => s.tilt[0]), 6);
  assert.equal(sharedScale([[readFeel(feel(0, 0))]], (s) => s.tilt[0]), null);
});

test("a trace maps time to x and the value to y around the middle line, clipped at the scale", () => {
  const s = [0, 1, 2].map((t) => readFeel(feel(t, [0, 3, -9][t])));
  const p = tracePath(s, (x) => x.tilt[0], { x0: 0, x1: 100, yMid: 50, half: 20, tEnd: 2, spanS: 2, scale: 6 });
  assert.equal(p, "0,50 50,40 100,70");
  assert.equal(tracePath(s.slice(0, 1), (x) => x.tilt[0], { x0: 0, x1: 100, yMid: 50, half: 20, tEnd: 2, spanS: 2, scale: 6 }), "");
});

test("foot contact becomes runs from touch-down to lift-off", () => {
  const feet = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]];
  const s = feet.map((f, t) => readFeel(feel(t, 0, f)));
  assert.deepEqual(contactRuns(s, 0, { x0: 0, x1: 30, tEnd: 3, spanS: 3 }), [[0, 20]]);
  assert.deepEqual(contactRuns(s, 1, { x0: 0, x1: 30, tEnd: 3, spanS: 3 }), []);
});

test("the event view of two frames: brighter is white, darker black, the rest mid grey, at the server's threshold", () => {
  const older = Float32Array.of(0.5, 0.5, 0.5, 0.5);
  const newer = Float32Array.of(0.6, 0.4, 0.52, 0.5);
  assert.deepEqual(Array.from(changePicture(older, newer, "events", 0.04)), [1, 0, 0.5, 0.5]);
  assert.throws(() => changePicture(older, newer, "events", undefined), /threshold/);
  assert.throws(() => changePicture(older, Float32Array.of(1), "events", 0.04), /size/);
});
