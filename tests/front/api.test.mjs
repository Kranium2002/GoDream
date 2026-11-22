import { test } from "node:test";
import assert from "node:assert/strict";
import { createApi } from "../../app/static/js/api.js";

const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function recorder(answer) {
  const calls = [];
  const fetchFn = async (url, init = {}) => {
    calls.push({ url, init });
    return answer(url, init);
  };
  return { calls, fetchFn };
}

test("state() GETs /api/state and returns the parsed body", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { status: "roaming" }));
  const r = await createApi(fetchFn).state();
  assert.equal(calls[0].url, "/api/state");
  assert.equal(calls[0].init.method ?? "GET", "GET");
  assert.deepEqual(r, { ok: true, status: 200, data: { status: "roaming" } });
});

test("demos(), duelStart(), duelStop() and story() follow the v1 contract", async () => {
  const { calls, fetchFn } = recorder(() => json(200, {}));
  const api = createApi(fetchFn);
  await api.demos();
  await api.duelStart("hills", { slope_deg: 12 }, 7);
  await api.duelStart("dodge", {});
  await api.duelStop();
  await api.story("ramp 1/x");
  assert.deepEqual(calls.map((c) => [c.url, c.init.method, c.init.body ? JSON.parse(c.init.body) : null]), [
    ["/api/demos", "GET", null],
    ["/api/duel/start", "POST", { demo: "hills", params: { slope_deg: 12 }, seed: 7 }],
    ["/api/duel/start", "POST", { demo: "dodge", params: {}, seed: null }],
    ["/api/duel/stop", "POST", {}],
    ["/api/story/ramp%201%2Fx", "GET", null],
  ]);
});

test("startRun() POSTs the declared spec and the cost confirmation flag", async () => {
  const { calls, fetchFn } = recorder(() => json(202, { run_id: "r1" }));
  const spec = { skill: "RAMP", parameters: {} };
  const r = await createApi(fetchFn).startRun(spec, true);
  assert.equal(calls[0].url, "/api/runs");
  assert.deepEqual(JSON.parse(calls[0].init.body), { declared_spec: spec, confirm_cost: true });
  assert.deepEqual(r, { ok: true, status: 202, data: { run_id: "r1" } });
});

test("infer() adds session_start: true only for the first window of a camera session", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { action: "NO_OP" }));
  const api = createApi(fetchFn);
  await api.infer(Float32Array.of(0.5), 1, 10, true);
  await api.infer(Float32Array.of(0.5), 1, 20, false);
  await api.infer(Float32Array.of(0.5), 1, 30);
  assert.deepEqual(JSON.parse(calls[0].init.body), { window: [0.5], frames: 1, captured_ms: 10, session_start: true });
  assert.equal("session_start" in JSON.parse(calls[1].init.body), false);
  assert.equal("session_start" in JSON.parse(calls[2].init.body), false);
});

test("runs() GETs /api/runs", async () => {
  const { calls, fetchFn } = recorder(() => json(200, ["r1", "r2"]));
  const r = await createApi(fetchFn).runs();
  assert.equal(calls[0].url, "/api/runs");
  assert.equal(calls[0].init.method, "GET");
  assert.deepEqual(r, { ok: true, status: 200, data: ["r1", "r2"] });
});

test("infer() POSTs the window as a plain list with its frame count and capture time", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { action: "NO_OP" }));
  const r = await createApi(fetchFn).infer(Float32Array.of(0.5, 0.25), 2, 1234.5);
  assert.equal(calls[0].url, "/api/infer");
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].init.body), { window: [0.5, 0.25], frames: 2, captured_ms: 1234.5 });
  assert.equal(r.data.action, "NO_OP");
});

test("run() GETs /api/runs/{run_id} with the id escaped", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { state: "running", steps: [] }));
  const r = await createApi(fetchFn).run("run 7/x");
  assert.equal(calls[0].url, "/api/runs/run%207%2Fx");
  assert.equal(r.data.state, "running");
});

test("an error answer carries the server's code, message and body", async () => {
  const body = { code: "cost_confirmation_required", message: "Confirm the cost.", usage: { requests: 3 } };
  const { fetchFn } = recorder(() => json(409, body));
  const r = await createApi(fetchFn).startRun({}, false);
  assert.deepEqual(r, { ok: false, status: 409, code: body.code, message: body.message, unavailable: false, data: body });
});

test("a 404 is unavailable and keeps the server's message", async () => {
  const { fetchFn } = recorder(() => json(404, { code: "not_found", message: "There is nothing at that address." }));
  const r = await createApi(fetchFn).demos();
  assert.equal(r.unavailable, true);
  assert.equal(r.message, "There is nothing at that address.");
});

test("a 503 whose code ends in _unavailable is unavailable; another 503 is not", async () => {
  const a = await createApi(async () => json(503, { code: "runs_unavailable", message: "No runner." })).runs();
  assert.equal(a.unavailable, true);
  assert.equal(a.message, "No runner.");
  const b = await createApi(async () => json(503, { code: "simulation_stopped", message: "Restart." })).runs();
  assert.equal(b.unavailable, false);
});

test("a failed fetch is a network error with status 0", async () => {
  const r = await createApi(async () => { throw new TypeError("Failed to fetch"); }).state();
  assert.equal(r.ok, false);
  assert.equal(r.status, 0);
  assert.equal(r.code, "network_error");
  assert.equal(r.unavailable, false);
  assert.match(r.message, /could not be reached/);
});

test("an error answer without a JSON body gets a code from its status", async () => {
  const r = await createApi(async () => new Response("<html>bad gateway</html>", { status: 502 })).state();
  assert.equal(r.ok, false);
  assert.equal(r.code, "http_502");
  assert.equal(r.data, null);
  assert.match(r.message, /502/);
});

test("a 200 answer that is not JSON is an error, not data", async () => {
  const r = await createApi(async () => new Response("hello", { status: 200 })).state();
  assert.equal(r.ok, false);
  assert.equal(r.code, "invalid_response");
});

test("robot() GETs /api/robot", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { bodies: [], geoms: [], meshes: [] }));
  const r = await createApi(fetchFn).robot();
  assert.equal(calls[0].url, "/api/robot");
  assert.equal(r.ok, true);
});

test("runCards() asks for the summary list and runEvents() for a run's recorded lines", async () => {
  const { calls, fetchFn } = recorder(() => json(200, []));
  const api = createApi(fetchFn);
  await api.runCards();
  await api.runEvents("v 0");
  assert.equal(calls[0].url, "/api/runs?summary=1");
  assert.equal(calls[1].url, "/api/runs/v%200/events");
});

test("drive(), stop() and reset() POST to the existing stage controls", async () => {
  const { calls, fetchFn } = recorder(() => json(200, {}));
  const api = createApi(fetchFn);
  await api.drive({ vel_x: 0.5, vel_y: 0, yaw_rate: 0 });
  await api.stop();
  await api.reset("slippery_floor");
  await api.reset();
  assert.deepEqual(calls.map((c) => [c.url, c.init.method, JSON.parse(c.init.body)]), [
    ["/api/drive", "POST", { vel_x: 0.5, vel_y: 0, yaw_rate: 0 }],
    ["/api/stop", "POST", {}],
    ["/api/reset", "POST", { preset: "slippery_floor" }],
    ["/api/reset", "POST", {}],
  ]);
});
