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

test("skills() GETs /api/skills", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { skills: [] }));
  const r = await createApi(fetchFn).skills();
  assert.equal(calls[0].url, "/api/skills");
  assert.equal(r.ok, true);
});

test("chat() POSTs the running message list as JSON", async () => {
  const { calls, fetchFn } = recorder(() => json(200, { reply: "ok", declared_spec: null, problems: [], missing: [] }));
  const messages = [{ role: "user", content: "teach it to cross this slope" }];
  const r = await createApi(fetchFn).chat(messages);
  assert.equal(calls[0].url, "/api/chat");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].init.body), { messages });
  assert.equal(r.data.reply, "ok");
});

test("startRun() POSTs the declared spec and the cost confirmation flag", async () => {
  const { calls, fetchFn } = recorder(() => json(202, { run_id: "r1" }));
  const spec = { skill: "RAMP", parameters: {} };
  const r = await createApi(fetchFn).startRun(spec, true);
  assert.equal(calls[0].url, "/api/runs");
  assert.deepEqual(JSON.parse(calls[0].init.body), { declared_spec: spec, confirm_cost: true });
  assert.deepEqual(r, { ok: true, status: 202, data: { run_id: "r1" } });
});

test("an error answer carries the server's code, message and body", async () => {
  const body = { code: "cost_confirmation_required", message: "Confirm the cost.", usage: { requests: 3 } };
  const { fetchFn } = recorder(() => json(409, body));
  const r = await createApi(fetchFn).startRun({}, false);
  assert.deepEqual(r, { ok: false, status: 409, code: body.code, message: body.message, unavailable: false, data: body });
});

test("a 404 is unavailable and keeps the server's message", async () => {
  const { fetchFn } = recorder(() => json(404, { code: "not_found", message: "There is nothing at that address." }));
  const r = await createApi(fetchFn).chat([]);
  assert.equal(r.unavailable, true);
  assert.equal(r.message, "There is nothing at that address.");
});

test("a 503 whose code ends in _unavailable is unavailable; another 503 is not", async () => {
  const a = await createApi(async () => json(503, { code: "chat_unavailable", message: "No API key." })).chat([]);
  assert.equal(a.unavailable, true);
  assert.equal(a.message, "No API key.");
  const b = await createApi(async () => json(503, { code: "simulation_stopped", message: "Restart." })).chat([]);
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
