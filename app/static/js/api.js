// HTTP API v0 client. Every call resolves (never throws) to
//   {ok: true, status, data}                                   on a 2xx JSON answer, or
//   {ok: false, status, code, message, unavailable, data}       otherwise (status 0: the server was not reached).
// unavailable: a 404, or a 503 whose code ends in "_unavailable" (the contract's "cannot work yet").

export function createApi(fetchFn) {
  async function call(path, body) {
    const init = body === undefined
      ? { method: "GET" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    let response;
    try {
      response = await fetchFn(path, init);
    } catch {
      return fail(0, "network_error", "The server could not be reached.", null);
    }
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    const status = response.status;
    if (response.ok) {
      return data !== null && typeof data === "object"
        ? { ok: true, status, data }
        : fail(status, "invalid_response", "The server's answer was not a JSON object.", null);
    }
    const code = typeof data?.code === "string" ? data.code : `http_${status}`;
    const message = typeof data?.message === "string" ? data.message : `The server answered ${status} without a readable message.`;
    return fail(status, code, message, data);
  }

  function fail(status, code, message, data) {
    const unavailable = status === 404 || (status === 503 && code.endsWith("_unavailable"));
    return { ok: false, status, code, message, unavailable, data };
  }

  return {
    state: () => call("/api/state"),
    skills: () => call("/api/skills"),
    robot: () => call("/api/robot"),
    drive: (command) => call("/api/drive", command),
    stop: () => call("/api/stop", {}),
    reset: (preset) => call("/api/reset", preset === undefined ? {} : { preset }),
    chat: (messages) => call("/api/chat", { messages }),
    runs: () => call("/api/runs"),
    runCards: () => call("/api/runs?summary=1"),
    runEvents: (runId) => call(`/api/runs/${encodeURIComponent(runId)}/events`),
    run: (runId) => call(`/api/runs/${encodeURIComponent(runId)}`),
    startRun: (declaredSpec, confirmCost) => call("/api/runs", { declared_spec: declaredSpec, confirm_cost: confirmCost }),
    // The camera window goes here and nowhere else. captured_ms: the newest frame's capture time, ms since the epoch.
    // sessionStart: the first window after Start camera, so the server resets the gate and the cooldown clock.
    infer: (window, frames, capturedMs, sessionStart = false) => call("/api/infer", {
      window: Array.from(window), frames, captured_ms: capturedMs, ...(sessionStart ? { session_start: true } : {}),
    }),
  };
}
