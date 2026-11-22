// The episode player's pure parts and the status areas. Every event below is synthesised test input (a fixture for
// pure functions); none of it is a run's data or is shown as one.
import { test } from "node:test";
import assert from "node:assert/strict";
import { captionView, episodesButton, episodesControl } from "../../app/static/js/episodes.js";
import { createNotes } from "../../app/static/js/notes.js";

const NOTE = "showing this run's own episodes: same scenarios and seeds, replayed on the stage; the run computes them in parallel in the background";

test("without an episodes key in /api/state the control is unavailable", () => {
  const c = episodesControl({ status: "roaming" });
  assert.deepEqual(c, { available: false, playing: false, runId: null, note: "" });
  assert.deepEqual(episodesButton(c, "run1"), { label: "Episodes: unavailable", disabled: true, action: null, pressed: false });
  assert.equal(episodesControl(null).available, false);
});

test("the button starts a run's episodes, or stops them when this run is the one playing", () => {
  const idle = episodesControl({ episodes: { playing: false, run_id: null, note: "" } });
  assert.equal(episodesButton(idle, "a").action, "start");
  const playing = episodesControl({ episodes: { playing: true, run_id: "a", note: NOTE } });
  assert.deepEqual(episodesButton(playing, "a"), { label: "Stop showing episodes", disabled: false, action: "stop", pressed: true });
  assert.equal(episodesButton(playing, "b").action, "start");
  assert.equal(episodesButton(playing, "a", true).disabled, true); // a request in flight
  assert.equal(playing.note, NOTE);
});

test("the caption shows skill, step, index (counted from 1), kind, action and the note as the event sends them", () => {
  const v = captionView({ state: "start", run_id: "a", skill: "DODGE", step: "attempts", index: 2, total: 40, kind: "ball", action: "DODGE_LEFT", caption: "A ball thrown from the left.", outcome: null, note: NOTE });
  assert.equal(v.hidden, false);
  assert.deepEqual(v.chip, { text: "Playing", tone: "accent" });
  assert.equal(v.title, "DODGE · real attempt 3 of 40 · ball");
  assert.equal(v.text, "Action: DODGE_LEFT. A ball thrown from the left.");
  assert.deepEqual(v.outcome, []);
  assert.equal(v.note, NOTE);
});

test("the caption shows the outcome at the end of an episode, and hides when the player is idle", () => {
  const v = captionView({ state: "end", skill: "RAMP", step: "evaluate", index: 0, total: null, kind: "slope 12 deg", action: "climb", caption: "", outcome: { progress_m: 1.23456, reached_top: true, fall: false, note: null }, note: NOTE });
  assert.deepEqual(v.chip, { text: "Ended", tone: "neutral" });
  assert.equal(v.title, "RAMP · held-out evaluation scene 1 · slope 12 deg");
  assert.deepEqual(v.outcome, [["progress_m", "1.235"], ["reached_top", "yes"], ["fall", "no"], ["note", "none"]]);
  assert.equal(captionView({ state: "end", skill: "MOVE", outcome: null }).noOutcome, "The event carried no outcome.");
  assert.equal(captionView({ state: "idle", note: NOTE }).hidden, true);
  assert.equal(captionView({ phase: "end" }).hidden, true); // the roaming stage's own episode events are not captions
});

test("a status area lists newest first, merges identical messages with a count and shows three until asked", () => {
  const n = createNotes({ visible: 3 });
  n.report("danger", "Inference failed: boom (infer_failed)", "Error");
  n.report("danger", "Inference failed: boom (infer_failed)", "Error");
  n.set("infer", "warn", "No decisions: no student.", "Unavailable");
  n.set("window", "warn", "The window settings are unavailable.", "Unavailable");
  n.report("danger", "Inference failed: boom (infer_failed)", "Error");
  n.report("warn", "Inference is unavailable: x", "Unavailable");
  let v = n.view();
  assert.deepEqual(v.items.map((i) => [i.text, i.count]), [["Inference is unavailable: x", 1], ["Inference failed: boom (infer_failed)", 3], ["The window settings are unavailable.", 1]]);
  assert.equal(v.hidden, 1);
  assert.equal(n.view(true).items.length, 4);
  assert.equal(n.set("infer", "warn", "No decisions: no student.", "Unavailable"), false); // unchanged: no re-render
  assert.equal(n.clear("infer"), true);
  v = n.view();
  assert.equal(v.total, 3);
  assert.equal(v.hidden, 0);
});
