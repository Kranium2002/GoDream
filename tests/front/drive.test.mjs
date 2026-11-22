import { test } from "node:test";
import assert from "node:assert/strict";
import { commandFields, cropRect, driveCommand } from "../../app/static/js/drive.js";

// The ranges /api/state publishes (config.yaml robot command fields); pure-function input only.
const fields = { vel_x: { unit: "m/s", min: -1, max: 1 }, vel_y: { unit: "m/s", min: -0.5, max: 0.5 }, yaw_rate: { unit: "rad/s", min: -1, max: 1 } };

test("commandFields reads the new commands key and the older command_fields", () => {
  assert.equal(commandFields({ commands: fields }), fields);
  assert.equal(commandFields({ command_fields: fields }), fields);
  assert.equal(commandFields({}), null);
});

test("driveCommand moves along one field by a share of its published range, every other field zero", () => {
  assert.deepEqual(driveCommand("forward", fields, 0.5), { vel_x: 0.5, vel_y: 0, yaw_rate: 0 });
  assert.deepEqual(driveCommand("back", fields, 0.5), { vel_x: -0.5, vel_y: 0, yaw_rate: 0 });
  assert.deepEqual(driveCommand("left", fields, 1), { vel_x: 0, vel_y: 0.5, yaw_rate: 0 });
  assert.deepEqual(driveCommand("right", fields, 1), { vel_x: 0, vel_y: -0.5, yaw_rate: 0 });
  assert.deepEqual(driveCommand("turn_left", fields, 0.25), { vel_x: 0, vel_y: 0, yaw_rate: 0.25 });
  assert.deepEqual(driveCommand("turn_right", fields, 0.25), { vel_x: 0, vel_y: 0, yaw_rate: -0.25 });
  assert.deepEqual(driveCommand("stop", fields, 1), { vel_x: 0, vel_y: 0, yaw_rate: 0 });
});

test("a direction the ranges do not allow, or a missing field, gives null", () => {
  assert.equal(driveCommand("back", { ...fields, vel_x: { min: 0, max: 1 } }, 0.5), null);
  assert.equal(driveCommand("left", { vel_x: fields.vel_x }, 0.5), null);
  assert.equal(driveCommand("sideways", fields, 0.5), null);
  assert.equal(driveCommand("forward", null, 0.5), null);
});

test("cropRect places the centre square on a preview shown with object-fit: contain", () => {
  // 640 x 480 shown in a 400 x 400 box: scale 0.625, image 400 x 300 starting at y 50; the square is 300 x 300 at x 50.
  assert.deepEqual(cropRect(640, 480, 400, 400), { left: 50, top: 50, size: 300 });
  assert.deepEqual(cropRect(480, 640, 300, 400), { left: 0, top: 50, size: 300 });
  assert.equal(cropRect(0, 0, 100, 100), null);
});
