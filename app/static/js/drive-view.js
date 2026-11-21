// "Try it in the simulation": drive the real simulated dog on the stage with the existing controls (POST /api/drive,
// /api/stop, /api/reset). Commands come from the ranges /api/state publishes; what is shown is what the server answered.
import { commandFields, driveCommand } from "./drive.js";

const KEYS = { w: "forward", s: "back", a: "left", d: "right", q: "turn_left", e: "turn_right", " ": "stop" };

export function initDriveView({ api, onOpen, onClose }) {
  const $ = (id) => document.getElementById(id);
  let fields = null, presets = [], current = null;
  const status = (text) => { $("drive-status").textContent = text; };

  function setState(state, capabilities) {
    fields = commandFields(state);
    const drive = capabilities.find((c) => c.name === "drive");
    $("drive-cap").hidden = !drive || drive.available;
    $("drive-cap").textContent = drive && !drive.available ? `Driving is unavailable: ${drive.reason}` : "";
    const names = Array.isArray(state?.available_presets) ? state.available_presets : Array.isArray(state?.presets) ? state.presets : [];
    current = typeof state?.preset === "string" ? state.preset : current;
    if (names.join() !== presets.join()) {
      presets = names;
      const sel = $("drive-preset");
      sel.replaceChildren(...names.map((n) => Object.assign(document.createElement("option"), { value: n, textContent: n })));
    }
    if (current) $("drive-preset").value = current;
  }

  const describe = (cmd) => Object.entries(cmd).map(([k, v]) => `${k} ${Number(v).toFixed(2)}${fields?.[k]?.unit ? ` ${fields[k].unit}` : ""}`).join(", ");

  async function go(dir) {
    if (!fields) return status("The server has not published its command ranges yet, so no command can be built.");
    const cmd = driveCommand(dir, fields, Number($("drive-speed").value));
    if (!cmd) return status("The published ranges do not allow that direction.");
    const r = dir === "stop" ? await api.stop() : await api.drive(cmd);
    if (!r.ok) return status(`${r.unavailable ? "Driving is unavailable" : "The server refused the command"}: ${r.message}`);
    status(r.data?.command ? `Sent: ${describe(r.data.command)}. The dog keeps this command until you change it.` : "The server accepted the command.");
  }

  async function reset() {
    const preset = $("drive-preset").value || undefined;
    status("Resetting the episode.");
    const r = await api.reset(preset);
    status(r.ok ? `A fresh episode started${r.data?.preset ? ` in the ${r.data.preset} world` : ""}.` : `The reset was refused: ${r.message}`);
  }

  function open() {
    $("drive").hidden = false;
    document.body.classList.add("drive-open");
    onOpen?.();
    $("drive").querySelector('[data-dir="forward"]').focus({ preventScroll: true });
  }
  function close() {
    $("drive").hidden = true;
    document.body.classList.remove("drive-open");
    onClose?.();
    $("try-sim").focus({ preventScroll: true });
  }

  for (const b of document.querySelectorAll("[data-dir]")) b.addEventListener("click", () => go(b.dataset.dir));
  $("drive-reset").addEventListener("click", reset);
  $("drive-close").addEventListener("click", close);
  $("try-sim").addEventListener("click", open);
  // Keys drive only while the pad is open and the person is not typing somewhere.
  addEventListener("keydown", (e) => {
    if ($("drive").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest?.("input, textarea, select, [contenteditable]")) return;
    if (e.key === "Escape") return close();
    const dir = KEYS[e.key.toLowerCase()];
    if (!dir) return;
    e.preventDefault();
    go(dir);
  });

  return { setState, open, close };
}
