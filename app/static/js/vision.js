// Camera frame -> the model's grey cells, exactly as Dev B defined it (plan, "Dev B contracts > Preprocessing"):
// not mirrored; centre square of side S = min(W, H); grey = luma . (R, G, B); area average over whole-pixel bins
// (rows floor(i*S/n) .. floor((i+1)*S/n) - 1, same for columns); divide; float32, row-major, top-left first.

// image_window from /api/state -> {ok: true, settings: {frames, height, width, hz, luma, divisor}} or
// {ok: false, problems: [str]}. Nothing is defaulted: an unreadable field is a problem.
export function readWindowSettings(raw) {
  if (raw === undefined || raw === null) return { ok: false, problems: ["The server does not publish image_window yet."] };
  if (typeof raw !== "object") return { ok: false, problems: ["image_window is not an object."] };
  const problems = [];
  const posInt = (k) => Number.isInteger(raw[k]) && raw[k] > 0 || problems.push(`${k} must be a whole number above 0.`);
  ["frames", "height", "width"].forEach(posInt);
  if (!(typeof raw.hz === "number" && raw.hz > 0 && Number.isFinite(raw.hz))) problems.push("hz must be a number above 0.");
  if (!(Array.isArray(raw.luma) && raw.luma.length === 3 && raw.luma.every(Number.isFinite))) problems.push("luma must be three numbers, the R, G and B weights.");
  if (!(typeof raw.normalization === "number" && raw.normalization > 0 && Number.isFinite(raw.normalization))) problems.push("normalization must be the number each grey value is divided by.");
  if (raw.order !== "oldest_first") problems.push(`order must be "oldest_first", the only order this page sends.`);
  if (raw.crop !== "centre_square" && raw.crop !== "center_square") problems.push(`crop must be "centre_square", the only crop this page makes.`);
  if (raw.mirrored !== false) problems.push("mirrored must be false: the contract's frames are not mirrored.");
  if (problems.length) return { ok: false, problems };
  const { frames, height, width, hz, luma } = raw;
  // What the student computes from the window (named only; the server encodes): present when the server says.
  const encoding = typeof raw.window_encoding === "string" && raw.window_encoding
    ? { windowEncoding: raw.window_encoding, encodingFrom: raw.window_encoding_from === "student" ? "student" : "config" } : {};
  // The encoding's threshold ("events": the event threshold; the diff encodings: the smallest divisor), when sent.
  const floor = typeof raw.diff_floor === "number" && raw.diff_floor > 0 && Number.isFinite(raw.diff_floor) ? { diffFloor: raw.diff_floor } : {};
  return { ok: true, settings: { frames, height, width, hz, luma: [...luma], divisor: raw.normalization, ...encoding, ...floor } };
}

export function preprocess(rgba, W, H, { height, width, luma, divisor }) {
  if (rgba.length !== W * H * 4) throw new Error(`The pixel data holds ${rgba.length} values, not ${W} x ${H} RGBA pixels.`);
  const S = Math.min(W, H);
  if (S < height || S < width) throw new Error(`The camera frame (${W} x ${H}) is smaller than the ${height} x ${width} window.`);
  const x0 = Math.floor((W - S) / 2);
  const y0 = Math.floor((H - S) / 2);
  const [wr, wg, wb] = luma;
  const edges = (n) => Array.from({ length: n + 1 }, (_, i) => Math.floor((i * S) / n));
  const rows = edges(height);
  const cols = edges(width);
  const out = new Float32Array(height * width);
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      let sum = 0;
      for (let y = rows[i]; y < rows[i + 1]; y++) {
        let p = ((y0 + y) * W + x0 + cols[j]) * 4;
        for (let x = cols[j]; x < cols[j + 1]; x++, p += 4) sum += wr * rgba[p] + wg * rgba[p + 1] + wb * rgba[p + 2];
      }
      out[i * width + j] = sum / ((rows[i + 1] - rows[i]) * (cols[j + 1] - cols[j])) / divisor;
    }
  }
  return out;
}
