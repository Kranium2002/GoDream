// Geometry for the viewer: the server's base64 mesh arrays, vertex normals, and MuJoCo primitives at their real sizes.
// A primitive is {positions: Float32Array, normals: Float32Array, indices: Uint32Array}.

function bytes(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let k = 0; k < s.length; k++) out[k] = s.charCodeAt(k);
  return out;
}
// Little-endian, as numpy writes on this machine; copied into an aligned buffer.
export const decodeFloat32 = (b64) => new Float32Array(bytes(b64).buffer);
export const decodeUint32 = (b64) => new Uint32Array(bytes(b64).buffer);

// Area-weighted vertex normals of a triangle mesh, unit length.
export function vertexNormals(positions, indices) {
  const n = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [indices[t] * 3, indices[t + 1] * 3, indices[t + 2] * 3];
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx; // twice the area, along the normal
    for (const v of [a, b, c]) { n[v] += cx; n[v + 1] += cy; n[v + 2] += cz; }
  }
  for (let v = 0; v < n.length; v += 3) {
    const l = Math.hypot(n[v], n[v + 1], n[v + 2]) || 1;
    n[v] /= l; n[v + 1] /= l; n[v + 2] /= l;
  }
  return n;
}

function build(verts, norms, idx) {
  return { positions: Float32Array.from(verts), normals: Float32Array.from(norms), indices: Uint32Array.from(idx) };
}

function box([hx, hy, hz]) {
  const v = [], n = [], i = [];
  const faces = [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]]; // axis, sign
  for (const [axis, sign] of faces) {
    const u = (axis + 1) % 3, w = (axis + 2) % 3, h = [hx, hy, hz], base = v.length / 3;
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const p = [0, 0, 0];
      p[axis] = sign * h[axis]; p[u] = a * h[u]; p[w] = b * h[w];
      const nn = [0, 0, 0]; nn[axis] = sign;
      v.push(...p); n.push(...nn);
    }
    i.push(...(sign > 0 ? [base, base + 1, base + 2, base, base + 2, base + 3] : [base, base + 2, base + 1, base, base + 3, base + 2]));
  }
  return build(v, n, i);
}

// A UV sphere of radii (rx, ry, rz); a capsule shifts the upper half up and the lower half down by h.
function roundShape(rx, ry, rz, h = 0, seg = 24, rings = 16) {
  const v = [], n = [], i = [];
  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI, z = Math.cos(phi), s = Math.sin(phi);
    const shift = h ? (r <= rings / 2 ? h : -h) : 0;
    for (let k = 0; k <= seg; k++) {
      const th = (k / seg) * 2 * Math.PI, x = s * Math.cos(th), y = s * Math.sin(th);
      v.push(rx * x, ry * y, rz * z + shift);
      const nx = x / rx, ny = y / ry, nz = z / rz, l = Math.hypot(nx, ny, nz) || 1;
      n.push(nx / l, ny / l, nz / l);
    }
  }
  for (let r = 0; r < rings; r++) for (let k = 0; k < seg; k++) {
    const a = r * (seg + 1) + k, b = a + seg + 1;
    i.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return build(v, n, i);
}

function cylinder(r, h, seg = 24) {
  const v = [], n = [], i = [];
  for (let k = 0; k <= seg; k++) {
    const th = (k / seg) * 2 * Math.PI, x = Math.cos(th), y = Math.sin(th);
    v.push(r * x, r * y, -h, r * x, r * y, h);
    n.push(x, y, 0, x, y, 0);
  }
  for (let k = 0; k < seg; k++) { const a = 2 * k; i.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
  for (const sign of [1, -1]) {
    const c = v.length / 3;
    v.push(0, 0, sign * h); n.push(0, 0, sign);
    for (let k = 0; k <= seg; k++) {
      const th = (k / seg) * 2 * Math.PI;
      v.push(r * Math.cos(th), r * Math.sin(th), sign * h); n.push(0, 0, sign);
    }
    for (let k = 0; k < seg; k++) i.push(...(sign > 0 ? [c, c + 1 + k, c + 2 + k] : [c, c + 2 + k, c + 1 + k]));
  }
  return build(v, n, i);
}

// MuJoCo sizes: box half extents; sphere radius; capsule and cylinder radius and half length along z; ellipsoid radii.
// A plane is the floor, drawn by the viewer itself: null.
export function primitive(type, size) {
  switch (type) {
    case "box": return box(size);
    case "sphere": return roundShape(size[0], size[0], size[0]);
    case "ellipsoid": return roundShape(size[0], size[1], size[2]);
    case "capsule": return roundShape(size[0], size[0], size[0], size[1]);
    case "cylinder": return cylinder(size[0], size[1]);
    default: return null;
  }
}
