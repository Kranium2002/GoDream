// The robot viewer: WebGL2, no library. It draws /api/robot placed by the simulator's pose events and nothing else:
// before the first pose there is no dog to draw. Orbit by drag, zoom by wheel or pinch, keyboard: arrows orbit,
// + and - zoom, 0 resets the view.
import { lookAt, multiply, normalMatrix, perspective } from "./mat.js";
import { createOrbit } from "./orbit.js";
import { buildScene, bodyMatrix } from "./scene.js";
import { createPoseBuffer } from "./pose.js";

const ROBOT_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uProj, uView, uModel;
uniform mat3 uNormalMat;
out vec3 vNormal;
out vec3 vWorld;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0);
  vWorld = w.xyz;
  vNormal = uNormalMat * aNormal;
  gl_Position = uProj * uView * w;
}`;

const ROBOT_FS = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorld;
uniform vec4 uColor;
uniform vec3 uEye, uLight;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;
  vec3 l = normalize(uLight), v = normalize(uEye - vWorld), h = normalize(l + v);
  float diff = max(dot(n, l), 0.0);
  float hemi = 0.5 + 0.5 * n.z;
  vec3 base = pow(uColor.rgb, vec3(2.2));
  vec3 c = base * (0.20 + 0.30 * hemi) + base * diff * 0.95;
  c += vec3(0.35) * pow(max(dot(n, h), 0.0), 40.0) * diff;
  c += vec3(0.10) * pow(1.0 - max(dot(n, v), 0.0), 3.0);
  outColor = vec4(pow(c, vec3(1.0 / 2.2)), uColor.a);
}`;

const FLOOR_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uProj, uView;
out vec3 vWorld;
void main() { vWorld = aPos; gl_Position = uProj * uView * vec4(aPos, 1.0); }`;

const FLOOR_FS = `#version 300 es
precision highp float;
in vec3 vWorld;
uniform vec3 uCenter, uShadow, uFloor, uSky;
uniform float uShadowR;
out vec4 outColor;
float grid(vec2 p, float step) {
  vec2 g = abs(fract(p / step - 0.5) - 0.5) / fwidth(p / step);
  return 1.0 - min(min(g.x, g.y), 1.0);
}
void main() {
  vec2 p = vWorld.xy;
  float d = distance(p, uCenter.xy);
  vec3 c = uFloor;
  c = mix(c, c + vec3(0.07), grid(p, 0.5));
  c = mix(c, c + vec3(0.04), grid(p, 0.1) * 0.5);
  float s = exp(-pow(distance(p, uShadow.xy) / uShadowR, 2.0) * 2.2);
  c *= 1.0 - 0.55 * s;
  float fade = smoothstep(2.5, 9.0, d);
  outColor = vec4(mix(c, uSky, fade), 1.0);
}`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`A shader did not compile: ${gl.getShaderInfoLog(s)}`);
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`The shaders did not link: ${gl.getProgramInfoLog(p)}`);
  const u = {};
  for (let k = 0; k < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); k++) {
    const name = gl.getActiveUniform(p, k).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}

const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255);

// createViewer(canvas, {background, floor}) -> {setRobot, pushPose, setShift, resetView, destroy}. Throws Error when
// WebGL2 is missing or a shader fails, so the page can fall back to the image stream.
export function createViewer(canvas, { background = "#101215", floor = "#1a1d22" } = {}) {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("This browser offers no WebGL2.");
  const robotP = program(gl, ROBOT_VS, ROBOT_FS);
  const floorP = program(gl, FLOOR_VS, FLOOR_FS);
  const sky = hex(background), floorRgb = hex(floor);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

  const START = { distance: 1.7, yaw: 0.75, pitch: 0.3 };
  const orbit = createOrbit({ target: [0, 0, 0.25], ...START, minDistance: 0.6, maxDistance: 8 });
  const poses = createPoseBuffer();
  let scene = null, vaos = new Map(), shift = 0, shiftGoal = 0, raf = 0, dirty = true;

  const floorBuf = gl.createBuffer();
  const floorVao = gl.createVertexArray();
  gl.bindVertexArray(floorVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, floorBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  function upload(geom) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    for (const [loc, data] of [[0, geom.positions], [1, geom.normals]]) {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geom.indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return { vao, count: geom.indices.length };
  }

  function setRobot(robot) {
    const next = buildScene(robot); // throws with the reason if unreadable
    for (const { vao } of vaos.values()) gl.deleteVertexArray(vao);
    vaos = new Map([...next.meshes].map(([k, g]) => [k, upload(g)]));
    scene = next;
    poses.clear();
    dirty = true;
  }

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; dirty = true; }
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    resize();
    const pose = scene ? poses.sample(now) : null;
    if (Math.abs(shiftGoal - shift) > 1e-3) { shift = reduceMotion.matches ? shiftGoal : shift + (shiftGoal - shift) * 0.12; dirty = true; }
    if (!dirty && !pose) return;
    dirty = false;
    if (pose && scene.rootBody !== null) {
      const p = pose.pos[scene.rootBody];
      if (p) orbit.setTarget([p[0], p[1], scene.floorZ + 0.22]); // the camera follows the dog the simulator moves
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(...sky, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const proj = perspective((35 * Math.PI) / 180, canvas.width / canvas.height, 0.05, 80);
    proj[8] = -shift; // lens shift: the dog sits beside the text, not under it
    const eye = orbit.eye();
    const view = lookAt(eye, orbit.target, [0, 0, 1]);
    const t = orbit.target, z = (scene?.floorZ ?? 0) - 0.0005, R = 30;

    gl.useProgram(floorP.p);
    gl.bindBuffer(gl.ARRAY_BUFFER, floorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.of(t[0] - R, t[1] - R, z, t[0] + R, t[1] - R, z, t[0] + R, t[1] + R, z,
      t[0] - R, t[1] - R, z, t[0] + R, t[1] + R, z, t[0] - R, t[1] + R, z), gl.DYNAMIC_DRAW);
    gl.uniformMatrix4fv(floorP.u.uProj, false, proj);
    gl.uniformMatrix4fv(floorP.u.uView, false, view);
    gl.uniform3f(floorP.u.uCenter, t[0], t[1], z);
    const root = pose && scene.rootBody !== null ? pose.pos[scene.rootBody] : null;
    gl.uniform3f(floorP.u.uShadow, root ? root[0] : 1e4, root ? root[1] : 1e4, z);
    gl.uniform1f(floorP.u.uShadowR, 0.42);
    gl.uniform3f(floorP.u.uFloor, ...floorRgb);
    gl.uniform3f(floorP.u.uSky, ...sky);
    gl.bindVertexArray(floorVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (!pose) return;
    gl.useProgram(robotP.p);
    gl.uniformMatrix4fv(robotP.u.uProj, false, proj);
    gl.uniformMatrix4fv(robotP.u.uView, false, view);
    gl.uniform3f(robotP.u.uEye, ...eye);
    gl.uniform3f(robotP.u.uLight, 0.5, 0.35, 0.8);
    const bodies = new Map();
    for (const d of scene.drawables) {
      if (!bodies.has(d.body)) bodies.set(d.body, bodyMatrix(pose, d.body));
      const model = multiply(bodies.get(d.body), d.local);
      const g = vaos.get(d.meshKey);
      gl.uniformMatrix4fv(robotP.u.uModel, false, model);
      gl.uniformMatrix3fv(robotP.u.uNormalMat, false, normalMatrix(model));
      gl.uniform4f(robotP.u.uColor, ...d.rgba);
      gl.bindVertexArray(g.vao);
      gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
  }

  // ---- input --------------------------------------------------------------------------------------------------
  const pointers = new Map();
  let pinch = null;
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]); });
  canvas.addEventListener("pointermove", (e) => {
    const last = pointers.get(e.pointerId);
    if (!last) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 1) {
      orbit.rotate(-(e.clientX - last[0]) * 0.008, (e.clientY - last[1]) * 0.006);
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) orbit.zoom(pinch / d);
      pinch = d;
    }
    dirty = true;
  });
  const lift = (e) => { pointers.delete(e.pointerId); pinch = null; };
  canvas.addEventListener("pointerup", lift);
  canvas.addEventListener("pointercancel", lift);
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); orbit.zoom(Math.exp(e.deltaY * 0.0012)); dirty = true; }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const keys = { ArrowLeft: () => orbit.rotate(0.12, 0), ArrowRight: () => orbit.rotate(-0.12, 0), ArrowUp: () => orbit.rotate(0, 0.08),
      ArrowDown: () => orbit.rotate(0, -0.08), "+": () => orbit.zoom(0.9), "=": () => orbit.zoom(0.9), "-": () => orbit.zoom(1.1), 0: resetView };
    if (!keys[e.key]) return;
    e.preventDefault();
    keys[e.key]();
    dirty = true;
  });

  function resetView() { Object.assign(orbit, { distance: START.distance, yaw: START.yaw, pitch: START.pitch }); dirty = true; }

  // ---- lifecycle: no drawing while the tab is hidden ---------------------------------------------------------
  const onVisibility = () => {
    cancelAnimationFrame(raf);
    if (!document.hidden) { dirty = true; raf = requestAnimationFrame(frame); }
  };
  document.addEventListener("visibilitychange", onVisibility);
  raf = requestAnimationFrame(frame);

  return {
    setRobot,
    pushPose(pose) { if (scene && Array.isArray(pose?.pos) && Array.isArray(pose?.quat)) poses.push(pose, performance.now()); },
    hasPose: () => Boolean(scene && poses.sample(performance.now())),
    bodyCount: () => scene?.bodyCount ?? 0,
    setShift(x) { shiftGoal = x; },
    resetView,
    destroy() { cancelAnimationFrame(raf); document.removeEventListener("visibilitychange", onVisibility); },
  };
}
