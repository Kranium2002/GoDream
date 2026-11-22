// The robot viewer: WebGL2, no library. It draws /api/robot placed by the simulator's pose events, plus the objects
// and markers of the contract's "scene" event, and nothing else: before the first pose there is no dog to draw. Orbit
// by drag, zoom by wheel or pinch, keyboard: arrows orbit, + and - zoom, 0 resets the view. Smooth by design: one
// requestAnimationFrame loop, poses played back on the simulator's clock (pose.js), the canvas size read from a
// ResizeObserver rather than from the layout on every frame, and no drawing while the view is inactive or hidden.
import { lookAt, multiply, normalMatrix, perspective } from "./mat.js";
import { createOrbit, headingOf, smooth, smoothAngle } from "./orbit.js";
import { buildScene, bodyMatrix, sceneItems } from "./scene.js";
import { createPoseBuffer } from "./pose.js";
import { createQuality } from "./quality.js";
import { primitive } from "./geometry.js";

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

// The scene event's objects as footprints on the floor, [x, y, r]: what the framing keeps in view. A sphere whose centre
// lies below the floor (a hill: a spherical cap) counts with the circle where it meets the floor.
function footprints(scene, floorZ) {
  const out = [];
  for (const o of Array.isArray(scene?.objects) ? scene.objects : []) {
    const p = o?.pos, s = o?.size;
    if (!Array.isArray(p) || !Array.isArray(s) || !p.slice(0, 3).every(Number.isFinite) || !Number.isFinite(s[0])) continue;
    let r = Math.abs(s[0]);
    if (o.shape === "sphere" && p[2] < floorZ) { const d = floorZ - p[2]; if (d >= r) continue; r = Math.sqrt(r * r - d * d); }
    else if (o.shape === "box" && Number.isFinite(s[1])) r = Math.hypot(s[0], s[1]);
    out.push([p[0], p[1], r]);
  }
  return out;
}

// Whether a world point lies inside one of the scene's spheres or boxes (boxes taken axis-aligned in their own frame
// without rotation, which is enough to keep the camera out of an obstacle).
function inside(scene, q) {
  for (const o of Array.isArray(scene?.objects) ? scene.objects : []) {
    const p = o?.pos, s = o?.size;
    if (!Array.isArray(p) || !Array.isArray(s)) continue;
    if (o.shape === "sphere" && Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < s[0] + 0.05) return true;
    if (o.shape === "box" && [0, 1, 2].every((k) => Math.abs(q[k] - p[k]) < (s[k] ?? 0) + 0.05)) return true;
  }
  return false;
}

// trained: draw the robot in its trained colours (geom trained_rgba), for the trained dog of a side-by-side run.
// createViewer(canvas, {background, floor, distance, yaw, pitch, lockHeading, frameScene, trained}) -> {setRobot, pushPose,
// setShift, resetView, destroy}. lockHeading: the camera keeps the heading the dog had at its first pose instead of
// turning with it, so two views of two dogs keep the same angle. frameScene: the view widens and moves to keep the
// scene's objects (a hill, an incoming object) in frame with the dog. Throws Error when WebGL2 is missing or a shader
// fails, so the page can fall back to the image stream.
export function createViewer(canvas, { background = "#101215", floor = "#1a1d22", distance = 1.7, yaw = 0.75, pitch = 0.3, lockHeading = false, frameScene = false, trained = false } = {}) {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("This browser offers no WebGL2.");
  const robotP = program(gl, ROBOT_VS, ROBOT_FS);
  const floorP = program(gl, FLOOR_VS, FLOOR_FS);
  const sky = hex(background), floorRgb = hex(floor);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

  const START = { distance, yaw, pitch, fit: 1.45 };
  let zoomK = 1; // the person's zoom while the framing sets the distance
  const orbit = createOrbit({ target: [0, 0, 0.25], ...START, minDistance: 0.6, maxDistance: 8 });
  const poses = createPoseBuffer();
  const quality = createQuality(); // the internal resolution scale; its rule is in quality.js
  let scene = null, vaos = new Map(), shift = [0, 0, 1], shiftGoal = [0, 0, 1], raf = 0, dirty = true, active = true;
  const extraVaos = new Map(); // scene-event shapes by type and size
  let size = { w: canvas.clientWidth, h: canvas.clientHeight }; // CSS pixels, kept by the ResizeObserver below
  // The follow camera: the trunk's position and heading, smoothed, so the dog keeps its place and size on screen and
  // the floor moving under it shows the walking. The person's drag turns the view around the dog (orbit.yaw).
  const FOLLOW_TAU_MS = 600;
  let follow = null, lastDraw = 0; // {x, y, heading}

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
    follow = null;
    dirty = true;
  }

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2) * quality.scale;
    const w = Math.max(1, Math.round(size.w * dpr)), h = Math.max(1, Math.round(size.h * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; dirty = true; }
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (shift.some((v, k) => Math.abs(shiftGoal[k] - v) > 1e-3)) {
      shift = reduceMotion.matches ? [...shiftGoal] : shift.map((v, k) => v + (shiftGoal[k] - v) * 0.12);
      dirty = true;
    }
    // Draw only when something changed: a pose pair still being interpolated, a camera move, a resize, a new robot.
    if (!dirty && !(scene && poses.moving())) return;
    dirty = false;
    const dt = Math.min(100, now - lastDraw);
    lastDraw = now;
    quality.frame(now);
    resize();
    const pose = scene ? poses.sample(now) : null;
    const p = pose && scene.rootBody !== null ? pose.pos[scene.rootBody] : null;
    if (p) {
      // The camera looks a little below the trunk, and rises with it (a ramp lifts the dog off the floor plane).
      const goal = { x: p[0], y: p[1], z: Math.max(scene.floorZ + 0.22, p[2] - 0.1), heading: headingOf(pose.quat[scene.rootBody]), d: START.distance };
      if (frameScene) {
        // The dog and every object within 15 m of it: the camera aims at the middle of them and backs off to fit them.
        const near = footprints(pose.scene, scene.floorZ).filter(([x, y, r]) => Math.hypot(x - p[0], y - p[1]) - r < 15);
        if (near.length) {
          const xs = [p[0] - 0.4, p[0] + 0.4, ...near.flatMap(([x, , r]) => [x - r, x + r])], ys = [p[1] - 0.4, p[1] + 0.4, ...near.flatMap(([, y, r]) => [y - r, y + r])];
          const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
          goal.x = (x0 + x1) / 2; goal.y = (y0 + y1) / 2;
          goal.d = Math.max(START.distance, START.fit * Math.hypot(x1 - x0, y1 - y0));
        }
        // An object above the dog (a punch glancing up off its head) stays in frame too: back off with its height.
        const top = Math.max(-Infinity, ...(pose.scene?.objects ?? []).filter((o) => Array.isArray(o?.pos) && Number.isFinite(o.pos[2]) && Number.isFinite(o.size?.[0]))
          .map((o) => o.pos[2] + Math.abs(o.size[0])));
        if (top - goal.z > 0) goal.d = Math.max(goal.d, 3.7 * (top - goal.z));
      }
      // A jump (a new episode placed the dog elsewhere) moves the camera at once instead of sweeping through the scene.
      const jump = !follow || reduceMotion.matches || Math.hypot(p[0] - follow.dog[0], p[1] - follow.dog[1]) > 1;
      if (jump) follow = { ...goal, dog: [p[0], p[1]] };
      else {
        const heading = lockHeading ? follow.heading : smoothAngle(follow.heading, goal.heading, dt, FOLLOW_TAU_MS);
        follow = { x: smooth(follow.x, goal.x, dt, FOLLOW_TAU_MS), y: smooth(follow.y, goal.y, dt, FOLLOW_TAU_MS), z: smooth(follow.z, goal.z, dt, FOLLOW_TAU_MS), heading, d: smooth(follow.d, goal.d, dt, FOLLOW_TAU_MS), dog: [p[0], p[1]] };
        if (Math.hypot(goal.x - follow.x, goal.y - follow.y, goal.z - follow.z, goal.d - follow.d) > 1e-4 || (!lockHeading && Math.abs(Math.sin(goal.heading - follow.heading)) > 1e-4)) dirty = true; // still catching up
      }
      orbit.setTarget([follow.x, follow.y, follow.z]);
      if (frameScene) orbit.distance = follow.d * zoomK;
      // Never from inside an obstacle: while the eye would sit in one, look from higher up.
      for (let k = 0; k < 12 && inside(pose.scene, orbit.eye(follow.heading)); k++) orbit.rotate(0, 0.08);
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(...sky, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const proj = perspective((35 * Math.PI) / 180, canvas.width / canvas.height, 0.05, 80);
    proj[8] = -shift[0]; // lens shift: the dog sits in the free part of the stage, not under the text
    proj[9] = -shift[1];
    proj[0] *= shift[2]; // framing scale: a smaller free region draws the dog smaller, the person's zoom untouched
    proj[5] *= shift[2];
    const eye = orbit.eye(follow ? follow.heading : 0);
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
      gl.uniform4f(robotP.u.uColor, ...(trained ? d.trainedRgba : d.rgba));
      gl.bindVertexArray(g.vao);
      gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
    }
    // The scene event's objects and markers, in the world frame: opaque ones first, then see-through ones blended.
    const items = sceneItems(pose.scene);
    for (const pass of [false, true]) {
      if (pass) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
      for (const it of items) {
        if (it.transparent !== pass) continue;
        const g = shapeVao(it.type, it.size);
        if (!g) continue;
        gl.uniformMatrix4fv(robotP.u.uModel, false, it.model);
        gl.uniformMatrix3fv(robotP.u.uNormalMat, false, normalMatrix(it.model));
        gl.uniform4f(robotP.u.uColor, ...it.rgba);
        gl.bindVertexArray(g.vao);
        gl.drawElements(gl.TRIANGLES, g.count, gl.UNSIGNED_INT, 0);
      }
      if (pass) { gl.disable(gl.BLEND); gl.depthMask(true); }
    }
    gl.bindVertexArray(null);
  }

  function shapeVao(type, sz) {
    const key = `${type}:${sz.join(",")}`;
    if (!extraVaos.has(key)) {
      const p = primitive(type, sz);
      if (!p) return null;
      if (extraVaos.size >= 64) { const [k, v] = extraVaos.entries().next().value; gl.deleteVertexArray(v.vao); extraVaos.delete(k); }
      extraVaos.set(key, upload(p));
    }
    return extraVaos.get(key);
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
      if (pinch) zoom(pinch / d);
      pinch = d;
    }
    dirty = true;
  });
  const lift = (e) => { pointers.delete(e.pointerId); pinch = null; };
  canvas.addEventListener("pointerup", lift);
  canvas.addEventListener("pointercancel", lift);
  // A plain wheel scrolls the page; Ctrl or Cmd with the wheel (and a trackpad pinch, which arrives that way) zooms.
  canvas.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    zoom(Math.exp(e.deltaY * 0.004));
    dirty = true;
  }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const keys = { ArrowLeft: () => orbit.rotate(0.12, 0), ArrowRight: () => orbit.rotate(-0.12, 0), ArrowUp: () => orbit.rotate(0, 0.08),
      ArrowDown: () => orbit.rotate(0, -0.08), "+": () => zoom(0.9), "=": () => zoom(0.9), "-": () => zoom(1.1), 0: resetView };
    if (!keys[e.key]) return;
    e.preventDefault();
    keys[e.key]();
    dirty = true;
  });

  function zoom(f) { if (frameScene) zoomK = Math.min(3, Math.max(0.35, zoomK * f)); else orbit.zoom(f); }
  function resetView() { Object.assign(orbit, { distance: START.distance, yaw: START.yaw, pitch: START.pitch }); zoomK = 1; dirty = true; }

  // ---- lifecycle: no drawing while the tab is hidden or the view is inactive (off screen) -----------------------
  const run = () => {
    cancelAnimationFrame(raf);
    raf = 0;
    if (active && !document.hidden) { dirty = true; raf = requestAnimationFrame(frame); }
  };
  document.addEventListener("visibilitychange", run);
  const sizer = new ResizeObserver(([e]) => {
    const box = e.contentBoxSize?.[0];
    size = box ? { w: box.inlineSize, h: box.blockSize } : { w: e.contentRect.width, h: e.contentRect.height };
    dirty = true;
  });
  sizer.observe(canvas);
  run();

  return {
    setRobot,
    pushPose(pose) { if (scene && Array.isArray(pose?.pos) && Array.isArray(pose?.quat)) { poses.push(pose, performance.now()); dirty = true; } },
    setScene(s) { poses.pushScene(s); dirty = true; },
    setActive(v) { if (active !== Boolean(v)) { active = Boolean(v); run(); } },
    resolutionScale: () => quality.scale,
    hasPose: () => Boolean(scene && poses.peek()),
    bodyCount: () => scene?.bodyCount ?? 0,
    setShift(x, y = 0, k = 1) { shiftGoal = [x, y, k]; },
    zoomBy(f) { zoom(f); dirty = true; },
    // Forget the poses drawn so far (a new run starts): the view shows the floor until the next pose.
    clear() { poses.clear(); follow = null; dirty = true; },
    resetView,
    // A new starting angle around the dog (radians from its heading), taken at once; the person's drag still turns it.
    // fit: how far the framing backs off per metre of the scene's footprint (frameScene).
    setYaw(yaw, fit = 1.45) { if (START.yaw !== yaw || START.fit !== fit) { START.yaw = yaw; START.fit = fit; orbit.yaw = yaw; dirty = true; } },
    destroy() { cancelAnimationFrame(raf); document.removeEventListener("visibilitychange", run); sizer.disconnect(); },
  };
}
