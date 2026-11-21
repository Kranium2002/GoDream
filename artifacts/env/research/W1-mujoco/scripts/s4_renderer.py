# Item 4: does mujoco.Renderer give an offscreen RGB frame with no visible window; which GL backend.
import os, sys, hashlib, traceback
print("MUJOCO_GL env:", repr(os.environ.get("MUJOCO_GL")))
import mujoco, numpy as np
print("mujoco", mujoco.__version__)
print("GLContext class:", mujoco.GLContext.__module__, mujoco.GLContext.__name__)
scene = sys.argv[1]
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
print("offscreen buffer (offwidth x offheight):", m.vis.global_.offwidth, m.vis.global_.offheight)
try:
    r = mujoco.Renderer(m, height=240, width=320)
    r.update_scene(d)  # default free camera
    img = r.render()
    print("frame:", type(img).__name__, img.shape, img.dtype, "min", img.min(), "max", img.max(),
          "std", round(float(img.std()), 2), "sha256", hashlib.sha256(img.tobytes()).hexdigest()[:16])
    # which OpenGL implementation is behind the GLFW context
    from OpenGL import GL
    with r._gl_context.as_current():
        print("GL_VENDOR:", GL.glGetString(GL.GL_VENDOR).decode())
        print("GL_RENDERER:", GL.glGetString(GL.GL_RENDERER).decode())
        print("GL_VERSION:", GL.glGetString(GL.GL_VERSION).decode())
    import glfw
    print("glfw lib version:", glfw.get_version_string().decode() if isinstance(glfw.get_version_string(), bytes) else glfw.get_version_string())
    print("window visible attrib:", glfw.get_window_attrib(r._gl_context._context, glfw.VISIBLE))
    # depth too
    r.enable_depth_rendering(); dep = r.render(); r.disable_depth_rendering()
    print("depth:", dep.shape, dep.dtype, "min", round(float(dep.min()), 3), "max", round(float(dep.max()), 3))
    if len(sys.argv) > 2:
        try:
            from PIL import Image  # optional
        except ImportError:
            # write a binary PPM so the frame can be inspected without extra deps
            with open(sys.argv[2], "wb") as f:
                f.write(b"P6 %d %d 255\n" % (img.shape[1], img.shape[0])); f.write(img.tobytes())
            print("wrote", sys.argv[2])
    r.close()
    print("RENDER_OK")
except Exception:
    traceback.print_exc()
    print("RENDER_FAILED")
