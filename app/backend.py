"""The one place the viewer server touches Dev B's code, so a change in Dev B's contract changes this file only.

Agreed with Dev B (message log, 2026-11-21 20:55 UTC): one module learned_judgment.declared with
    skill_catalog(config)        -> [{skill, description, params, actions, outcomes, available, missing}]
    declaration_schema(config)   -> JSON Schema of a declaration {skill, params, source_text}
    check_declaration(d, config) -> [{code, field, message}], empty when valid
    resolve_declaration(d, config), parse_request(text, config): not used by the server yet.
If it fails to import, the catalogue lists config.yaml's skills, each unavailable, naming what is missing; no
parameter, action or outcome is made up here. Also here: the run files and runner (runs_dir, runner), the image-window
contract (image_window), the DODGE actions (actions) and live inference through
learned_judgment.runtime (inference).
"""
import importlib
import logging
import math
import threading
from pathlib import Path

from learned_judgment.skills import available_skills

ROOT = Path(__file__).resolve().parents[1]  # the repository root; config.yaml paths are relative to it
MODULE = "learned_judgment.declared"
NEEDED = ("skill_catalog", "declaration_schema", "check_declaration")
log = logging.getLogger(__name__)


def declared():
    """(Dev B's declaration module, []) once it imports with everything the server needs, else (None, [what is
    missing, one plain phrase each])."""
    try:
        module = importlib.import_module(MODULE)
    except ImportError as e:
        if e.name == MODULE:
            return None, [f"{MODULE} (Dev B, not pushed yet)"]
        log.warning("%s does not import: %s", MODULE, e)
        return None, [f"{MODULE} (Dev B) does not import: it needs {e.name}"]
    absent = [name for name in NEEDED if not callable(getattr(module, name, None))]
    if absent:
        return None, [f"{MODULE}.{name} (Dev B)" for name in absent]
    return module, []


def runs_dir(config: dict):
    """(the directory of run folders, []) from config.yaml observed.runs_dir (Dev B), relative to the repository root,
    or (None, [what is missing]). Each run is a folder <run_id>/ with events.jsonl and result.json."""
    name = (config.get("observed") or {}).get("runs_dir")
    return (ROOT / name, []) if name else (None, ["config.yaml observed.runs_dir (Dev B)"])


def runner(config: dict):
    """(runner, []) once Dev B's learning-run entry point is wired here, else (None, [what is missing]).

    A runner has quote(declaration, config) -> dict, the TabPFN usage and cost a run would spend, shown to the person
    before anything is spent; and start(declaration, config) -> run_id, which returns at once: the run itself goes in
    a process of its own, writing <runs_dir>/<run_id>/events.jsonl and result.json, so it never holds up the
    simulation thread or the HTTP server.
    """
    # WIRE: Dev B's runner entry point goes here once published. learned_judgment.observed (2026-11-21) is not one yet:
    # its quote, teach and train commands need rows and dream files collected beforehand by
    # learned_judgment.skills.dodge_vision, take no declaration, and nothing freezes the gate thresholds afterwards.
    return None, ["Dev B's learning-run entry point that starts a run from a declaration and quotes its TabPFN cost "
                  "first (learned_judgment.observed needs pre-collected rows and dream files and takes no declaration)"]


def image_window(config: dict):
    """The image-window contract for the browser, from config.yaml vision (Dev B), in the encoding agreed with the
    front lane; None without a vision block."""
    v = config.get("vision")
    if not v:
        return None
    return {"frames": v["frames"], "height": v["frame_size"], "width": v["frame_size"], "hz": v["frame_rate_hz"],
            "order": "oldest_first", "normalization": v["pixel_scale"], "crop": "centre_square",
            "luma": v["luma_weights"], "mirrored": False}


def actions(config: dict):
    """(the actions with their commands, hold_s) from config.yaml skills.dodge_vision (Dev B), unchanged; (None, None)
    without them."""
    dodge = (config.get("skills") or {}).get("dodge_vision") or {}
    return dodge.get("actions"), dodge.get("hold_s")


RUNTIME = "learned_judgment.runtime"
_runtime_lock = threading.Lock()  # Dev B's runtime keeps gate and cooldown state: one call at a time
_loaded = {}  # the loaded student: "dir", "model" (a DodgeRuntime), "window" and "last_t" (its newest capture time)


def inference(config: dict):
    """(infer(window, t_capture_s) -> dict, reset(), window size {frames, height, width}, []) once Dev B's runtime has
    loaded the frozen student that config.yaml app.infer.student_dir names; else (None, None, None, [missing]).

    learned_judgment.runtime.DodgeRuntime.load(directory, config) reads student.json, network.npz and network.json,
    which `python -m learned_judgment.observed train` writes to <observed.runs_dir>/<run>/student/, and needs gate
    thresholds frozen in student.json. The window size is the student's (student.json input), and must match
    config.yaml vision, which the browser follows. Its infer(window, t_capture_s) refuses a time earlier than the
    previous call's, so the times passed on never go backwards; reset() starts a new camera session. One student is
    loaded per server and every call holds one lock: the gate keeps state.
    """
    published = image_window(config)
    missing = [] if published else ["config.yaml vision (Dev B)"]
    try:
        runtime = importlib.import_module(RUNTIME)
    except ImportError as e:
        runtime = None
        missing.append(f"{RUNTIME} (Dev B, not pushed yet)" if e.name == RUNTIME
                       else f"{RUNTIME} (Dev B) does not import: it needs {e.name}")
    else:
        if not hasattr(runtime, "DodgeRuntime"):
            missing.append(f"{RUNTIME}.DodgeRuntime (Dev B)")
    directory = config["app"]["infer"]["student_dir"]
    if not directory:
        missing.append("config.yaml app.infer.student_dir naming a frozen student directory, which Dev B's observed "
                       "run exports (none exists yet)")
    elif not (ROOT / directory / "student.json").is_file():
        missing.append("student.json in the directory config.yaml app.infer.student_dir names")
    if missing:
        return None, None, None, missing
    with _runtime_lock:
        if _loaded.get("dir") != directory:
            try:
                model = runtime.DodgeRuntime.load(ROOT / directory, config)
            except RuntimeError:  # Dev B's "no frozen thresholds" (shortcut: retried on every call until fixed)
                return None, None, None, ["gate thresholds frozen in the student's student.json (Dev B calibrates "
                                          "them on simulation dev trajectories)"]
            except Exception:
                log.exception("The student at app.infer.student_dir does not load")
                return None, None, None, ["a student at app.infer.student_dir that loads (the server log has why)"]
            if (model.frames, model.frame_size) != (published["frames"], published["height"]):
                return None, None, None, [f"a student whose window ({model.frames} frames of {model.frame_size} x "
                                          f"{model.frame_size}) matches config.yaml vision"]
            _loaded.update(dir=directory, model=model, last_t=-math.inf,
                           window={"frames": model.frames, "height": model.frame_size, "width": model.frame_size})
        model, window = _loaded["model"], _loaded["window"]

    def run(frames, t_capture_s):
        with _runtime_lock:
            t = _loaded["last_t"] = max(t_capture_s, _loaded["last_t"])  # a skewed frame age must not step back
            return model.infer(frames, t)

    def reset():
        with _runtime_lock:
            model.reset()
            _loaded["last_t"] = -math.inf

    return run, reset, window, []


def skill_catalog(config: dict) -> list:
    """Dev B's catalogue unchanged, or every skill named in config.yaml skills marked unavailable with the reason."""
    module, missing = declared()
    if module is not None:
        return module.skill_catalog(config)
    registered = set(available_skills())  # the existing SkillSpec registry (learned_judgment.skills)
    return [{"skill": name.upper(), "description": "", "params": {}, "actions": [], "outcomes": [],
             "available": False,
             "missing": missing + ([] if name in registered else [f"{name.upper()} SkillSpec (Dev B)"])}
            for name in config["skills"]]
