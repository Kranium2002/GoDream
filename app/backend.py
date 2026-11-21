"""The one place the viewer server touches Dev B's code, so a change in Dev B's contract changes this file only.

Agreed with Dev B (message log, 2026-11-21 20:55 UTC): one module learned_judgment.declared with
    skill_catalog(config)        -> [{skill, description, params, actions, outcomes, available, missing}]
    declaration_schema(config)   -> JSON Schema of a declaration {skill, params, source_text}
    check_declaration(d, config) -> [{code, field, message}], empty when valid
    resolve_declaration(d, config), parse_request(text, config): not used by the server yet.
If it fails to import, the catalogue lists config.yaml's skills, each unavailable, naming what is missing; no
parameter, action or outcome is made up here. Also here: the run files and runner (runs_dir, runner), the image-window
contract (image_window) and live inference through learned_judgment.runtime (inference).
"""
import importlib
import logging
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
    # WIRE: Dev B's runner entry point goes here once published; none is named yet, so none is guessed.
    return None, ["Dev B's learning-run entry point with its TabPFN cost quote (not published yet)"]


def image_window(config: dict):
    """The image-window contract for the browser, from config.yaml vision (Dev B), in the encoding agreed with the
    front lane; None without a vision block."""
    v = config.get("vision")
    if not v:
        return None
    return {"frames": v["frames"], "height": v["frame_size"], "width": v["frame_size"], "hz": v["frame_rate_hz"],
            "order": "oldest_first", "normalization": v["pixel_scale"], "crop": "centre_square",
            "luma": v["luma_weights"], "mirrored": False}


RUNTIME = "learned_judgment.runtime"
_runtime_lock = threading.Lock()  # Dev B's runtime keeps gate and cooldown state: one call at a time
_loaded = {}  # "path": the model path learned_judgment.runtime has loaded


def inference(config: dict):
    """(infer(window, t_capture_s) -> dict, reset(), image_window, []) once Dev B's runtime imports,
    config.yaml app.infer.model_path names a frozen model that loads and config.yaml has vision; else
    (None, None, image_window, [what is missing]).

    Agreed with Dev B: learned_judgment.runtime.load(path) once, then infer(window, t_capture_s) per window (float32
    (frames, height, width), oldest first, in [0, 1]) and reset() when a camera session starts. If Dev B's shape
    differs, this function is the one place to change.
    """
    window = image_window(config)
    missing = [] if window else ["config.yaml vision (Dev B)"]
    try:
        runtime = importlib.import_module(RUNTIME)
    except ImportError as e:
        runtime = None
        missing.append(f"{RUNTIME} (Dev B, not pushed yet)" if e.name == RUNTIME
                       else f"{RUNTIME} (Dev B) does not import: it needs {e.name}")
    else:
        missing += [f"{RUNTIME}.{name} (Dev B)" for name in ("load", "infer", "reset")
                    if not callable(getattr(runtime, name, None))]
    path = config["app"]["infer"]["model_path"]
    if not path:
        missing.append("config.yaml app.infer.model_path naming the frozen SNN export from Dev B")
    elif not (ROOT / path).is_file():
        missing.append("the frozen model file that config.yaml app.infer.model_path names")
    if missing:
        return None, None, window, missing
    with _runtime_lock:
        if _loaded.get("path") != path:
            try:
                runtime.load(str(ROOT / path))
            except Exception:  # shortcut: retried on every call while broken; fine until the model is handed over
                log.exception("The frozen model at app.infer.model_path does not load")
                return None, None, window, ["a frozen model at app.infer.model_path that loads (see the server log)"]
            _loaded["path"] = path

    def run(frames, t_capture_s):
        with _runtime_lock:
            return runtime.infer(frames, t_capture_s)

    def reset():
        with _runtime_lock:
            runtime.reset()

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
