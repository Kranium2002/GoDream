"""Check 7: the modules of the DODGE lane import cleanly and keep the lane's rules.

FIXTURE CHECK - mechanics only. Imports learned_judgment.skills and learned_judgment.skills.dodge
in a fresh interpreter, confirms their public names, and reads their source for two rules:
no print in library code, and no skill name in the registry module.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c7_imports.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import ast  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402

from fixture_dodge import Checks  # noqa: E402

checks = Checks()

print("\n== imports ==")
modules = {}
for name in ("learned_judgment.skills", "learned_judgment.skills.dodge"):
    module = importlib.import_module(name)
    modules[name] = module
    exported = getattr(module, "__all__", ())
    missing = [item for item in exported if not hasattr(module, item)]
    print(f"  {name}: __all__ = {list(exported)}")
    checks.check(f"{name} imports and exports every name in __all__", not missing, f"missing {missing}" if missing else "")

skills, dodge = modules["learned_judgment.skills"], modules["learned_judgment.skills.dodge"]
print("\n== public signatures ==")
for function in (skills.load_skill, skills.available_skills, dodge.DodgeSpec.__init__, dodge.DodgeSpec.sample_situations,
                 dodge.DodgeSpec.physics_guess, dodge.DodgeSpec.objective, dodge.DodgeSpec.episode,
                 dodge.DodgeSpec.loss, dodge.DodgeSpec.success):
    print(f"  {function.__qualname__}{inspect.signature(function)}")
checks.check("DodgeSpec is the registered SPEC_CLASS", dodge.SPEC_CLASS is dodge.DodgeSpec and dodge.DodgeSpec.skill_name == "dodge")
checks.check("DodgeSpec implements every abstract method", not inspect.isabstract(dodge.DodgeSpec))

print("\n== lane rules in the source ==")
for name, module in modules.items():
    tree = ast.parse(inspect.getsource(module))
    prints = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"]
    checks.check(f"{name}: no print call", not prints, f"lines {prints}" if prints else "")
registry_strings = [
    node.value for node in ast.walk(ast.parse(inspect.getsource(skills)))
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and "dodge" in node.value.lower()
]
checks.check("the registry module names no skill", not registry_strings, f"{registry_strings}" if registry_strings else "")

checks.finish()
