"""PLUMBING wrapper of the pipeline command line for c6: ``learned_judgment.pipeline.main`` with the
fixture opener (fixture skill on the fixture world) instead of open_run (the Go2 simulator).

Usage (from the repository root):
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\fixture_cli.py [--fixture-stop-after N] <pipeline arguments>

``--fixture-stop-after N`` makes should_stop() true once N episodes of the current command have
finished, to exercise the stopped exit code. Every billable tabpfn-client entry point is replaced
by a tripwire before main runs, so this process cannot send a prediction request. Fixture data
only; nothing here is evidence about the robot, the simulator, TabPFN or any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import fixture_opener  # noqa: E402

import tabpfn_client  # noqa: E402
from tabpfn_client.client import ServiceClient  # noqa: E402
from tabpfn_client.service_wrapper import InferenceClient  # noqa: E402

from learned_judgment.pipeline import main  # noqa: E402


def _tripwire(label: str):  # noqa: ANN202
    def tripwire(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError(f"billable entry point {label} was called")

    return tripwire


for owner, names in (
    (tabpfn_client.TabPFNRegressor, ("fit", "_fit", "predict", "_predict", "submit_fit", "submit_predict", "wait_for_predict")),
    (InferenceClient, ("fit", "fit_with_result", "predict", "predict_chunks", "submit_predict")),
    (ServiceClient, ("fit", "fit_with_result", "_fit", "predict", "predict_chunks", "submit_predict", "_predict",
                     "_upload_train_set", "_upload_test_set", "_submit_fit_job", "_submit_predict_job")),
):
    for name in names:
        if hasattr(owner, name):
            setattr(owner, name, _tripwire(f"{owner.__name__}.{name}"))

arguments = sys.argv[1:]
stop_after = None
if arguments[:1] == ["--fixture-stop-after"]:
    stop_after = int(arguments[1])
    arguments = arguments[2:]


def opener(config, skill_name, **kwargs):  # noqa: ANN001, ANN003, ANN201
    if stop_after is not None:
        finished = {"episodes": 0}
        progress, should_stop = kwargs["progress"], kwargs["should_stop"]

        def counting(event):  # noqa: ANN001, ANN202
            if event["event"] == "progress":
                finished["episodes"] += 1
            progress(event)

        kwargs["progress"] = counting
        kwargs["should_stop"] = lambda: should_stop() or finished["episodes"] >= stop_after
    return fixture_opener(config, skill_name, **kwargs)


print("PLUMBING CHECK wrapper: pipeline.main with the fixture world; not the Go2 simulator", flush=True)
raise SystemExit(main(arguments, opener=opener))
