"""Authenticate tabpfn-client with the Prior Labs API key kept in this project's .env.

Usage:
    import prior_labs
    prior_labs.connect()
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

tabpfn-client only reads a credential from TABPFN_TOKEN and does not load .env files, so the
key stored here as PRIOR_LABS_API_KEY has to be handed over explicitly. connect() does that
inside the process: the key is never printed, logged, or written to disk.
"""

import os
from pathlib import Path

import httpx
from dotenv import dotenv_values

ENV_VAR = "PRIOR_LABS_API_KEY"
ENV_FILE = Path(__file__).resolve().parent / ".env"

_SDK_ENV_VAR = "TABPFN_TOKEN"


def connect(env_file: str | Path = ENV_FILE) -> None:
    """Check the API key with Prior Labs' server and authenticate this process.

    The key is taken from the PRIOR_LABS_API_KEY environment variable if set, otherwise from
    `env_file`. Safe to call more than once.

    Raises:
        RuntimeError: If the key is missing, rejected, belongs to an account with an
            unverified email, or the server cannot be reached.
    """
    key = os.environ.get(ENV_VAR, "").strip()
    source = f"the {ENV_VAR} environment variable"
    if not key:
        # interpolate=False: take the value exactly as written, even if it contains "${...}".
        key = (dotenv_values(env_file, interpolate=False).get(ENV_VAR) or "").strip()
        source = str(env_file)
    if not key:
        raise RuntimeError(f"{ENV_VAR} is not set in the environment or in {env_file}")

    import tabpfn_client

    # init() validates the token against the server; set_access_token() would not.
    os.environ[_SDK_ENV_VAR] = key
    try:
        tabpfn_client.init()
    except (RuntimeError, httpx.HTTPError) as exc:
        raise RuntimeError(
            f"Prior Labs authentication failed with the key from {source} "
            f"({type(exc).__name__}): {exc}"
        ) from exc
    finally:
        # The client now holds the token in memory, so keep it out of child processes.
        os.environ.pop(_SDK_ENV_VAR, None)
