"""Environment access. Everything is read lazily, never at import time."""

import os

from dotenv import find_dotenv, load_dotenv

# usecwd=True: look for .env from the directory the app is run in (walking up),
load_dotenv(find_dotenv(usecwd=True))


def get_env(
    name: str, default: str | None = None, *, required: bool = False
) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"{name} is not set in environment or .env file")
    return value


def get_api_key(provider: str = "") -> str:
    name = f"{provider}_API_KEY" if provider else "API_KEY"
    return get_env(name, required=True)  # type: ignore[return-value]


def get_relay_url() -> str:
    return get_env("RELAY_URL", required=True)  # type: ignore[return-value]
