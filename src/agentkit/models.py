"""Chat model construction (Gemini via its OpenAI-compatible endpoint, through the relay)."""

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model

from .getenv import get_env
from .relay import get_async_http_client, get_http_client

DEFAULT_MODEL = "google_vertexai:gemini-3.6-flash"


def _build(model: str | None = None, **settings: Any):
    # Precedence: explicit `model` argument > LLM_MODEL env > DEFAULT_MODEL
    # write default model as "openai:gpt-4o", "google_vertexai:gemini-3.6-flash"
    return init_chat_model(
        model=model or get_env("LLM_MODEL", DEFAULT_MODEL)
        configurable_fields=("model", "model_provider"),
        http_client=get_http_client(),
        http_async_client=get_async_http_client(),
        **settings,
    )


@lru_cache(maxsize=None)  # one shared instance per model name
def _default(model: str | None = None):
    return _build(model)


def get_model(model: str | None = None, **settings: Any):
    """Shared instance for a model name, or a fresh one when ChatOpenAI kwargs are
    given (e.g. temperature=0.2, top_p=0.9)."""
    return _build(model, **settings) if settings else _default(model)
