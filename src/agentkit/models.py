"""Chat model construction (Gemini via its OpenAI-compatible endpoint, through the relay)."""

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from .getenv import get_api_key, get_env
from .relay import get_async_http_client, get_http_client

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.6-flash"


def _build(model: str | None = None, **settings: Any) -> ChatOpenAI:
    # Precedence: explicit `model` argument > LLM_MODEL env > DEFAULT_MODEL
    return ChatOpenAI(
        model=model or get_env("LLM_MODEL", DEFAULT_MODEL),  # type: ignore[arg-type]
        api_key=get_api_key("GOOGLE"),  # type: ignore[arg-type]
        base_url=GEMINI_OPENAI_URL,
        http_client=get_http_client(),
        http_async_client=get_async_http_client(),
        **settings,
    )


@lru_cache(maxsize=None)  # one shared instance per model name
def _default(model: str | None = None) -> ChatOpenAI:
    return _build(model)


def get_model(model: str | None = None, **settings: Any) -> ChatOpenAI:
    """Shared instance for a model name, or a fresh one when ChatOpenAI kwargs are
    given (e.g. temperature=0.2, top_p=0.9)."""
    return _build(model, **settings) if settings else _default(model)
