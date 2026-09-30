"""Chat model construction (Gemini via its OpenAI-compatible endpoint, through the relay)."""

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from .getenv import get_api_key, get_env
from .relay import get_async_http_client, get_http_client

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _build(model: str = "gemini-3.6-flash", **settings: Any) -> ChatOpenAI:
    return ChatOpenAI(
        model=get_env("LLM_MODEL", model),  # type: ignore[arg-type]
        api_key=get_api_key("GOOGLE"),  # type: ignore[arg-type]
        base_url=GEMINI_OPENAI_URL,
        http_client=get_http_client(),
        http_async_client=get_async_http_client(),
        **settings,
    )


@lru_cache(maxsize=1)
def _default(model="gemini-3.6-flash") -> ChatOpenAI:
    return _build(model=model)


def get_model(model="gemini-3.6-flash", **settings: Any) -> ChatOpenAI:
    """Shared default model, or a fresh one when ChatOpenAI kwargs are given
    (e.g. temperature=0.2, top_p=0.9)."""
    return _build(model=model, **settings) if settings else _default(model=model)
