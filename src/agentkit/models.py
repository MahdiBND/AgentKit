"""Chat model construction (Gemini via its OpenAI-compatible endpoint, through the relay)."""

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model, BaseChatModel
from .relay_registry import get_handler_for, register_google
from .getenv import get_env

register_google()


def _build(model: str | None = None, **settings: Any) -> BaseChatModel:
    # Precedence: explicit `model` argument > LLM_MODEL env > DEFAULT_MODEL
    DEFAULT_MODEL = "google_genai:gemini-3.6-flash"  # in case nothing is set anywhere

    model_id = model or get_env("LLM_MODEL", DEFAULT_MODEL)

    return _config_relay(model_id, **settings)


def _config_relay(model_id, **settings) -> BaseChatModel:
    probe = init_chat_model(model_id, **settings)
    handler = get_handler_for(probe)

    if handler:
        return init_chat_model(model_id, **handler(), **settings)

    return probe


@lru_cache(maxsize=None)  # one shared instance per model name
def _default(model: str | None = None) -> BaseChatModel:
    return _build(model)


def get_model(model: str | None = None, **settings: Any) -> BaseChatModel:
    """Shared instance for a model name, or a fresh one when ChatOpenAI kwargs are
    given (e.g. temperature=0.2, top_p=0.9)."""
    return _build(model, **settings) if settings else _default(model)
