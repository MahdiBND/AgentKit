"""Langfuse tracing. Auto-enabled when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set."""
from .getenv import get_env
from .log import logger


def tracing_enabled() -> bool:
    return bool(get_env("LANGFUSE_PUBLIC_KEY") and get_env("LANGFUSE_SECRET_KEY"))


def get_callbacks() -> list:
    if not tracing_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as e:  # never let observability break the app
        logger.warning("Langfuse unavailable, continuing without tracing: %s", e)
        return []


def trace_metadata(session_id: str | None = None, user_id: str | None = None,
                   tags: list[str] | None = None) -> dict:
    meta: dict = {}
    if session_id:
        meta["langfuse_session_id"] = session_id
    if user_id:
        meta["langfuse_user_id"] = user_id
    if tags:
        meta["langfuse_tags"] = tags
    return meta
