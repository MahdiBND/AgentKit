"""Gemini embeddings via langchain-google-genai, routed through the relay."""

from collections.abc import Sequence
from functools import lru_cache

from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .getenv import get_api_key, get_env
from .log import logger
from .relay import get_async_http_client, get_http_client

DEFAULT_DIMENSIONS = 1536


def _default_model() -> str:
    return get_env("EMBED_MODEL", "gemini-embedding-2")  # type: ignore[return-value]


@lru_cache(maxsize=8)
def get_embeddings(
    model: str | None = None,
    dimensions: int = DEFAULT_DIMENSIONS,
    task_type: str | None = None,
) -> GoogleGenerativeAIEmbeddings:
    """LangChain Embeddings object (use it directly with vector stores)."""
    api_key = get_api_key("GOOGLE")
    emb = GoogleGenerativeAIEmbeddings(
        model=model or _default_model(),
        google_api_key=api_key,
        output_dimensionality=dimensions,
        task_type=task_type,
    )
    # `client_args` is shared by the sync and async clients, so it can't carry a
    # sync-only transport. Swap in a genai client wired to the relay instead.
    emb.client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            httpx_client=get_http_client(),
            httpx_async_client=get_async_http_client(),
        ),
    )
    return emb


def embed(
    input: str | Sequence[str],
    *,
    model: str | None = None,
    dimensions: int = DEFAULT_DIMENSIONS,
    task_type: str | None = None,
) -> list[float] | list[list[float]]:
    """str -> one vector; list[str] -> list of vectors."""
    emb = get_embeddings(model, dimensions, task_type)
    if isinstance(input, str):
        logger.debug("embed: 1 query, dims=%d", dimensions)
        return emb.embed_query(input)
    logger.debug("embed: %d documents, dims=%d", len(input), dimensions)
    return emb.embed_documents(list(input))


async def aembed(
    input: str | Sequence[str],
    *,
    model: str | None = None,
    dimensions: int = DEFAULT_DIMENSIONS,
    task_type: str | None = None,
) -> list[float] | list[list[float]]:
    emb = get_embeddings(model, dimensions, task_type)
    if isinstance(input, str):
        return await emb.aembed_query(input)
    return await emb.aembed_documents(list(input))
