"""Vector stores + a retriever tool for RAG.

Backends are pluggable: the default is in-memory (no extra deps). Add your own with
`register_backend("pgvector", factory)`, where factory(embeddings, name) -> VectorStore.
Select with `get_vector_store(backend=...)` or the VECTOR_BACKEND env var.
"""

from collections.abc import Callable
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .embed import get_embeddings
from .getenv import get_env
from .log import logger

BackendFactory = Callable[[Embeddings, str], VectorStore]


def _memory(embeddings: Embeddings, name: str) -> VectorStore:
    return InMemoryVectorStore(embeddings)


def _chroma(embeddings: Embeddings, name: str) -> VectorStore:
    from langchain_chroma import Chroma  # pip install langchain-chroma

    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=get_env("CHROMA_PATH"),  # None = ephemeral
    )


## TODO: add postgres (and supabase if possible)

_BACKENDS: dict[str, BackendFactory] = {"memory": _memory, "chroma": _chroma}


def register_backend(name: str, factory: BackendFactory) -> None:
    _BACKENDS[name] = factory
    get_vector_store.cache_clear()


@lru_cache(maxsize=None)
def get_vector_store(name: str = "default", backend: str | None = None) -> VectorStore:
    """Same (name, backend) always returns the same store instance."""
    backend = backend or get_env("VECTOR_BACKEND", "memory")  # type: ignore[assignment]
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown vector backend '{backend}'. Available: {sorted(_BACKENDS)}"
        )
    logger.info("vectorstore: name=%s backend=%s", name, backend)
    return _BACKENDS[backend](get_embeddings(), name)


def retriever_tool(
    store: VectorStore | None = None,
    *,
    name: str = "search_knowledge",
    description: str = "Search the knowledge base for passages relevant to a query.",
    k: int = 4,
) -> BaseTool:
    """Tool for `new_agent(..., tools=[retriever_tool(store)])` (RAG)."""
    store = store or get_vector_store()

    def _format(docs) -> str:
        if not docs:
            return "No relevant results found."
        return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

    def search(query: str) -> str:
        return _format(store.similarity_search(query, k=k))

    async def asearch(query: str) -> str:
        return _format(await store.asimilarity_search(query, k=k))

    return StructuredTool.from_function(
        func=search, coroutine=asearch, name=name, description=description
    )
