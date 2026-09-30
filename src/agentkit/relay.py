"""HTTP clients that route every request through a relay (e.g. a Cloudflare Worker).

The original destination is sent in `x-relay-target` / `x-relay-path` headers.
"""

from functools import lru_cache
from urllib.parse import urlparse

import httpx

from .getenv import get_relay_url
from .log import logger


def _rewrite(request: httpx.Request, relay: str) -> None:
    parsed = urlparse(str(request.url))

    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    target = f"{parsed.scheme}://{parsed.netloc}"
    logger.debug("relay: %s%s", target, parsed.path)

    request.url = httpx.URL(relay)
    request.headers["host"] = urlparse(relay).netloc
    request.headers["x-relay-target"] = target
    request.headers["x-relay-path"] = path


class RelayTransport(httpx.HTTPTransport):
    def __init__(self, relay_url: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.relay_url = relay_url

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _rewrite(request, self.relay_url)
        return super().handle_request(request)


class AsyncRelayTransport(httpx.AsyncHTTPTransport):
    def __init__(self, relay_url: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.relay_url = relay_url

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _rewrite(request, self.relay_url)
        return await super().handle_async_request(request)


def create_http_client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        transport=RelayTransport(get_relay_url()),
        timeout=timeout,
        follow_redirects=True,
    )


def create_async_client(timeout: float = 60.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=AsyncRelayTransport(get_relay_url()),
        timeout=timeout,
        follow_redirects=True,
    )


@lru_cache(maxsize=1)
def get_http_client() -> httpx.Client:
    """Shared sync client (connection pooling)."""
    return create_http_client()


@lru_cache(maxsize=1)
def get_async_http_client() -> httpx.AsyncClient:
    """Shared async client."""
    return create_async_client()
