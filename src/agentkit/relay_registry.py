from typing import Callable
from .relay import get_async_http_client, get_http_client

_RELAY_HANDLERS: dict[type, Callable[[], dict]] = {}


def register_google():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from google import genai
        from google.genai import types

        def handler() -> dict:
            http_options = types.HttpOptions(
                httpx_client=get_http_client(),
                httpx_async_client=get_async_http_client(),
            )
            return {"client": genai.Client(http_options=http_options)}

        _RELAY_HANDLERS[ChatGoogleGenerativeAI] = handler
    except ImportError:
        pass


def get_handler_for(probe):
    for cls, handler in _RELAY_HANDLERS.items():
        if isinstance(probe, cls):
            return handler
