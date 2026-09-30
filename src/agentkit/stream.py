import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI


from .log import logger
from .sessions import SessionStore
from .tracing import get_callbacks, trace_metadata


def _text(message: BaseMessage) -> str:
    text = message.text
    # Newer langchain-core: `.text` is a str subclass that is also callable (deprecated).
    # Check for str first so we never call it. Older versions return a bound method.
    if isinstance(text, str):
        return str(text)
    if callable(text):
        text = text()
    return text if isinstance(text, str) else str(message.content)


@dataclass
class StreamEvent:
    """One event from `Agent.stream` / `Agent.astream`.

    type        fields set
    ----------  ------------------------------------------------------------
    text        text     - a chunk of the model's answer as it is generated
    tool_call   name, args - the model decided to call a tool
    tool_result name, content - the tool finished
    final       output   - the complete result (str, or `output_type` instance);
                           always the last event
    """

    type: Literal["text", "tool_call", "tool_result", "final"]
    text: str = ""
    name: str | None = None
    args: dict | None = None
    content: Any = None
    output: Any = None


class _StreamState:
    """Turns raw langgraph stream chunks into StreamEvents and tracks the messages."""

    def __init__(self, input_messages: list[BaseMessage]) -> None:
        self.messages: list[BaseMessage] = list(input_messages)
        self.last_ai: AIMessage | None = None

    def feed(self, mode: str, data: Any) -> Iterator[StreamEvent]:
        if mode == "messages":
            chunk, meta = data
            if (
                isinstance(chunk, AIMessageChunk)
                and meta.get("langgraph_node") == "model"
            ):
                text = _text(chunk)
                if text:
                    yield StreamEvent("text", text=text)
        elif mode == "updates" and isinstance(data, dict):
            for update in data.values():
                msgs = update.get("messages") if isinstance(update, dict) else None
                for m in msgs if isinstance(msgs, list) else []:
                    self.messages.append(m)
                    if isinstance(m, AIMessage):
                        self.last_ai = m
                        for call in m.tool_calls:
                            yield StreamEvent(
                                "tool_call", name=call["name"], args=call["args"]
                            )
                    elif isinstance(m, ToolMessage):
                        yield StreamEvent("tool_result", name=m.name, content=m.content)

    def final_text(self) -> str:
        return _text(self.last_ai) if self.last_ai is not None else ""
