"""Stream event types and the parser that produces them from LangGraph chunks."""
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage

from .messages import message_text


@dataclass
class StreamEvent:
    """One event from `Agent.stream` / `Agent.astream`.

    type         fields set
    -----------  ------------------------------------------------------------
    text         text     - a chunk of the model's answer as it is generated
    tool_call    name, args - the model decided to call a tool
    tool_result  name, content - the tool finished
    final        output   - the complete result (str, or `output_type` instance);
                            always the last event
    """

    type: Literal["text", "tool_call", "tool_result", "final"]
    text: str = ""
    name: str | None = None
    args: dict | None = None
    content: Any = None
    output: Any = None


class StreamState:
    """Turns raw LangGraph chunks (stream_mode=["messages", "updates"]) into
    StreamEvents, and collects the messages produced along the way."""

    def __init__(self, input_messages: list[BaseMessage]) -> None:
        self.messages: list[BaseMessage] = list(input_messages)
        self.last_ai: AIMessage | None = None

    def feed(self, mode: str, data: Any) -> Iterator[StreamEvent]:
        if mode == "messages":
            chunk, meta = data
            if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model":
                text = message_text(chunk)
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
                            yield StreamEvent("tool_call", name=call["name"], args=call["args"])
                    elif isinstance(m, ToolMessage):
                        yield StreamEvent("tool_result", name=m.name, content=m.content)

    def final_text(self) -> str:
        return message_text(self.last_ai) if self.last_ai is not None else ""
