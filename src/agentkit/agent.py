import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage

from .log import logger
from .messages import message_text
from .models import get_model
from .sessions import SessionStore
from .streaming import StreamEvent, StreamState
from .tracing import build_config


class Agent:
    """Thin wrapper around a LangChain agent: logging, tracing, optional sessions.

    Stateless unless both `session_store` is set and `session_id` is passed.
    """

    def __init__(
        self,
        instructions: str,
        *,
        model: Any | None = None,
        model_name: str = "gemini-3.6-flash",
        tools: Sequence[Any] | None = None,
        output_type: type | None = None,
        model_settings: dict[str, Any] | None = None,
        session_store: SessionStore | None = None,
        name: str = "agent",
    ) -> None:
        """
        output_type:    pydantic model / dataclass / TypedDict. When set, `invoke`
                        returns an instance of it instead of a string.
        model_settings: ChatOpenAI kwargs, e.g. {"temperature": 0.2, "top_p": 0.9}.
                        Only applies to the default model.
        """
        if model is not None and model_settings:
            raise ValueError(
                "model_settings only applies to the default model; "
                "configure your own model instead"
            )

        self.name = name
        self.sessions = session_store
        self.output_type = output_type
        self.graph = create_agent(
            model=model or get_model(model=model_name, **(model_settings or {})),
            system_prompt=instructions,
            tools=list(tools or []),
            response_format=output_type,
        )

    # -- internals -------------------------------------------------------
    def _messages(self, text: str, session_id: str | None) -> list[BaseMessage]:
        history: list[BaseMessage] = []
        if session_id:
            if self.sessions is None:
                logger.warning(
                    "session_id given but no session_store; running stateless"
                )
            else:
                history = self.sessions.load(session_id)
        return [*history, HumanMessage(text)]

    def _config(
        self, session_id: str | None, user_id: str | None, config: dict | None
    ) -> dict:
        return build_config(
            config, session_id=session_id, user_id=user_id, tags=[self.name]
        )

    def _save(
        self, session_id: str | None, messages: list[BaseMessage], started: float
    ) -> None:
        if session_id and self.sessions is not None:
            self.sessions.save(session_id, messages)
        logger.info(
            "%s: done in %.2fs (session=%s)",
            self.name,
            time.time() - started,
            session_id,
        )

    def _finish(self, result: dict, session_id: str | None, started: float) -> Any:
        messages = result["messages"]
        self._save(session_id, messages, started)
        if self.output_type is not None:
            return result["structured_response"]
        return message_text(messages[-1])

    # -- public API ------------------------------------------------------
    def invoke(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        config: dict | None = None,
    ) -> Any:
        started = time.time()
        result = self.graph.invoke(
            {"messages": self._messages(text, session_id)},
            config=self._config(session_id, user_id, config),
        )
        return self._finish(result, session_id, started)

    async def ainvoke(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        config: dict | None = None,
    ) -> Any:
        started = time.time()
        result = await self.graph.ainvoke(
            {"messages": self._messages(text, session_id)},
            config=self._config(session_id, user_id, config),
        )
        return self._finish(result, session_id, started)

    def stream(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        config: dict | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield StreamEvents; the last one is always type "final".

        With `output_type` set there is nothing to stream token by token, so this
        runs `invoke` and yields a single "final" event.
        """
        if self.output_type is not None:
            yield StreamEvent(
                "final",
                output=self.invoke(
                    text, session_id=session_id, user_id=user_id, config=config
                ),
            )
            return

        started = time.time()
        state = StreamState(self._messages(text, session_id))
        for mode, data in self.graph.stream(
            {"messages": list(state.messages)},
            config=self._config(session_id, user_id, config),
            stream_mode=["messages", "updates"],
        ):
            yield from state.feed(mode, data)
        self._save(session_id, state.messages, started)
        yield StreamEvent("final", output=state.final_text())

    async def astream(
        self,
        text: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        config: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async version of `stream`."""
        if self.output_type is not None:
            yield StreamEvent(
                "final",
                output=await self.ainvoke(
                    text, session_id=session_id, user_id=user_id, config=config
                ),
            )
            return

        started = time.time()
        state = StreamState(self._messages(text, session_id))
        async for mode, data in self.graph.astream(
            {"messages": list(state.messages)},
            config=self._config(session_id, user_id, config),
            stream_mode=["messages", "updates"],
        ):
            for event in state.feed(mode, data):
                yield event
        self._save(session_id, state.messages, started)
        yield StreamEvent("final", output=state.final_text())


def new_agent(instructions: str, **kwargs: Any) -> Agent:
    """Options are\n
    `model`: str = "gemini-3.6-flash"\n
    `tools`: Sequence[Any] | None = None\n
    `output_type`: type | None = None\n
    `model_settings`: dict[str, Any] | None = None\n
    `session_store`: SessionStore | None = None\n
    `name`: str = "agent"\n
    """
    return Agent(instructions, **kwargs)
