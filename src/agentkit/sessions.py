"""Session storage. Implement `SessionStore` to add Redis/SQL/etc. later."""
from threading import Lock
from typing import Protocol

from langchain_core.messages import BaseMessage


class SessionStore(Protocol):
    def load(self, session_id: str) -> list[BaseMessage]: ...
    def save(self, session_id: str, messages: list[BaseMessage]) -> None: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, list[BaseMessage]] = {}
        self._lock = Lock()

    def load(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._data.get(session_id, []))

    def save(self, session_id: str, messages: list[BaseMessage]) -> None:
        with self._lock:
            self._data[session_id] = list(messages)
