from langchain_core.messages import BaseMessage


def message_text(message: BaseMessage) -> str:
    """Plain text of a message (joins multi-part content)."""
    text = message.text
    # Newer langchain-core: `.text` is a str subclass that is also callable (deprecated).
    # Check for str first so we never call it. Older versions return a bound method.
    if isinstance(text, str):
        return str(text)
    if callable(text):
        text = text()
    return text if isinstance(text, str) else str(message.content)
