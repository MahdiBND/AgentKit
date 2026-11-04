import os

import pytest

from agentkit import new_agent, configure_logging

pytestmark = pytest.mark.integration

configure_logging()


@pytest.fixture
def agent():
    """Create an agent using the configured relay transport."""
    if not os.getenv("RELAY_URL"):
        pytest.skip("RELAY_URL is not configured")

    return new_agent("You are a helpful assistant. Be concise.")


def test_agent_invoke_through_relay(agent):
    """Verify that a real LLM request succeeds through the relay."""
    response = agent.invoke("Say hi in one sentence.")

    assert response is not None
    assert isinstance(response, str)
    assert response.strip()


def test_agent_stream_through_relay(agent):
    """Verify that streaming works through the relay."""
    chunks = list(agent.stream("Count from one to five."))

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunks).strip()


@pytest.mark.asyncio
async def test_agent_ainvoke_through_relay(agent):
    """Verify asynchronous invocation through the relay."""
    response = await agent.ainvoke("Say hi in one sentence.")

    assert isinstance(response, str)
    assert response.strip()


@pytest.mark.asyncio
async def test_agent_astream_through_relay(agent):
    """Verify asynchronous streaming through the relay."""
    chunks = []

    async for chunk in agent.astream("Count from one to five."):
        chunks.append(chunk)

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunks).strip()
