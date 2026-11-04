# Agentkit

A small base library for LLM apps, so you don't have to set up LangChain, agents, networking, a relay, env loading, tracing and embeddings from scratch every time.

- **Agents** with tools, structured output, streaming and optional sessions
- **Relay**: every HTTP call (chat + embeddings) goes through your relay (e.g. a Cloudflare Worker)
- **Embeddings and vector stores** with pluggable backends, plus a ready-made RAG tool
- **Logging and tracing**: stdlib logging, plus Langfuse that turns on automatically
- **Lazy setup**: importing the package never fails, config is only read when used

Requires Python 3.10+. Models run on Gemini through its OpenAI-compatible endpoint.

## Install

In another project (editable, while you're still changing it):

```bash
uv add agentkit
```

From git:

```bash
uv add git+https://github.com/MahdiBND/AgentKit
```

Optional extra for a persistent vector store:

```bash
uv add "agentkit[chroma]"
```

## Configure

agentkit reads a `.env` file from the directory you run your app in (walking upward), or plain environment variables. Copy `env.example` to get started.

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Gemini API key |
| `RELAY_URL` | yes | URL of your relay |
| `LLM_MODEL` | no | Default chat model (code-level `model=` overrides it) |
| `EMBED_MODEL` | no | Embedding model (default `gemini-embedding-2`) |
| `LOG_LEVEL` | no | Used by `configure_logging()` (default `INFO`) |
| `VECTOR_BACKEND` | no | `memory` (default) or `chroma` |
| `CHROMA_PATH` | no | Persist directory for Chroma; unset = in-memory |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | no | Tracing is enabled when both keys are set |

Missing required values fail immediately, with a clear message, when you create an agent or client.

## Quick start

```python
from agentkit import new_agent

agent = new_agent("You are a helpful assistant.")
print(agent.invoke("Say hi in one sentence."))
```

### Tools

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny, 12C"

agent = new_agent("Use tools when needed.", tools=[get_weather])
agent.invoke("Weather in Oslo?")
```

### Structured output

```python
from pydantic import BaseModel

class CodeResponse(BaseModel):
    code: str

agent = new_agent("You write code.", output_type=CodeResponse)
agent.invoke("print one")   # -> CodeResponse(code="print(1)")
```

`output_type` can be a pydantic model, dataclass or TypedDict. Without it, `invoke` returns a string.

### Model and model settings

Each agent can use its own model: a small, cheap one for exploring files, a big one for planning. different providers.

```python
from agentkit import new_agent

explorer = new_agent("Explore the directory.", model="openai:gpt-4o")
planner  = new_agent("Write a detailed plan.", model="google_genai:gemini-3.7")
```

Resolution order for the model name: the `model=` you pass in code, then `LLM_MODEL`, then the built-in default.

Sampling settings can be passed to the default model:

```python
agent = new_agent(
    "You write code.",
    model_settings={"temperature": 0.2, "top_p": 0.9},
)
```

`model_settings` only applies to the default model. If you pass your own `model=`, configure it there:

```python
new_agent("...", model=get_model(model="<big-model>", temperature=0.2))
```

### Streaming

`stream` / `astream` yield events, including tool calls and results, not only text. The last event is always `final`.

```python
for e in agent.stream("Weather in Oslo?"):
    if e.type == "text":
        print(e.text, end="", flush=True)
    elif e.type == "tool_call":
        print(f"\n[calling {e.name}({e.args})]")
    elif e.type == "tool_result":
        print(f"[{e.name} -> {e.content}]")
    elif e.type == "final":
        answer = e.output

# async
async for e in agent.astream("..."):
    ...
```

| `e.type` | Fields |
|---|---|
| `text` | `text`: chunk of the answer |
| `tool_call` | `name`, `args` |
| `tool_result` | `name`, `content` |
| `final` | `output`: full answer |

With `output_type` set, there is nothing to stream token by token, so `stream` runs `invoke` and yields a single `final` event holding the object.

If you stop reading before `final`, the session (if any) is not saved for that turn.

### Sessions

Stateless by default. Pass a store, then a `session_id`:

```python
from agentkit import new_agent, InMemorySessionStore

agent = new_agent("Be brief.", session_store=InMemorySessionStore())
agent.invoke("My name is Sam", session_id="u1")
agent.invoke("What's my name?", session_id="u1")   # remembers
```

Without `session_id` (or without a store) behavior is unchanged. To persist sessions elsewhere (Redis, SQL, ...), write a class with two methods and pass it as `session_store=`:

```python
class MyStore:
    def load(self, session_id: str) -> list[BaseMessage]: ...
    def save(self, session_id: str, messages: list[BaseMessage]) -> None: ...
```

### Embeddings

```python
from agentkit import embed, aembed, get_embeddings

embed("hello")             # -> list[float]        (1536 dims by default)
embed(["a", "b"])          # -> list[list[float]]
await aembed("hello")

embed("hello", dimensions=768, task_type="RETRIEVAL_QUERY")
get_embeddings()           # LangChain Embeddings object, for vector stores
```

### Vector stores and RAG

```python
from agentkit import new_agent, get_vector_store, retriever_tool

store = get_vector_store("docs")          # in-memory by default
store.add_texts(["cats are mammals", "python is a language"])

agent = new_agent(
    "Answer using the knowledge base.",
    tools=[retriever_tool(store, k=4)],
)
agent.invoke("What is python?")
```

- `retriever_tool`: returns a LangChain tool with both sync and async versions, so it works with invoke and ainvoke. It returns numbered passages, or "No relevant results found." when there are none.
- `get_vector_store(name)` returns the same instance for the same name, so in-memory contents persist across calls.
- Choose a backend with `VECTOR_BACKEND` or `get_vector_store(backend="chroma")`.
- Add your own backend:

```python
from agentkit import register_backend

register_backend("pgvector", lambda embeddings, name: MyPGVector(embeddings, name))
```

### Logging and tracing

```python
from agentkit import configure_logging

configure_logging()        # or configure_logging("DEBUG")
```

The library logs to the `agentkit` logger and is silent until you call this. Langfuse tracing needs no code: set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. `session_id` and `user_id` passed to `invoke` / `stream` are attached to the trace.

```python
agent.invoke("hi", session_id="u1", user_id="sam")
```

## How it's organised

```
src/agentkit/
  getenv.py       env + .env loading
  relay.py        HTTP clients that route through the relay
  relay_registry  register providers that need relay (gemini is registered by default)
  models.py       chat model construction (get_model)
  embed.py        embeddings (langchain-google-genai, via the relay)
  vectorstore.py  vector store backends + retriever tool
  agent.py        Agent / new_agent (invoke, stream, sessions)
  streaming.py    StreamEvent + parsing of LangGraph chunks
  messages.py     message text helper
  sessions.py     SessionStore protocol + in-memory store
  tracing.py      Langfuse callbacks and metadata
  log.py          logging setup
```

Dependencies point one way: `agent.py` uses the smaller modules, and none of them import `agent.py`. Anything replaceable (tracer, stream format, session storage, vector backend) lives in exactly one file.

## Notes

- **Relay:** requests are rewritten to `RELAY_URL`, with the original destination in `x-relay-target` and `x-relay-path` headers. Your relay must forward them and pass streamed responses through without buffering, otherwise streaming arrives all at once.
- **Embedding client:** `langchain-google-genai` shares one `client_args` between its sync and async clients, so agentkit attaches a relay-wired `genai` client to the wrapper instead. If a future version changes how that wrapper builds its client, `embed.py` is where to look.