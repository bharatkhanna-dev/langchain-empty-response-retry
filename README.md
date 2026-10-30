# langchain-empty-response-retry

A small, model-agnostic [LangChain](https://github.com/langchain-ai/langchain) agent middleware that retries with a short corrective nudge when the model returns a **successful completion with no text and no tool calls**.

## Why

Models occasionally return an empty completion, most often at the end of long tool-calling chains. The agent loop treats it as a finished turn and the user receives nothing. Because no exception is raised, `ModelRetryMiddleware` never fires.

`EmptyResponseRetryMiddleware` detects this by content, not by provider-specific finish reasons, and retries the request with a short nudge.

## Install

```bash
pip install langchain-empty-response-retry
```

## Usage

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_empty_response_retry import EmptyResponseRetryMiddleware


@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return "Refunds take 5 days."


agent = create_agent(
    model="openai:gpt-4o",  # any chat model
    tools=[search],
    middleware=[EmptyResponseRetryMiddleware(max_retries=2)],
)

result = agent.invoke({"messages": [{"role": "user", "content": "How long do refunds take?"}]})
```

### What a retry looks like

On an empty completion the request is retried with one nudge appended. Only the accepted response is kept in the agent state.

```text
model call 1: [Human: "How long do refunds take?"]                     -> AIMessage("")
model call 2: [Human: "How long do refunds take?",
               Human: "Your previous response was empty. Respond to ..."] -> AIMessage("About 5 days.")

final state:  [Human: "How long do refunds take?", AI: "About 5 days."]
```

### Failing loudly or using a fallback reply

```python
# Raise EmptyModelResponseError after the last retry:
EmptyResponseRetryMiddleware(max_retries=1, on_exhausted="error")

# Or reply with fixed text instead of an empty message:
EmptyResponseRetryMiddleware(on_exhausted="Sorry, I couldn't generate a reply. Please try again.")
```

## How it works

On every model call the middleware:

1. Calls the handler. A response is **not** empty if it has text, any `tool_calls` or `invalid_tool_calls`, or a `structured_response`. Reasoning-only responses count as empty.
2. If empty, retries with `original messages + [nudge]`. The blank attempt is not included (some providers reject empty assistant turns) and nudges do not accumulate across attempts.
3. When retries are exhausted, `on_exhausted` decides the outcome.

The predicate is exported for reuse:

```python
from langchain_core.messages import AIMessage
from langchain_empty_response_retry import is_empty_ai_message

is_empty_ai_message(AIMessage(content="  "))  # True
```

## Configuration

| Constructor arg | Default | Description |
|---|---|---|
| `max_retries` | `2` | Retries after the first attempt. `0` disables retries but still applies `on_exhausted`. |
| `nudge` | see `DEFAULT_NUDGE` | Text of the corrective message. |
| `nudge_role` | `"human"` | `"human"` or `"system"`. Mid-conversation system messages are rejected by some providers. |
| `on_exhausted` | `"return"` | `"return"` returns the last response unchanged, `"error"` raises `EmptyModelResponseError`, any other string is returned to the user as the reply. |
| `is_empty` | `None` | Custom predicate `(AIMessage) -> bool` replacing the default check. |

## Ordering

Middleware order is outermost first. Recommended placement:

- Place **inside** any middleware that trims or rewrites message history, so the nudge is not trimmed away.
- Put a `ModelCallLimitMiddleware` outermost as a hard ceiling; retries across several guards multiply model calls.

## Limitations

- **Streaming:** the retry happens after the first attempt has already streamed. Final state is correct, but a live `stream_mode="messages"` consumer may have seen the rejected attempt.
- Does not fix blank responses caused by context overflow or hard safety blocks; retries will also be blank.
- Each retry is a full model call. Keep `max_retries` small.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
