"""Internal helpers for normalising model-call handler results."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


def result_messages(response: Any) -> list[BaseMessage]:
    """Return the message list from any shape a handler may return."""
    if isinstance(response, AIMessage):
        return [response]
    inner = getattr(response, "model_response", None)  # ExtendedModelResponse
    if inner is not None:
        response = inner
    return list(getattr(response, "result", None) or [])


def last_ai(response: Any) -> AIMessage | None:
    """Return the last ``AIMessage`` in the response, if any."""
    for message in reversed(result_messages(response)):
        if isinstance(message, AIMessage):
            return message
    return None


def structured_response(response: Any) -> Any:
    """Return the structured response carried by the handler result, if any."""
    inner = getattr(response, "model_response", response)
    return getattr(inner, "structured_response", None)
