"""Retry model calls that complete successfully but return nothing."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ._common import last_ai, structured_response

logger = logging.getLogger(__name__)

DEFAULT_NUDGE = (
    "Your previous response was empty. Respond to the latest message, "
    "or call a tool if one is needed."
)


class EmptyModelResponseError(RuntimeError):
    """Raised when retries are exhausted and ``on_exhausted="error"``."""


def is_empty_ai_message(msg: AIMessage, *, whitespace_is_empty: bool = True) -> bool:
    """Return True if ``msg`` has no text and no (valid or invalid) tool calls.

    Whitespace-only text counts as empty unless ``whitespace_is_empty`` is False.
    """
    if msg.tool_calls or getattr(msg, "invalid_tool_calls", None):
        return False
    text = msg.text
    return not (text.strip() if whitespace_is_empty else text)


class EmptyResponseRetryMiddleware(AgentMiddleware):
    """Retry with a short corrective nudge when the model returns an empty completion."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        nudge: str = DEFAULT_NUDGE,
        nudge_role: Literal["human", "system"] = "human",
        on_exhausted: str = "return",
        is_empty: Callable[[AIMessage], bool] | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            max_retries: Retries after the first attempt. Must be >= 0.
            nudge: Text of the corrective message appended on retry.
            nudge_role: Role of the nudge message. ``"human"`` is the safe default.
            on_exhausted: ``"return"`` returns the last response unchanged,
                ``"error"`` raises ``EmptyModelResponseError``, any other string
                is returned as the reply.
            is_empty: Optional predicate overriding the default emptiness check.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if nudge_role not in ("human", "system"):
            raise ValueError("nudge_role must be 'human' or 'system'")
        super().__init__()
        self.max_retries = max_retries
        self.nudge = nudge
        self.nudge_role = nudge_role
        self.on_exhausted = on_exhausted
        self.is_empty = is_empty or is_empty_ai_message

    def _is_empty_response(self, response: Any) -> bool:
        if structured_response(response) is not None:
            return False
        last = last_ai(response)
        return last is None or self.is_empty(last)

    def _retry_request(self, request: ModelRequest) -> ModelRequest:
        nudge = (
            SystemMessage(content=self.nudge)
            if self.nudge_role == "system"
            else HumanMessage(content=self.nudge)
        )
        return request.override(messages=[*request.messages, nudge])

    def _exhausted(self, response: Any) -> Any:
        if self.on_exhausted == "return":
            return response
        if self.on_exhausted == "error":
            raise EmptyModelResponseError(
                f"model returned an empty response after {self.max_retries + 1} attempt(s)"
            )
        return ModelResponse(result=[AIMessage(content=self.on_exhausted)])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        current = request
        response: Any = None
        for attempt in range(self.max_retries + 1):
            response = handler(current)
            if not self._is_empty_response(response):
                return response
            if attempt < self.max_retries:
                logger.warning("empty model response; retrying (attempt %d)", attempt + 1)
                current = self._retry_request(request)
        return self._exhausted(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        current = request
        response: Any = None
        for attempt in range(self.max_retries + 1):
            response = await handler(current)
            if not self._is_empty_response(response):
                return response
            if attempt < self.max_retries:
                logger.warning("empty model response; retrying (attempt %d)", attempt + 1)
                current = self._retry_request(request)
        return self._exhausted(response)
