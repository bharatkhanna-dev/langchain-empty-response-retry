import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_empty_response_retry import (
    DEFAULT_NUDGE,
    EmptyModelResponseError,
    EmptyResponseRetryMiddleware,
    is_empty_ai_message,
)

from .conftest import make_request, scripted_async_handler, scripted_handler

BLANK = AIMessage(content="")
OK = AIMessage(content="hello")


def test_non_empty_returns_same_object():
    mw = EmptyResponseRetryMiddleware()
    resp = ModelResponse(result=[OK])
    handler, calls = scripted_handler(resp)
    assert mw.wrap_model_call(make_request([HumanMessage("hi")]), handler) is resp
    assert len(calls) == 1


@pytest.mark.parametrize(
    "msg",
    [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "t", "args": "{bad", "id": "1", "error": "x", "type": "invalid_tool_call"}
            ],
        ),
    ],
)
def test_tool_calls_are_not_empty(msg):
    assert not is_empty_ai_message(msg)


@pytest.mark.parametrize(
    "content",
    [
        "   \n",
        [{"type": "text", "text": "  "}],
        [{"type": "reasoning", "reasoning": "thinking"}],
    ],
)
def test_blank_variants_are_empty(content):
    assert is_empty_ai_message(AIMessage(content=content))


def test_structured_response_is_not_empty():
    mw = EmptyResponseRetryMiddleware()
    resp = ModelResponse(result=[BLANK], structured_response={"a": 1})
    handler, calls = scripted_handler(resp)
    assert mw.wrap_model_call(make_request([HumanMessage("hi")]), handler) is resp
    assert len(calls) == 1


def test_empty_then_ok_retries_with_single_nudge():
    mw = EmptyResponseRetryMiddleware()
    base = [HumanMessage("hi")]
    request = make_request(base)
    handler, calls = scripted_handler(BLANK, OK)
    result = mw.wrap_model_call(request, handler)
    assert result.result == [OK]
    assert len(calls) == 2
    retry_msgs = calls[1].messages
    assert retry_msgs[:-1] == base
    assert isinstance(retry_msgs[-1], HumanMessage)
    assert retry_msgs[-1].content == DEFAULT_NUDGE
    assert BLANK not in retry_msgs
    assert request.messages == base


def test_nudges_do_not_accumulate():
    mw = EmptyResponseRetryMiddleware(max_retries=2)
    handler, calls = scripted_handler(BLANK, BLANK, BLANK)
    mw.wrap_model_call(make_request([HumanMessage("hi")]), handler)
    assert len(calls) == 3
    for retry in calls[1:]:
        assert sum(m.content == DEFAULT_NUDGE for m in retry.messages) == 1


def test_exhausted_return_error_and_fallback():
    last = ModelResponse(result=[BLANK])
    handler_last, _ = scripted_handler(last)
    mw = EmptyResponseRetryMiddleware(max_retries=0)
    assert mw.wrap_model_call(make_request([HumanMessage("hi")]), handler_last) is last

    handler_err, _ = scripted_handler(BLANK)
    with pytest.raises(EmptyModelResponseError):
        EmptyResponseRetryMiddleware(max_retries=0, on_exhausted="error").wrap_model_call(
            make_request([HumanMessage("hi")]), handler_err
        )

    handler_fb, _ = scripted_handler(BLANK)
    out = EmptyResponseRetryMiddleware(max_retries=0, on_exhausted="Sorry.").wrap_model_call(
        make_request([HumanMessage("hi")]), handler_fb
    )
    assert out.result[0].content == "Sorry."


def test_bare_ai_message_from_handler():
    mw = EmptyResponseRetryMiddleware()
    calls = []

    def handler(req):
        calls.append(req)
        return OK

    assert mw.wrap_model_call(make_request([HumanMessage("hi")]), handler) is OK
    assert len(calls) == 1


def test_system_nudge_role():
    mw = EmptyResponseRetryMiddleware(nudge_role="system", nudge="go")
    handler, calls = scripted_handler(BLANK, OK)
    mw.wrap_model_call(make_request([HumanMessage("hi")]), handler)
    assert isinstance(calls[1].messages[-1], SystemMessage)


def test_custom_predicate():
    mw = EmptyResponseRetryMiddleware(is_empty=lambda m: m.text == "bad")
    handler, calls = scripted_handler(AIMessage(content="bad"), OK)
    mw.wrap_model_call(make_request([HumanMessage("hi")]), handler)
    assert len(calls) == 2


def test_invalid_config():
    with pytest.raises(ValueError):
        EmptyResponseRetryMiddleware(max_retries=-1)
    with pytest.raises(ValueError):
        EmptyResponseRetryMiddleware(nudge_role="assistant")  # type: ignore[arg-type]


async def test_async_retry_and_exhaustion():
    mw = EmptyResponseRetryMiddleware(max_retries=1)
    handler, calls = await _run(mw, BLANK, OK)
    assert len(calls) == 2

    handler, calls = scripted_async_handler(BLANK, BLANK)
    await mw.awrap_model_call(make_request([HumanMessage("hi")]), handler)
    assert len(calls) == 2

    handler, _ = scripted_async_handler(BLANK)
    with pytest.raises(EmptyModelResponseError):
        await EmptyResponseRetryMiddleware(max_retries=0, on_exhausted="error").awrap_model_call(
            make_request([HumanMessage("hi")]), handler
        )


async def _run(mw, *replies):
    handler, calls = scripted_async_handler(*replies)
    await mw.awrap_model_call(make_request([HumanMessage("hi")]), handler)
    return handler, calls
