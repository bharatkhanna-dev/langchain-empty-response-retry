from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_empty_response_retry import EmptyResponseRetryMiddleware


class FakeModel(GenericFakeChatModel):
    calls: int = 0

    def _generate(self, messages, *args, **kwargs):
        self.calls += 1
        return super()._generate(messages, *args, **kwargs)


def test_works_inside_create_agent_and_nudge_is_not_persisted():
    model = FakeModel(messages=iter([AIMessage(content=""), AIMessage(content="Hello!")]))
    agent = create_agent(model=model, tools=[], middleware=[EmptyResponseRetryMiddleware()])
    out = agent.invoke({"messages": [HumanMessage("hi")]})

    assert model.calls == 2
    assert [m.content for m in out["messages"]] == ["hi", "Hello!"]
