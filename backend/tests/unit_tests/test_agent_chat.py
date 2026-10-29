import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import graph


def _text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


@pytest.mark.anyio
async def test_agent_replies_to_a_message() -> None:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Reply with the single word pong.")]}
    )

    reply = result["messages"][-1]
    print(reply)
    assert isinstance(reply, AIMessage)
    assert _text(reply).strip()
