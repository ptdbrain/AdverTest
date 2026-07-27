import pytest

from src.agents.graph import agent


@pytest.mark.asyncio
async def test_agent_basic_flow():
    result = await agent.ainvoke({"query": "Hello"})
    assert "response" in result


@pytest.mark.asyncio
async def test_agent_state_structure():
    result = await agent.ainvoke({"query": "Test query"})
    assert isinstance(result, dict)
    assert "query" in result


@pytest.mark.asyncio
async def test_agent_routes_calculation_to_safe_tool():
    result = await agent.ainvoke({"query": "calculate 2 + 3 * 4"})

    assert result["route"] == "calculate"
    assert result["tool_result"] == "14"
    assert "14" in result["response"]


@pytest.mark.asyncio
async def test_agent_routes_knowledge_query_to_search_tool():
    result = await agent.ainvoke({"query": "What is LangGraph?"})

    assert result["route"] == "search"
    assert "stateful" in result["tool_result"]


@pytest.mark.asyncio
async def test_agent_handles_invalid_calculation_without_crashing():
    result = await agent.ainvoke({"query": "calculate 2 + unknown"})

    assert result["route"] == "calculate"
    assert result["error"].startswith("Calculation failed:")
    assert result["response"].startswith("I could not calculate that:")
