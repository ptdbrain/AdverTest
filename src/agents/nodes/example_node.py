from src.agents.state import AgentState
from src.agents.tools.example_tool import calculate, search_knowledge


async def analyze_node(state: AgentState) -> dict:
    """Classify a query into a tool route or a direct response."""
    query = state.get("query", "")
    normalized = query.strip().lower()
    if normalized.startswith("calculate ") or normalized.startswith("calc "):
        route = "calculate"
    elif "langgraph" in normalized or "agent" in normalized or "knowledge" in normalized:
        route = "search"
    else:
        route = "respond"

    return {"analysis": f"Analysis route: {route}", "route": route}


async def tool_node(state: AgentState) -> dict:
    """Run the selected tool and convert failures into structured state."""
    query = state.get("query", "")
    route = state.get("route")

    try:
        if route == "calculate":
            expression = query.split(" ", 1)[1]
            result = calculate.invoke(expression)
        elif route == "search":
            result = search_knowledge.invoke(query)
        else:
            return {}
    except Exception as exc:
        return {"error": f"Tool failed: {exc}"}

    if isinstance(result, str) and result.startswith("Calculation error:"):
        return {"error": f"Calculation failed: {result}"}
    return {"tool_result": result}


async def respond_node(state: AgentState) -> dict:
    """Create a response from tool output or the analysis result."""
    analysis = state.get("analysis", "")
    error = state.get("error")

    if error and error.startswith("Calculation failed:"):
        return {"response": f"I could not calculate that: {error}"}
    if error:
        return {"response": f"I could not complete the request: {error}"}

    tool_result = state.get("tool_result")
    if tool_result:
        return {"response": tool_result}
    return {"response": f"Result based on analysis: {analysis}"}
