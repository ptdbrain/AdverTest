from langgraph.graph import END, StateGraph

from src.agents.nodes.example_node import analyze_node, respond_node, tool_node
from src.agents.state import AgentState


def should_continue(state: AgentState) -> str:
    """Route to a tool or finish with a direct response."""
    if state.get("error"):
        return END
    return state.get("route", "respond")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("calculate", tool_node)
    graph.add_node("search", tool_node)
    graph.add_node("respond", respond_node)

    # Add edges
    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", should_continue)
    graph.add_edge("calculate", "respond")
    graph.add_edge("search", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


agent = build_graph()
