import ast
import operator

from langchain_core.tools import tool

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@tool
def search_knowledge(query: str) -> str:
    """Search the small local knowledge base for a query."""
    knowledge = {
        "langgraph": "LangGraph is a framework for building stateful, multi-step agents.",
        "agent": "An agent combines state, decision-making nodes, edges, and tools.",
    }
    normalized = query.lower()
    for keyword, answer in knowledge.items():
        if keyword in normalized:
            return answer
    return f"No indexed knowledge found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a numeric expression without using eval."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
        return f"Calculation error: {exc}"


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST using a fixed safe operator allowlist."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Only numeric constants are supported")
    if isinstance(node, ast.UnaryOp):
        operator_fn = _SAFE_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator_fn(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        operator_fn = _SAFE_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator_fn(_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"Unsupported expression: {type(node).__name__}")
