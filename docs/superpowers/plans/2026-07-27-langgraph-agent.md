# LangGraph Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the placeholder two-node graph with a small, testable LangGraph agent that demonstrates Chapter 4 state, nodes, conditional edges, tools, error handling, and ReAct-style tool routing.

**Architecture:** The graph analyzes a query, routes calculation requests to the safe calculator tool, routes knowledge requests to the deterministic knowledge tool, and otherwise produces a direct response. Tool failures are returned as user-facing responses without crashing the API. LLM integration remains optional until a real OpenAI key is configured.

**Tech Stack:** Python 3.13, LangGraph, LangChain Core tools, FastAPI, pytest, Ruff.

## Global Constraints

- Do not call a paid LLM from automated tests.
- Keep the existing FastAPI contract and `DAY01` untouched.
- Keep tools typed, documented, validated, and safe.

### Task 1: State and tool routing behavior

**Files:**
- Modify: `src/agents/state.py`
- Modify: `src/agents/nodes/example_node.py`
- Modify: `src/agents/tools/example_tool.py`
- Modify: `src/agents/graph.py`
- Test: `tests/test_agents/test_graph.py`

- [ ] Add tests for calculation routing, knowledge routing, direct response, and invalid calculation handling.
- [ ] Run the focused tests and confirm the new behavior fails before implementation.
- [ ] Add the state fields and node/tool implementations needed by the failing tests.
- [ ] Run the focused tests, then the full suite and Ruff.

### Task 2: Documentation and delivery

**Files:**
- Modify: `WORKLOG.md`

- [ ] Record the Chapter 4 implementation and verification results.
- [ ] Commit on `develop`, push `develop`, fast-forward merge into `main`, and push `main`.
