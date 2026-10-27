"""LangGraph flowchart: plan, search locally, look, check, maybe refine, then answer."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .llm import ModelCaller
from .nodes import make_nodes, route
from .state import AgentState


def build_graph(caller: ModelCaller, embedder, index, *, top_k: int = 5, min_gap_s: float = 6.0):
    plan, search_node, look, check, refine, answer = make_nodes(
        caller, embedder, index, top_k=top_k, min_gap_s=min_gap_s
    )
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("search", search_node)
    graph.add_node("look", look)
    graph.add_node("check", check)
    graph.add_node("refine", refine)
    graph.add_node("answer", answer)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "look")
    graph.add_edge("look", "check")
    graph.add_conditional_edges("check", route, {"answer": "answer", "refine": "refine"})
    graph.add_edge("refine", "search")
    graph.add_edge("answer", END)
    return graph.compile()


def run_request(
    request: str,
    caller: ModelCaller,
    embedder,
    index,
    *,
    retries_left: int = 2,
    top_k: int = 5,
    min_gap_s: float = 6.0,
) -> AgentState:
    graph = build_graph(caller, embedder, index, top_k=top_k, min_gap_s=min_gap_s)
    return graph.invoke(
        {
            "request": request,
            "search_phrase": "",
            "hits": [],
            "verdict": False,
            "reason": "",
            "matched_indices": [],
            "retries_left": retries_left,
            "attempts": [],
            "answer": "",
        }
    )
