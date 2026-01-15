from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.nodes import (
    supervisor_preview_node,
    plan_node,
    column_inference_node,
    user_confirmation_node,
    codegen_node,
    exec_node,
    traceback_node,
    repair_codegen_node,
    results_node,
)


def _route_after_confirmation(state: AgentState) -> str:
    """
    If confirmed_config exists -> proceed to codegen.
    Else we stay done for this turn (waiting for user confirm/modification).
    """
    if state.get("confirmed_config"):
        return "codegen"
    return "end"


def _route_after_exec(state: AgentState) -> str:
    if state.get("exec_error"):
        # allow repair loop if attempts remain
        attempt = int(state.get("attempt", 0) or 0)
        max_attempts = int(state.get("max_attempts", 2) or 2)
        if attempt < max_attempts:
            return "traceback"
        return "results"
    return "results"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("preview", supervisor_preview_node)
    g.add_node("plan", plan_node)
    g.add_node("infer_columns", column_inference_node)
    g.add_node("confirm", user_confirmation_node)
    g.add_node("codegen", codegen_node)
    g.add_node("exec", exec_node)
    g.add_node("traceback", traceback_node)
    g.add_node("repair", repair_codegen_node)
    g.add_node("results", results_node)

    g.set_entry_point("preview")

    g.add_edge("preview", "plan")
    g.add_edge("plan", "infer_columns")
    g.add_edge("infer_columns", "confirm")

    g.add_conditional_edges("confirm", _route_after_confirmation, {
        "codegen": "codegen",
        "end": END,
    })

    g.add_edge("codegen", "exec")

    g.add_conditional_edges("exec", _route_after_exec, {
        "traceback": "traceback",
        "results": "results",
    })

    g.add_edge("traceback", "repair")
    g.add_edge("repair", "exec")
    g.add_edge("results", END)

    return g.compile()
