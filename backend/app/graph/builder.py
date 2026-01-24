from __future__ import annotations

from langgraph.graph import StateGraph, END
from app.graph.qa import is_probably_qa

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
    qa_node,
)


# -----------------------------
# Routing helpers
# -----------------------------

def _route_start(state: AgentState) -> str:
    """
    IMPORTANT: Prevent re-running column inference on every /chat call.

    - If we already have a confirmed_config, we can proceed directly to codegen/exec.
      (Typical when user sends another message after a successful confirm, or state is persisted.)
    - If we already have a proposed_config, go directly to confirm to interpret user_message
      ("confirm" or "modify ..."), without re-inferring columns.
    - Otherwise this is the first turn for the dataset_id: run preview -> plan -> infer.
    """
    if is_probably_qa(state.get("user_message", "")):
        return "qa"    
    if state.get("confirmed_config"):
        return "codegen"
    if state.get("proposed_config"):
        return "confirm"
    return "preview"


def _route_after_confirmation(state: AgentState) -> str:
    """
    If confirmed_config exists -> proceed to codegen.
    Else we end this turn (waiting for user confirm/modification).
    """
    if state.get("confirmed_config"):
        return "codegen"
    return "end"


def _route_after_exec(state: AgentState) -> str:
    if state.get("exec_error"):
        attempt = int(state.get("attempt", 0) or 0)
        max_attempts = int(state.get("max_attempts", 2) or 2)
        if attempt < max_attempts:
            return "traceback"
        return "results"
    return "results"


def build_graph():
    g = StateGraph(AgentState)

    # A no-op start node so we can route based on what's already in state.
    # This is what stops the graph from re-running preview/plan/infer every time.
    async def start_node(state: AgentState) -> AgentState:
        return state

    g.add_node("start", start_node)

    g.add_node("preview", supervisor_preview_node)
    g.add_node("plan", plan_node)
    g.add_node("infer_columns", column_inference_node)
    g.add_node("confirm", user_confirmation_node)
    g.add_node("codegen", codegen_node)
    g.add_node("exec", exec_node)
    g.add_node("traceback", traceback_node)
    g.add_node("repair", repair_codegen_node)
    g.add_node("results", results_node)
    g.add_node("qa", qa_node)

    # Entry point is now "start", not "preview"
    g.set_entry_point("start")

    # Route based on existing state
    g.add_conditional_edges(
        "start",
        _route_start,
        {
            "preview": "preview",
            "confirm": "confirm",
            "codegen": "codegen",
            "qa": "qa",
        },
    )

    # First-turn flow
    g.add_edge("preview", "plan")
    g.add_edge("plan", "infer_columns")
    g.add_edge("infer_columns", "confirm")

    # Confirm flow
    g.add_conditional_edges(
        "confirm",
        _route_after_confirmation,
        {
            "codegen": "codegen",
            "end": END,
        },
    )

    # Execute flow
    g.add_edge("codegen", "exec")

    g.add_conditional_edges(
        "exec",
        _route_after_exec,
        {
            "traceback": "traceback",
            "results": "results",
        },
    )

    g.add_edge("traceback", "repair")
    g.add_edge("repair", "exec")
    g.add_edge("results", END)
    g.add_edge("qa", END)

    return g.compile()
