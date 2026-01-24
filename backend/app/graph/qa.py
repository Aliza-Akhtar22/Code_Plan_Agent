from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from app.graph.llm import chat_text
from app.graph.prompts import FORECAST_QA_PROMPT
from app.graph.state import AgentState


# -----------------------------
# Simple intent heuristics (sync-safe for routing)
# -----------------------------

_CONFIRM_WORDS = {"confirm", "confirmed", "go ahead", "proceed"}
_MODIFY_HINTS = {"change", "modify", "update", "set", "use ds", "use y", "regressor", "regressors", "add regressor"}
_RUN_HINTS = {"run", "execute", "generate code", "codegen", "train", "fit", "forecast now", "start forecasting"}

_QA_STARTERS = (
    "what", "why", "how", "which", "can you", "should i", "do you think",
    "explain", "meaning", "interpret", "suggest", "recommend"
)

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def is_probably_qa(user_message: str) -> bool:
    """
    Heuristic classifier: True if message looks like a question/explanation request,
    and NOT a direct confirm/modify/run instruction.
    """
    m = _norm(user_message)
    if not m:
        return False

    # Hard exclusions: direct control messages
    if m in _CONFIRM_WORDS:
        return False
    if any(h in m for h in _RUN_HINTS):
        return False

    # Modification messages are tricky because they include "regressor".
    # If it's clearly an instruction ("regressor is T", "add regressor X"), treat as non-QA.
    # Otherwise, Q about regressors ("what regressors are meaningful") should be QA.
    instruction_patterns = [
        r"\bregressors?\s+(are|is)\s+\w+",
        r"\badd\s+regressors?\b",
        r"\buse\s+\w+\s+as\s+(ds|y)\b",
        r"\bset\s+\w+\s+as\s+(ds|y)\b",
    ]
    if any(re.search(p, m) for p in instruction_patterns):
        return False

    # Positive signals
    if "?" in m:
        return True

    if m.startswith(_QA_STARTERS):
        return True

    # If message contains "explain", "interpret", "meaning", treat as QA
    if any(k in m for k in ["explain", "interpret", "meaning", "what do the results mean", "what does this mean"]):
        return True

    return False


# -----------------------------
# Context building
# -----------------------------

def _summarize_results(exec_output: Any) -> Dict[str, Any]:
    """
    Produce a compact summary that is safe to pass into the LLM.
    """
    if not isinstance(exec_output, dict):
        return {}

    forecast = exec_output.get("forecast")
    if not isinstance(forecast, list):
        forecast = []

    preview_rows = forecast[:5]
    keys = list(preview_rows[0].keys()) if preview_rows else []

    return {
        "training_rows": exec_output.get("training_rows"),
        "input_rows": exec_output.get("input_rows"),
        "forecast_rows": len(forecast),
        "forecast_preview_first_5": preview_rows,
        "forecast_columns": keys,
        "config_used": exec_output.get("config_used"),
    }


def build_qa_context(state: AgentState) -> Dict[str, Any]:
    """
    Build a single structured context object for Q&A.
    """
    ctx: Dict[str, Any] = {
        "dataset_preview": state.get("df_preview"),
        "plan_text": state.get("plan_text"),
        "proposed_config": state.get("proposed_config"),
        "confirmed_config": state.get("confirmed_config"),
        "results_summary": _summarize_results(state.get("exec_output")),
        "last_error": state.get("exec_error"),
    }
    return ctx


# -----------------------------
# Main QA entry point
# -----------------------------

async def answer_forecast_qa(state: AgentState) -> str:
    """
    Uses FORECAST_QA_PROMPT to answer user questions using available context.
    """
    user_message = state.get("user_message") or ""
    ctx = build_qa_context(state)

    user = f"""USER QUESTION:
{user_message}

CONTEXT (JSON-like):
{ctx}
"""
    return await chat_text(FORECAST_QA_PROMPT, user)
