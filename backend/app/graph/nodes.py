from __future__ import annotations

import traceback as tb
from typing import Any, Dict

import pandas as pd
import numpy as np
from prophet import Prophet

from app.core.profiling import preview_payload
from app.graph.state import AgentState, ColumnConfig
from app.graph.prompts import (
    SUPERVISOR_PLAN_PROMPT,
    COLUMN_INFERENCE_PROMPT,
    CONFIRMATION_INTERPRETER_PROMPT,
    CODEGEN_PROMPT,
    REPAIR_PROMPT,
)
from app.graph.llm import chat_text, chat_json


def _format_preview_for_llm(state: AgentState) -> str:
    prev = state["df_preview"]
    return f"""DATASET PREVIEW (top 5 rows):
{prev["head"]}

COLUMN PROFILE:
{prev["profile"]}

COLUMNS:
{prev["columns"]}
"""


async def supervisor_preview_node(state: AgentState) -> AgentState:
    # Ensure preview exists (in case you call graph differently later)
    if "df_preview" not in state:
        state["df_preview"] = preview_payload(state["df"])
    return state


async def plan_node(state: AgentState) -> AgentState:
    user = _format_preview_for_llm(state)
    plan_text = await chat_text(SUPERVISOR_PLAN_PROMPT, user)
    state["plan_text"] = plan_text

    # This message is early-stage; we also propose columns next.
    state["assistant_message"] = plan_text
    return state


async def column_inference_node(state: AgentState) -> AgentState:
    user = _format_preview_for_llm(state)
    j = await chat_json(COLUMN_INFERENCE_PROMPT, user)

    proposed: ColumnConfig = {
        "model": "prophet",
        "ds_col": j.get("ds_col", "") or "",
        "y_col": j.get("y_col", "") or "",
        "regressors": j.get("regressors", []) or [],
        "freq": j.get("freq", "D") or "D",
        "periods": int(j.get("periods", 30) or 30),
    }
    state["proposed_config"] = proposed

    rationale = j.get("rationale", "")
    msg = (
        f"{state.get('assistant_message','')}\n\n"
        f"Proposed configuration:\n"
        f"- model: Prophet\n"
        f"- ds: {proposed.get('ds_col','')}\n"
        f"- y: {proposed.get('y_col','')}\n"
        f"- regressors: {proposed.get('regressors',[])}\n"
        f"- freq: {proposed.get('freq','D')}\n"
        f"- periods: {proposed.get('periods',30)}\n"
    )
    if rationale:
        msg += f"\nRationale: {rationale}\n"

    msg += "\nReply with 'confirm' to proceed, or tell me changes in natural language (e.g., 'use Date as ds, Sales as y, add Price regressor, forecast 60 days')."

    state["assistant_message"] = msg
    return state


async def user_confirmation_node(state: AgentState) -> AgentState:
    proposed = state.get("proposed_config") or {}
    user_msg = state.get("user_message", "").strip()

    # If user hasn't provided any message (unlikely), ask
    if not user_msg:
        state["assistant_message"] = "Please confirm the proposed ds/y/regressors, or specify changes."
        return state

    # Interpret confirmation/modification
    user = f"""proposed_config = {proposed}\n\nuser_message = {user_msg}"""
    j = await chat_json(CONFIRMATION_INTERPRETER_PROMPT, user)

    action = (j.get("action") or "").lower().strip()
    config = j.get("config") or proposed
    msg_to_user = j.get("message_to_user") or ""

    # Normalize config
    confirmed: ColumnConfig = {
        "model": "prophet",
        "ds_col": config.get("ds_col", proposed.get("ds_col", "")) or "",
        "y_col": config.get("y_col", proposed.get("y_col", "")) or "",
        "regressors": config.get("regressors", proposed.get("regressors", [])) or [],
        "freq": config.get("freq", proposed.get("freq", "D")) or "D",
        "periods": int(config.get("periods", proposed.get("periods", 30)) or 30),
    }

    if action == "confirm":
        state["confirmed_config"] = confirmed
        state["assistant_message"] = (
            "Confirmed configuration:\n"
            f"- model: Prophet\n"
            f"- ds: {confirmed['ds_col']}\n"
            f"- y: {confirmed['y_col']}\n"
            f"- regressors: {confirmed['regressors']}\n"
            f"- freq: {confirmed['freq']}\n"
            f"- periods: {confirmed['periods']}\n\n"
            "Generating code and running the forecast now."
        )
        return state

    if action == "modify":
        state["proposed_config"] = confirmed
        state["assistant_message"] = (
            (msg_to_user + "\n\n" if msg_to_user else "")
            + "Updated proposed configuration:\n"
            f"- model: Prophet\n"
            f"- ds: {confirmed['ds_col']}\n"
            f"- y: {confirmed['y_col']}\n"
            f"- regressors: {confirmed['regressors']}\n"
            f"- freq: {confirmed['freq']}\n"
            f"- periods: {confirmed['periods']}\n\n"
            "Reply with 'confirm' to proceed, or specify further changes."
        )
        return state

    # ask_clarifying or fallback
    state["assistant_message"] = msg_to_user or "I’m not sure if you are confirming or changing the config. Please reply 'confirm' or specify ds/y/regressors."
    return state


async def codegen_node(state: AgentState) -> AgentState:
    config = state.get("confirmed_config")
    if not config:
        # Not confirmed yet: do not generate code
        return state

    user = f"confirmed_config = {config}"
    code = await chat_text(CODEGEN_PROMPT, user)
    state["generated_code"] = code
    return state


def _safe_exec_run(code: str, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute generated code in a restricted-ish environment.
    Note: This is not a sandbox. For production, run in containerized sandbox.
    """
    allowed_globals: Dict[str, Any] = {
        "__builtins__": {
            "__import__": __import__,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
            "print": print,
        },
        "pd": pd,
        "np": np,
        "Prophet": Prophet,
    }
    local_vars: Dict[str, Any] = {}
    exec(code, allowed_globals, local_vars)

    run_fn = local_vars.get("run") or allowed_globals.get("run")
    if not callable(run_fn):
        raise ValueError("Generated code did not define a callable function named `run(df, config)`.")

    return run_fn(df, config)


async def exec_node(state: AgentState) -> AgentState:
    code = state.get("generated_code") or ""
    config = state.get("confirmed_config") or {}

    if not code or not config:
        return state

    try:
        out = _safe_exec_run(code, state["df"], dict(config))
        state["exec_output"] = out
        state["exec_error"] = None
        state["traceback"] = None
    except Exception as e:
        state["exec_output"] = None
        state["exec_error"] = f"{type(e).__name__}: {e}"
        state["traceback"] = tb.format_exc()

    return state


async def traceback_node(state: AgentState) -> AgentState:
    # Keep as separate node (you wanted it explicitly)
    if state.get("exec_error"):
        # assistant_message can reflect an internal retry; we’ll keep it minimal
        state["assistant_message"] = (
            "I hit an execution error while running the generated Prophet code. "
            "I’m regenerating a corrected version and retrying."
        )
    return state


async def repair_codegen_node(state: AgentState) -> AgentState:
    if not state.get("exec_error"):
        return state

    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 2) or 2)
    if attempt >= max_attempts:
        return state

    failing_code = state.get("generated_code") or ""
    trace = state.get("traceback") or state.get("exec_error") or ""
    user = f"FAILING CODE:\n{failing_code}\n\nTRACEBACK:\n{trace}"
    repaired = await chat_text(REPAIR_PROMPT, user)

    state["generated_code"] = repaired
    state["attempt"] = attempt + 1
    return state


async def results_node(state: AgentState) -> AgentState:
    if state.get("exec_output"):
        out = state["exec_output"]
        state["assistant_message"] = (
            "Forecast completed.\n\n"
            f"Training rows used: {out.get('training_rows')}\n"
            f"Input rows: {out.get('input_rows')}\n\n"
            "Forecast (head):\n"
            f"{out.get('forecast_head')}\n\n"
            "Forecast (tail):\n"
            f"{out.get('forecast_tail')}\n"
        )
        return state

    if state.get("exec_error") and int(state.get("attempt", 0) or 0) >= int(state.get("max_attempts", 2) or 2):
        state["assistant_message"] = (
            "I could not successfully execute the generated code after retries.\n\n"
            f"Error: {state.get('exec_error')}\n\n"
            "If you want, paste the columns you intend for ds/y/regressors and I will tighten the generation."
        )
        return state

    # If not executed yet (e.g., waiting for confirm)
    return state
