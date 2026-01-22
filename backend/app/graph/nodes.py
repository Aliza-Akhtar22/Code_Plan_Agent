from __future__ import annotations

import traceback as tb
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import re
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


def _normalize_config(raw: Dict[str, Any], fallback: Dict[str, Any]) -> ColumnConfig:
    """
    Normalize config strictly, using fallback for missing fields.
    Ensures stable types (regressors=list[str], periods=int, freq=str).
    """
    regressors = raw.get("regressors", fallback.get("regressors", []))
    if regressors is None:
        regressors = []
    if not isinstance(regressors, list):
        regressors = [regressors]
    regressors = [str(r).strip() for r in regressors if str(r).strip()]

    freq = (raw.get("freq", fallback.get("freq", "D")) or "D").strip()
    if freq not in ("D", "W", "M"):
        freq = "D"

    periods_val = raw.get("periods", fallback.get("periods", 30))
    try:
        periods = int(periods_val or 30)
    except Exception:
        periods = 30
    if periods <= 0:
        periods = 30

    return {
        "model": "prophet",
        "ds_col": (raw.get("ds_col", fallback.get("ds_col", "")) or "").strip(),
        "y_col": (raw.get("y_col", fallback.get("y_col", "")) or "").strip(),
        "regressors": regressors,
        "freq": freq,
        "periods": periods,
    }


async def supervisor_preview_node(state: AgentState) -> AgentState:
    # Ensure preview exists (in case you call graph differently later)
    if "df_preview" not in state:
        state["df_preview"] = preview_payload(state["df"])
    return state


async def plan_node(state: AgentState) -> AgentState:
    user = _format_preview_for_llm(state)
    plan_text = await chat_text(SUPERVISOR_PLAN_PROMPT, user)
    state["plan_text"] = plan_text
    state["assistant_message"] = plan_text
    return state


async def column_inference_node(state: AgentState) -> AgentState:
    user = _format_preview_for_llm(state)
    j = await chat_json(COLUMN_INFERENCE_PROMPT, user)

    proposed: ColumnConfig = _normalize_config(j or {}, {})
    state["proposed_config"] = proposed

    rationale = (j or {}).get("rationale", "")
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

    msg += (
        "\nReply with 'confirm' to proceed, or tell me changes in natural language "
        "(e.g., 'use Date as ds, Sales as y, add Price regressor, forecast 2 months')."
    )

    state["assistant_message"] = msg
    return state


async def user_confirmation_node(state: AgentState) -> AgentState:
    proposed = state.get("proposed_config") or {}
    user_msg = (state.get("user_message") or "").strip()
    msg_norm = user_msg.lower().strip()

    if not user_msg:
        state["assistant_message"] = "Please confirm the proposed ds/y/regressors, or specify changes."
        return state

    yes_tokens = {"yes", "y", "sure", "yeah", "yep"}
    no_tokens = {"no", "n", "nope"}
    confirm_tokens = {"confirm", "confirmed", "go ahead", "proceed"}

    # ---------------------------------------------------------
    # 1) If we have pending_config from a clarifying question:
    #    - yes/no acts on it
    #    - ANY OTHER message is treated as a NEW instruction
    #      (do NOT block the user behind yes/no)
    # ---------------------------------------------------------
    pending = state.get("pending_config")
    if pending:
        if msg_norm in yes_tokens:
            updated = _normalize_config(pending, proposed)
            state["proposed_config"] = updated
            state["confirmed_config"] = updated
            state.pop("pending_config", None)

            state["assistant_message"] = (
                "Confirmed configuration:\n"
                f"- model: Prophet\n"
                f"- ds: {updated['ds_col']}\n"
                f"- y: {updated['y_col']}\n"
                f"- regressors: {updated['regressors']}\n"
                f"- freq: {updated['freq']}\n"
                f"- periods: {updated['periods']}\n\n"
                "Generating code and running the forecast now."
            )
            return state

        if msg_norm in no_tokens:
            state.pop("pending_config", None)
            state["assistant_message"] = (
                "Okay — I won’t apply that pending change. "
                "Please specify the exact update you want (e.g., 'forecast 2 months') or reply 'confirm'."
            )
            return state

        # IMPORTANT: user gave a new instruction; drop pending and continue processing normally
        state.pop("pending_config", None)

    # ---------------------------------------------------------
    # 2) Heuristic: detect forecast horizon like:
    #    - "forecast next 2 months"
    #    - "forecast for 8 weeks"
    #    - "forecast 60 days"
    #
    #    We update BOTH freq and periods deterministically:
    #      days  -> freq="D", periods=N
    #      weeks -> freq="W", periods=N
    #      months-> freq="M", periods=N
    #      years -> freq="M", periods=N*12
    # ---------------------------------------------------------
    m = re.search(
        r"(?:forecast\s*)?(?:for\s*)?(?:next\s*)?(\d+)\s*(day|days|d|week|weeks|w|month|months|m|year|years|y)\b",
        msg_norm,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)

        if unit in ("day", "days", "d"):
            freq = "D"
            periods = n
        elif unit in ("week", "weeks", "w"):
            freq = "W"
            periods = n
        elif unit in ("month", "months", "m"):
            freq = "M"
            periods = n
        else:  # year/years/y
            freq = "M"
            periods = n * 12

        updated = _normalize_config({"freq": freq, "periods": periods}, proposed)
        state["proposed_config"] = updated

        state["assistant_message"] = (
            "Updated proposed configuration:\n"
            f"- model: Prophet\n"
            f"- ds: {updated['ds_col']}\n"
            f"- y: {updated['y_col']}\n"
            f"- regressors: {updated['regressors']}\n"
            f"- freq: {updated['freq']}\n"
            f"- periods: {updated['periods']}\n\n"
            "Reply with 'confirm' to proceed, or specify further changes."
        )
        return state

    # ---------------------------------------------------------
    # 3) Direct confirm (no pending_config)
    # ---------------------------------------------------------
    if msg_norm in confirm_tokens:
        confirmed = _normalize_config(proposed, proposed)
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

    # ---------------------------------------------------------
    # 4) Otherwise, interpret via LLM (modify / ask_clarifying)
    # ---------------------------------------------------------
    user = f"""proposed_config = {proposed}\n\nuser_message = {user_msg}"""
    j = await chat_json(CONFIRMATION_INTERPRETER_PROMPT, user)

    action = (j.get("action") or "").lower().strip()
    msg_to_user = (j.get("message_to_user") or "").strip()

    if action == "confirm":
        confirmed = _normalize_config(proposed, proposed)
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
        raw_cfg = j.get("config") or {}
        updated = _normalize_config(raw_cfg, proposed)
        state["proposed_config"] = updated
        state["assistant_message"] = (
            (msg_to_user + "\n\n" if msg_to_user else "")
            + "Updated proposed configuration:\n"
            f"- model: Prophet\n"
            f"- ds: {updated['ds_col']}\n"
            f"- y: {updated['y_col']}\n"
            f"- regressors: {updated['regressors']}\n"
            f"- freq: {updated['freq']}\n"
            f"- periods: {updated['periods']}\n\n"
            "Reply with 'confirm' to proceed, or specify further changes."
        )
        return state

    if action == "ask_clarifying":
        raw_cfg = j.get("config") or {}
        pending_cfg = _normalize_config(raw_cfg, proposed)
        state["pending_config"] = pending_cfg

        summary = (
            "Pending update (reply 'yes' to apply, 'no' to ignore):\n"
            f"- model: Prophet\n"
            f"- ds: {pending_cfg.get('ds_col','')}\n"
            f"- y: {pending_cfg.get('y_col','')}\n"
            f"- regressors: {pending_cfg.get('regressors',[])}\n"
            f"- freq: {pending_cfg.get('freq','D')}\n"
            f"- periods: {pending_cfg.get('periods',30)}\n"
        )

        state["assistant_message"] = (
            (msg_to_user.strip() + "\n\n" if msg_to_user else "")
            + summary
        )
        return state

    state["assistant_message"] = msg_to_user or "Please reply 'confirm' or specify the change explicitly."
    return state


async def codegen_node(state: AgentState) -> AgentState:
    config = state.get("confirmed_config")
    if not config:
        return state

    user = f"confirmed_config = {config}"
    code = await chat_text(CODEGEN_PROMPT, user)
    state["generated_code"] = code
    print(code)
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
    if state.get("exec_error"):
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

    return state
