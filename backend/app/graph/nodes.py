from __future__ import annotations

import traceback as tb
from typing import Any, Dict, List
from app.graph.qa import answer_forecast_qa

import numpy as np
import pandas as pd
import re
from prophet import Prophet

from app.core.profiling import preview_payload
from app.graph.llm import chat_json, chat_text
from app.graph.prompts import (
    CODEGEN_PROMPT,
    COLUMN_INFERENCE_PROMPT,
    CONFIRMATION_INTERPRETER_PROMPT,
    REPAIR_PROMPT,
    SUPERVISOR_PLAN_PROMPT,
)
from app.graph.state import AgentState, ColumnConfig


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


def _render_config_block(cfg: Dict[str, Any], title: str = "Updated proposed configuration:") -> str:
    regs = cfg.get("regressors", [])
    return (
        f"{title}\n"
        f"- model: Prophet\n"
        f"- ds: {cfg.get('ds_col','')}\n"
        f"- y: {cfg.get('y_col','')}\n"
        f"- regressors: {regs}\n"
        f"- freq: {cfg.get('freq','D')}\n"
        f"- periods: {cfg.get('periods',30)}\n"
    )


def _final_ui_message(cfg: Dict[str, Any]) -> str:
    return (
        f"{_render_config_block(cfg)}\n"
        "Reply with 'confirm' to proceed, or specify further changes."
    )


def _colnames(state: AgentState) -> List[str]:
    prev = state.get("df_preview") or {}
    cols = prev.get("columns") or []
    return [str(c) for c in cols]


def _parse_regressor_override(user_msg: str, state: AgentState) -> Dict[str, Any] | None:
    """
    Explicit regressor instructions should REPLACE the regressor list.

    Examples:
    - "T is my regressor" -> ["T"]
    - "regressor is T" -> ["T"]
    - "regressors are T, rh" -> ["T","rh"]

    IMPORTANT:
    - Do NOT scan the entire message for column names (prevents picking up target like 'p').
    - Exclude ds_col and y_col if they appear.
    """
    msg = (user_msg or "").strip()
    if not msg:
        return None

    msg_l = msg.lower()

    # Replacement triggers
    replace_intent = any(
        k in msg_l
        for k in [
            "regressors are",
            "regressor is",
            "is my regressor",
            "as regressor",
            "as regressors",
        ]
    )
    if not replace_intent:
        return None

    cols = _colnames(state)
    cols_l = {c.lower(): c for c in cols}

    # Avoid using ds/y as regressors
    proposed = state.get("proposed_config") or {}
    ds_col = (proposed.get("ds_col") or "").strip()
    y_col = (proposed.get("y_col") or "").strip()
    blocked = {ds_col, y_col, ds_col.lower(), y_col.lower()}

    # 1) Prefer parsing AFTER the regressor phrase
    candidates_text = ""
    for marker in ["regressors are", "regressor is", "is my regressor", "as regressors", "as regressor"]:
        idx = msg_l.find(marker)
        if idx != -1:
            candidates_text = msg[idx + len(marker):].strip()
            break

    picked: List[str] = []

    if candidates_text:
        # Split candidates list
        tokens = re.split(r"[,\n;/]+|\band\b", candidates_text, flags=re.IGNORECASE)
        tokens = [t.strip(" .:-_()[]{}\"'") for t in tokens if t.strip()]
        for t in tokens:
            key = t.lower()
            if key in cols_l:
                c = cols_l[key]
                if c not in blocked and c.lower() not in blocked:
                    picked.append(c)

    # 2) If still nothing, use tight regex near the regressor phrase (NOT whole-message scan)
    if not picked:
        m = re.search(r"\bregressor\s+is\s+([a-zA-Z0-9_.]+)\b", msg_l)
        if not m:
            m = re.search(r"\b([a-zA-Z0-9_.]+)\s+is\s+my\s+regressor\b", msg_l)
        if m:
            token = m.group(1).strip().lower()
            if token in cols_l:
                c = cols_l[token]
                if c not in blocked and c.lower() not in blocked:
                    picked.append(c)

    # Deduplicate while preserving order
    seen = set()
    picked = [x for x in picked if not (x in seen or seen.add(x))]

    return {"regressors": picked}


def _parse_add_regressor(user_msg: str, state: AgentState) -> Dict[str, Any] | None:
    msg_l = (user_msg or "").lower()
    if "add" not in msg_l or "regressor" not in msg_l:
        return None

    cols = _colnames(state)
    cols_l = {c.lower(): c for c in cols}

    # pick first column mentioned after "add"
    for c in cols:
        if re.search(rf"\b{re.escape(c.lower())}\b", msg_l):
            return {"add_regressor": cols_l.get(c.lower(), c)}
    return None


async def supervisor_preview_node(state: AgentState) -> AgentState:
    if "df_preview" not in state:
        state["df_preview"] = preview_payload(state["df"])
    return state


async def plan_node(state: AgentState) -> AgentState:
    # Keep the supervisor plan stored (optional), but do not show it in UI
    user = _format_preview_for_llm(state)
    plan_text = await chat_text(SUPERVISOR_PLAN_PROMPT, user)
    state["plan_text"] = plan_text
    state["assistant_message"] = ""  # keep UI clean
    return state


async def column_inference_node(state: AgentState) -> AgentState:
    user = _format_preview_for_llm(state)
    j = await chat_json(COLUMN_INFERENCE_PROMPT, user)

    proposed: ColumnConfig = _normalize_config(j or {}, {})
    state["proposed_config"] = proposed

    state["assistant_message"] = _final_ui_message(proposed)
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

    # 1) pending yes/no
    pending = state.get("pending_config")
    if pending:
        if msg_norm in yes_tokens:
            updated = _normalize_config(pending, proposed)
            state["proposed_config"] = updated
            state["confirmed_config"] = updated
            state.pop("pending_config", None)
            state["assistant_message"] = "Generating code and running the forecast now."
            return state

        if msg_norm in no_tokens:
            state.pop("pending_config", None)
            state["assistant_message"] = "Okay — please specify the exact update you want or reply 'confirm'."
            return state

        state.pop("pending_config", None)

    # 2) horizon heuristic
    m = re.search(
        r"(?:forecast\s*)?(?:for\s*)?(?:next\s*)?(\d+)\s*"
        r"(day|days|d|week|weeks|w|month|months|m|year|years|y|quarter|quarters|qtr|qtrs|q)\b",
        msg_norm,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)

        base_freq = (proposed.get("freq") or "D").strip()
        if base_freq not in ("D", "W", "M"):
            base_freq = "D"        

        if unit in ("day", "days", "d"):
            freq, periods = "D", n
        elif unit in ("week", "weeks", "w"):
            freq, periods = "W", n
        elif unit in ("month", "months", "m"):
            freq, periods = "M", n
        elif unit in ("year", "years", "y"):
            freq, periods = "M", n * 12            
        else:
            if base_freq == "D":
                freq, periods = "D", n * 90
            elif base_freq == "W":
                freq, periods = "W", n * 13
            else:
                freq, periods = "M", n * 3

        updated = _normalize_config({"freq": freq, "periods": periods}, proposed)
        state["proposed_config"] = updated
        state["assistant_message"] = _final_ui_message(updated)
        return state

    # 3) deterministic regressor override (REPLACE)
    reg_override = _parse_regressor_override(user_msg, state)
    if reg_override is not None:
        # If user explicitly said regressor but we couldn't map, ask a clean clarifying question
        if not reg_override.get("regressors"):
            cols = _colnames(state)
            state["assistant_message"] = (
                "I couldn't identify that regressor column in your dataset. "
                f"Available columns are: {cols}\n"
                "Please type the exact column name for the regressor."
            )
            return state

        updated = _normalize_config(reg_override, proposed)
        state["proposed_config"] = updated
        state["assistant_message"] = _final_ui_message(updated)
        return state

    # 4) deterministic add regressor (ADD)
    add = _parse_add_regressor(user_msg, state)
    if add:
        updated_regs = list(proposed.get("regressors", []) or [])
        r = add["add_regressor"]
        if r not in updated_regs:
            updated_regs.append(r)
        updated = _normalize_config({"regressors": updated_regs}, proposed)
        state["proposed_config"] = updated
        state["assistant_message"] = _final_ui_message(updated)
        return state

    # 5) direct confirm
    if msg_norm in confirm_tokens:
        confirmed = _normalize_config(proposed, proposed)
        state["confirmed_config"] = confirmed
        state["assistant_message"] = "Generating code and running the forecast now."
        return state

    # 6) interpret via LLM (modify / ask_clarifying)
    user = f"""proposed_config = {proposed}\n\nuser_message = {user_msg}"""
    j = await chat_json(CONFIRMATION_INTERPRETER_PROMPT, user)

    action = (j.get("action") or "").lower().strip()
    msg_to_user = (j.get("message_to_user") or "").strip()

    if action == "confirm":
        confirmed = _normalize_config(proposed, proposed)
        state["confirmed_config"] = confirmed
        state["assistant_message"] = "Generating code and running the forecast now."
        return state

    if action == "modify":
        raw_cfg = j.get("config") or {}

        # IMPORTANT: If user text said "X is my regressor" but LLM returned many,
        # override with deterministic parser (replace semantics).
        reg_override2 = _parse_regressor_override(user_msg, state)
        if reg_override2 is not None and reg_override2.get("regressors"):
            raw_cfg["regressors"] = reg_override2["regressors"]

        updated = _normalize_config(raw_cfg, proposed)
        state["proposed_config"] = updated
        state["assistant_message"] = _final_ui_message(updated)
        return state

    if action == "ask_clarifying":
        raw_cfg = j.get("config") or {}
        pending_cfg = _normalize_config(raw_cfg, proposed)
        state["pending_config"] = pending_cfg
        state["assistant_message"] = msg_to_user or "Could you clarify what change you want?"
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
        out = state["exec_output"] or {}
        cfg = (out.get("config_used") or state.get("confirmed_config") or {}) if isinstance(out, dict) else {}
        ds_col = (cfg.get("ds_col") or "ds")
        y_col = (cfg.get("y_col") or "y")

        forecast_rows = out.get("forecast") if isinstance(out, dict) else None
        if not isinstance(forecast_rows, list):
            forecast_rows = []

        preview_n = min(10, len(forecast_rows))
        preview_rows = forecast_rows[:preview_n]

        state["assistant_message"] = (
            "Forecast completed.\n\n"
            f"Training rows used: {out.get('training_rows')}\n"
            f"Input rows: {out.get('input_rows')}\n\n"
            f"Future forecast rows returned: {len(forecast_rows)}\n\n"
            f"Forecast columns: {ds_col}, {y_col}_forecast, {y_col}_lower, {y_col}_upper\n\n"
            "Forecast preview (first few future rows):\n"
            f"{preview_rows}"
        )
        return state

    if state.get("exec_error") and int(state.get("attempt", 0) or 0) >= int(state.get("max_attempts", 2) or 2):
        state["assistant_message"] = (
            "I could not successfully execute the generated code after retries.\n\n"
            f"Error: {state.get('exec_error')}\n\n"
            "If you want, paste the columns you intend for ds/y/regressors and I will tighten the generation."
        )
        return state
    
async def qa_node(state: AgentState) -> AgentState:
    answer = await answer_forecast_qa(state)
    state["assistant_message"] = answer
    return state
