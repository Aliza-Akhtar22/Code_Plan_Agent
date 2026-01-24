from __future__ import annotations

import io
import uuid
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from app.core.profiling import preview_payload
from app.core.storage import DatasetStore
from app.graph.builder import build_graph
from app.graph.state import AgentState

router = APIRouter()

# In-memory MVP stores
store = DatasetStore()
graph = build_graph()

# Persist AgentState per dataset_id across chat turns (MVP memory store)
STATE_STORE: Dict[str, AgentState] = {}


class ChatRequest(BaseModel):
    dataset_id: str = Field(..., description="Returned from /upload")
    message: str = Field(..., description="User message")
    show_code: bool = Field(False, description="If true, include generated code when available")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a CSV file, store it server-side (in-memory MVP), and return dataset_id and head preview.
    """
    content = await file.read()
    dataset_id = str(uuid.uuid4())

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to read CSV: {type(e).__name__}: {e}",
        }

    store.put(dataset_id, df)

    # New dataset => new conversation state
    STATE_STORE.pop(dataset_id, None)

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "filename": file.filename,
        "preview": preview_payload(df),
    }


@router.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Main chat endpoint. Uses LangGraph to progress through:
    preview -> plan -> infer columns -> confirm/modify -> codegen -> exec -> repair on error -> results

    IMPORTANT (MVP):
    - Persists AgentState per dataset_id so "confirm" does not re-infer a new config and override user changes.
    """
    df = store.get(req.dataset_id)
    if df is None:
        return {"ok": False, "error": "Invalid dataset_id. Upload first via /upload."}

    prev_state = STATE_STORE.get(req.dataset_id)

    if prev_state:
        # Rehydrate existing state for this dataset_id
        state: AgentState = dict(prev_state)  # shallow copy is fine
        state["df"] = df
        state["df_preview"] = preview_payload(df)  # refresh (optional but safe)
        state["user_message"] = req.message
        state["show_code"] = req.show_code

        # Reset execution envelope for this turn
        state["attempt"] = 0
        state["exec_output"] = None
        state["exec_error"] = None
        state["traceback"] = None
    else:
        # First message for this dataset_id
        state = AgentState(
            dataset_id=req.dataset_id,
            user_message=req.message,
            df=df,
            df_preview=preview_payload(df),
            attempt=0,
            max_attempts=2,
            show_code=req.show_code,
            # Plan metadata defaults (optional; nodes.py will set/update these)
            plan_version=0,
            plan_last_updated="",
            plan_text="",
        )

    final_state = await graph.ainvoke(state)

    # Persist for next turn (store without df to reduce memory usage)
    to_store: AgentState = dict(final_state)
    to_store.pop("df", None)
    STATE_STORE[req.dataset_id] = to_store

    response: Dict[str, Any] = {
        "ok": True,
        "assistant_message": final_state.get("assistant_message") or "",
        "preview": final_state.get("df_preview"),
        # NEW: return plan information explicitly (useful for a dedicated UI panel)
        "plan_text": final_state.get("plan_text"),
        "plan_version": final_state.get("plan_version", 0),
        "plan_last_updated": final_state.get("plan_last_updated"),
        "proposed_config": final_state.get("proposed_config"),
        "confirmed_config": final_state.get("confirmed_config"),
        "results": final_state.get("exec_output"),
        "error": final_state.get("exec_error"),
    }

    if req.show_code:
        response["generated_code"] = final_state.get("generated_code")

    return response
