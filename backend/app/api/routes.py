from __future__ import annotations

import io
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from app.core.storage import DatasetStore
from app.core.profiling import preview_payload
from app.graph.builder import build_graph
from app.graph.state import AgentState

router = APIRouter()
store = DatasetStore()
graph = build_graph()


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
    """
    df = store.get(req.dataset_id)
    if df is None:
        return {"ok": False, "error": "Invalid dataset_id. Upload first via /upload."}

    # Initialize state each chat turn.
    # MVP: stateless chat; we re-run inference each time and interpret user confirmation.
    # Next step: persist state per dataset_id or session_id.
    state: AgentState = AgentState(
        dataset_id=req.dataset_id,
        user_message=req.message,
        df=df,
        df_preview=preview_payload(df),
        attempt=0,
        max_attempts=2,
        show_code=req.show_code,
    )

    final_state = await graph.ainvoke(state)

    response: Dict[str, Any] = {
        "ok": True,
        "assistant_message": final_state.get("assistant_message") or "",
        "preview": final_state.get("df_preview"),
        "proposed_config": final_state.get("proposed_config"),
        "confirmed_config": final_state.get("confirmed_config"),
        "results": final_state.get("exec_output"),
        "error": final_state.get("exec_error"),
    }

    if req.show_code:
        response["generated_code"] = final_state.get("generated_code")

    return response
