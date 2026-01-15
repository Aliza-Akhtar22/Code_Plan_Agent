from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class ColumnConfig(TypedDict, total=False):
    model: str
    ds_col: str
    y_col: str
    regressors: List[str]
    freq: str
    periods: int


class AgentState(TypedDict, total=False):
    dataset_id: str
    user_message: str
    assistant_message: str

    # Data
    df: pd.DataFrame
    df_preview: Dict[str, Any]

    # Plan/config
    plan_text: str
    proposed_config: ColumnConfig
    confirmed_config: ColumnConfig

    # Code execution
    generated_code: str
    exec_output: Optional[Dict[str, Any]]
    exec_error: Optional[str]
    traceback: Optional[str]

    # Retry controls
    attempt: int
    max_attempts: int

    # UI controls
    show_code: bool
