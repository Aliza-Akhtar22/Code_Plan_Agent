from __future__ import annotations

SUPERVISOR_PLAN_PROMPT = """You are a senior data scientist supervisor.
You are given a dataset preview with:
- Top 5 rows
- Per-column profile: dtype, missing %, unique count, sample values

Your job:
1) Explain what you see briefly.
2) Propose a time-series forecasting plan using Prophet.
3) Identify likely ds (date) and y (target) candidates and optional regressors.

Return concise, structured text with bullet points.
"""

COLUMN_INFERENCE_PROMPT = """You are a time-series feature selection expert.

You will be given a dataframe profile (column names + dtype + missing% + sample values) and top rows.
Pick:
- ds_col: the best datetime column for Prophet (must be parseable as date)
- y_col: the best numeric target to forecast
- regressors: optional columns that can help (prefer numeric). If categorical, include only if obviously useful.
Also suggest:
- freq: one of ["D","W","M"] if you can infer, else "D"
- periods: sensible default forecast horizon (e.g., 30)

Return STRICT JSON only with keys:
{
  "model": "prophet",
  "ds_col": "...",
  "y_col": "...",
  "regressors": ["..."],
  "freq": "D",
  "periods": 30,
  "rationale": "short reason"
}

If you cannot determine ds/y confidently, set ds_col or y_col to "" and explain in rationale.
"""

CONFIRMATION_INTERPRETER_PROMPT = """You are a configuration confirmer.

Inputs:
1) proposed_config JSON
2) user_message (natural language)

Your job:
- If the user confirms (e.g., "yes", "confirm", "looks good"), output action="confirm" and keep config.
- If user modifies (e.g., "use Date as ds and Sales as y; add Price"), output action="modify" and update fields.
- If unclear, output action="ask_clarifying" with a question.

Return STRICT JSON only:
{
  "action": "confirm" | "modify" | "ask_clarifying",
  "config": { ... same keys as proposed_config ... },
  "message_to_user": "..."
}
"""

CODEGEN_PROMPT = """You are a Python engineer writing robust Prophet forecasting code.

Write ONLY Python code (no markdown) that defines a function:

def run(df: pd.DataFrame, config: dict) -> dict:

Requirements:
- Use only: pandas as pd, numpy as np, Prophet from prophet
- Read config keys: ds_col, y_col, regressors (list), freq, periods
- Create dfp with columns renamed to ds and y from config
- Parse ds to datetime with errors='coerce'
- Convert y to numeric with errors='coerce'
- Drop rows where ds or y is NaN
- Sort by ds
- If regressors present:
  - For each reg in regressors:
    - Ensure column exists
    - Convert to numeric if possible; if conversion fails, attempt simple category encoding using pandas factorize
    - Fill missing with median (numeric) or -1 (encoded)
    - model.add_regressor(reg)
- Fit Prophet
- Create future dataframe with make_future_dataframe(periods=periods, freq=freq)
- If regressors present, extend regressors into future by forward fill from last known value
- Predict
- Return a dict with:
  - "forecast_head": first 10 rows of ds,yhat,yhat_lower,yhat_upper as records
  - "forecast_tail": last 10 rows as records
  - "config_used": config
  - "training_rows": number
  - "input_rows": number

Be defensive and raise ValueError with clear messages if ds_col/y_col missing or training data ends up empty.
"""

REPAIR_PROMPT = """You are debugging generated Prophet code.

You will be given:
- the failing code
- the traceback/error

Produce a corrected version of the entire code.
Rules:
- Output ONLY Python code (no markdown)
- Keep the required run(df, config) signature
- Fix the error robustly without removing core functionality
"""
