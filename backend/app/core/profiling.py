from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


def _sample_values(series: pd.Series, k: int = 3) -> List[Any]:
    vals = series.dropna().unique().tolist()
    return vals[:k]


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "columns": {}}
    n = max(int(df.shape[0]), 1)

    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        dtype = str(s.dtype)
        unique = int(s.nunique(dropna=True))
        out["columns"][col] = {
            "dtype": dtype,
            "missing_count": missing,
            "missing_pct": round((missing / n) * 100.0, 2),
            "unique_count": unique,
            "sample_values": _sample_values(s),
        }
    return out


def preview_payload(df: pd.DataFrame) -> Dict[str, Any]:
    head = df.head(5).to_dict(orient="records")
    return {
        "head": head,
        "profile": profile_dataframe(df),
        "columns": list(df.columns),
    }
