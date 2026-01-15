from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class DatasetStore:
    """
    MVP in-memory dataset store.
    For production: store in disk (parquet) or DB keyed by dataset_id + user/session.
    """
    _data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def put(self, dataset_id: str, df: pd.DataFrame) -> None:
        self._data[dataset_id] = df

    def get(self, dataset_id: str) -> Optional[pd.DataFrame]:
        return self._data.get(dataset_id)
