"""Betting rule 抽象介面。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class BettingRule(ABC):
    """單一固定注碼規則。

    子類只負責 select()；結算由 BetEvaluator 統一處理。
    """

    rule_id: str = ""
    name: str = ""
    description: str = ""
    version: int = 1

    # 子類可覆寫：select 需要的欄位
    required_cols: Sequence[str] = ()

    def is_available(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        return all(c in df.columns for c in self.required_cols)

    @abstractmethod
    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        """回傳符合條件的列（可多列／場）。

        ctx 可傳入 overlay_threshold、score_col 等執行期參數。
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }