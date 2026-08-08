import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HorseMicroTracer:
    """單匹馬「微觀逐行印出」(Micro-Tracing) 數據洩漏檢驗器

    檢驗標準：
    1. 第 1 場比賽：特徵值必須是 NaN 或預設 Baseline（不可包含當場比賽結果）。
    2. 第 N 場比賽：特徵值絕對不能包含第 N 場比賽自己的結果！
       如果第 N 場跑第 1 名，而第 N 場的「歷史勝率」欄位當場立刻上升，
       代表 shift(1) 未發揮作用，產生當場數據洩漏 (Data Leakage)。
    """

    def __init__(
        self,
        date_col: str = "date",
        horse_id_col: str = "horse_id",
        target_col: str = "placing",
        race_id_col: str = "race_id",
    ):
        self.date_col = date_col
        self.horse_id_col = horse_id_col
        self.target_col = target_col
        self.race_id_col = race_id_col

    def _resolve_date_column(self, df: pd.DataFrame) -> str:
        """自動辨識 DataFrame 中的日期欄位名稱"""
        if self.date_col in df.columns:
            return self.date_col
        elif "race_date" in df.columns:
            return "race_date"
        else:
            raise KeyError(
                f"【錯誤】DataFrame 中找不到日期欄位 '{self.date_col}' 或 'race_date'！"
            )

    def find_candidate_horses(
        self, df: pd.DataFrame, min_races: int = 5, max_races: int = 10
    ) -> List[str]:
        """過濾出出賽次數介於 min_races 與 max_races 之間的馬匹 ID 清單"""
        if self.horse_id_col not in df.columns:
            raise KeyError(
                f"【錯誤】DataFrame 中找不到馬匹識別欄位 '{self.horse_id_col}'！"
            )

        race_counts = df.groupby(self.horse_id_col).size()
        candidates = race_counts[
            (race_counts >= min_races) & (race_counts <= max_races)
        ].index.tolist()
        return candidates

    def auto_detect_rolling_cols(self, df: pd.DataFrame) -> List[str]:
        """自動偵測 DataFrame 中屬於滾動/平滑/歷史統計的欄位"""
        keywords = ["rolling", "smooth", "rate", "stat", "hist", "mean", "avg", "win_rate"]
        exclude_cols = {
            self.horse_id_col,
            self.race_id_col,
            self.date_col,
            "race_date",
            self.target_col,
            "is_win",
            "is_top3",
            "relevance_score",
            "finish_time_sec",
            "win_odds",
        }

        detected = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords) and pd.api.types.is_numeric_dtype(df[col]):
                detected.append(col)

        return detected

    def trace_horse(
        self,
        df: pd.DataFrame,
        horse_id: Optional[str] = None,
        min_races: int = 5,
        max_races: int = 10,
        rolling_cols: Optional[List[str]] = None,
        baseline_val: Optional[float] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, bool, Dict[str, Any]]:
        """執行單匹馬微觀逐行印出與洩漏檢驗

        :param df: 包含賽事與特徵的 DataFrame
        :param horse_id: 指定馬匹 ID (若為 None，自動挑選 5~10 場出賽的馬匹)
        :param min_races: 自動篩選時的最少出賽次數
        :param max_races: 自動篩選時的最大出賽次數
        :param rolling_cols: 欲檢驗的滾動特徵欄位清單 (若為 None 自動偵測)
        :param baseline_val: 第 1 場特徵允許的 Baseline 預設值 (若為 None 則預期為 NaN)
        :param verbose: 是否列印格式化表格與日誌
        :return: (trace_df, is_passed, report_dict)
        """
        date_col_used = self._resolve_date_column(df)

        # 1. 決定目標馬匹
        if horse_id is None:
            candidates = self.find_candidate_horses(df, min_races, max_races)
            if not candidates:
                all_counts = df.groupby(self.horse_id_col).size()
                candidates = all_counts[all_counts >= 2].index.tolist()
                if not candidates:
                    raise ValueError("【錯誤】DataFrame 中無任何出賽次數 >= 2 的馬匹可供微觀檢驗！")
                logger.warning(
                    f"⚠️ 未找到出賽 {min_races}~{max_races} 次的馬匹，自動改選出賽次數={all_counts[candidates[0]]} 的馬匹: {candidates[0]}"
                )
            horse_id = candidates[0]

        # 2. 過濾特定馬匹並按日期嚴格遞增排序
        horse_df = df[df[self.horse_id_col] == horse_id].copy()
        if horse_df.empty:
            raise ValueError(f"【錯誤】在 DataFrame 中找不到馬匹 ID: '{horse_id}'")

        horse_df[date_col_used] = pd.to_datetime(horse_df[date_col_used])
        horse_df = horse_df.sort_values(by=date_col_used).reset_index(drop=True)

        if "is_win" not in horse_df.columns and self.target_col in horse_df.columns:
            horse_df["is_win"] = (horse_df[self.target_col] == 1).astype(int)

        # 3. 確定評估的滾動欄位
        if rolling_cols is None or len(rolling_cols) == 0:
            rolling_cols = self.auto_detect_rolling_cols(horse_df)

        if not rolling_cols:
            raise ValueError("【錯誤】未指定且未能自動偵測出任何滾動統計特徵欄位 (rolling_cols)！")

        # 4. 逐行比對與數據洩漏自動判讀
        leakage_flags = []
        detailed_reports = []
        is_passed = True

        for idx, row in horse_df.iterrows():
            race_no = idx + 1
            row_placing = row.get(self.target_col, np.nan)
            row_is_win = row.get("is_win", 0 if row_placing != 1 else 1)

            row_leak_cols = []

            for col in rolling_cols:
                val = row[col]

                # 檢驗標準 1：第 1 場比賽，特徵必須是 NaN 或 Baseline
                if race_no == 1:
                    if pd.notna(val):
                        if baseline_val is not None and np.isclose(val, baseline_val):
                            pass
                        else:
                            if row_is_win == 1 and np.isclose(val, 1.0):
                                row_leak_cols.append(f"{col}(第1場洩漏當場勝率:{val})")

                # 檢驗標準 2：第 N 場比賽，特徵絕對不能包含第 N 場自己的賽果
                else:
                    prev_races = horse_df.iloc[:idx]
                    prev_wins = prev_races["is_win"].values if "is_win" in prev_races.columns else (prev_races[self.target_col] == 1).values

                    # 若前 N-1 場勝率為 0，而第 N 場拿到第 1 名：特徵不可於當場立刻飆升 > 0
                    if np.sum(prev_wins) == 0 and row_is_win == 1:
                        if pd.notna(val) and val > 0.001:
                            row_leak_cols.append(f"{col}(第{race_no}場洩漏當場冠軍:值飆升至{val:.4f})")

                    # 若前 N-1 場全勝(勝率1.0)，而第 N 場落敗：特徵當場不可陡降
                    elif np.all(prev_wins == 1) and row_is_win == 0 and len(prev_wins) > 0:
                        if pd.notna(val) and val < 0.999:
                            row_leak_cols.append(f"{col}(第{race_no}場洩漏當場敗績:值陡降至{val:.4f})")

            if row_leak_cols:
                is_passed = False
                leakage_flags.append("🚨 LEAKAGE!")
                detailed_reports.append(f"第 {race_no} 場存在洩漏: " + ", ".join(row_leak_cols))
            else:
                leakage_flags.append("✅ OK")

        horse_df["_leakage_status"] = leakage_flags

        # 5. 格式化主控台肉眼比對印出 (Micro-Tracing Output)
        if verbose:
            if is_passed:
                print("🎉 [檢驗結果: 通過] 特徵 shift(1) 生效，第 N 場無當場賽果洩漏問題！")
            else:
                print("❌ [檢驗結果: 警告] 偵測到當場數據洩漏 (Data Leakage)！")
                for r in detailed_reports:
                    print(f"   ⚠️ {r}")
            print("=" * 90 + "\n")

        report = {
            "horse_id": horse_id,
            "total_races": len(horse_df),
            "is_passed": is_passed,
            "evaluated_rolling_cols": rolling_cols,
            "leakage_details": detailed_reports,
        }

        return horse_df, is_passed, report


def inspect_horse_rolling_features(
    df: pd.DataFrame,
    horse_id: Optional[str] = None,
    rolling_cols: Optional[List[str]] = None,
    min_races: int = 5,
    max_races: int = 10,
) -> Tuple[pd.DataFrame, bool]:
    """快捷 API：執行單匹馬微觀逐行比對 (Micro-Tracing)"""
    tracer = HorseMicroTracer()
    trace_df, is_passed, _ = tracer.trace_horse(
        df=df,
        horse_id=horse_id,
        rolling_cols=rolling_cols,
        min_races=min_races,
        max_races=max_races,
        verbose=True,
    )
    return trace_df, is_passed