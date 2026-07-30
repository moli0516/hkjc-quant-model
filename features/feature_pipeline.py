import gc
import os
import sqlite3
from typing import List, Optional
import pandas as pd

from features.generators import load_all_generators
from features.utils import LeakageGuard


class FeaturesPipeline:
    """HKJC Quant 量化特徵工程主管道 (動態掃描 Plugin 模式 + 零拷貝 & 零碎片化極速優化)。"""

    def __init__(self, key_cols: Optional[List[str]] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]
        # 🚀 自動掃描並依優先權排序載入所有 Generator
        self.generators = load_all_generators(key_cols=self.key_cols)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("[FeaturesPipeline] 輸入的 DataFrame 為空！")

        print(
            f"🚀 [Pipeline] 開始執行特徵生成 (共自動載入 {len(self.generators)} 個 Generators, 輸入筆數: {len(df)})"
        )

        # -------------------------------------------------------------------------
        # 🔒 [防洩漏靈魂步驟 1] 保留原始 Index 順序，並強制按【時間】嚴格排序
        # -------------------------------------------------------------------------
        original_index = df.index

        if "date" in df.columns:
            # 確保按照時間序列排序，避免 Rolling/Expanding 計算時發生未來的資料洩漏
            working_df = df.sort_values(["date", "race_id", "horse_id"]).copy()
        else:
            working_df = df.copy()

        # -------------------------------------------------------------------------
        # 🛡️ [防洩漏靈魂步驟 2] 驗證時間與 Target 資料合規性
        # -------------------------------------------------------------------------
        if hasattr(LeakageGuard, "check_dataframe"):
            LeakageGuard.check_dataframe(working_df)

        collected_feature_dfs: List[pd.DataFrame] = []

        # 💡 用於追蹤已存在的特徵欄位名稱 (包含原始 df 的欄位)，排除 key_cols 避免干擾
        existing_cols = set(working_df.columns)

        for gen in self.generators:
            gen_name = gen.__class__.__name__
            print(f"  ⚡ 正在執行 Generator: {gen_name}...")

            # 執行 Generator 生成特徵
            feat_df = gen.generate(working_df)

            if feat_df is None or feat_df.empty:
                print(f"  ⚠️ [Warning] {gen_name} 未產出任何特徵，跳過。")
                continue

            # ---------------------------------------------------------------------
            # 💡 [欄位防重名與數據清理]
            # ---------------------------------------------------------------------
            # 1. 只挑出非 Key 欄位且尚未在 existing_cols 中出現過的「新特徵欄位」
            new_feature_cols = [
                col
                for col in feat_df.columns
                if col not in self.key_cols and col not in existing_cols
            ]

            if not new_feature_cols:
                # 檢查是否因為缺少必要欄位而直接 returned 特徵空殼
                gen_cols_non_key = [
                    col for col in feat_df.columns if col not in self.key_cols
                ]
                if not gen_cols_non_key:
                    print(
                        f"  ⚠️ [Warning] {gen_name} 因缺少輸入必要欄位，未生成任何新特徵。"
                    )
                else:
                    print(
                        f"  ℹ️ {gen_name} 產出的欄位 ({gen_cols_non_key}) 皆已存在於輸入數據中，跳過。"
                    )
                continue

            # 僅保留純新特徵欄位 (Key 欄位將在最後統一拼合)
            clean_feat_df = feat_df[new_feature_cols].copy()

            # 2. 自動將 float64 轉為 float32 以節省記憶體並防止碎片化
            float64_cols = clean_feat_df.select_dtypes(
                include=["float64"]
            ).columns
            if len(float64_cols) > 0:
                clean_feat_df[float64_cols] = clean_feat_df[
                    float64_cols
                ].astype("float32")

            # 3. 更新已存在的欄位集合
            existing_cols.update(new_feature_cols)

            # 4. 收集結果 DataFrame
            collected_feature_dfs.append(clean_feat_df)

            # 5. 選擇性動態將新特徵併回 working_df，供後續有依賴關係的 Generator 使用
            # (例如 PaceStrategyGenerator 依賴 RunningPositionGenerator 的產出)
            working_df = pd.concat([working_df, clean_feat_df], axis=1)

            # 手動釋放暫存記憶體
            gc.collect()

        if not collected_feature_dfs:
            raise RuntimeError(
                "[FeaturesPipeline] 没有任何 Generator 成功生成特徵！"
            )

        # -------------------------------------------------------------------------
        # 🚀 [零碎片化特徵合併、Key 欄位保留與 Index 恢復]
        # -------------------------------------------------------------------------
        print("📦 [Pipeline] 正在高效併合所有特徵矩陣 (含 Key 欄位)...")

        # 1. 提取 Key 欄位 (確保包含 self.key_cols，例如 race_id, horse_id)
        present_keys = [
            col for col in self.key_cols if col in working_df.columns
        ]
        keys_df = working_df[present_keys].copy()

        # 2. 一次性併合 Keys 與所有生成出的特徵 DataFrame
        generated_features_df = pd.concat(collected_feature_dfs, axis=1)
        final_features_df = pd.concat([keys_df, generated_features_df], axis=1)

        # 🔒 [防洩漏與對齊靈魂步驟 3] 恢復為傳入時的原始 Index 順序
        final_features_df = final_features_df.reindex(original_index)

        print(
            f"✅ [Pipeline] 特徵工程完成！總共產出 {final_features_df.shape[1]} 個欄位 (含 Keys: {present_keys})，筆數: {len(final_features_df)}"
        )

        return final_features_df