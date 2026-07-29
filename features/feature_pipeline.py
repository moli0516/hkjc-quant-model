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

        working_df = df.copy()
        collected_feature_dfs: List[pd.DataFrame] = []
        
        # 💡 用於追蹤已存在的特徵欄位名稱 (包含原始 df 的欄位)
        existing_cols = set(working_df.columns)

        for gen in self.generators:
            gen_name = gen.__class__.__name__
            try:
                gen_features = gen.generate(working_df)
                
                LeakageGuard.validate_feature_dataframe(
                    gen_features, self.key_cols
                )

                # 1. 提煉特徵欄位 (排除 key_cols)
                feature_cols = [
                    c for c in gen_features.columns if c not in self.key_cols
                ]

                if feature_cols:
                    gen_features_aligned = gen_features.reindex(working_df.index)

                    # 2. 轉 float32
                    for col in feature_cols:
                        if gen_features_aligned[col].dtype == "float64":
                            gen_features_aligned[col] = gen_features_aligned[col].astype("float32")

                    # 🚨 關鍵防護：檢查是否有與先前重複的特徵名稱！
                    duplicate_cols = [c for c in feature_cols if c in existing_cols]
                    if duplicate_cols:
                        print(f"  ⚠️ [{gen_name}] 警告：發現重複特徵 {duplicate_cols}，將自動排除重複欄位")
                        # 覆蓋或過濾掉重複欄位
                        feature_cols = [c for c in feature_cols if c not in duplicate_cols]

                    if feature_cols:
                        pure_features = gen_features_aligned[feature_cols]
                        collected_feature_dfs.append(pure_features)

                        # 更新 existing_cols 記錄
                        existing_cols.update(feature_cols)

                        # 併入 working_df 供後續 Generator 使用
                        working_df = pd.concat([working_df, pure_features], axis=1)

                print(
                    f"  ✅ [{gen_name}] 成功 (新特徵數: {len(feature_cols)})"
                )

            except Exception as e:
                print(f"  ❌ [{gen_name}] 執行失敗: {e}")
                raise e
            finally:
                gc.collect()

        print("⚡ [Pipeline] 正在一次性拼裝最終特徵矩陣...")
        base_keys = working_df[self.key_cols]
        feature_matrix = pd.concat([base_keys] + collected_feature_dfs, axis=1)

        LeakageGuard.validate_feature_dataframe(feature_matrix, self.key_cols)
        
        del working_df, collected_feature_dfs
        gc.collect()

        total_features = feature_matrix.shape[1] - len(self.key_cols)
        print(
            f"🎉 [Pipeline] 特徵工程完成！總特徵數: {total_features} 個"
        )
        return feature_matrix

    def run_and_save(
        self,
        df: pd.DataFrame,
        output_type: str = "sqlite",
        destination: str = "data/features.db",
        table_name: str = "feature_matrix",
    ) -> pd.DataFrame:
        features_df = self.run(df)
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        output_type_lower = output_type.lower()
        if output_type_lower == "sqlite":
            with sqlite3.connect(destination) as conn:
                features_df.to_sql(
                    table_name, conn, if_exists="replace", index=False
                )
            print(f"💾 已存入 SQLite: '{destination}' (Table: {table_name})")

        elif output_type_lower in ["parquet", "pq"]:
            features_df.to_parquet(destination, index=False)
            print(f"💾 已存入 Parquet: '{destination}'")

        else:
            raise ValueError(f"不支援的 output_type: '{output_type}'")

        return features_df