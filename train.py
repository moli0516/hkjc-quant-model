import logging
import pandas as pd
import json
import os
import itertools
from typing import List, Tuple, Dict, Any
from models.inference import Quant_inference_engine
from config.settings import settings
from backtests.oot_backtest_place import PreTrainedBacktester

# 設定日誌
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureSearchPipeline:
    def __init__(self, base_features: List[str], candidate_features: List[str], max_add_features: int = 2):
        self.logger = logging.getLogger("FeatureSearchPipeline")
        self.logger.setLevel(logging.INFO)
        
        self.base_features = base_features
        self.candidate_features = candidate_features
        self.max_add_features = max_add_features
        self.target = settings.target
        
        # 內部狀態儲存
        self.df_all = pd.DataFrame()
        self.df_hist = pd.DataFrame()
        self.search_results = []
        self.best_run = {}

    def load_and_split_data(self) -> "FeatureSearchPipeline":
        """步驟 1 & 2: 載入歷史大盤數據並嚴格切分時間軸"""
        self.logger.info("💾 正在載入歷史大盤數據...")
        self.df_all = pd.read_parquet(settings.features_parquet_path)
        
        self.logger.info("✂️ 正在進行時間序列切分...")
        self.df_hist = self.df_all[self.df_all['date'] < '2025-07-01'].copy()
        return self

    def generate_combinations(self) -> List[Tuple[str, ...]]:
        """基於候選特徵生成窮舉組合清單"""
        combinations = [()]  # 初始包含空元組作為 Baseline
        for r in range(1, self.max_add_features + 1):
            combinations.extend(list(itertools.combinations(self.candidate_features, r)))
        return combinations

    def evaluate_single_combination(self, combo: Tuple[str, ...]) -> Dict[str, Any]:
        """核心評估方法：串聯訓練、預測與回測"""
        current_features = list(self.base_features) + list(combo)
        
        # 實例化推論引擎與訓練
        engine = Quant_inference_engine(features=current_features, target=self.target)
        metrics = engine.train_model(self.df_hist, val_day=180)
        
        # 實例化回測器與預測
        tester = PreTrainedBacktester(model=engine.model, target_col=self.target)
        df_recent = tester.load_recent_test_data()
        
        df_recent['pred_prob'] = engine.predict(df_recent[current_features], df_recent['race_unique_id'])
        df_recent['pred_rank'] = df_recent.groupby('race_unique_id')['pred_prob'].rank(ascending=False, method='min')
        
        # 執行回測與指標計算
        tester.run_backtest(df_recent)
        
        top1_horses = df_recent[df_recent['pred_rank'] == 1]
        win_rate = top1_horses[self.target].mean() if len(top1_horses) > 0 else 0.0
        
        return {
            "combo": combo,
            "features_list": current_features,
            "win_rate": win_rate,
            "metrics": metrics,
            "engine": engine
        }

    def run_search(self) -> "FeatureSearchPipeline":
        """執行特徵組合窮舉搜尋主迴圈"""
        if self.df_hist.empty:
            self.load_and_split_data()
            
        all_combos = self.generate_combinations()
        total_combos = len(all_combos)
        self.logger.info(f"🧩 總共將評估 {total_combos} 組特徵組合（含 Baseline）...")
        
        for idx, combo in enumerate(all_combos, 1):
            combo_name = "Baseline (None)" if not combo else " + ".join(combo)
            self.logger.info(f"⏳ [{idx}/{total_combos}] 正在評估組合: {combo_name}")
            
            try:
                res = self.evaluate_single_combination(combo)
                self.search_results.append(res)
            except Exception as e:
                self.logger.error(f"❌ 組合 [{combo_name}] 運行失敗: {str(e)}")
                continue
                
        # 排序並定位最優成果
        df_res = pd.DataFrame(self.search_results).sort_values(by="win_rate", ascending=False).reset_index(drop=True)
        self.best_run = df_res.iloc[0].to_dict()
        
        self._print_summary(df_res)
        return self

    def _print_summary(self, df_res: pd.DataFrame):
        """列印 Top 5 結果報表"""
        print("\n" + "="*60)
        print("🏆 特徵組合回測尋優結束 - TOP 5 最佳組合")
        print("="*60)
        for i, row in df_res.head(5).iterrows():
            c_name = "Baseline (None)" if not row['combo'] else " + ".join(row['combo'])
            print(f"Rank {i+1}: OOT勝率 {row['win_rate']:.2%} | 組合: [{c_name}]")
        print("="*60 + "\n")

    def save_best_artifacts(self, model_dir: str = "models"):
        """將最終選定勝率最高之模型與配置進行持久化儲存"""
        if not self.best_run:
            self.logger.error("⚠️ 無最優模型紀錄，請先執行 run_search()")
            return
            
        self.logger.info(f"✨ 正在儲存最優模型配置...")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "xgbranker_best.json")
        features_path = os.path.join(model_dir, "features_config.json")
        
        # 儲存
        self.best_run["engine"].save_model(model_path)
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(self.best_run["features_list"], f, ensure_ascii=False, indent=4)
            
        self.logger.info(f"📝 最佳特徵配置已成功儲存至 {features_path}")
        self._log_best_metrics()

    def _log_best_metrics(self):
        """輸出最優模型的詳細指標與特徵重要性"""
        metrics = self.best_run["metrics"]
        self.logger.info(f"🏆 最佳訓練樹棵數 (Trees): {metrics['best_iteration']}")
        self.logger.info(f"📊 驗證集最佳 NDCG 分數: {metrics['best_score']:.4f}")
        
        sorted_importance = sorted(metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        self.logger.info("🔥 最佳模型關鍵特徵重要性：")
        for feat, imp in sorted_importance:
            self.logger.info(f"  - {feat}: {imp:.4f}")

    @staticmethod
    def analyze_feature_differentiation(file_path: str):
        """靜態方法：對最終產出的預測明細進行特徵顯著性檢定"""
        if not os.path.exists(file_path):
            logging.warning(f"⚠️ 找不到預測明細檔案 {file_path}，跳過特徵顯著性分析。")
            return
            
        df = pd.read_parquet(file_path)
        df['is_top_choice'] = (df['pred_rank'] == 1).astype(int)
        
        # 讀取剛剛儲存的配置
        with open("models/features_config.json", "r") as f:
            current_features = json.load(f)
        
        check_features = ['j_track_smoothed_place_rate', 'h_smoothed_rolling_5_place_rate', 'weight_delta']
        check_features = [f for f in check_features if f in df.columns and f in current_features]
        
        if not check_features:
            return
            
        comparison = df.groupby('is_top_choice')[check_features].mean()
        
        print("\n=== 🧪 最佳模型特徵辨識能力分析 ===")
        print("0: 非首選馬 (Others), 1: 模型首選馬 (Top Choice)")
        print(comparison)
        
        diff = comparison.loc[1] - comparison.loc[0]
        print("\n=== 特徵顯著性差異 (正值代表 Top1 馬擁有更優勢的特徵) ===")
        print(diff)


# ==================== 🎬 執行入口 ====================
if __name__ == "__main__":
    # 定義特徵群組
    base_feats = settings.base_features
    candidate_feats = settings.candidate_features
    
    # 1. 初始化管線（這裡先給預設值 1，後續會在迴圈中動態修改）
    pipeline = FeatureSearchPipeline(
        base_features=base_feats, 
        candidate_features=candidate_feats, 
        max_add_features=1
    )
    
    # 2. 🚀 關鍵優化：先載入數據（只讀取一次硬碟，節省大量時間）
    pipeline.load_and_split_data()
    
    # 用來收集所有數量輪次的全域結果
    global_records = []
    absolute_best_win_rate = -1.0
    absolute_best_features = list(base_feats)
    # 3. 🔄 進行數量迭代（嘗試新增 1 到 2 個特徵）
    for num_features in [1, 2, 3]:
        logging.info(f"\n" + "="*70)
        logging.info(f"🔥 啟動特徵尋優 - 嘗試組合：【基準特徵 + 任意 {num_features} 個候選特徵】")
        logging.info("="*70)
        
        # 動態修改管線內部的限制參數
        pipeline.max_add_features = num_features
        
        # 執行當前數量的尋優搜尋（內部會自動選出該輪最優模型）
        pipeline.run_search()
        
        # ✨ 關鍵安全調整：直接從 pipeline 每次跑完認證的 best_run 內撈取當輪的第一名
        if hasattr(pipeline, "best_run") and pipeline.best_run:
            res = pipeline.best_run
            
            current_combo = res.get("combo")
            current_win_rate = res.get("win_rate")
            current_features_list = res.get("features_list", [])
            
            if current_win_rate is not None:
                if isinstance(current_combo, (list, tuple, set)):
                    combo_str = " + ".join(current_combo) if current_combo else "Baseline (None)"
                else:
                    combo_str = str(current_combo)

                global_records.append({
                    "added_count": num_features,
                    "combination": combo_str,
                    "oot_win_rate": float(current_win_rate)
                })
                
                # ✨ 追蹤跨數量（全域）最高勝率的特徵組合，鎖定全場總冠軍
                if float(current_win_rate) > absolute_best_win_rate:
                    absolute_best_win_rate = float(current_win_rate)
                    absolute_best_features = current_features_list

    # ============================================================
    # 👑 跨維度終極大決戰 - 綜合 TOP 5 最佳組合（不分數量）
    # ============================================================
    if global_records:
        # 1. 建立 DataFrame 
        df_global = pd.DataFrame(global_records)
        
        # 2. 🚨 強制將勝率欄位轉換為數值型態（如果是 None 會變成 NaN，如果是字串會強轉）
        df_global['oot_win_rate'] = pd.to_numeric(df_global['oot_win_rate'], errors='coerce')
        
        # 3. 排除因轉換失敗產生的 NaN 紀錄
        df_global = df_global.dropna(subset=['oot_win_rate']).reset_index(drop=True)
        
        if not df_global.empty:
            # 依據勝率由高到低重新大排行
            df_global = df_global.sort_values(by="oot_win_rate", ascending=False).reset_index(drop=True)
            
            print("\n" + "="*70)
            print("🏆 全域特徵組合尋優結束 - 跨數量 TOP 5 終極組合總排行")
            print("="*70)
            for rank in range(min(5, len(df_global))):
                row = df_global.iloc[rank]
                val = float(row['oot_win_rate'])
                print(f"Rank {rank+1}: OOT勝率 {val:.2%} | (已加 {row['added_count']} 個特徵) | 組合: [{row['combination']}]")
            print("="*70)
        else:
            # 💡 如果萬一真的還是空，我們把原始 global_records 印出來看發生什麼事
            print("\n⚠️ 警告：所有輪次收集到的 oot_win_rate 皆為無效值，無法列印總排行。")
            print("🔍 偵錯資訊 - 原始收集到的資料前3筆為：", global_records[:3])

    # 4. 儲存全域最優的模型與特徵配置（注意：這會儲存最後留在 pipeline.best_model 裡的那套）
    pipeline.save_best_artifacts()
    
    # 5. 執行最後的分析
    predictions_path = settings.features_parquet_path.parent / "oot_july_predictions.parquet"
    FeatureSearchPipeline.analyze_feature_differentiation(predictions_path)