import json
import pandas as pd
import pathlib
import fastparquet
from config.settings import settings
import numpy as np

class Feature_pipeline:
    def __init__(self):
        self.json_path = settings.flattened_json_dir
        print(f"📂 JSON 來源路徑: {self.json_path}")
        self.alphas = settings.smoothing_alphas
        # 讀入原始資料並依時間、賽事 ID 排序，確保時間序列計算的正確性
        self.df = self._load_original_json_df().sort_values(by=["date", "races.race_id"]).reset_index(drop=True)
        self.bl_win = 1/14
        self.bl_place = 3/14

    def _load_original_json_df(self):
        combine_data = []
        for file in self.json_path.iterdir():
            if file.is_file() and file.suffix == '.json':
                print(f"📖 讀取檔案: {file.name}")
                file_path = self.json_path / file.name
                with open(file_path, "r", encoding='utf-8') as f:
                    raw_json = json.load(f)
                    if isinstance(raw_json, list):
                        combine_data.extend(raw_json)
                    else:
                        combine_data.append(raw_json)
        df = pd.json_normalize(combine_data)
        return df

    def _build_win_placing(self):
        self.df['is_win'] = (self.df['placing'] == 1).astype(int)
        self.df['is_place'] = (self.df['placing'] < 4).astype(int)

    def _build_rating_features(self):
        class_avg_map = {
            1: 100,
            2: 85,
            3: 70,
            4: 50,
            5: 30
        }
        self.df["rating_is_real"] = self.df["rating"].notna()
        mask = (self.df['rating_is_real'] == False)
    
        # 針對每一班的缺失評分進行填充
        for cls, avg_val in class_avg_map.items():
            self.df.loc[mask & (self.df['class'] == cls), 'rating'] = avg_val
        
        # 確保填充後 rating 為 float 類型
        self.df['rating'] = self.df['rating'].astype(float)

    def _build_h_speed_z_features(self):
        # 1. 計算絕對速度 (Length / Finished Time)
        self.df["h_speed"] = self.df["length"] / self.df["finished_time_sec"]
        
        # 2. 計算單場賽事內的速度 Z-score (這部分可以使用當場數據，因為是同場競爭對比)
        self.df["h_speed_z"] = self.df.groupby("races.race_id")["h_speed"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        ).fillna(0)
        
        # 3. 按馬匹分組計算歷史速度特徵 (必須 shift(1) 避免洩漏)
        h_grpby_z = self.df.groupby("horse_id")["h_speed_z"]
        
        # 過去 2 場與 15 場的移動平均
        r2_mean = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).mean())
        r15_mean = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
        
        # 4. 核心特徵：Anchor (長期速度基線)
        self.df["h_mean_speed_z_15"] = r15_mean.fillna(0)
        
        # 5. 核心特徵：Momentum (短期相對於長期的變化)
        self.df["h_speed_z_momentum"] = (r2_mean - r15_mean).fillna(0)
        
        # 6. 穩定性特徵：保留 std 作為波動指標 (std 若為 NaN 填充為 0)
        self.df["h_rolling_2_speed_z_std"] = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).std()).fillna(0)
        self.df["h_rolling_15_speed_z_std"] = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).std()).fillna(0)
        
        # 7. 馬匹成熟度特徵 (累積出賽次數，需 shift 避免計算當場)
        self.df["h_race_count_history"] = h_grpby_z.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0)

    def _build_advanced_texture_and_weight_features(self):
        """
        [新增與優化特徵]
        1. h_smoothed_rolling_5_texture_place_rate_sand (沙地)
        2. h_smoothed_rolling_5_texture_win_rate_turf (草地)
        3. weight_impact_score
        """
        # 判斷場地類型：假設 'ST' (沙田) 且 track_type 為 'Dirt'/'Sand' 或依據你的 track_type 定義
        # 這裡採取安全定義：只要 track_type 含有 'track' 或 'turf' 為草地，其餘或含有 'sand'/'dirt' 為沙地
        # 你可以根據實際的 `track_type` 欄位值進行調整
        self.df['is_turf'] = self.df['track_type'].astype(str).str.lower().str.contains('草地').astype(int)
        self.df['is_sand'] = (~self.df['is_turf'].astype(bool)).astype(int)

        grouped = self.df.sort_values(['horse_id', 'date']).groupby('horse_id')

        # 1. 沙地 Place 率 (過往 5 場) - 嚴格執行 shift(1)
        sand_place_numerator = (grouped.apply(lambda x: (x['is_place'] * x['is_sand']).shift(1).rolling(window=5, min_periods=1).sum())).reset_index(level=0, drop=True)
        sand_place_denominator = (grouped.apply(lambda x: x['is_sand'].shift(1).rolling(window=5, min_periods=1).sum())).reset_index(level=0, drop=True)
        self.df['h_smoothed_rolling_5_texture_place_rate_sand'] = (sand_place_numerator / (sand_place_denominator + 1e-6)).fillna(self.bl_place)

        # 2. 草地 Win 率 (過往 5 場) - 嚴格執行 shift(1)
        turf_win_numerator = (grouped.apply(lambda x: (x['is_win'] * x['is_turf']).shift(1).rolling(window=5, min_periods=1).sum())).reset_index(level=0, drop=True)
        turf_win_denominator = (grouped.apply(lambda x: x['is_turf'].shift(1).rolling(window=5, min_periods=1).sum())).reset_index(level=0, drop=True)
        self.df['h_smoothed_rolling_5_texture_win_rate_turf'] = (turf_win_numerator / (turf_win_denominator + 1e-6)).fillna(self.bl_win)

        # 3. 負重影響分數 (當前負重 / 過去 5 場平均 rank_weight)
        # 使用 rank_weight (馬匹淨重) 或 weight (負磅)，這裡採用過往平均負重進行對比
        avg_weight_last_5 = grouped['weight'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        # 若無歷史數據，則影響分數為 1.0 (無影響)
        self.df['weight_impact_score'] = (self.df['weight'] / (avg_weight_last_5 + 1e-6)).fillna(1.0)

    def _build_smoothing_features(self, src, target):
        alpha = self.alphas.get(src, settings._data.get("smoothing_params", {}).get("default_alpha", 20))
        bl = self.bl_win if "win" in target else self.bl_place
        grp_src = self.df.groupby(src)[target]
        
        grp_cnt = grp_src.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0)
        grp_sum = grp_src.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        
        return ((grp_sum + alpha * bl) / (grp_cnt + alpha)).fillna(bl)

    def _build_smoothing_rolling_n_features(self, src, target, n):
        alpha = self.alphas.get(src, settings._data.get("smoothing_params", {}).get("default_alpha", 20))
        bl = self.bl_win if "win" in target else self.bl_place
        grp_src = self.df.groupby(src)[target]
        
        grp_cnt = grp_src.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).count()).fillna(0)
        grp_sum = grp_src.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()).fillna(0)
        
        return ((grp_sum + alpha * bl) / (grp_cnt + alpha)).fillna(bl)

    def _build_advanced_draw_features(self):
        # 1. 跑道偏差因子
        track_bias_map = {'A': 1.2, 'B': 1.1, 'C': 0.8, 'C+3': 0.7}
        self.df['track_bias_factor'] = self.df['track_type'].map(track_bias_map).fillna(1.0)
        self.df['adj_draw'] = self.df['draw'] * self.df['track_bias_factor']
        
        # 2. 檔位與速度的交互特徵
        self.df['draw_speed_interaction'] = self.df['draw'] * self.df['h_mean_speed_z_15']
        
        # 3. 相對評分偏差 (Relative Rating)
        self.df['rating'] = self.df['rating'].astype(float)
        race_avg_rating = self.df.groupby('races.race_id')['rating'].transform('mean')
        
        self.df['rating_vs_race_avg'] = self.df['rating'] - race_avg_rating
        self.df['rating_strength_score'] = self.df['rating_vs_race_avg'] * self.df['rating_is_real'].astype(int)

    def _build_rank_weight_features(self):
        # 標準化評分優勢 (使用 race_unique_id 分組)
        self.df['z_rating_vs_race_avg'] = self.df.groupby('race_unique_id')['rating_vs_race_avg'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        ).fillna(0)
        
        # 交互權重特徵
        self.df['rating_x_rank_weight'] = self.df['rating_vs_race_avg'] * self.df['rank_weight']
        self.df['jockey_adaptability_x_rank_weight'] = self.df['j_track_smoothed_place_rate'] * self.df['rank_weight']
        self.df['form_x_rank_weight'] = self.df['h_smoothed_rolling_5_place_rate'] * self.df['rank_weight']

    def _build_weight_features(self):
        self.df = self.df.sort_values(by=['horse_id', 'race_unique_id'])
    
        # 1. 最近 3 場的平均馬匹淨重量 (rank_weight)
        self.df['avg_rank_weight_last_3'] = (
            self.df.groupby('horse_id')['rank_weight']
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )
        
        # 填充缺失值 (若為首場，以當前淨重代替)
        self.df['avg_rank_weight_last_3'] = self.df['avg_rank_weight_last_3'].fillna(self.df['rank_weight'])
        self.df['weight_delta'] = self.df['rank_weight'] - self.df['avg_rank_weight_last_3']
        
        # 2. 負荷比例 (負重 weight / 淨重 rank_weight)
        self.df['load_ratio'] = self.df['weight'] / (self.df['rank_weight'] + 1e-6)
        
        # 3. 重量與淨重交叉特徵
        self.df['delta_x_rank'] = self.df['weight_delta'] * self.df['rank_weight']
        self.df['load_ratio_x_rank'] = self.df['load_ratio'] * self.df['rank_weight']

    def run(self):
        # 1. 建立基準標籤與 ID
        self._build_win_placing()
        self.df["race_unique_id"] = self.df["date"].astype(str) + "_" + self.df["races.race_id"].astype(str)
        
        # 2. 建立馬匹速度特徵 (這是很多交叉特徵的基底，必須先跑)
        self._build_h_speed_z_features()
        
        # 3. 建立評分與基本特徵 (填充 rating，計算基礎偏差)
        self._build_rating_features()
        
        # 4. 賠率與高維度 ID 交互組合
        self.df['win_odds_inv'] = 1 / (self.df['odds'] + 1e-6)
        self.df['log_win_odds'] = np.log1p(self.df['odds'])
        self.df["jockey_trainer"] = self.df["jockey"] + "_" + self.df["trainer"]
        self.df["track_detailed"] = self.df["venue"].astype(str) + "_" + self.df["track_texture"].astype(str)
        self.df["rail_detailed"] = self.df["track_detailed"].astype(str) + "_" + self.df["track_type"].astype(str)
        self.df["yeild_detailed"] = self.df["venue"].astype(str) + "_" + self.df["races.track_condition"].astype(str)
        self.df["env_core"] = self.df["track_detailed"].astype(str) + "_" + self.df["length"].astype(str)
        self.df["env_detail"] = self.df["env_core"] + "_" + self.df["track_type"].astype(str) + "_" + self.df["races.track_condition"].astype(str)
        
        self.df["horse_track_detailed"] = self.df["horse_id"].astype(str) + "_" + self.df["track_detailed"].astype(str)
        self.df["horse_env_core"] = self.df["horse_id"].astype(str) + "_" + self.df["env_core"].astype(str)
        self.df["horse_yeild_detailed"] = self.df["horse_id"].astype(str) + "_" + self.df["yeild_detailed"].astype(str)
        
        self.df["jockey_track_detailed"] = self.df["jockey"].astype(str) + "_" + self.df["track_detailed"].astype(str)
        self.df["jockey_env_core"] = self.df["jockey"].astype(str) + "_" + self.df["env_core"].astype(str)
        self.df["jockey_yeild_detailed"] = self.df["jockey"].astype(str) + "_" + self.df["yeild_detailed"].astype(str)

        self.df["trainer_track_detailed"] = self.df["trainer"].astype(str) + "_" + self.df["track_detailed"].astype(str)
        self.df["trainer_yeild_detailed"] = self.df["trainer"].astype(str) + "_" + self.df["yeild_detailed"].astype(str)
        
        self.df["actual_rank_score"] = self.df["placing"].map({1: 15, 2: 7, 3: 3, 4: 1}).fillna(0).astype(int)

        # 5. 計算全局平滑特徵 (Global Smoothing)
        smooths = {
            "j_smoothed_win_rate": ("jockey", "is_win"),
            "t_smoothed_win_rate": ("trainer", "is_win"),
            "jt_smoothed_win_rate": ("jockey_trainer", "is_win"),
            "h_smoothed_win_rate": ("horse_id", "is_win"),
            "d_smoothed_win_rate": ("draw", "is_win"),
            "j_smoothed_place_rate": ("jockey", "is_place"),
            "t_smoothed_place_rate": ("trainer", "is_place"),
            "jt_smoothed_place_rate": ("jockey_trainer", "is_place"),
            "h_smoothed_place_rate": ("horse_id", "is_place"),
            "d_smoothed_place_rate": ("draw", "is_place"),
            "h_track_smoothed_win_rate": ("horse_track_detailed", "is_win"),
            "h_track_smoothed_place_rate": ("horse_track_detailed", "is_place"),
            "h_env_smoothed_win_rate": ("horse_env_core", "is_win"),
            "h_env_smoothed_place_rate": ("horse_env_core", "is_place"),
            "h_yield_smoothed_win_rate": ("horse_yeild_detailed", "is_win"),
            "h_yield_smoothed_place_rate": ("horse_yeild_detailed", "is_place"),
            "j_track_smoothed_place_rate": ("jockey_track_detailed", "is_place"),
            "j_env_smoothed_place_rate": ("jockey_env_core", "is_place"),
            "j_yield_smoothed_place_rate": ("jockey_yeild_detailed", "is_place"),
            "t_track_smoothed_place_rate": ("trainer_track_detailed", "is_place"),
            "t_yield_smoothed_place_rate": ("trainer_yeild_detailed", "is_place")
        }
        
        for name, (src, target) in smooths.items():
            self.df[name] = self._build_smoothing_features(src, target)
            print(f"✅ Feature {name} ({src} -> {target}) built successfully.")

        # 6. 計算滑動視窗平滑特徵 (Rolling Smoothing)
        smooth_rollings_n = {
            "j_smoothed_rolling_30_win_rate": ("jockey", "is_win", 30),
            "t_smoothed_rolling_30_win_rate": ("trainer", "is_win", 30),
            "jt_smoothed_rolling_15_win_rate": ("jockey_trainer", "is_win", 15),
            "h_smoothed_rolling_5_win_rate": ("horse_id", "is_win", 5),
            "j_smoothed_rolling_30_place_rate": ("jockey", "is_place", 30),
            "t_smoothed_rolling_30_place_rate": ("trainer", "is_place", 40),
            "jt_smoothed_rolling_15_place_rate": ("jockey_trainer", "is_place", 15),
            "h_smoothed_rolling_5_place_rate": ("horse_id", "is_place", 5)
        }
        
        for name, (src, target, n) in smooth_rollings_n.items():
            self.df[name] = self._build_smoothing_rolling_n_features(src, target, n)
            print(f"🔄 Feature {name} (Rolling-{n}) built successfully.")

        # 7. 建立優化後的沙/草特徵與負重影響特徵 (你的新策略核心)
        self._build_advanced_texture_and_weight_features()

        # 7. 計算檔位與評分優勢基本衍生 (產生 rating_vs_race_avg 和 h_mean_speed_z_15 相關)
        self._build_advanced_draw_features()

        # 8. 計算重量特徵 (產生 weight_delta, load_ratio 等)
        self._build_weight_features()

        # 9. 最後一步：計算重度依賴其他平滑特徵的「交叉權重特徵」
        self._build_rank_weight_features()
        
        # 最終檢查與匯出
        print(f"📅 資料集最大日期: {self.df['date'].max()}")
        self.df.to_parquet(settings.features_parquet_path, index=False)
        print(f"🎉 特徵工程完成！檔案已儲存至 {settings.features_parquet_path}")

if __name__ == "__main__":
    feature_pipeline = Feature_pipeline()
    feature_pipeline.run()