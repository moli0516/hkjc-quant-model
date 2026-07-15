import json
import pandas as pd
import pathlib
import fastparquet

class Feature_pipeline:
    def __init__(self, json_path):
        self.json_path = json_path
        self.df = self._load_original_json_df(self.json_path).sort_values(by=[ "date", "races.race_id"]).reset_index(drop=True)
        self.bl_win = 1/14
        self.bl_place = 3/14
    def _load_original_json_df(self, json_path):
        combine_data = []
        for file in json_path.iterdir():
            print(file.name)
            if file.is_file():
                file_path = json_path / file.name
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
    def _build_draw_features(self):
        self.df['is_inside_d'] = (self.df['draw'] < 5).astype(int)
        self.df["is_medium_d"] = ((self.df["draw"] > 4) & (self.df["draw"] < 9)).astype(int)
        self.df['is_outside_d'] = (8 < self.df['draw']).astype(int)
    def _build_h_speed_z_features(self):
        # 1. Absolute speed
        self.df["h_speed"] = self.df["length"] / self.df["finished_time_sec"]
        
        # 2. Z score of speed in every race (注意：這裡仍需以當前比賽為準)
        self.df["h_speed_z"] = self.df.groupby("races.race_id")["h_speed"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        )
        
        # !!!IMPORTANT!!! GROUP BY HORSE ID
        h_grpby_z = self.df.groupby("horse_id")["h_speed_z"]
        
        # 3. 基礎平滑：確保所有 rolling 計算前都已經 shift(1) 以防洩漏
        # 使用 min_periods=1 允許馬匹數據不足時也能計算
        r2_mean = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).mean())
        r15_mean = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
        
        # 4. 核心特徵：Anchor (長期基線)
        self.df["h_mean_speed_z_15"] = r15_mean.fillna(0)
        
        # 5. 核心特徵：Momentum (短期相對於長期的變化)
        # 若 R2 為空，則動量為 0，代表無近期資訊
        self.df["h_speed_z_momentum"] = (r2_mean - r15_mean).fillna(0)
        
        # 6. 穩定性特徵：保留 std 作為波動指標
        self.df["h_rolling_2_speed_z_std"] = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).std()).fillna(0)
        self.df["h_rolling_15_speed_z_std"] = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).std()).fillna(0)
        
        # 7. (選配) 加入馬匹的成熟度資訊，幫助模型判斷數據的可信度
        self.df["h_race_count_history"] = h_grpby_z.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0)
        
    def _build_smoothing_features(self, src, target, alpha):
        bl = self.bl_win if "win" in target else self.bl_place
        grp_src = self.df.groupby(src)[target]
        grp_cnt = grp_src.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0) #can be used again in the place part
        grp_sum = grp_src.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        print(f"Feature {src} to {target} has been finished building")
        return ((grp_sum + alpha * bl) / (grp_cnt + alpha)).fillna(bl)
    
    def _build_smoothing_rolling_n_features(self, src, target, alpha, n):
        bl = self.bl_win if "win" in target else self.bl_place
        grp_src = self.df.groupby(src)[target]
        grp_cnt = grp_src.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).count()).fillna(0) #can be used again in the place part
        grp_sum = grp_src.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()).fillna(0)
        print(f"Feature {src} to {target} with rolling-{n}  has been finished building")
        return ((grp_sum + alpha * bl) / (grp_cnt + alpha)).fillna(bl)
    
    def run(self):
        self._build_win_placing()
        self._build_h_speed_z_features()
        self._build_draw_features()
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
        smooths = {"j_smoothed_win_rate": ("jockey", "is_win", 20),
                   "t_smoothed_win_rate": ("trainer", "is_win", 20),
                   "jt_smoothed_win_rate": ("jockey_trainer", "is_win", 40),
                   "h_smoothed_win_rate": ("horse_id", "is_win", 8),
                   "d_smoothed_win_rate": ("draw", "is_win", 30),
                   "j_smoothed_place_rate": ("jockey", "is_place", 20),
                   "t_smoothed_place_rate": ("trainer", "is_place", 20),
                   "jt_smoothed_place_rate": ("jockey_trainer", "is_place", 40),
                   "h_smoothed_place_rate": ("horse_id", "is_place", 8),
                   "d_smoothed_place_rate": ("draw", "is_place", 30),
        # --- 新增：馬匹與特定場地條件 (Alpha 保守) ---
            # 馬匹在大跑道類型的表現 (如：谷草、沙泥)
                    "h_track_smoothed_win_rate": ("horse_track_detailed", "is_win", 25),
                    "h_track_smoothed_place_rate": ("horse_track_detailed", "is_place", 25),
                    
                    # 馬匹在特定路程的表現 (核心：如谷草1200)
                    "h_env_smoothed_win_rate": ("horse_env_core", "is_win", 35),
                    "h_env_smoothed_place_rate": ("horse_env_core", "is_place", 35),
                    
                    # 今晚黏地終極武器：馬匹在特定黏地下的表現 (最稀疏，Alpha 最大)
                    "h_yield_smoothed_win_rate": ("horse_yeild_detailed", "is_win", 40),
                    "h_yield_smoothed_place_rate": ("horse_yeild_detailed", "is_place", 40),

                    # --- 新增：騎師與特定場地條件 (Alpha 較大膽) ---
                    "j_track_smoothed_place_rate": ("jockey_track_detailed", "is_place", 15),
                    "j_env_smoothed_place_rate": ("jockey_env_core", "is_place", 20),
                    "j_yield_smoothed_place_rate": ("jockey_yeild_detailed", "is_place", 25),

                    # --- 新增：練馬師與特定場地條件 ---
                    "t_track_smoothed_place_rate": ("trainer_track_detailed", "is_place", 20),
                    "t_yield_smoothed_place_rate": ("trainer_yeild_detailed", "is_place", 30)}
        for smooth in smooths:
            self.df[smooth] = self._build_smoothing_features(*smooths[smooth])
        smooth_rollings_n = smooths = {"j_smoothed_rolling_30_win_rate": ("jockey", "is_win", 20, 30),
                    "t_smoothed_rolling_30_win_rate": ("trainer", "is_win", 20, 30),
                    "jt_smoothed_rolling_15_win_rate": ("jockey_trainer", "is_win", 40, 15),
                    "h_smoothed_rolling_5_win_rate": ("horse_id", "is_win", 8, 5),
                    "j_smoothed_rolling_30_place_rate": ("jockey", "is_place", 20, 30),
                    "t_smoothed_rolling_30_place_rate": ("trainer", "is_place", 20, 40),
                    "jt_smoothed_rolling_15_place_rate": ("jockey_trainer", "is_place", 40, 15),
                    "h_smoothed_rolling_5_place_rate": ("horse_id", "is_place", 8, 5)}
        
        for smooth_rolling_n in smooth_rollings_n:
            self.df[smooth_rolling_n] = self._build_smoothing_rolling_n_features(*smooth_rollings_n[smooth_rolling_n])
        print(self.df['date'].max())
        self.df.to_parquet('horse_win_rate.parquet')
        

if __name__ == "__main__":
    json_path = pathlib.Path.cwd() / "data" / "cleaned_json" / "flatten"
    feature_pipeline = Feature_pipeline(json_path)
    feature_pipeline.run()
