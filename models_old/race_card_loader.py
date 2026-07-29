import json
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Race_card:
    def __init__(self, raw_data, settings_obj = settings):
        self.settings = settings_obj
        self.raw_data = raw_data
        self.df_rc = pd.DataFrame(raw_data)

    def _merge_rc_horse_id(self):
        rating_path = self.settings.rating_path
        raw_json = None
        with open(rating_path, "r", encoding='utf-8') as f:
            raw_json = json.load(f)
        rating_df = pd.json_normalize(
            raw_json
        )
        self.df = self.df.merge(
            rating_df[["horse_name", "horse_id"]], 
            on=["horse_name"], 
            how="left"
        )
    
    def _preprocess_basic(self):
        self.df_rc['date'] = pd.to_datetime(self.df_rc['date'])

    def _combine_composite_key(self):
        self.df_rc["track_detailed"] = self.df_rc["venue"].astype(str) + "_" + self.df_rc["track_texture"].astype(str)
        self.df_rc["rail_detailed"] = self.df_rc["venue"].astype(str) + "_" + self.df_rc["track_texture"].astype(str) + "_" + self.df_rc["track_type"].astype(str)
        self.df_rc["yeild_detailed"] = self.df_rc["venue"].astype(str) + "_" + self.df_rc["races.track_condition"].astype(str)
        self.df_rc["env_core"] = self.df_rc["venue"].astype(str) + "_" + self.df_rc["track_texture"].astype(str) + "_" + self.df_rc["length"].astype(float).astype(str)
        self.df_rc["env_detail"] = self.df_rc["env_core"] + "_" + self.df_rc["track_type"].astype(str) + "_" + self.df_rc["races.track_condition"].astype(str)

        self.df["horse_track_detailed"] = self.df["horse_id"].astype(str) + "_" + self.df["track_detailed"].astype(str)
        self.df["horse_env_core"] = self.df["horse_id"].astype(str) + "_" + self.df["env_core"].astype(str)
        self.df["horse_yeild_detailed"] = self.df["horse_id"].astype(str) + "_" + self.df["yeild_detailed"].astype(str)

        self.df_rc["jockey_track_detailed"] = self.df_rc["jockey"].astype(str) + "_" + self.df_rc["track_detailed"]
        self.df_rc["jockey_env_core"] = self.df_rc["jockey"].astype(str) + "_" + self.df_rc["env_core"]
        self.df_rc["jockey_yeild_detailed"] = self.df_rc["jockey"].astype(str) + "_" + self.df_rc["yeild_detailed"]

        self.df_rc["trainer_track_detailed"] = self.df_rc["trainer"].astype(str) + "_" + self.df_rc["track_detailed"]
        self.df_rc["trainer_yeild_detailed"] = self.df_rc["trainer"].astype(str) + "_" + self.df_rc["yeild_detailed"]

    def _mapping(self, df_hist_raw: pd.DataFrame, latest_features_col: list):
        df_hist_sorted = df_hist_raw.sort_values("date")
        for col in latest_features_col:
    # 1. 判斷該特徵的 Key 是馬匹、騎師、練馬師還是組合
            if col.startswith("h_track"):
                mapping_key = "horse_track_detailed"
            elif col.startswith("h_env"):
                mapping_key = "horse_env_core"
            elif col.startswith("h_yield"):
                mapping_key = "horse_yeild_detailed"
            elif col.startswith("h_"):
                mapping_key = "horse_id"
            elif col.startswith("j_track"):
                mapping_key = "jockey_track_detailed"
            elif col.startswith("j_env"):
                mapping_key = "jockey_env_core"
            elif col.startswith("j_yield"):
                mapping_key = "jockey_yeild_detailed"
            elif col.startswith("j_"):
                mapping_key = "jockey"
            elif col.startswith("d_"):
                mapping_key = "draw"
            elif col.startswith("t_track"):
                mapping_key = "trainer_track_detailed"
            elif col.startswith("t_yield"):
                mapping_key = "trainer_yeild_detailed"
            elif col.startswith("t_"):
                mapping_key = "trainer"
            elif col.startswith("jt_"):
                mapping_key = "jockey_trainer"
            else:
                mapping_key = "horse_id" # 預設
        latest_map = df_hist_sorted.groupby(mapping_key)[col].last().to_dict()
        self.df_rc[col] = self.df_rc[mapping_key].map(latest_map)
    
    def _fill_missing_values(self, df_hist: pd.DataFrame, target_cols: list):
        for col in target_cols:
            median_val = df_hist[col].median()
            self.df_rc[col] = self.df_rc[col].fillna(median_val)
    
    def run(self):
        self._preprocess_basic()
        self._merge_rc_horse_id()
        self._combine_composite_key()