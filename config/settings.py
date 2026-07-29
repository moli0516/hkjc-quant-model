import json
import pathlib

class Settings:
    def __init__(self):
        self.root_dir = pathlib.Path(__file__).parent.parent
        self.config_path = self.root_dir / "config" / "settings.json"
        self._data = self._load()

    def _load(self):
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)
        
    @property
    def active_features(self):
        return self._data.get("active_features", [])
    
    @property
    def base_features(self):
        return self._data.get("base_features", [])
    
    @property
    def candidate_features(self):
        return self._data.get("candidate_features", [])
    
    @property
    def smoothing_alphas(self):
        return self._data.get("smoothing_params", {}).get("alphas", {})
    
    @property
    def raw_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get("raw_json_dir", "")
    
    @property
    def raw_races_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_races_json_dir", "")
        
    @property
    def raw_sectional_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_sectional_json_dir", "")
    
    @property
    def raw_horses_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_horses_json_dir", "")

    @property
    def flattened_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get("flattened_json_dir", "")

    @property
    def normalized_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get("normalized_json_dir", "")

    @property
    def horses_dir(self):
        return self.normalized_json_dir / self._data.get("paths", {}).get("horsess_dir", "")

    @property
    def races_dir(self):
        return self.normalized_json_dir / self._data.get("paths", {}).get("races_dir", "")

    @property
    def rating_path(self):
        return self.root_dir / self._data.get("paths", {}).get("rating_json_path", "")
    
    @property
    def features_parquet_path(self):
        return self.root_dir / self._data.get("paths", {}).get("features_parquet_path", "")
    
    @property
    def today_rc_path(self):
        return self.root_dir / self._data.get("paths", {}).get("today_rc_json_path", "")
    
    @property
    def target(self):
        return self._data.get("target", [])
        
    def get_feature_group(self, group_name):
        return self._data.get("feature_groups", {}).get(group_name, [])

    # ------------------ 新增: DataLoader 屬性 ------------------

    @property
    def data_loader_config(self):
        """獲取完整 data_loader 設定字典"""
        return self._data.get("data_loader", {})

    @property
    def id_cols(self):
        """主鍵與識別欄位"""
        return self.data_loader_config.get("id_cols", ["race_id", "horse_id", "horse_name"])

    @property
    def target_cols(self):
        """目標/標籤欄位"""
        return self.data_loader_config.get("target_cols", ["placing", "is_win", "is_top3"])

    @property
    def eval_cols(self):
        """評估與特徵排除欄位"""
        return self.data_loader_config.get("eval_cols", ["win_odds", "draw", "jockey", "trainer", "date"])

    @property
    def categorical_cols(self):
        """類別型特徵欄位"""
        return self.data_loader_config.get("categorical_cols", ["brand_prefix", "course_type", "track_draw_key"])

settings = Settings()