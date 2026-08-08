import json
import pathlib

class Settings:
    def __init__(self):
        self.root_dir = pathlib.Path(__file__).parent.parent
        self.config_dir = self.root_dir / "config"
        self.pointer_file = self.config_dir / ".active_config"
        self.config_path = self._resolve_config_path()
        self._data = self._load()

    def _resolve_config_path(self) -> pathlib.Path:
        """解析當前應該載入的設定檔路徑，若有持久化檔案則優先讀取"""
        if self.pointer_file.exists():
            try:
                active_name = self.pointer_file.read_text(encoding="utf-8").strip()
                active_path = self.config_dir / active_name
                if active_path.exists():
                    return active_path
            except Exception:
                pass
        # 預設載入 settings.json
        return self.config_dir / "settings.json"

    def _load(self):
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def switch_config(self, config_name_or_path: str) -> str:
        """
        在進程內切換設定檔，並寫入持久化指針檔案 (.active_config) 以供下次執行自動讀取
        """
        target_path = pathlib.Path(config_name_or_path)
        if not target_path.is_absolute():
            target_path = self.config_dir / target_path

        if not target_path.exists():
            raise FileNotFoundError(f"找不到指定的設定檔: {target_path}")

        self.config_path = target_path
        self._data = self._load()

        # 寫入持久化指針
        try:
            self.pointer_file.write_text(self.config_path.name, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ 無法持久化儲存設定檔選擇: {e}")

        return self.config_path.name

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
    def raw_trackworks_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_trackworks_json_dir", "")

    @property
    def raw_trails_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_trails_json_dir", "")

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

    # ------------------ DataLoader 屬性 ------------------

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
    def default_params(self):
        """最優模型參數"""
        return self._data.get("default_params", {})

    @property
    def categorical_cols(self):
        """類別型特徵欄位"""
        return self.data_loader_config.get("categorical_cols", ["brand_prefix", "course_type", "track_draw_key"])
    
    @property
    def banned_features(self):
        return self._data.get("banned_features", [])

settings = Settings()