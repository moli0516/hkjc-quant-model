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
        return self.config_dir / "settings.json"

    def _load(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def switch_config(self, config_name_or_path: str) -> str:
        target_path = pathlib.Path(config_name_or_path)
        if not target_path.is_absolute():
            target_path = self.config_dir / target_path

        if not target_path.exists():
            raise FileNotFoundError(f"找不到指定的設定檔: {target_path}")

        self.config_path = target_path
        self._data = self._load()

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
        return self.raw_json_dir / self._data.get("paths", {}).get(
            "raw_races_json_dir", ""
        )
      
    @property  
    def raw_horses_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get(
            "raw_horses_json_dir", ""
        )

    @property
    def raw_sectional_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get(
            "raw_sectional_json_dir", ""
        )

    @property
    def raw_trackworks_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get(
            "raw_trackworks_json_dir", ""
        )

    @property
    def raw_trails_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get(
            "raw_trails_json_dir", ""
        )

    @property
    def flattened_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get(
            "flattened_json_dir", ""
        )

    @property
    def normalized_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get(
            "normalized_json_dir", ""
        )

    @property
    def horses_dir(self):
        return self.normalized_json_dir / self._data.get("paths", {}).get(
            "horsess_dir", ""
        )

    @property
    def races_dir(self):
        return self.normalized_json_dir / self._data.get("paths", {}).get(
            "races_dir", ""
        )

    @property
    def rating_path(self):
        return self.root_dir / self._data.get("paths", {}).get(
            "rating_json_path", ""
        )

    @property
    def features_parquet_path(self):
        return self.root_dir / self._data.get("paths", {}).get(
            "features_parquet_path", ""
        )

    @property
    def today_rc_path(self):
        return self.root_dir / self._data.get("paths", {}).get(
            "today_rc_json_path", ""
        )

    @property
    def predictions_dir(self):
        """Walk-forward predictions 冷儲存目錄"""
        rel = self._data.get("paths", {}).get("predictions_dir", "data/predictions")
        return self.root_dir / rel

    @property
    def evaluation_config(self) -> dict:
        return self._data.get("evaluation", {})

    @property
    def auto_save_predictions(self) -> bool:
        return bool(self.evaluation_config.get("auto_save_predictions", True))

    @property
    def predictions_format(self) -> str:
        return str(self.evaluation_config.get("predictions_format", "parquet"))

    @property
    def diagnosis_stake(self) -> float:
        return float(self.evaluation_config.get("diagnosis_stake", 1.0))

    @property
    def target(self):
        return self._data.get("target", [])

    def get_feature_group(self, group_name):
        return self._data.get("feature_groups", {}).get(group_name, [])

    @property
    def data_loader_config(self):
        return self._data.get("data_loader", {})

    @property
    def id_cols(self):
        return self.data_loader_config.get(
            "id_cols", ["race_id", "horse_id", "horse_name"]
        )

    @property
    def target_cols(self):
        return self.data_loader_config.get(
            "target_cols", ["placing", "is_win", "is_top3"]
        )

    @property
    def eval_cols(self):
        return self.data_loader_config.get(
            "eval_cols", ["win_odds", "draw", "jockey", "trainer", "date"]
        )

    @property
    def default_params(self):
        return self._data.get("default_params", {})

    @property
    def categorical_cols(self):
        return self.data_loader_config.get(
            "categorical_cols",
            ["brand_prefix", "course_type", "track_draw_key"],
        )

    @property
    def banned_features(self):
        return self._data.get("banned_features", [])


settings = Settings()