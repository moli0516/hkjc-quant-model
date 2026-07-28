import json
import pathlib
import re
from datetime import datetime
import pandas as pd
from config.settings import settings


class HorseCleaner:

    def __init__(self):
        pass

    # ==========================================
    # 工具函數 (Static Methods)
    # ==========================================
    @staticmethod
    def parse_origin_age(val: str | None) -> tuple[str | None, int | None]:
        """解析出生地與年齡（例："澳洲 / 3" -> ("澳洲", 3)）"""
        if not val or pd.isna(val):
            return None, None
        parts = str(val).split("/")
        origin = parts[0].strip() if len(parts) > 0 else None
        age = None
        if len(parts) > 1:
            try:
                age = int(parts[1].strip())
            except ValueError:
                age = None
        return origin, age

    @staticmethod
    def parse_color_sex(val: str | None) -> tuple[str | None, str | None]:
        """解析毛色與性別（例："棗 / 閹" -> ("棗", "閹")）"""
        if not val or pd.isna(val):
            return None, None
        parts = str(val).split("/")
        color = parts[0].strip() if len(parts) > 0 else None
        sex = parts[1].strip() if len(parts) > 1 else None
        return color, sex

    @staticmethod
    def parse_stakes(val: str | None) -> float | None:
        """解析金額（例："$2,650,450" -> 2650450.0）"""
        if not val or pd.isna(val):
            return None
        # 移除非數字字符（保留小數點）
        clean_val = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(clean_val) if clean_val else 0.0
        except ValueError:
            return None

    @staticmethod
    def parse_placing_records(
        val: str | None,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """解析冠亞季冠總次數（例："1-7-4-27" -> (1, 7, 4, 27)）"""
        if not val or pd.isna(val):
            return None, None, None, None
        parts = str(val).split("-")
        if len(parts) == 4:
            try:
                return (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                )
            except ValueError:
                pass
        return None, None, None, None

    @staticmethod
    def parse_date(val: str | None) -> str | None:
        """轉換日期格式（例："21/05/2026" -> "2026-05-21"）"""
        if not val or pd.isna(val):
            return None
        clean_val = str(val).strip()
        try:
            return datetime.strptime(clean_val, "%d/%m/%Y").strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            return None

    @staticmethod
    def clean_int(val) -> int | None:
        """安全清理整數"""
        if val is None or pd.isna(val):
            return None
        try:
            return int(float(str(val).strip()))
        except ValueError:
            return None

    # ==========================================
    # 主清洗入口
    # ==========================================
    def process(
        self, horses_dir=settings.raw_horses_json_dir
    ) -> pd.DataFrame:
        horses_dir = pathlib.Path(horses_dir)
        horses_list = []

        horse_files = list(horses_dir.glob("*.json"))
        print(
            f"🔍 [HorseCleaner] 找到 {len(horse_files)} 個馬匹 Raw JSON 檔案..."
        )

        for file_path in horse_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                origin, age = self.parse_origin_age(
                    raw_data.get("origin_age")
                )
                color, sex = self.parse_color_sex(raw_data.get("color_sex"))
                wins, seconds, thirds, total_runs = (
                    self.parse_placing_records(raw_data.get("placing_records"))
                )

                horses_list.append({
                    "horse_code": raw_data.get("horse_code"),
                    "origin": origin,
                    "age": age,
                    "color": color,
                    "sex": sex,
                    "import_type": raw_data.get("import_type"),
                    "season_stakes": self.parse_stakes(
                        raw_data.get("season_stakes")
                    ),
                    "total_stakes": self.parse_stakes(
                        raw_data.get("total_stakes")
                    ),
                    "wins": wins,
                    "seconds": seconds,
                    "thirds": thirds,
                    "total_runs": total_runs,
                    "recent_10_races_count": self.clean_int(
                        raw_data.get("recent_10_races_count")
                    ),
                    "current_location": raw_data.get("current_location"),
                    "location_arrival_date": self.parse_date(
                        raw_data.get("location_arrival_date")
                    ),
                    "import_date": self.parse_date(raw_data.get("import_date")),
                    "trainer": raw_data.get("trainer"),
                    "owner": raw_data.get("owner"),
                    "current_rating": self.clean_int(
                        raw_data.get("current_rating")
                    ),
                    "season_start_rating": self.clean_int(
                        raw_data.get("season_start_rating")
                    ),
                    "sire": raw_data.get("sire"),
                    "dam": raw_data.get("dam"),
                    "damsire": raw_data.get("damsire"),
                })

            except Exception as e:
                print(
                    f"❌ [HorseCleaner] 解析檔案失敗 [{file_path.name}]: {e}"
                )

        df_horses = pd.DataFrame(horses_list)
        if not df_horses.empty:
            df_horses = df_horses.drop_duplicates(subset=["horse_code"])

        return df_horses