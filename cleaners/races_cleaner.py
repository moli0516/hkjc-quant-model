import json
import pathlib
import re
import pandas as pd
from config.settings import settings


class RaceCleaner:

    def __init__(self, rating_json_path=settings.rating_path):
        self.rating_json_path = rating_json_path
        self.rating_df = self._load_rating_df()

    def _load_rating_df(self) -> pd.DataFrame | None:
        """載入並預處理 Rating 資料"""
        if not pathlib.Path(self.rating_json_path).exists():
            return None
        try:
            with open(self.rating_json_path, "r", encoding="utf-8") as f:
                rating_json = json.load(f)
            df = pd.json_normalize(rating_json)
            if "horse_name" in df.columns:
                df["clean_horse_name"] = df["horse_name"].apply(
                    lambda x: self.extract_horse_info(x)[0]
                )
            return df
        except Exception as e:
            print(f"【警告】載入 Rating 資料失敗: {e}")
            return None

    # ==========================================
    # 工具函數 (Static Methods)
    # ==========================================
    @staticmethod
    def extract_horse_info(raw_name: str) -> tuple[str | None, str | None]:
        """解析馬名與烙號/horse_id"""
        if not isinstance(raw_name, str) or pd.isna(raw_name):
            return None, None

        match = re.search(r"\(([A-Z0-9]{3,5})\)", raw_name)
        horse_id = match.group(1) if match else None

        clean_name = (
            re.sub(r"\s*\([A-Z0-9]{3,5}\)", "", raw_name)
            .replace("\xa0", "")
            .strip()
        )
        return clean_name, horse_id

    @staticmethod
    def clean_head_horse_dist(hhd) -> float | None:
        """勝負距離轉為 float 馬身數"""
        if pd.isna(hhd) or hhd is None:
            return None
        if isinstance(hhd, (int, float)):
            return float(hhd)

        hhd_str = str(hhd).strip()
        if "-" in hhd_str and "/" in hhd_str:
            try:
                parts = hhd_str.split("-")
                frac = parts[1].split("/")
                return float(parts[0]) + float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        elif "/" in hhd_str:
            try:
                frac = hhd_str.split("/")
                return float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        else:
            margin_map = {
                "---": 0.0,
                "平頭馬": 0.0,
                "鼻位": 0.05,
                "短馬頭位": 0.1,
                "頭位": 0.2,
                "頸位": 0.3,
                "多個馬身": 99.0,
                "多個馬位": 99.0,
                "未能完成賽事": None,
                "退出": None,
            }
            return margin_map.get(
                hhd_str, float(hhd_str) if hhd_str.isdigit() else 0.25
            )

    @staticmethod
    def convert_min_to_sec(time_val) -> float | None:
        """時間轉秒數 (解析 1:10.12 或 23.88)"""
        if pd.isna(time_val) or time_val in ["---", "", None]:
            return None
        time_str = str(time_val).strip()
        try:
            if ":" in time_str:
                parts = time_str.split(":")
                return float(parts[0]) * 60.0 + float(parts[1])
            else:
                return float(time_str)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def extract_basic_info(
        basic_info_str: str,
    ) -> tuple[str | None, int | None]:
        """解析 basic_info -> 班次 (class) 與 路程 (distance)"""
        if not isinstance(basic_info_str, str) or pd.isna(basic_info_str):
            return None, None

        length_match = re.search(r"(\d{3,4})\s*米", basic_info_str)
        distance = int(length_match.group(1)) if length_match else None

        class_map = {
            "第一班": "1",
            "第二班": "2",
            "第三班": "3",
            "第四班": "4",
            "第五班": "5",
            "一級賽": "G1",
            "二級賽": "G2",
            "三級賽": "G3",
            "國際一級賽": "G1",
            "國際二級賽": "G2",
            "國際三級賽": "G3",
        }
        race_class = None
        for k, v in class_map.items():
            if k in basic_info_str:
                race_class = v
                break

        return race_class, distance

    @staticmethod
    def extract_track_info(text: str) -> tuple[str, str]:
        """解析 track_info -> 跑道材質 & 賽道 (A/B/C)"""
        if not isinstance(text, str) or pd.isna(text):
            return "未知", "N/A"
        clean_text = (
            text.replace('"', "").replace("“", "").replace("”", "").strip()
        )
        if "全天候" in clean_text or "泥地" in clean_text:
            return "泥地", "N/A"
        match = re.search(r"([^\s-]+)\s*-\s*([A-Za-z0-9+]+)", clean_text)
        if match:
            return match.group(1), match.group(2)
        return clean_text, "N/A"

    @staticmethod
    def clean_draw_value(val) -> int | None:
        """檔位清洗範例（防空字串、float 轉型）"""
        if val is None:
            return None
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["none", "null", "-", "--", ""]:
            return None
        try:
            return int(float(val_str))
        except (ValueError, TypeError):
            return None

    # ==========================================
    # 主清洗入口
    # ==========================================
    def process(
        self, races_dir=settings.raw_races_json_dir
    ) -> dict[str, pd.DataFrame]:
        races_dir = pathlib.Path(races_dir)
        races_list = []
        results_list = []

        hkjc_special_codes = [
            "DISQ",
            "DNF",
            "FE",
            "ML",
            "PU",
            "TNP",
            "TO",
            "UR",
            "VOID",
            "WR",
            "WV",
            "WV-A",
            "WX",
            "WX-A",
            "WXNR",
            "退出",
        ]

        race_files = list(races_dir.glob("*.json"))
        print(f"🔍 [RaceCleaner] 找到 {len(race_files)} 個賽事 Raw JSON 檔案...")

        for file_path in race_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                date = raw_data.get("date")
                venue = raw_data.get("venue")

                for race in raw_data.get("races") or []:
                    race_no = int(race["race_id"])
                    race_key = f"{date}_{venue}_{race_no}"

                    race_class, distance = self.extract_basic_info(
                        race.get("basic_info")
                    )
                    track_texture, track_type = self.extract_track_info(
                        race.get("track_info")
                    )

                    races_list.append({
                        "race_id": race_key,
                        "date": date,
                        "venue": venue,
                        "race_no": race_no,
                        "race_class": race_class,
                        "distance": distance,
                        "track_condition": race.get("track_condition"),
                        "track_texture": track_texture,
                        "track_type": track_type,
                    })

                    for horse in race.get("horses") or []:
                        placing_raw = str(horse.get("placing", "")).strip()

                        if (
                            placing_raw in hkjc_special_codes
                            or not placing_raw
                        ):
                            continue

                        clean_name, horse_id = self.extract_horse_info(
                            horse.get("horse_name")
                        )
                        time_raw = horse.get("finish_time") or horse.get(
                            "finished_time"
                        )
                        margin_raw = horse.get("margin") or horse.get(
                            "head_horse_dist"
                        )

                        placing_match = re.search(r"(\d+)", placing_raw)
                        placing = (
                            int(placing_match.group(1))
                            if placing_match
                            else None
                        )

                        results_list.append({
                            "race_id": race_key,
                            "horse_id": horse.get("horse_id"),
                            "horse_name": clean_name,
                            "placing": placing,
                            "draw": self.clean_draw_value(horse.get("draw")),
                            "jockey": horse.get("jockey"),
                            "trainer": horse.get("trainer"),
                            "actual_weight": (
                                float(horse.get("actual_weight"))
                                if horse.get("actual_weight")
                                else None
                            ),
                            "declared_weight": (
                                float(horse.get("body_weight"))
                                if horse.get("body_weight")
                                else None
                            ),
                            "win_odds": (
                                float(horse.get("odds"))
                                if horse.get("odds")
                                else None
                            ),
                            "finish_time_sec": self.convert_min_to_sec(
                                time_raw
                            ),
                            "margin_len": self.clean_head_horse_dist(
                                margin_raw
                            ),
                        })
            except Exception as e:
                print(f"❌ [RaceCleaner] 解析檔案失敗 [{file_path.name}]: {e}")

        df_races = pd.DataFrame(races_list).drop_duplicates(subset=["race_id"])
        df_results = pd.DataFrame(results_list)

        # 進行 Rating 資料合併
        if self.rating_df is not None and not df_results.empty:
            df_results = df_results.merge(
                self.rating_df[["clean_horse_name", "rating"]],
                left_on="horse_name",
                right_on="clean_horse_name",
                how="left",
            ).drop(columns=["clean_horse_name"], errors="ignore")

        return {"races": df_races, "race_results": df_results}