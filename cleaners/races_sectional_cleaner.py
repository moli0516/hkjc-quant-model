import json
import os
import pathlib
import re
import pandas as pd
from config.settings import settings
from database.db_manager import DBManager


class BatchDataCleanerNormalized:

    def __init__(self, rating_json_path=settings.rating_path):
        self.rating_json_path = rating_json_path
        self.rating_df = self._load_rating_df()

    def _load_rating_df(self):
        """載入並預處理 Rating 資料"""
        if not os.path.exists(self.rating_json_path):
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
    def extract_horse_info(raw_name):
        """解析馬名與烙號 (horse_id)"""
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
    def clean_head_horse_dist(hhd):
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
    def convert_min_to_sec(time_val):
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
    def extract_basic_info(basic_info_str):
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
    def extract_track_info(text):
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
    def clean_draw_value(val):
        if val is None:
            return None
        # 轉成字串並去除前後空白
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["none", "null", "-", "--", ""]:
            return None
        try:
            # 先轉成 float 再轉 int，可以完美相容 5, "5", "5.0", " 5 "
            return int(float(val_str))
        except (ValueError, TypeError):
            return None

    # ==========================================
    # 目錄自動掃描與清洗邏輯
    # ==========================================
    def process_all_files(
        self,
        races_dir=settings.raw_races_json_dir,
        sectionals_dir=settings.raw_sectional_json_dir,
    ):
        races_dir = pathlib.Path(races_dir)
        sectionals_dir = pathlib.Path(sectionals_dir)

        races_list = []
        results_list = []
        sec_list = []

        hkjc_special_codes = [
            "DISQ", "DNF", "FE", "ML", "PU", "TNP", "TO",
            "UR", "VOID", "WR", "WV", "WV-A", "WX", "WX-A", "WXNR", "退出",
        ]

        # ----------------------------------------------------
        # 1. 掃描並處理 raw_races_json_dir
        # ----------------------------------------------------
        race_files = list(races_dir.glob("*.json"))
        print(f"🔍 找到 {len(race_files)} 個賽事 Raw JSON 檔案...")

        for file_path in race_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                date = raw_data.get("date")
                venue = raw_data.get("venue")

                # 【防護修正】使用 (raw_data.get("races") or []) 防止 key 存在但值為 null 的狀況
                for race in raw_data.get("races") or []:
                    race_no = int(race["race_id"])
                    race_key = f"{date}_{venue}_{race_no}"

                    # 解析 races 表
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

                    # 解析 race_results 表
                    for horse in race.get("horses") or []:
                        placing_raw = str(horse.get("placing", "")).strip()

                        if placing_raw in hkjc_special_codes or not placing_raw:
                            continue

                        clean_name, horse_id = self.extract_horse_info(
                            horse.get("horse_name")
                        )
                        time_raw = horse.get("finish_time") or horse.get("finished_time")
                        margin_raw = horse.get("margin") or horse.get("head_horse_dist")

                        placing_match = re.search(r"(\d+)", placing_raw)
                        placing = int(placing_match.group(1)) if placing_match else None

                        results_list.append({
                            "race_id": race_key,
                            "horse_id": horse_id,
                            "horse_name": clean_name,
                            "placing": placing,
                            "draw": (
                                self.clean_draw_value(horse.get("draw"))
                            ),
                            "jockey": horse.get("jockey"),
                            "trainer": horse.get("trainer"),
                            "actual_weight": (
                                float(horse.get("body_weight"))
                                if horse.get("body_weight")
                                else None
                            ),
                            "declared_weight": (
                                float(horse.get("declared_weight"))
                                if horse.get("declared_weight")
                                else None
                            ),
                            "win_odds": (
                                float(horse.get("odds")) if horse.get("odds") else None
                            ),
                            "finish_time_sec": self.convert_min_to_sec(time_raw),
                            "margin_len": self.clean_head_horse_dist(margin_raw),
                        })
            except Exception as e:
                print(f"❌ 解析賽事檔案失敗 [{file_path.name}]: {e}")

        # ----------------------------------------------------
        # 2. 掃描並處理 raw_sectional_json_dir
        # ----------------------------------------------------
        sec_files = list(sectionals_dir.glob("*.json"))
        print(f"🔍 找到 {len(sec_files)} 個分段時間 Raw JSON 檔案...")

        for file_path in sec_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    sec_data = json.load(f)

                if not sec_data or not isinstance(sec_data, dict):
                    continue

                date = sec_data.get("date")
                venue = sec_data.get("venue")
                
                # 【關鍵修正】加上 "sectionals" key
                items = (
                    sec_data.get("sectionals")
                    or sec_data.get("races")
                    or sec_data.get("sectional_data")
                    or []
                )

                for item in items:
                    sec_race_no = item.get("race_no") or item.get("race_id")
                    if not sec_race_no or not date or not venue:
                        continue

                    race_key = f"{date}_{venue}_{int(sec_race_no)}"

                    # 提取馬匹層級
                    horse_list = (
                        item.get("sectional_data")
                        or item.get("horses")
                        or item.get("details")
                        or []
                    )

                    for horse_sec in horse_list:
                        clean_name, horse_id = self.extract_horse_info(
                            horse_sec.get("horse_name") or horse_sec.get("name")
                        )

                        # 提取分段細節層級
                        details = (
                            horse_sec.get("sectional_details")
                            or horse_sec.get("sectionals")
                            or []
                        )

                        for idx, detail in enumerate(details, 1):
                            sec_no = detail.get("section_no") or idx
                            position = detail.get("position") or detail.get("pos")
                            sec_time = detail.get("sectional_time") or detail.get("time")
                            margin = detail.get("margin") or detail.get("behind")

                            sec_list.append({
                                "race_id": race_key,
                                "horse_name": clean_name,
                                "horse_id": horse_id,
                                "section_no": int(sec_no),
                                "position": (
                                    int(position)
                                    if str(position).isdigit()
                                    else None
                                ),
                                "sectional_time_sec": self.convert_min_to_sec(
                                    sec_time
                                ),
                                "margin_behind": str(margin) if margin else None,
                            })
            except Exception as e:
                print(f"❌ 解析分段檔案失敗 [{file_path.name}]: {e}")

        # ----------------------------------------------------
        # 3. 轉換 DataFrames 並輸出
        # ----------------------------------------------------
        df_races = pd.DataFrame(races_list).drop_duplicates(subset=["race_id"])
        df_results = pd.DataFrame(results_list)
        df_sectionals = pd.DataFrame(sec_list)

        if self.rating_df is not None and not df_results.empty:
            df_results = df_results.merge(
                self.rating_df[["clean_horse_name", "rating"]],
                left_on="horse_name",
                right_on="clean_horse_name",
                how="left",
            ).drop(columns=["clean_horse_name"], errors="ignore")

        return {
            "races": df_races,
            "race_results": df_results,
            "race_sectionals": df_sectionals,
        }


# ==========================================
# 主執行進入點
# ==========================================
if __name__ == "__main__":
    db = DBManager()
    db.init_db()

    cleaner = BatchDataCleanerNormalized()
    print("🚀 開始自動掃描並清洗所有 RAW JSON 檔案...")
    tables = cleaner.process_all_files()

    print("\n📊 --- 清洗統計 ---")
    print(f"賽事總數 (races): {len(tables['races'])}")
    print(f"馬匹賽果總數 (race_results): {len(tables['race_results'])}")
    print(f"分段數據總數 (race_sectionals): {len(tables['race_sectionals'])}")

    print("\n💾 正在寫入資料庫...")
    # 傳入 replace_mode=True 確保重複執行時清空舊資料再重新寫入
    db.insert_dataframes(tables)
    print("✨ 全部數據已成功寫入資料庫！")
    #sec_dir = settings.raw_sectional_json_dir  # 請替換為你的分段 JSON 資料夾路徑
    #sample_file = list(sec_dir.glob("*.json"))[0]

    #with open(sample_file, "r", encoding="utf-8") as f:
    #    sample_data = json.load(f)

    #print(f"📄 測試檔案: {sample_file.name}")
    #print("🔍 頂層 Keys:", list(sample_data.keys()) if isinstance(sample_data, dict) else "List structure")
    #print("🔍 內容預覽 (前 500 字):")
    #print(json.dumps(sample_data, ensure_ascii=False, indent=2)[:500])