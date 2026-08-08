import json
import logging
import pathlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from config.settings import settings
from database.db_manager import DBManager

logger = logging.getLogger(__name__)


class TrailsCleaner:
    """HKJC 試閘 (Barrier Trials) 數據清洗器 (Trails Data Cleaner)
    
    負責解析 raw_json/trails 目錄下的試閘 JSON 資料，將組別 metadata 與參賽馬匹細節
    解構為對齊標準 Schema 的 DataFrames，並處理馬名/烙號對照、時間轉換與馬身差距解析。
    """

    def __init__(
        self,
        rating_json_path: Optional[Union[str, pathlib.Path]] = None,
        horse_mapping: Optional[Dict[str, str]] = None,
        db: Optional[DBManager] = None,
    ):
        self.db = db or DBManager()
        self.rating_json_path = pathlib.Path(
            rating_json_path or settings.rating_path
        )
        self.horse_mapping = (
            horse_mapping
            if horse_mapping is not None
            else self.db.get_horse_name_id_mapping()
        )

    # ==========================================
    # 工具函數 (Static Methods & Helpers)
    # ==========================================
    @staticmethod
    def extract_horse_info(raw_name: str) -> Tuple[Optional[str], Optional[str]]:
        """解析馬名與括號內的烙號 (例: "活影 (B100)" -> ("活影", "B100"))"""
        if not isinstance(raw_name, str) or pd.isna(raw_name):
            return None, None

        match = re.search(r"\(([A-Z0-9]{3,5})\)", raw_name)
        horse_code = match.group(1) if match else None

        clean_name = (
            re.sub(r"\s*\([A-Z0-9]{3,5}\)", "", raw_name)
            .replace("\xa0", "")
            .strip()
        )
        return clean_name, horse_code

    @staticmethod
    def parse_date(val: Optional[str]) -> Optional[str]:
        """轉換日期格式（支援 '2019/01/03', '2019-01-03', '03/01/2019' -> '2019-01-03'）"""
        if not val or pd.isna(val):
            return None
        clean_val = str(val).strip()
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(clean_val, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def convert_min_to_sec(time_val: Any) -> Optional[float]:
        """時間轉秒數 (解析 '0.59.53' -> 59.53, '1.00.11' -> 60.11, 或 '1:10.12' -> 70.12)"""
        if pd.isna(time_val) or time_val in ["---", "", None]:
            return None
        time_str = str(time_val).strip()
        try:
            if ":" in time_str:
                parts = time_str.split(":")
                return float(parts[0]) * 60.0 + float(parts[1])
            elif time_str.count(".") == 2:  # 處理 0.59.53 / 1.00.11 / 1.01.3 格式 (分.秒.毫秒)
                parts = time_str.split(".")
                return float(parts[0]) * 60.0 + float(parts[1]) + float(parts[2]) / 100.0
            else:
                return float(time_str)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def clean_margin_dist(margin_val: Any) -> Optional[float]:
        """將試閘差距文字轉為 float 馬身數 (例: 'Neck' -> 0.3, '3-3/4L' -> 3.75, '14-1/2L' -> 14.5)"""
        if pd.isna(margin_val) or margin_val is None:
            return None
        if isinstance(margin_val, (int, float)):
            return float(margin_val)

        m_str = str(margin_val).strip().upper().replace("L", "").strip()
        if not m_str:
            return 0.0  # 頭馬差距為空字串，預設為 0.0 馬身

        margin_map = {
            "---": 0.0,
            "DH": 0.0,
            "NOSE": 0.05,
            "N": 0.05,
            "SHORT HEAD": 0.1,
            "SH": 0.1,
            "HEAD": 0.2,
            "H": 0.2,
            "NECK": 0.3,
            "NK": 0.3,
            "ML": 99.0,
        }
        if m_str in margin_map:
            return margin_map[m_str]

        if "-" in m_str and "/" in m_str:
            try:
                parts = m_str.split("-")
                frac = parts[1].split("/")
                return float(parts[0]) + float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        elif "/" in m_str:
            try:
                frac = m_str.split("/")
                return float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        else:
            try:
                return float(m_str)
            except ValueError:
                return 0.25

    @staticmethod
    def extract_basic_info(basic_info_str: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
        """從 basic_info (例: '第 1 組 - 從化草地 - 1000米') 解析 跑道/場地 與 路程"""
        if not isinstance(basic_info_str, str) or not basic_info_str.strip():
            return None, None

        length_match = re.search(r"(\d{3,4})\s*米", basic_info_str)
        distance = int(length_match.group(1)) if length_match else None

        parts = [p.strip() for p in basic_info_str.split("-")]
        track_desc = parts[1] if len(parts) >= 2 else None

        return track_desc, distance

    @staticmethod
    def clean_draw_value(val: Any) -> Optional[int]:
        """檔位清洗 (防空字串、轉型整數)"""
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
        self,
        trails_dir: Optional[Union[str, pathlib.Path]] = settings.raw_trails_json_dir,
        debug: bool = True,
        sample_size: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """清洗試閘 raw JSON 數據，生成組別 (trials) 與試閘馬匹結果 (trial_results) 兩個標準 DataFrame"""
        if trails_dir is None:
            trails_dir = getattr(settings, "raw_trails_json_dir", settings.raw_json_dir / "trails")
        trails_dir = pathlib.Path(trails_dir)

        trials_list: List[Dict[str, Any]] = []
        trial_results_list: List[Dict[str, Any]] = []

        parse_stats = {
            "total_files": 0,
            "corrupted_files": 0,
            "total_trials": 0,
            "total_horse_results": 0,
            "id_matched_by_code": 0,
            "id_matched_by_map": 0,
            "id_unknown": 0,
        }

        if not trails_dir.exists():
            logger.warning(f"⚠️ [TrailsCleaner] 目錄不存在: {trails_dir}，回傳空 DataFrames。")
            empty_trials = pd.DataFrame(columns=[
                "trial_id", "date", "group_no", "basic_info", "venue", 
                "track_type", "distance", "track_condition", "finish_time_sec",
                "sec1_time", "sec2_time", "sec3_time", "sec4_time"
            ])
            empty_results = pd.DataFrame(columns=[
                "trial_id", "horse_id", "horse_code", "horse_name", "placing", 
                "draw", "jockey", "trainer", "gear", "margin_len", "running_position", 
                "finish_time_sec", "result_remark", "performance_comment", "is_withdrawn"
            ])
            return {"trials": empty_trials, "trial_results": empty_results}

        trail_files = list(trails_dir.glob("*.json"))
        parse_stats["total_files"] = len(trail_files)
        logger.info(f"🔍 [TrailsCleaner] 找到 {parse_stats['total_files']} 個試閘 Raw JSON 檔案...")

        for file_path in trail_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data or not isinstance(raw_data, dict):
                    continue

                trial_date = self.parse_date(raw_data.get("date"))
                if not trial_date:
                    continue

                trials_nodes = raw_data.get("trials") or []
                for trial in trials_nodes:
                    group_no = trial.get("group_no")
                    if group_no is None:
                        continue

                    parse_stats["total_trials"] += 1
                    trial_id = f"{trial_date}_G{int(group_no)}"

                    parsed_track, parsed_dist = self.extract_basic_info(trial.get("basic_info"))
                    dist = trial.get("distance") or parsed_dist
                    track_type = trial.get("track_type") or parsed_track

                    sec_times = trial.get("sectional_times") or []
                    sec1 = self.convert_min_to_sec(sec_times[0]) if len(sec_times) > 0 else None
                    sec2 = self.convert_min_to_sec(sec_times[1]) if len(sec_times) > 1 else None
                    sec3 = self.convert_min_to_sec(sec_times[2]) if len(sec_times) > 2 else None
                    sec4 = self.convert_min_to_sec(sec_times[3]) if len(sec_times) > 3 else None

                    trials_list.append({
                        "trial_id": trial_id,
                        "date": trial_date,
                        "group_no": int(group_no),
                        "basic_info": trial.get("basic_info"),
                        "venue": trial.get("venue"),
                        "track_type": track_type,
                        "distance": int(dist) if dist else None,
                        "track_condition": trial.get("track_condition"),
                        "finish_time_sec": self.convert_min_to_sec(trial.get("finish_time")),
                        "sec1_time": sec1,
                        "sec2_time": sec2,
                        "sec3_time": sec3,
                        "sec4_time": sec4,
                    })

                    horses_nodes = trial.get("horses") or []
                    for idx, horse in enumerate(horses_nodes, 1):
                        parse_stats["total_horse_results"] += 1

                        raw_name = horse.get("horse_name")
                        clean_name, extracted_code = self.extract_horse_info(raw_name)
                        
                        raw_id = horse.get("horse_id")
                        horse_code = extracted_code or (raw_id.split("_")[-1] if raw_id and "_" in raw_id else raw_id)

                        # 馬匹 ID 解析與補全邏輯
                        horse_id = None
                        if raw_id:
                            horse_id = raw_id
                            parse_stats["id_matched_by_code"] += 1
                        elif clean_name in self.horse_mapping:
                            horse_id = self.horse_mapping[clean_name]
                            parse_stats["id_matched_by_map"] += 1
                        elif horse_code:
                            horse_id = f"HK_{horse_code}"
                            parse_stats["id_matched_by_code"] += 1
                        else:
                            parse_stats["id_unknown"] += 1

                        placing = idx  # 預設按陣列名次排序

                        trial_results_list.append({
                            "trial_id": trial_id,
                            "horse_id": horse_id if horse_id else "UNKNOWN",
                            "horse_code": horse_code,
                            "horse_name": clean_name,
                            "placing": placing,
                            "draw": self.clean_draw_value(horse.get("draw")),
                            "jockey": horse.get("jockey"),
                            "trainer": horse.get("trainer"),
                            "gear": horse.get("gear"),
                            "margin_len": self.clean_margin_dist(horse.get("margin")),
                            "running_position": horse.get("running_position"),
                            "finish_time_sec": self.convert_min_to_sec(horse.get("finish_time")),
                            "result_remark": horse.get("result_remark"),
                            "performance_comment": horse.get("performance_comment"),
                            "is_withdrawn": bool(horse.get("is_withdrawn", False)),
                        })

            except Exception as e:
                parse_stats["corrupted_files"] += 1
                logger.error(f"❌ [TrailsCleaner] 解析檔案失敗 [{file_path.name}]: {e}")

        df_trials = pd.DataFrame(trials_list)
        df_results = pd.DataFrame(trial_results_list)

        raw_trials_len = len(df_trials)
        raw_results_len = len(df_results)

        if not df_trials.empty:
            df_trials = df_trials.drop_duplicates(subset=["trial_id"]).reset_index(drop=True)

        if not df_results.empty:
            df_results = df_results.drop_duplicates(
                subset=["trial_id", "horse_name", "placing"]
            ).reset_index(drop=True)

        if debug:
            self.display_debug_report(
                df_trials, df_results, parse_stats, raw_trials_len, raw_results_len, sample_size
            )

        return {"trials": df_trials, "trial_results": df_results}

    def display_debug_report(
        self,
        df_trials: pd.DataFrame,
        df_results: pd.DataFrame,
        stats: Dict[str, int],
        raw_trials_len: int,
        raw_results_len: int,
        sample_size: int = 5,
    ) -> None:
        """印出除錯日誌、數據品質報告與 DataFrame 隨機樣本"""
        print("\n" + "=" * 80)
        print("🛠️  [TrailsCleaner DEBUG REPORT] 試閘數據清洗除錯與品質審核")
        print("=" * 80)
        print(f" 📂 掃描 Raw 檔案總數: {stats['total_files']} | 損毀/解析失敗: {stats['corrupted_files']}")
        print(f" 📊 解析總計: 試閘組別 (trials)={stats['total_trials']} | 參賽紀錄 (trial_results)={stats['total_horse_results']}")
        print(f" 🔍 馬匹 ID 匹配來源: 直接/提取烙號={stats['id_matched_by_code']} | 名稱對照={stats['id_matched_by_map']} | 無法辨識(UNKNOWN)={stats['id_unknown']}")
        print(f" 🧹 組別去重: 清洗前 {raw_trials_len} 條 ➔ 去重後 {len(df_trials)} 條")
        print(f" 🧹 結果去重: 清洗前 {raw_results_len} 條 ➔ 去重後 {len(df_results)} 條")

        if df_trials.empty or df_results.empty:
            print(" ⚠️  [警告] 產出的 DataFrame 為空，請檢查路徑或 Raw Data Structure！")
            print("=" * 80 + "\n")
            return

        print("\n📈 [trial_results 欄位缺失值統計 (Missing Values Audit)]")
        missing_summary = pd.DataFrame({
            "Missing Count": df_results.isnull().sum(),
            "Missing Ratio (%)": (df_results.isnull().sum() / len(df_results) * 100).round(2)
        })
        print(missing_summary.to_string())

        print(f"\n👀 [trial_results 數據樣本預覽 (Top {sample_size} Rows)]")
        print("-" * 80)
        print(df_results.head(sample_size).to_string(index=False))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaner = TrailsCleaner()
    dfs = cleaner.process(debug=True, sample_size=5)