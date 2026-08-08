import json
import logging
import pathlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from database.db_manager import DBManager

from config.settings import settings

logger = logging.getLogger(__name__)


class TrackworkCleaner:
    """HKJC 晨操與試閘數據清洗器 (Trackwork Data Cleaner)"""

    def __init__(
        self,
        rating_json_path: Optional[Union[str, pathlib.Path]] = None,
        horse_mapping: Optional[Dict[str, str]] = None,
        db: Optional[DBManager] = DBManager()
    ):  
        self.db = db
        self.rating_json_path = pathlib.Path(
            rating_json_path or settings.rating_path
        )
        self.horse_mapping = (
            horse_mapping
            if horse_mapping is not None
            else self.db.get_horse_name_id_mapping()
        )

    def _load_horse_mapping(self) -> Dict[str, str]:
        """從 Rating 資料與 Raw Horses/Races 資料建立 馬名 -> 馬匹烙號 (horse_id) 對照表"""
        mapping = {}

        # 1. 從 Rating JSON 載入
        if self.rating_json_path.exists():
            try:
                with open(self.rating_json_path, "r", encoding="utf-8") as f:
                    rating_json = json.load(f)
                for entry in rating_json:
                    h_name = entry.get("horse_name")
                    h_id = entry.get("horse_id") or entry.get("horse_code")
                    if h_name and h_id:
                        clean_name = (
                            re.sub(r"\s*\([A-Z0-9]{3,5}\)", "", str(h_name))
                            .replace("\xa0", "")
                            .strip()
                        )
                        mapping[clean_name] = str(h_id).strip()
            except Exception as e:
                logger.warning(
                    f"⚠️ [TrackworkCleaner] 載入 Rating 對照表失敗: {e}"
                )

        # 2. 從 Raw Horses 資料夾補全
        horses_dir = pathlib.Path(settings.raw_horses_json_dir)
        if horses_dir.exists():
            for fpath in horses_dir.glob("*.json"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    h_code = data.get("horse_code")
                    h_name = data.get("horse_name") or data.get("name")
                    if h_code and h_name:
                        clean_name = (
                            re.sub(r"\s*\([A-Z0-9]{3,5}\)", "", str(h_name))
                            .replace("\xa0", "")
                            .strip()
                        )
                        if clean_name not in mapping:
                            mapping[clean_name] = str(h_code).strip()
                except Exception:
                    continue

        logger.info(f"💡 [TrackworkCleaner] 馬名-烙號對照表加載完成，共 {len(mapping)} 條對映")
        return mapping

    @staticmethod
    def parse_date(val: Optional[str]) -> Optional[str]:
        """轉換日期格式（支援 '20260805', '21/05/2026' -> '2026-05-21' 或 '2026-05-21'）"""
        if not val or pd.isna(val):
            return None
        clean_val = str(val).strip()
        for fmt in ("%Y%m%d", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(clean_val, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def convert_min_to_sec(time_val: Any) -> Optional[float]:
        """時間轉秒數 (解析 1:10.12, 1.01.3 或 23.88)"""
        if pd.isna(time_val) or time_val in ["---", "", None]:
            return None
        time_str = str(time_val).strip()
        try:
            if ":" in time_str:
                parts = time_str.split(":")
                return float(parts[0]) * 60.0 + float(parts[1])
            elif time_str.count(".") == 2:  # 處理 1.01.3 格式 (分.秒.毫秒)
                parts = time_str.split(".")
                return float(parts[0]) * 60.0 + float(parts[1]) + float(parts[2]) / 10.0
            else:
                return float(time_str)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def parse_workouts_text(workouts_str: str) -> Dict[str, Any]:
        """從 Workouts 文字中解析總秒數、分段秒數、騎師/助手資訊與晨操/試閘路程 (Distance)"""
        res = {
            "work_time_sec": None,
            "rider": None,
            "distance": None,
            "sectional_count": 0
        }
        if not workouts_str:
            return res

        s = str(workouts_str).strip()

        # 1. 提取括號內的總時間，例如 (58.0), (1.01.3), (28.8), (1.11.90)
        total_time_match = re.search(r"\((1?\d?\.\d{2}\.\d{1,2}|\d{1,2}\.\d{1,2})\)", s)
        if total_time_match:
            time_raw = total_time_match.group(1)
            res["work_time_sec"] = TrackworkCleaner.convert_min_to_sec(time_raw)

        # 2. 提取括號內的騎師/助手/練馬師，例如 (艾兆禮), (助手), (副練馬師)
        rider_match = re.search(r"\(([^0-9\.\)\(\s]{2,10})\)", s)
        if rider_match:
            res["rider"] = rider_match.group(1)

        # 3. 提取路程 Distance（例如：1200M, 1000m, 800M, 1200 米）
        dist_match = re.search(r"\b(\d{3,4})\s*(?:M|m|米)?\b", s)
        if dist_match:
            res["distance"] = int(dist_match.group(1))

        # 4. 計算分段次數
        clean_desc = re.sub(r"\([^\)]*\)", "", s)
        sectionals = re.findall(r"\b(\d{2}\.\d{1,2})\b", clean_desc)
        res["sectional_count"] = len(sectionals)

        return res

    def process(
        self,
        trackwork_dir: Optional[Union[str, pathlib.Path]] = settings.raw_trackworks_json_dir,
        debug: bool = True,
        sample_size: int = 5,
    ) -> pd.DataFrame:
        """清洗晨操 raw JSON 數據，生成標準化 DataFrame"""
        if trackwork_dir is None:
            trackwork_dir = settings.raw_json_dir / "trackwork"
        else:
            trackwork_dir = pathlib.Path(trackwork_dir)

        records = []
        parse_stats = {
            "total_files": 0,
            "corrupted_files": 0,
            "total_raw_records": 0,
            "id_matched_by_code": 0,
            "id_matched_by_map": 0,
            "id_unknown": 0,
        }

        if not trackwork_dir.exists():
            logger.warning(
                f"⚠️ [TrackworkCleaner] 目錄不存在: {trackwork_dir}，回傳空 DataFrame。"
            )
            return pd.DataFrame(records)

        trackwork_files = list(trackwork_dir.glob("*.json"))
        parse_stats["total_files"] = len(trackwork_files)
        logger.info(
            f"🔍 [TrackworkCleaner] 找到 {parse_stats['total_files']} 個晨操 Raw JSON 檔案..."
        )

        horse_code_pattern = re.compile(r"^[A-Z0-9]{4,15}$", re.IGNORECASE)

        for file_path in trackwork_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                if isinstance(raw_data, list):
                    items = raw_data
                    global_work_date = None
                elif isinstance(raw_data, dict):
                    items = (
                        raw_data.get("records")
                        or raw_data.get("Records")
                        or raw_data.get("trackwork")
                        or raw_data.get("items")
                        or raw_data.get("data")
                        or [raw_data]
                    )
                    global_work_date = raw_data.get("date") or raw_data.get("work_date")
                else:
                    continue

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    parse_stats["total_raw_records"] += 1
                    
                    # 💡 修復 1: 相容爬蟲原生的 _crawled_date 與 date
                    raw_work_date = (
                        item.get("_crawled_date")
                        or item.get("work_date")
                        or item.get("date")
                        or global_work_date
                    )
                    work_date = self.parse_date(raw_work_date)

                    raw_horse = (
                        item.get("Horse")
                        or item.get("horse_id")
                        or item.get("horse_code")
                        or ""
                    )
                    raw_horse_str = str(raw_horse).strip()

                    horse_id = None
                    horse_name = item.get("horse_name")

                    # A. 若輸入值本身符合烙號規則 (如 "M011")
                    if horse_code_pattern.match(raw_horse_str) and not any(
                        "\u4e00" <= c <= "\u9fff" for c in raw_horse_str
                    ):
                        horse_id = raw_horse_str
                        parse_stats["id_matched_by_code"] += 1
                    else:
                        horse_name = raw_horse_str
                        # B. 從查表對照組補全 ID
                        if horse_name in self.horse_mapping:
                            horse_id = self.horse_mapping[horse_name]
                            parse_stats["id_matched_by_map"] += 1
                        else:
                            # C. 兜底搜尋 JSON 內部其他可能鍵值
                            horse_id = item.get("horse_id") or item.get("horse_code")
                            if horse_id:
                                parse_stats["id_matched_by_code"] += 1
                            else:
                                parse_stats["id_unknown"] += 1

                    # 💡 修復 2: 解析 Workouts 文字獲得時間與人名
                    workouts_text = item.get("Workouts") or item.get("workout_desc") or ""
                    parsed_workouts = self.parse_workouts_text(workouts_text)

                    # 💡 修復 3: 完美對齊爬蟲鍵值 (Racecourse_Track, Type, Gear, Trainer)
                    records.append({
                        "horse_id": horse_id if horse_id else "UNKNOWN",
                        "horse_name": horse_name,
                        "work_date": work_date,
                        "track_type": (
                            item.get("Racecourse_Track")
                            or item.get("track_type")
                            or item.get("venue")
                            or item.get("track")
                        ),
                        "workout_type": (
                            item.get("Type")
                            or item.get("workout_type")
                            or item.get("work_type")
                            or item.get("description")
                        ),
                        "distance": parsed_workouts["distance"],
                        "work_time_sec": (
                            parsed_workouts["work_time_sec"]
                            or self.convert_min_to_sec(item.get("work_time") or item.get("time"))
                        ),
                        "gear": item.get("Gear") or item.get("gear"),
                        "rider": (
                            item.get("Trainer")
                            or parsed_workouts["rider"]
                            or item.get("rider")
                            or item.get("jockey")
                        ),
                        "remarks": item.get("Workouts") or item.get("remarks") or item.get("comment"),
                    })

            except Exception as e:
                parse_stats["corrupted_files"] += 1
                logger.error(
                    f"❌ [TrackworkCleaner] 解析檔案失敗 [{file_path.name}]: {e}"
                )

        df_trackwork = pd.DataFrame(records)
        raw_len = len(df_trackwork)

        if not df_trackwork.empty:
            df_trackwork = df_trackwork.drop_duplicates(
                subset=["horse_id", "work_date", "workout_type", "work_time_sec"]
            ).reset_index(drop=True)

        dedup_len = len(df_trackwork)

        if debug:
            self.display_debug_report(df_trackwork, parse_stats, raw_len, dedup_len, sample_size)

        return df_trackwork

    def display_debug_report(
        self,
        df: pd.DataFrame,
        stats: Dict[str, int],
        raw_len: int,
        dedup_len: int,
        sample_size: int = 5,
    ) -> None:
        """印出調試日誌、數據品質報告與 DataFrame 隨機樣本"""
        print("\n" + "=" * 80)
        print("🛠️  [TrackworkCleaner DEBUG REPORT] 晨操數據清洗除錯與品質審核")
        print("=" * 80)
        print(f" 📂 掃描 Raw 檔案總數: {stats['total_files']} | 損毀/解析失敗: {stats['corrupted_files']}")
        print(f" 📊 解析紀錄總筆數: {stats['total_raw_records']} 條")
        print(f" 🔍 馬匹 ID 匹配來源: 直接烙號={stats['id_matched_by_code']} | 名稱對照={stats['id_matched_by_map']} | 無法辨識(UNKNOWN)={stats['id_unknown']}")
        print(f" 🧹 重複項清理: 清洗前 {raw_len} 條 ➔ 去重後 {dedup_len} 條 (移除 {raw_len - dedup_len} 條重複紀錄)")
        
        if df.empty:
            print(" ⚠️  [警告] 產出的 DataFrame 為空，請檢查路徑或 Raw Data Structure！")
            print("=" * 80 + "\n")
            return

        print("\n📈 [欄位缺失值統計 (Missing Values Audit)]")
        missing_summary = pd.DataFrame({
            "Missing Count": df.isnull().sum(),
            "Missing Ratio (%)": (df.isnull().sum() / len(df) * 100).round(2)
        })
        print(missing_summary.to_string())

        print(f"\n👀 [DataFrame 數據樣本預覽 (Top {sample_size} Rows)]")
        print("-" * 80)
        print(df.head(sample_size).to_string(index=False))

        print(f"\n🎲 [DataFrame 隨機採樣 (Random {sample_size} Samples)]")
        print("-" * 80)
        print(df.sample(min(sample_size, len(df))).to_string(index=False))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # 獨立除錯測試腳本入口
    logging.basicConfig(level=logging.INFO)
    cleaner = TrackworkCleaner()
    df_result = cleaner.process(debug=True, sample_size=5)