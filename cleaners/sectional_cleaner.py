import json
import pathlib
import re
import pandas as pd
from config.settings import settings


class SectionalCleaner:

    @staticmethod
    def extract_horse_info(raw_name: str) -> tuple[str | None, str | None]:
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
    def convert_min_to_sec(time_val) -> float | None:
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
                "DH": 0.0,    # Dead Heat (平頭馬)
                "N": 0.05,    # Nose (鼻位)
                "SH": 0.1,    # Short Head (短馬頭位)
                "H": 0.2,     # Head (頭位)
                "N": 0.3,     # Neck (頸位 - 如果怕跟 Nose 衝突，通常可以用 K 或 NK)
                "ML": 99.0,   # Multiple Lengths / Many Lengths (多個馬身)
                "DNF": None,  # Did Not Finish (未能完成賽事)
                "WV": None,   # Withdrawn (退出)
            }
            return margin_map.get(
                hhd_str, float(hhd_str) if hhd_str.isdigit() else 0.25
            )

    def process(
        self, sectionals_dir=settings.raw_sectional_json_dir
    ) -> pd.DataFrame:
        sectionals_dir = pathlib.Path(sectionals_dir)
        sec_list = []

        sec_files = list(sectionals_dir.glob("*.json"))
        print(
            f"🔍 [SectionalCleaner] 找到 {len(sec_files)} 個分段時間 Raw JSON 檔案..."
        )

        for file_path in sec_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    sec_data = json.load(f)

                if not sec_data or not isinstance(sec_data, dict):
                    continue

                date = sec_data.get("date")
                venue = sec_data.get("venue")

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

                        details = (
                            horse_sec.get("sectional_details")
                            or horse_sec.get("sectionals")
                            or []
                        )

                        for idx, detail in enumerate(details, 1):
                            sec_no = detail.get("section_no") or idx
                            position = detail.get("position") or detail.get(
                                "pos"
                            )
                            sec_time = detail.get(
                                "sectional_time"
                            ) or detail.get("time")
                            margin = detail.get("margin") or detail.get(
                                "behind"
                            )

                            sec_list.append({
                                "race_id": race_key,
                                "horse_name": clean_name,
                                "horse_id": horse_sec.get("horse_id"),
                                "section_no": int(sec_no),
                                "position": (
                                    int(position)
                                    if str(position).isdigit()
                                    else None
                                ),
                                "sectional_time_sec": self.convert_min_to_sec(
                                    sec_time
                                ),
                                "margin_behind": (
                                    self.clean_head_horse_dist(margin)
                                ),
                            })
            except Exception as e:
                print(
                    f"❌ [SectionalCleaner] 解析分段檔案失敗 [{file_path.name}]: {e}"
                )

        return pd.DataFrame(sec_list)