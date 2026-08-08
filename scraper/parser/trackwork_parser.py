from typing import Any, Dict, List
import re

class TrackworkJsonParser:
    """直接解析原生 Trackwork JSON 結構的極速 Parser"""

    @staticmethod
    def parse(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析 TrackworkOneDayRecords API 回傳之 JSON 結構
        
        :param raw_json: 原始 JSON 字典
        :return: 結構化後的字典列表
        """
        if not raw_json or "Records" not in raw_json:
            return []

        cleaned_records = []
        # 正則表達式：判斷是否為香港賽馬會標準馬匹烙號格式 (例如 M011, H123, J045)
        horse_code_pattern = re.compile(r"^[A-Z]\d{3}$")

        for record in raw_json.get("Records", []):
            raw_horse = record.get("Horse", "")
            raw_horse_str = str(raw_horse).strip() if raw_horse else ""

            # 當 "Horse" 為標準烙號 (如 M011) 時，標記為 horse_code，否則為 horse_name
            is_code_only = bool(horse_code_pattern.match(raw_horse_str))
            
            horse_name = None if is_code_only else raw_horse_str
            horse_code = raw_horse_str if is_code_only else None

            cleaned_records.append({
                "horse_name": horse_name,
                "horse_code": horse_code,
                "trainer": record.get("Trainer"),
                "work_type": record.get("Type"),
                "racecourse_track": record.get("Racecourse_Track"),
                "workouts": record.get("Workouts"),
                "gear": record.get("Gear"),
            })

        return cleaned_records