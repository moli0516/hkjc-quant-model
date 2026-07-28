import json
import pathlib
from typing import Any, Dict
import aiofiles
from config.settings import settings

class DataManager:
    def __init__(self, json_path: pathlib.Path = settings.raw_json_dir):
        self.json_path = json_path
        self.json_path.mkdir(parents=True, exist_ok=True)

    def _get_date_file_path(self, date_str: str) -> pathlib.Path:
        """格式化日期檔名：YYYY-MM-DD.json"""
        formatted_date = f"{date_str[:4]}-{date_str[5:7]}-{date_str[8:]}"
        return self.json_path / f"{formatted_date}.json"

    def check_file_exist(self, key_id: str, file_type: str = "horse") -> bool:
        """通用檢查檔案是否存在"""
        clean_id = key_id.strip().upper()

        if file_type == "horse":
            file_path = settings.raw_horses_json_dir / f"{clean_id}.json"
        elif file_type == "race":
            file_path = self._get_date_file_path(clean_id)

        return file_path.is_file()

    async def save_races_json(self, date_str: str, params: Dict[str, Any]) -> None:
        """非同步儲存單日賽果 JSON"""
        file_path = self._get_date_file_path(date_str)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)

    async def save_normal_json(self, file_name: str, params: Any) -> None:
        """非同步儲存一般 JSON (修正舊版路徑 Bug)"""
        file_path = self.json_path / f"{file_name}.json"
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)
