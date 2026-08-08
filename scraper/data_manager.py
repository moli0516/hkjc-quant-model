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
        """格式化日期檔名：YYYY-MM-DD.json，並存放在 races 目錄下"""
        formatted_date = f"{date_str[:4]}-{date_str[5:7]}-{date_str[8:]}"
        return settings.raw_races_json_dir / f"{formatted_date}.json"

    def check_file_exist(self, key_id: str, file_type: str = "horse") -> bool:
        """通用檢查檔案是否存在"""
        clean_id = key_id.strip().upper()

        if file_type == "horse":
            file_path = settings.raw_horses_json_dir / f"{clean_id}.json"
        elif file_type == "race":
            file_path = self._get_date_file_path(clean_id)
        elif file_type == "sectional":
            formatted_date = f"{clean_id[:4]}-{clean_id[5:7]}-{clean_id[8:]}"
            file_path = settings.raw_sectional_json_dir / f"{formatted_date}.json"
        elif file_type == "trackwork":
            file_path = settings.raw_trackworks_json_dir / f"{clean_id}.json"
        else:
            file_path = self.json_path / f"{clean_id}.json"

        return file_path.is_file()

    async def save_races_json(self, date_str: str, params: Dict[str, Any]) -> None:
        """非同步儲存單日賽果 JSON"""
        file_path = self._get_date_file_path(date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)

    async def save_normal_json(self, file_name: str, params: Any) -> None:
        """非同步儲存一般 JSON (修正舊版路徑 Bug)"""
        file_path = self.json_path / f"{file_name}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)

    async def save_trackwork_json(self, file_name: str, params: Any) -> None:
        """非同步儲存晨操 JSON"""
        target_dir = settings.raw_trackworks_json_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{file_name}.json"
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)