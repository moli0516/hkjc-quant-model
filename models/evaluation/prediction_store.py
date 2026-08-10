"""Walk-forward predictions 冷儲存。

把 evaluate() 產出的 predictions DataFrame 與執行 meta 持久化，
之後可離線重跑 diagnostics / betting rules / H1，無需重訓。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionStore:
    """OOP 冷儲存：predictions.parquet + meta.json。

    用法::

        store = PredictionStore()  # 讀 settings.predictions_dir
        run_dir = store.save(preds, meta={"model_name": "xgb_ranker", ...})
        preds2, meta2 = store.load()          # 最新一次
        preds3, meta3 = store.load(run_id="wf_20260810_190512")
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        file_format: str = "parquet",
    ):
        """
        Parameters
        ----------
        root_dir : 儲存根目錄。None 時從 settings.predictions_dir 讀取。
        file_format : 'parquet' | 'csv'（無 pyarrow 時可退回 csv）
        """
        if root_dir is None:
            try:
                from config.settings import settings

                root_dir = settings.predictions_dir
            except Exception:
                root_dir = Path("data/predictions")

        self.root_dir = Path(root_dir)
        self.file_format = (file_format or "parquet").lower().strip()
        if self.file_format not in {"parquet", "csv"}:
            raise ValueError("file_format 僅支援 'parquet' 或 'csv'")

        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 路徑工具
    # ------------------------------------------------------------------
    def _run_dir(self, run_id: str) -> Path:
        return self.root_dir / run_id

    def _data_path(self, run_dir: Path) -> Path:
        ext = "parquet" if self.file_format == "parquet" else "csv"
        return run_dir / f"predictions.{ext}"

    def _meta_path(self, run_dir: Path) -> Path:
        return run_dir / "meta.json"

    @staticmethod
    def make_run_id(prefix: str = "wf") -> str:
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------
    # 寫入
    # ------------------------------------------------------------------
    def save(
        self,
        predictions: pd.DataFrame,
        meta: Optional[dict] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """儲存 predictions 與 meta，回傳 run 目錄路徑。"""
        if predictions is None or predictions.empty:
            raise ValueError("predictions 為空，拒絕寫入。")

        run_id = run_id or self.make_run_id()
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        data_path = self._data_path(run_dir)
        meta_path = self._meta_path(run_dir)

        df = predictions.copy()
        # 序列化友善：Timestamp → 字串可在 load 再 parse
        self._write_frame(df, data_path)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "n_rows": int(len(df)),
            "n_races": int(df["race_id"].nunique()) if "race_id" in df.columns else None,
            "columns": list(df.columns),
            "format": self.file_format,
            "data_file": data_path.name,
        }
        if meta:
            payload["meta"] = meta

        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "💾 Predictions 已冷儲存 | run_id=%s | rows=%d | path=%s",
            run_id,
            len(df),
            run_dir,
        )
        return run_dir

    def _write_frame(self, df: pd.DataFrame, path: Path) -> None:
        if self.file_format == "parquet":
            try:
                df.to_parquet(path, index=False)
                return
            except Exception as e:
                logger.warning("parquet 寫入失敗 (%s)，改存 CSV。", e)
                path = path.with_suffix(".csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 讀取
    # ------------------------------------------------------------------
    def load(
        self,
        run_id: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        載入 predictions。

        - path: 直接指定 run 目錄或 predictions 檔案
        - run_id: 指定 run 名稱
        - 皆空: 載入最新一次 run
        """
        if path is not None:
            run_dir = self._resolve_path(Path(path))
        elif run_id is not None:
            run_dir = self._run_dir(run_id)
            if not run_dir.exists():
                raise FileNotFoundError(f"找不到 run: {run_dir}")
        else:
            run_dir = self.latest_run_dir()
            if run_dir is None:
                raise FileNotFoundError(
                    f"在 {self.root_dir} 找不到任何已儲存的 predictions run。"
                )

        meta = self._read_meta(run_dir)
        data_path = self._find_data_file(run_dir, meta)
        df = self._read_frame(data_path)

        # 還原常用型別
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("train_end", "test_end"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info(
            "📂 已載入 predictions | run=%s | rows=%d",
            run_dir.name,
            len(df),
        )
        return df, meta

    def _resolve_path(self, path: Path) -> Path:
        path = Path(path)
        if path.is_dir():
            return path
        if path.is_file():
            return path.parent
        # 相對 root
        candidate = self.root_dir / path
        if candidate.is_dir():
            return candidate
        if candidate.is_file():
            return candidate.parent
        raise FileNotFoundError(f"找不到 path: {path}")

    def _read_meta(self, run_dir: Path) -> dict:
        meta_path = self._meta_path(run_dir)
        if not meta_path.exists():
            return {"run_id": run_dir.name, "meta": {}}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _find_data_file(self, run_dir: Path, meta: dict) -> Path:
        name = meta.get("data_file")
        if name:
            p = run_dir / name
            if p.exists():
                return p
        for cand in ("predictions.parquet", "predictions.csv"):
            p = run_dir / cand
            if p.exists():
                return p
        raise FileNotFoundError(f"{run_dir} 內找不到 predictions 資料檔。")

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # 列表 / 最新
    # ------------------------------------------------------------------
    def list_runs(self) -> list[dict]:
        """列出所有 run（新到舊）。"""
        runs = []
        if not self.root_dir.exists():
            return runs
        for d in sorted(self.root_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta = self._read_meta(d)
            runs.append(
                {
                    "run_id": d.name,
                    "path": str(d),
                    "saved_at": meta.get("saved_at"),
                    "n_rows": meta.get("n_rows"),
                    "n_races": meta.get("n_races"),
                    "model_name": (meta.get("meta") or {}).get("model_name"),
                }
            )
        return runs

    def latest_run_dir(self) -> Optional[Path]:
        runs = self.list_runs()
        if not runs:
            return None
        return Path(runs[0]["path"])

    def print_runs(self) -> None:
        runs = self.list_runs()
        if not runs:
            print(f"（{self.root_dir} 尚無任何 cold-stored predictions）")
            return
        print(f"📁 Predictions store: {self.root_dir}")
        for r in runs:
            print(
                f"  • {r['run_id']} | rows={r.get('n_rows')} | "
                f"races={r.get('n_races')} | model={r.get('model_name')} | "
                f"at={r.get('saved_at')}"
            )