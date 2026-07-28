from typing import Optional
import pandas as pd


class TrackEncoder:
    """跑道條件、路程與檔位組合的編碼與清洗器。"""

    @staticmethod
    def categorize_course_type(
        track_type_col: pd.Series, feature_name: Optional[str] = None
    ) -> pd.Series:
        """將 track_type 統一歸類為 TURF (草地) 或 AWT (全天候/泥地)。"""

        def _clean(val):
            if pd.isna(val):
                return "UNKNOWN"
            s = str(val).upper().strip()
            # 增強匹配條件，涵蓋 HKJC 常見的全天候/泥地標示
            if any(
                kw in s
                for kw in ["ALL WEATHER", "DIRT", "AWT", "ALL-WEATHER"]
            ):
                return "AWT"
            return "TURF"

        res = track_type_col.apply(_clean)
        return res.rename(
            feature_name if feature_name else "course_type_clean"
        )

    @staticmethod
    def create_track_draw_combo(
        venue_col: pd.Series,
        track_type_col: pd.Series,
        draw_col: pd.Series,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """建立賽場+跑道+檔位組合 Key（例如：ST_TURF_DRAW_1）。"""
        course_clean = TrackEncoder.categorize_course_type(track_type_col)
        combo = (
            venue_col.astype(str)
            + "_"
            + course_clean.astype(str)
            + "_DRAW_"
            + draw_col.astype(str)
        )
        return combo.rename(
            feature_name if feature_name else "track_draw_combo_key"
        )