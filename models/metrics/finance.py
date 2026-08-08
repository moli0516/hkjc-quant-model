# models/metrics/finance.py
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinanceMetrics:
    """
    賽馬量化投資與財務指標計算模組 (具備完整 Logging 追蹤、溫度校準與凱利公式動態注碼)
    """

    @staticmethod
    def _softmax_with_temperature(logits: np.ndarray, temperature: float = 2.5) -> np.ndarray:
        """帶有溫度係數的 Softmax，將原始 Logits 校準為真實平滑機率分佈"""
        scaled_logits = logits / max(temperature, 1e-5)
        exp_scores = np.exp(scaled_logits - np.max(scaled_logits))  # 數值穩定防溢位
        return exp_scores / exp_scores.sum()

    @staticmethod
    def calculate_betting_performance(
        df: pd.DataFrame, 
        pred_score_col: str = "pred_score", 
        odds_col: str = "win_odds", 
        target_placing_col: str = "placing",
        group_col: str = "race_id",
        stake: float = 10.0,
        strategy: str = "tiered_ev",
        min_win_prob: float = 0.12,          # 🛡️ 門檻 1: 預測勝率不低於 12%
        max_odds: float = 18.0,              # 🛡️ 門檻 2: 賠率超過 18 倍不投注獨贏
        use_kelly: bool = True,
        kelly_fraction: float = 0.1667,      # 1/6 Kelly 降低波動
        initial_bankroll: float = 1000.0,
        max_stake_pct: float = 0.03,         # 單場注碼上限 3%
        min_ev: float = 1.05,                # 最低要求 5% 期望溢價
        temperature: float = 2.5             # 溫度校準參數
    ) -> Dict[str, Any]:
        """計算投注下的財務表現"""
        if df.empty or odds_col not in df.columns or target_placing_col not in df.columns:
            logger.warning("FinanceMetrics 收到空的 DataFrame 或缺少必要的賠率/名次欄位。")
            return {
                "total_bets": 0, "hit_count": 0, "win_rate": 0.0, 
                "total_stake": 0.0, "total_return": 0.0, "net_profit": 0.0, 
                "roi": 0.0, "final_bankroll": float(initial_bankroll)
            }

        total_stake = 0.0
        total_return = 0.0
        bet_count = 0
        hit_count = 0
        current_bankroll = float(initial_bankroll)

        for race_id, group in df.groupby(group_col):
            if group.empty:
                continue

            valid_group = group.dropna(subset=[odds_col, pred_score_col]).copy()
            if len(valid_group) < 2:
                continue

            # 1. 執行 Temperature Scaling 機率校準
            scores = valid_group[pred_score_col].values
            probs = FinanceMetrics._softmax_with_temperature(scores, temperature=temperature)
            valid_group["model_prob"] = probs

            # 2. 挑選模型預測首馬 (Top 1)
            top_horse = valid_group.sort_values(by=pred_score_col, ascending=False).iloc[0]
            
            actual_placing = top_horse[target_placing_col]
            odds = float(top_horse[odds_col])
            model_p = float(top_horse["model_prob"])

            # 3. 計算 EV (Expected Value)
            ev = model_p * odds

            # 🛡️ 嚴格風控 1: EV 門檻過濾
            if ev < min_ev:
                continue

            # 🛡️ 嚴格風控 2: 分層雙重過濾 (tiered_ev)
            if strategy == "tiered_ev":
                if model_p < min_win_prob or odds > max_odds:
                    continue

            # 4. 計算 Kelly 注碼
            if use_kelly:
                b = odds - 1.0
                q = 1.0 - model_p
                f_star = (b * model_p - q) / b if b > 0 else 0.0

                if f_star <= 0:
                    continue

                kelly_pct = min(f_star * kelly_fraction, max_stake_pct)
                current_stake = current_bankroll * kelly_pct

                if current_stake < 1.0: # 低於門檻不下注
                    continue
            else:
                current_stake = float(stake)

            current_stake = min(current_stake, current_bankroll)
            if current_stake <= 0:
                continue

            total_stake += current_stake
            bet_count += 1

            # 5. 結算
            if actual_placing == 1:
                return_amount = current_stake * odds
                total_return += return_amount
                hit_count += 1
                current_bankroll += (return_amount - current_stake)
            else:
                current_bankroll -= current_stake

        net_profit = total_return - total_stake
        roi = (net_profit / total_stake) if total_stake > 0 else 0.0
        win_rate = (hit_count / bet_count) if bet_count > 0 else 0.0

        return {
            "total_bets": int(bet_count),
            "hit_count": int(hit_count),
            "win_rate": float(win_rate),
            "total_stake": float(total_stake),
            "total_return": float(total_return),
            "net_profit": float(net_profit),
            "roi": float(roi),
            "final_bankroll": float(current_bankroll)
        }