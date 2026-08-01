# models/metrics/finance.py
import logging
import numpy as np
import pandas as pd

# 建立專屬的 logger
logger = logging.getLogger(__name__)

class FinanceMetrics:
    """
    賽馬量化投資與財務指標計算模組 (具備完整 Logging 追蹤與凱利公式動態注碼)
    """

    @staticmethod
    def calculate_betting_performance(
        df: pd.DataFrame, 
        pred_score_col: str = "pred_score", 
        odds_col: str = "win_odds", 
        target_placing_col: str = "placing",
        group_col: str = "race_id",
        stake: float = 10.0,
        use_kelly: bool = False,
        kelly_fraction: float = 0.5,
        initial_bankroll: float = 1000.0,
        max_stake_pct: float = 0.05
    ) -> dict:
        """
        計算投注下的財務表現。
        支援固定金額平注法 (Flat Betting) 或凱利公式動態注碼 (Kelly Criterion)。
        """
        if df.empty or odds_col not in df.columns or target_placing_col not in df.columns:
            logger.warning("FinanceMetrics 收到空的 DataFrame 或缺少必要的賠率/名次欄位。")
            return {"total_bets": 0, "hit_count": 0, "win_rate": 0.0, "total_stake": 0.0, "total_return": 0.0, "net_profit": 0.0, "roi": 0.0}

        total_stake = 0.0
        total_return = 0.0
        bet_count = 0
        hit_count = 0
        
        # 追蹤凱利動態資金池（若未啟用凱利，則僅作記錄參考）
        current_bankroll = initial_bankroll

        mode_str = f"凱利公式 (Fraction: {kelly_fraction})" if use_kelly else f"固定平注 (Stake: {stake})"
        logger.info(f"開始進行財務回測計算，總資料列數: {len(df)}，模式: {mode_str}")

        # 依據賽事分組
        for race_id, group in df.groupby(group_col):
            if group.empty:
                continue

            # 排除賠率或預測分數為 NaN 的馬匹
            valid_group = group.dropna(subset=[odds_col, pred_score_col])
            if valid_group.empty:
                logger.debug(f"賽事 {race_id} 沒有合法的有效賠率資料，跳過。")
                continue

            # 1. 將整場模型分數透過 Softmax 轉化為真實機率分佈 (p)
            scores = valid_group[pred_score_col].values
            exp_scores = np.exp(scores - np.max(scores)) # 防止數值溢位
            probs = exp_scores / exp_scores.sum()
            
            # 將計算出的機率暫存回臨時 DataFrame
            eval_group = valid_group.copy()
            eval_group['model_prob'] = probs

            # 2. 挑選出模型預測分數最高（模型心中的第一名）的馬匹
            top_horse = eval_group.sort_values(by=pred_score_col, ascending=False).iloc[0]
            
            actual_placing = top_horse[target_placing_col]
            odds = top_horse[odds_col]
            model_p = top_horse['model_prob']

            # 3. 決定本場下注金額 (Stake)
            if use_kelly:
                # 凱利公式計算: f* = (bp - q) / b
                b = odds - 1.0
                q = 1.0 - model_p
                f_star = (b * model_p - q) / b if b > 0 else 0.0
                
                # 若期望值 <= 0 或計算出的比例 <= 0，則不上注
                if f_star <= 0:
                    logger.debug(f"[略過] 賽事 {race_id} | 期望值為負或無優勢 (EV <= 1)，不上注。")
                    continue
                
                # 應用 Fractional Kelly 與單場風控上限比例
                kelly_pct = min(f_star * kelly_fraction, max_stake_pct)
                current_stake = current_bankroll * kelly_pct
                
                # 確保資金足夠且大於最小單位
                if current_stake <= 0:
                    continue
            else:
                current_stake = stake

            total_stake += current_stake
            bet_count += 1
            
            # 4. 結算勝負與回報
            if actual_placing == 1:
                return_amount = current_stake * odds
                total_return += return_amount
                hit_count += 1
                current_bankroll += (return_amount - current_stake) # 增加淨利
                logger.debug(f"[命中] 賽事 {race_id} | 投注馬匹勝出！賠率: {odds} | 投入: {current_stake:.2f} | 獲得回報: {return_amount:.2f}")
            else:
                current_bankroll -= current_stake # 扣除虧損
                logger.debug(f"[未中] 賽事 {race_id} | 實際名次: {actual_placing} | 投入: {current_stake:.2f}")

        net_profit = total_return - total_stake
        roi = (net_profit / total_stake) if total_stake > 0 else 0.0
        win_rate = (hit_count / bet_count) if bet_count > 0 else 0.0

        # 輸出結構化摘要日誌
        logger.info(
            f"財務回測完成 => 總投注場次: {bet_count} | 命中場次: {hit_count} | "
            f"勝率: {win_rate*100:.2f}% | 總投入: ${total_stake:.2f} | "
            f"淨損益: ${net_profit:.2f} | ROI: {roi*100:.2f}%"
        )

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