import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_place_safe_stake(prob, place_odds, current_bankroll, initial_bankroll, confidence_level):
    """專為『位置市場』設計的穩健型凱利注碼控制"""
    b = place_odds - 1
    q = 1 - prob
    
    # 計算位置凱利值
    print
    kelly = (b * prob - q) / b if b > 0 else 0
    
    if kelly <= 0:
        return 0.0
    
    # 位置市場波動較低，採用 Fractional Kelly 控防回撤
    if confidence_level == 'high':
        factor = 0.08  # 1/12.5 凱利
    elif confidence_level == 'medium':
        factor = 0.04  # 1/25 凱利
    else:
        factor = 0.0
        
    if current_bankroll < (initial_bankroll * 0.95):
        factor *= 0.5
        
    stake = kelly * factor * current_bankroll
    max_allowed = current_bankroll * 0.025  # 單場最高限制 2.5%
    
    return min(stake, max_allowed)

def categorize_place_bet_aligned(row):
    place_odds = 1 + 0.25 * (row['odds'] - 1)
    
    # 修正 1：放寬對超熱門馬的限制。位置賠率低至 1.10 (獨贏 1.4) 也可以納入
    # 因為我們需要這些高勝率的基石來穩定資金曲線
    if place_odds < 1.10 or place_odds > 4.5:
        return 'none'
        
    # 必須是模型推薦的 Top 1 首選馬
    if row['pred_rank'] == 1:
        # 修正 2：不再看絕對的 0.48，而是看模型對這匹馬的信心是否顯著大於市場預期
        # 市場預期的位置勝率大約是 1 / place_odds
        market_implied_prob = 1.0 / place_odds
        
        # 計算模型高出市場的「優勢邊際 (Edge)」
        edge = row['pred_prob'] - market_implied_prob
        
        # 只有當模型算出的勝率，比市場賠率隱含的勝率高出 7% 以上時，才認定為高信心
        if edge > 0.07:
            return 'high'
        elif edge > 0.02:
            return 'medium'
            
    return 'none'

def run_place_backtest(file_path, initial_bankroll=10000):
    if not os.path.exists(file_path):
        print(f"❌ 找不到預測明細檔案: {file_path}")
        return

    df = pd.read_parquet(file_path)
    df = df.sort_values(by=['date', 'races.race_id']).reset_index(drop=True)
    
    # 套用【完全對齊】後的過濾器
    df['bet_type'] = df.apply(categorize_place_bet_aligned, axis=1)
    filtered = df[df['bet_type'] != 'none'].copy()
    
    current_bankroll = initial_bankroll
    history = [initial_bankroll]
    peak_bankroll = initial_bankroll
    
    total_bets = 0
    wins = 0
    
    print(f"=== 🏇 啟動『位置(Place)』全邏輯對齊回測引擎 ===")
    
    for idx, row in filtered.iterrows():
        if current_bankroll > peak_bankroll:
            peak_bankroll = current_bankroll
            
        current_drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
        
        # 安全熔斷
        if current_drawdown > 0.10:
            print(f"⚠️ 觸發安全熔斷！當前資產自最高點回撤達 {current_drawdown*100:.2f}%。")
            break
            
        # 再次計算對齊的位置賠率（確保與過濾器一致）
        place_odds = 1 + 0.25 * (row['odds'] - 1)
        
        # 計算注碼 (代入位置賠率)
        stake = get_place_safe_stake(row['pred_prob'], place_odds, current_bankroll, initial_bankroll, row['bet_type'])
        
        if stake > 0:
            stake = max(stake, 10)
        else:
            continue
            
        total_bets += 1
        is_place_win = row['placing'] in [1, 2, 3]
        
        if is_place_win:
            wins += 1
            # 派彩完全使用位置賠率計算
            payout = (place_odds - 1) * stake
        else:
            payout = -stake
            
        current_bankroll += payout
        history.append(current_bankroll)
        
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    print(f"\n================ 位置回測結果 (對齊版) ================")
    print(f"過濾後總下注場次: {total_bets} 場")
    print(f"位置實際命中率  : {win_rate:.2f}%")
    print(f"最終資產總額    : {current_bankroll:.2f} HKD")
    print(f"總淨損益率      : {((current_bankroll - initial_bankroll)/initial_bankroll)*100:.2f}%")
    print(f"======================================================")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Place Aligned Bankroll', color='#2ca02c', linewidth=2)
    plt.axhline(y=initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
    plt.title("Place Betting Strategy - Fully Aligned Odds & Filter")
    plt.xlabel("Number of Bets")
    plt.ylabel("Bankroll (HKD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
 
if __name__ == "__main__":
    PATH = r'd:\git-repos\hkjc-quant-model\data\oot_july_predictions.parquet'
    run_place_backtest(PATH)