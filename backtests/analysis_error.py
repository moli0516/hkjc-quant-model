import pandas as pd

def generate_error_report(file_path, output_file='error_analysis.csv'):
    # 1. 載入資料
    df = pd.read_parquet(file_path)
    
    # 2. 篩選出模型認為會贏（pred_rank == 1）但實際輸掉（placing != 1）的場次
    errors = df[(df['pred_rank'] == 1) & (df['placing'] != 1)].copy()
    
    # 3. 加入虧損診斷指標
    # 假設我們在這些場次都下注了
    errors['error_type'] = 'False Positive' # 模型誤判為勝
    errors['odds_group'] = pd.cut(errors['odds'], bins=[0, 2, 3, 5, 10, 100])
    
    # 4. 排序：找出那些「賠率最高、模型最有信心卻輸掉」的慘烈場次
    errors = errors.sort_values(by=['pred_prob', 'odds'], ascending=[False, False])
    
    # 5. 匯出 CSV 供您人工查看
    columns_to_show = ['race_unique_id', 'horse_id', 'odds', 'pred_prob', 'placing', 'odds_group']
    # 如果您的 DataFrame 有其他特徵（如馬匹負磅、練馬師、騎師），請添加到上方列表中
    
    errors[columns_to_show].to_csv(output_file, index=False)
    
    print(f"==================== 🔍 錯誤分析報告已生成 ====================")
    print(f"共分析出 {len(errors)} 場誤判場次")
    print(f"清單已儲存至: {output_file}")
    print(f"前 5 場最慘烈誤判樣本:")
    print(errors[columns_to_show].head(5))

if __name__ == "__main__":
    PATH = 'd:/git-repos/hkjc-quant-model/data/oot_july_predictions.parquet'
    generate_error_report(PATH)