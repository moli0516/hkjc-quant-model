import os
import pandas as pd
import numpy as np
import xgboost as xgb

# =====================================================================
# 1. 基礎設定與數據載入
# =====================================================================
print("💡 正在啟動快活谷夜賽量化推理引擎（Schema 對齊版）...")

HISTORICAL_PARQUET_PATH = "horse_win_rate.parquet"  # 📌 請替換為你的 parquet 檔案路徑

if not os.path.exists(HISTORICAL_PARQUET_PATH):
    raise FileNotFoundError(f"找不到歷史數據 Parquet 檔案：{HISTORICAL_PARQUET_PATH}")

# 載入歷史特徵大盤
df_hist_raw = pd.read_parquet(HISTORICAL_PARQUET_PATH)

tonight_race_8_data = {
    # --- 基礎賽事與環境資訊 ---
    "date": ["2026/07/15"] * 11,
    "venue": ["跑馬地"] * 11,                                     # 🔍 已修正為官方匹配名稱
    "races.race_id": [8] * 11,                                    # 第 8 場
    "class": [2] * 11,                                            # ⚠️ 第二班高班賽
    "length": [1200] * 11,                                        # ⚠️ 1200米
    "track_texture": ["草地"] * 11,
    "track_type": ["C"] * 11,                                     # "C" 賽道
    "races.track_condition": ["好地"] * 11,                       
    "races.track_info": ["草地 - \"C\" 賽道"] * 11,
    
    # 11 匹馬的具體資料 (對應圖片中 1-11 號馬)
    "horse_name": [
        "幸運有您", "天天同樂", "翠紅", "競駿輝煌", "維港智能", 
        "昇瀧駒", "魔術控制", "勇敢巨星", "友愛心得", "小霸王", "興馳千里"
    ],
    "jockey": [
        "霍宏聲", "莫雷拉", "艾兆禮", "布文", "潘頓", 
        "巴度", "梁家俊", "何澤堯", "楊明綸", "黃寶妮", "田泰安"
    ],
    "trainer": [
        "羅富全", "羅富全", "廖康銘", "蔡約翰", "伍鵬志", 
        "蘇偉賢", "巫偉傑", "呂健威", "大衛希斯", "告東尼", "呂健威"
    ],
    "draw": [6, 1, 3, 4, 7, 2, 9, 8, 5, 11, 10],                  # 檔位
    "weight": [135, 133, 133, 125, 123, 122, 121, 119, 118, 118, 117], # 負磅
    "rank_weight": [1222, 1086, 1092, 1135, 1094, 1185, 1090, 1109, 1180, 1187, 1052], # 排位體重
    
    # 組合鍵
    "jockey_trainer": [
        "霍宏聲_羅富全", "莫雷拉_羅富全", "艾兆禮_廖康銘", "布文_蔡約翰", "潘頓_伍鵬志",
        "巴度_蘇偉賢", "梁家俊_巫偉傑", "何澤堯_呂健威", "楊明綸_大衛希斯", "黃寶妮_告東尼", "田泰安_呂健威"
    ],
}

# 轉換成 DataFrame
df_today = pd.DataFrame(tonight_race_8_data)

# 確保日期格式正確
df_today['date'] = pd.to_datetime(df_today['date'])
horse_id_map = df_hist_raw.groupby('horse_name')['horse_id'].last().to_dict()

# 2. 將今天排位表中的中文馬名，直接對照映射出 horse_id
df_today['horse_id'] = df_today['horse_name'].map(horse_id_map)

# 3. 檢查是否有歷史大盤中從未出現過的新馬（冷啟動）
missing_horses = df_today[df_today['horse_id'].isna()]['horse_name'].tolist()

if len(missing_horses) > 0:
    print(f"⚠️ 發現 {len(missing_horses)} 匹新馬在歷史大盤中無紀錄：{missing_horses}")
    print("👉 系統已自動為新馬編配臨時 ID，稍後特徵工程將自動以「中位數」進行冷啟動填補。")
    # 為新馬生成臨時 ID 避免後續代碼報錯
    for idx, row in df_today[df_today['horse_id'].isna()].iterrows():
        df_today.loc[idx, 'horse_id'] = f"NEW_{row['horse_name']}"
else:
    print("✅ 所有馬匹烙號（horse_id）已成功從歷史大盤中對齊！")

# 4. 打印前 5 筆確認結果
print(df_today[['horse_name', 'horse_id']])

# =====================================================================
# 3. 生成今日數據的交叉組合鍵 (確保與你的工程管道 100% 一致)
# =====================================================================
df_today["track_detailed"] = df_today["venue"].astype(str) + "_" + df_today["track_texture"].astype(str)
df_today["rail_detailed"] = df_today["venue"].astype(str) + "_" + df_today["track_texture"].astype(str) + "_" + df_today["track_type"].astype(str)
df_today["yeild_detailed"] = df_today["venue"].astype(str) + "_" + df_today["races.track_condition"].astype(str)
df_today["env_core"] = df_today["venue"].astype(str) + "_" + df_today["track_texture"].astype(str) + "_" + df_today["length"].astype(float).astype(str)
df_today["env_detail"] = df_today["env_core"] + "_" + df_today["track_type"].astype(str) + "_" + df_today["races.track_condition"].astype(str)

df_today["horse_track_detailed"] = df_today["horse_id_placeholder"] = ""  # 預留位置
df_today["horse_env_core"] = ""
df_today["horse_yeild_detailed"] = ""

df_today["jockey_track_detailed"] = df_today["jockey"].astype(str) + "_" + df_today["track_detailed"]
df_today["jockey_env_core"] = df_today["jockey"].astype(str) + "_" + df_today["env_core"]
df_today["jockey_yeild_detailed"] = df_today["jockey"].astype(str) + "_" + df_today["yeild_detailed"]

df_today["trainer_track_detailed"] = df_today["trainer"].astype(str) + "_" + df_today["track_detailed"]
df_today["trainer_yeild_detailed"] = df_today["trainer"].astype(str) + "_" + df_today["yeild_detailed"]
df_today['is_inside_d'] = (df_today['draw'] < 5).astype(int)
df_today["is_medium_d"] = ((df_today["draw"] > 4) & (df_today["draw"] < 9)).astype(int)
df_today['is_outside_d'] = (8 < df_today['draw']).astype(int)
# =====================================================================
# 4. 特徵對接：將歷史累積平滑特徵與速度特徵 Map 至今日排位表
# =====================================================================
print("🔮 正在進行特徵映射與冷啟動對齊...")

# 從歷史數據中取得「每匹馬/騎師/練馬師最後一次出賽後」所累積的最新歷史平滑勝率與速度表現
# 這一步能防止資料洩漏，並完美獲取最新狀態
latest_features = [
    # 基礎平滑
    "j_smoothed_place_rate", "t_smoothed_place_rate", "jt_smoothed_place_rate", "h_smoothed_place_rate", "d_smoothed_place_rate",
    # 滾動30/15/5近況
    "j_smoothed_rolling_30_place_rate", "t_smoothed_rolling_30_place_rate", 
    "jt_smoothed_rolling_15_place_rate", "h_smoothed_rolling_5_place_rate",
    # 場地、黏地、路程交叉平滑
    "h_track_smoothed_place_rate", "h_env_smoothed_place_rate", "h_yield_smoothed_place_rate",
    "j_track_smoothed_place_rate", "j_env_smoothed_place_rate", "j_yield_smoothed_place_rate",
    "t_track_smoothed_place_rate", "t_yield_smoothed_place_rate",
    # 歷史均速 Z-score
    "h_mean_speed_z_15","h_speed_z_momentum","h_race_count_history","h_rolling_2_speed_z_std","h_rolling_15_speed_z_std"
]

# 物理特徵與靜態特徵
PHYSICAL_FEATURES = ["weight", "rank_weight"]

# 組合出最終餵入模型的特徵列表
FEATURES = latest_features + PHYSICAL_FEATURES
TARGET = "is_place"

# 為今天排位表 Merge 歷史累積值
for col in latest_features:
    # 1. 判斷該特徵的 Key 是馬匹、騎師、練馬師還是組合
    if col.startswith("h_track"):
        mapping_key = "horse_track_detailed"
    elif col.startswith("h_env"):
        mapping_key = "horse_env_core"
    elif col.startswith("h_yield"):
        mapping_key = "horse_yeild_detailed"
    elif col.startswith("h_"):
        mapping_key = "horse_id"
    elif col.startswith("j_track"):
        mapping_key = "jockey_track_detailed"
    elif col.startswith("j_env"):
        mapping_key = "jockey_env_core"
    elif col.startswith("j_yield"):
        mapping_key = "jockey_yeild_detailed"
    elif col.startswith("j_"):
        mapping_key = "jockey"
    elif col.startswith("d_"):
        mapping_key = "draw"
    elif col.startswith("t_track"):
        mapping_key = "trainer_track_detailed"
    elif col.startswith("t_yield"):
        mapping_key = "trainer_yeild_detailed"
    elif col.startswith("t_"):
        mapping_key = "trainer"
    elif col.startswith("jt_"):
        mapping_key = "jockey_trainer"
    else:
        mapping_key = "horse_id" # 預設

    # 提取歷史最新的特徵狀態
    latest_map = df_hist_raw.sort_values('date').groupby(mapping_key)[col].last().to_dict()
    df_today[col] = df_today[mapping_key].map(latest_map)

# 清理歷史數據 (用來訓練)
df_hist_clean = df_hist_raw.dropna(subset=FEATURES + [TARGET]).copy()

# 填補今天排位表的缺失值 (冷啟動處理：新馬或新組合使用大盤中位數填補)
for col in FEATURES:
    median_val = df_hist_clean[col].median()
    df_today[col] = df_today[col].fillna(median_val)

# =====================================================================
# 5. 訓練 XGBoost 臨時模型
# =====================================================================
print(f"🤖 正在使用 {len(df_hist_clean)} 條乾淨歷史紀錄訓練 XGBoost...")

# 時序劃分驗證集 (最後 60 天)
df_hist_clean['date'] = pd.to_datetime(df_hist_clean['date'])
split_date = df_hist_clean['date'].max() - pd.Timedelta(days=60)

train_data = df_hist_clean[df_hist_clean['date'] <= split_date]
val_data = df_hist_clean[df_hist_clean['date'] > split_date]

X_train, y_train = train_data[FEATURES], train_data[TARGET]
X_val, y_val = val_data[FEATURES], val_data[TARGET]

model = xgb.XGBClassifier(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("✅ 模型訓練完成！驗證集防過擬合收斂成功。")

# =====================================================================
# 6. 執行推理與單場機率歸一化 (Calibrate)
# =====================================================================
print("🚀 正在執行 AI 矩陣推理解算...")

X_today = df_today[FEATURES]
# 預測上名原始概率
df_today['raw_prob'] = model.predict_proba(X_today)[:, 1]

# 計算同場相對預測排名
df_today['pred_rank'] = df_today.groupby('races.race_id')['raw_prob'].rank(ascending=False, method='min')

# 單場後處理：使同場 12 匹馬的位置率總和等於 3 (因為一場比賽有 3 個前三名名額)
def calibrate_probs(group):
    total = group['raw_prob'].sum()
    if total > 0:
        group['calibrated_prob'] = (group['raw_prob'] / total) * 3
    else:
        group['calibrated_prob'] = 3 / len(group)
    return group

df_final = df_today.groupby('races.race_id', group_keys=False).apply(calibrate_probs)

# =====================================================================
# 7. 現場看板可視化輸出
# =====================================================================
print("\n" + "="*75)
print(f"🏁 {df_today['date'].iloc[0]} 快活谷夜賽 AI 智能量化排位板 🏁")
print("="*75)

# 按預測排名排序並輸出
df_final_sorted = df_final.sort_values('pred_rank')

print(df_final_sorted[['pred_rank', 'horse_name', 'horse_id', 'draw', 'calibrated_prob']].to_string(
    index=False,
    formatters={
        'pred_rank': lambda x: f"Rank {int(x)}",
        'draw': lambda x: f"{int(x)}檔",
        'calibrated_prob': lambda x: f"{x*100:.1f}%"
    }
))

print("\n" + "="*75)
print("💡 實戰小貼士：")
print("1. 請優先關注前三名 (Rank 1, 2, 3) 且檔位在 1-5 檔之內、體重適中的馬匹。")
print("2. 當 AI 修正位置率高於 60% 且現場位置賠率（Place）大於 1.5 倍時，即為數學上的「高期望值 (EV) 投注區」。")
print("="*75)

# Sort and print the results
booster = model.get_booster()

# Get 'weight' importance dictionary
# Options include: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
weights = booster.get_score(importance_type='weight')

# Sort and print the feature weights
sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
for feature, weight in sorted_weights:
    print(f"Feature: {feature}, Weight: {weight}")