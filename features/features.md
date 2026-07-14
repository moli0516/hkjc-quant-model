## 📂 模組一：`race_feature` folder（賽事與環境特徵）

*此表的主鍵（Unique Key）為 `race_id`。負責捕捉當天、當場比賽的客觀物理環境與大盤雜訊。*


| 特徵欄位名稱                    | 數據類型 | Finished? |
| ------------------------------- | -------- | --------- |
| `draw_placing_rate`             | 浮點數   | Done      |
| `draw_win_rate`                 | 浮點數   | Done      |

## 📂 模組二：`horses_features` folder（馬匹、人為與歷史滾動特徵）

*此表的主鍵為 `race_id` + `horse_id`。這是你模型的靈魂。**請注意：計算以下所有滾動與歷史特徵前，必須先 Merge `date` 進來，依時間正序排列，並嚴格執行 `.shift(1)` 防漏工程。***

### 🐎 1. 馬匹物理與天賦維度


| 特徵欄位名稱             | 數據類型 | Finished? |
| ------------------------ | -------- | --------- |
| `win_rate`               | 浮點數   | Done      |
| `place_rate`             | 浮點數   | Done      |
| `win_z`                  | 浮點數   | Done      |
| `place_z`                | 浮點數   | Done      |
| `smoothed_win_rate`      | 浮點數   | Done      |
| `smoothed_placing_rate`  | 浮點數   | Done      |
| `speed`                  | 浮點數   | Done      |
| `speed_z`                | 浮點數   | Done      |
| `rolling_2_speed_z`      | 浮點數   | Done      |
| `rolling_2_speed_z_std`  | 浮點數   | Done      |
| `smoothed_h_track_win_rate`     | 浮點數   | Done      |
| `smoothed_h_track_placing_rate` | 浮點數   | Done      |
| `smoothed_h_rail_win_rate`      | 浮點數   | Done      |
| `smoothed_h_rail_placing_rate`  | 浮點數   | Done      |

### 🏇 2. 人為形勢與利益博弈維度


| 特徵欄位名稱                | 數據類型 | Finished? |
| --------------------------- | -------- | --------- |
| `j_win_rate`                | 浮點數   | Done      |
| `j_placing_rate`            | 浮點數   | **Done    |
| `30_t_win_rate`             | 浮點數   | Done      |
| `30_t_placing_rate`         | 浮點數   | Done      |
| `jt_win_rate`               | 浮點數   | Done      |
| `jt_placing_rate`           | 浮點數   | Done      |
| `rolling_30_j_win_rate`     | 浮點數   | Done      |
| `rolling_30_j_placing_rate` | 浮點數   | Done      |
| `rolling_30_t_win_rate`     | 浮點數   | Done      |
| `rolling_30_t_placing_rate` | 浮點數   | Done      |
| `smoothed_jt_win_rate`      | 浮點數   | Done      |
| `smoothed_jt_placing_rate`  | 浮點數   | Done      |

### ⚖️ 3. 當場博弈與相對化維度

*這組特徵需要你在最後的 `03_Model_Training.ipynb` 中，將 Races 和 Horses 拼接後，透過 `groupby('race_id')` 即時計算。*


| 特徵欄位名稱             | 數據類型 | 底層邏輯與物理意義                                                                                                                                                    |
| ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `actual_weight_z`        | 浮點數   | **當場負磅 Z-Score**：這匹馬今天的負磅（如 133 磅）在當場 14 匹馬中是偏重（正數）還是偏輕（負數）。負重直接影響物理加速度。                                           |
| `draw_advantage`         | 浮點數   | **當場檔位優勢指數**：結合當天跑道偏差，計算這匹馬今天排到的檔位（如 1 檔或 12 檔）在歷史上的特定勝率。                                                               |
| `horse_scene_win_rate_z` | 浮點數   | **分場景當場對手 Z-Score**：這匹馬在「特定路程+特定場地」的歷史勝率，與**同場其餘 13 匹對手**進行當場 Z-Score 標準化。這是捕捉**「降維打擊/泥地專門怪」**的核心武器。 |
| `odds`                   | 浮點數   | **臨場獨贏賠率**：直接保留原始賠率或轉換為隱含機率 ($1 / \text{odds}$)。這是全市場資金共識的結晶，包含了所有幕後未知的「開工心情」訊號。                              |

---

## 🛠️ 開工前的工程建議

當你在 `horses_features.ipynb` 建立這些欄位時，你可以直接把這份表格當成你的 **Checklist**。

1. **先做物理，再做人為：** 先把馬匹的 `speed_z` 滾動平均做出來，確保 Bug Free。
2. **死守 $K$ 因子：** 在算 `jt_prob_encoded`（騎練組合）時，務必引入平滑因子 $K$（建議設在 $10 \sim 20$ 之間），強行壓制長年份大數據帶來的極端統計波動。

這份特徵矩陣一旦合體完成，你的 XGBoost 模型就同時擁有了「物理天賦」、「近期狀態」、「人情世故」與「當場博弈」的全面視角。你準備先從哪一個維度的特徵代碼開始動手寫起？