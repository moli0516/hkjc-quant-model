# 🐎 HKJC Quant: Feature Engineering Pipeline (量化特徵工程管線)

這是一個專為香港賽馬（HKJC）量化預測設計的高效、防洩漏（Leakage-Proof）、且具備極高擴充性的特徵工程管線。本模組負責將原始的賽事與馬匹數據，轉化為可用於機器學習模型的高品質特徵矩陣。

本專案的核心設計理念為：**「絕對的時間隔離（防數據洩漏）、零碎片化的記憶體管理，以及隨插即用的動態擴充架構。」**

---

## 🏗️ 1. 核心工作流程 (Pipeline Workflow)

整個特徵工程的工作流程由 `BaseTargetBuilder` 與 `FeaturesPipeline` 兩個核心類別協同完成。完整的資料流如下：

### Step 1: 基礎骨架與目標變數建立 (`BaseTargetBuilder`)

* **資料載入與清洗：** 接收來自 SQLite 的 Raw Data，統一欄位命名，並濾除退跑 (WV)、無效名次 (DNF, PU 等) 及空值。
* **時間排序防護：** 強制將數據按 `race_date`, `race_id`, `horse_id` 進行升冪排序，這是防止未來數據洩漏的第一道防線。
* **動態時序遮罩 (Temporal Guard)：** 嚴格比對馬匹的 `import_date` 與 `race_date`。若抵港日晚於賽事日，強制將其遮蔽為 `NaT`，防止未到的資訊穿越時空。
* **Target 生成：** 統一生成機器學習模型的標籤：`target_win` (獨贏), `target_place` (位置), `target_rank_score` (名次分數)。

### Step 2: 進入主幹道 (`FeaturesPipeline`)

* **初始化與掃描：** 實例化 Pipeline 時，會自動掃描 `generators/` 目錄下的所有可用模組。
* **特徵迭代生成：** 依照各 Generator 定義的優先級（`EXECUTION_ORDER`）逐一派發工作。
* **記憶體與效能優化 (Zero-Fragmentation)：**
* 過濾已存在的重複欄位，確保不覆寫。
* 自動將新生成的 `float64` 降維轉型為 `float32`，大幅降低記憶體消耗。
* 將每次迴圈產出的純特徵暫存於 List，最後使用 `pd.concat` 一次性水平併合（Zero-copy），避免 Pandas DataFrame 不斷 append 造成的記憶體碎片化 (Fragmentation)。
* **對齊與輸出：** 將生成的特徵與原始的 `key_cols` (如 `race_id`, `horse_id`) 對齊，並強制恢復傳入時的原始 Index，輸出最終完整的特徵矩陣。

---

## 🔌 2. 熱插拔特徵生成器機制 (Hot-Swappable Generators)

本系統無需在 Pipeline 中手動 `import` 或註冊任何特徵腳本。所有 Generator 均採用「熱插拔 (Plugin)」架構。

### 實現原理 (`generators/__init__.py`)

1. **動態模組探索：** 使用 Python 內建的 `pkgutil.iter_modules` 與 `importlib.import_module`，動態掃描 `generators` 資料夾下所有不以 `_` 開頭的 Python 檔案。
2. **類別反射 (Reflection)：** 透過 `inspect.getmembers` 尋找繼承或命名以 `Generator` 結尾的 Class。
3. **執行序控制 (Execution Order)：** 實例化後，系統會讀取類別內的 `EXECUTION_ORDER` 常數（預設 500）進行排序。這保證了具有相依性的特徵（例如：需先計算走位，才能計算跑法策略）能依照正確的順序被執行。

> **💡 開發者指南：** 若要新增特徵，只需在 `generators/` 目錄下建立一個新的 `.py` 檔案，編寫一個包含 `generate(self, df)` 方法且設定好 `EXECUTION_ORDER` 的類別，Pipeline 下次執行時將會**自動載入並套用**，完全無需修改核心程式碼。

---

## 📋 3. 特徵生成器與產出目錄 (Generators & Features Catalog)

以下為目前系統內建的各個 Generator 模組、用途解釋，以及其產出的特徵定義總覽：

### 🐎 馬匹基本狀態與資歷


| Generator                         | 用途解釋                               | 產出 Feature 欄位           | 定義                                         |
| --------------------------------- | -------------------------------------- | --------------------------- | -------------------------------------------- |
| **`HorseProfileGenerator`**       | 計算馬匹在香港的服役年資。             | `est_years_in_hk`           | 該場比賽當下，馬匹已抵港服役的實際年數。     |
| **`InjuryRestGenerator`**         | 捕捉馬匹的參賽節奏、久休復出與適應期。 | `days_since_last_race`      | 距離上一場出賽的天數。                       |
|                                   |                                        | `is_layoff_60d` / `90d`     | 是否休息超過 60 天 / 90 天（久休復出標記）。 |
|                                   |                                        | `days_since_import`         | 該場比賽當下距離抵港日期的天數。             |
| **`BodyWeightRecoveryGenerator`** | 監測馬匹體重變化與密集出賽疲勞度。     | `horse_weight_vs_hist_mean` | 當前體重與過去 3 場平均體重的差值。          |
|                                   |                                        | `horse_weight_abs_change`   | 與上一場體重的絕對變化量。                   |
|                                   |                                        | `is_heavy_workload_14d`     | 是否在 14 天內連續出賽的高強度賽程標記。     |

### 📈 歷史績效與實力指標 (滾動統計)


| Generator                       | 用途解釋                                            | 產出 Feature 欄位                    | 定義                                              |
| ------------------------------- | --------------------------------------------------- | ------------------------------------ | ------------------------------------------------- |
| **`HorseRollingGenerator`**     | 馬匹歷史表現的滾動統計（透過 Shift 隔離防洩漏）。   | `horse_rolling_pos_mean_{3,5,10}`    | 過去 3/5/10 場的平均名次。                        |
|                                 |                                                     | `horse_rolling_pos_std_{3,5,10}`     | 過去 3/5/10 場名次的標準差。                      |
|                                 |                                                     | `horse_rolling_win_rate_{3,5,10}`    | 過去 3/5/10 場的滾動平滑勝率。                    |
|                                 |                                                     | `horse_rolling_top3_rate_{3,5,10}`   | 過去 3/5/10 場的滾動平滑上名率。                  |
|                                 |                                                     | `horse_rolling_weight_mean_{3,5,10}` | 過去 3/5/10 場的平均負重。                        |
|                                 |                                                     | `horse_weight_change`                | 本場與上場馬匹體重的淨差額。                      |
| **`ClassPerformanceGenerator`** | 分析馬匹在各班次 (Class 1-5) 的適應力及升降班表現。 | `horse_class_win_rate`               | 馬匹於該班次的歷史貝氏平滑勝率。                  |
|                                 |                                                     | `horse_class_top3_rate`              | 馬匹於該班次的歷史貝氏平滑上名率。                |
|                                 |                                                     | `is_class_up` / `is_class_down`      | 該場是否為升班戰 / 降班戰。                       |
|                                 |                                                     | `horse_hist_class_up_count`          | 歷史累積升班次數。                                |
|                                 |                                                     | `horse_hist_class_down_count`        | 歷史累積降班次數。                                |
| **`TrackDistanceGenerator`**    | 針對特定路程與場地類型的歷史適應性。                | `horse_dist_win_rate`                | 該馬匹於相同距離的歷史平滑勝率。                  |
|                                 |                                                     | `horse_track_win_rate`               | 該馬匹於相同場地(草/泥)的歷史平滑勝率。           |
| **`RatingClassGenerator`**      | 簡單的班次變化數字特徵。                            | `class_change`                       | 班次數字與上一場的差異。                          |
| **`RatingMomentumGenerator`**   | 評分優勢與動態趨勢 (純硬實力)。                     | `rating_diff_from_race_mean`         | 本馬評分與同場對手平均評分的差值。                |
|                                 |                                                     | `rating_race_z`                      | 本馬評分在同場比賽中的 Z-Score。                  |
|                                 |                                                     | `rating_momentum_3`                  | 本場評分與 3 場前評分的變化量（上升或下降趨勢）。 |

### 🏇 人馬關係與幕後團隊 (Alpha & Synergy)


| Generator                           | 用途解釋                                            | 產出 Feature 欄位             | 定義                                                     |
| ----------------------------------- | --------------------------------------------------- | ----------------------------- | -------------------------------------------------------- |
| **`HumanSireGenerator`**            | 騎師、練馬師與種馬的全域績效。                      | `jockey_rolling_win_rate_10`  | 騎師近 10 場滾動勝率。                                   |
|                                     |                                                     | `trainer_rolling_win_rate_20` | 練馬師近 20 場滾動勝率。                                 |
|                                     |                                                     | `sire_global_win_rate`        | 該種馬 (父系) 子嗣的全域平滑勝率。                       |
| **`JockeyTrainerSynergyGenerator`** | 長期合作與特定專屬組合的默契表現。                  | `jt_combo_win_rate_smooth`    | 該「騎師+練馬師」組合的歷史平滑勝率。                    |
|                                     |                                                     | `horse_jockey_combo_win_rate` | 該「馬匹+騎師」專屬組合的歷史平滑勝率。                  |
| **`JockeyTrainerAlphaGenerator`**   | 騎師與練馬師組合帶來的附加價值 (Alpha) 及換人效應。 | `alpha_jt_combo_win_rate`     | 組合全域勝率 (限 1-5 班)。                               |
|                                     |                                                     | `alpha_jt_combo_top3_rate`    | 組合全域上名率 (限 1-5 班)。                             |
|                                     |                                                     | `alpha_jt_synergy_alpha`      | 騎練組合勝率 - 騎師個人勝率 (衡量練馬師帶給騎師的加成)。 |
|                                     |                                                     | `alpha_is_jockey_switched`    | 是否更換騎師。                                           |
|                                     |                                                     | `alpha_jockey_upgrade_alpha`  | 新騎師勝率與舊騎師勝率的差值 (判斷是否為「加強配備」)。  |
| **`JTRecentFormGenerator`**         | 團隊近況狀態 (Recent Form)。                        | `jockey_recent_win_rate_5`    | 騎師近 5 場勝率。                                        |
|                                     |                                                     | `jt_combo_recent_top3_rate_5` | 騎練組合近 5 場上名率。                                  |
| **`SynergyFitnessGenerator`**       | 針對人馬配合度做向量化優化特徵。                    | `is_jockey_changed`           | 上一場與本場是否換人。                                   |
|                                     |                                                     | `pair_ride_count`             | 該騎師策騎該馬匹的歷史累積次數。                         |
|                                     |                                                     | `pair_win_rate`               | 該人馬組合的歷史平滑勝率。                               |

### ⏱️ 速度、步速與賽事細節 (Speed & Pace)


| Generator                     | 用途解釋                       | 產出 Feature 欄位                                     | 定義                                                         |
| ----------------------------- | ------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| **`SectionalSpeedGenerator`** | 分段時間與追趕能力轉化。       | `speed_mps_overall`                                   | 全場平均速度 (公尺/秒)。                                     |
|                               |                                | `sectional_time_last`                                 | 提取最後一段有效的分段時間。                                 |
|                               |                                | `speed_mps_last_sectional`                            | 末腳 400m 的衝刺速度 (公尺/秒)。                             |
|                               |                                | `position_gain_first_to_last`                         | 起步走位與最終走位的名次變化 (追趕能力)。                    |
|                               |                                | `horse_rolling_last_sec_speed_mean_3`                 | 過去 3 場的平均末腳速度。                                    |
| **`SectionalBurstGenerator`** | 末腳爆發力指標。               | `burst_ratio_last_sec`                                | 末段速度 / 全場速度 (>1 代表具備強烈後勁)。                  |
|                               |                                | `horse_rolling_burst_ratio_3`                         | 過去 3 場的平均末腳爆發比率。                                |
| **`SpeedFeatureGenerator`**   | 同場標準化速度指數 (Z-Score)。 | `finish_time_race_z`                                  | 完賽時間在同場比賽中的 Z-Score (越小代表越快)。              |
|                               |                                | `last_400m_speed_z`                                   | 末腳速度在同場中的 Z-Score。                                 |
|                               |                                | `early_pace_expenditure_z`                            | 早段消耗體力(首段時間)的同場 Z-Score。                       |
|                               |                                | `horse_rolling_speed_z_mean_5`                        | 馬匹過去 5 場的完賽速度 Z-Score 滾動平均。                   |
| **`PaceStrategyGenerator`**   | 跑法推演與預期賽事步速壓力。   | `horse_avg_sec1_pos_3`                                | 過去 3 場首段走位的平均名次。                                |
|                               |                                | `is_front_runner`                                     | 預測是否為前置型馬 (首段名次平均 <= 3.5)。                   |
|                               |                                | `race_front_runner_count`                             | 該場比賽中共有多少匹前置型對手 (衡量搶放壓力)。              |
|                               |                                | `is_front_runner_race_front_runner_count_interaction` | `is_front_runner` 與同場對手數量的乘積，表示遭受的步速壓力。 |

### 📊 市場與相對指標 (Context & Market)


| Generator                      | 用途解釋                                              | 產出 Feature 欄位                   | 定義                                                |
| ------------------------------ | ----------------------------------------------------- | ----------------------------------- | --------------------------------------------------- |
| **`ContextRelativeGenerator`** | 將絕對數值轉為同場中的相對優勢。                      | `weight_diff_from_race_avg`         | 實際負重與同場平均負重的差值。                      |
|                                |                                                       | `odds_rank_in_race`                 | 獨贏賠率在同場的排名 (1=大熱門)。                   |
|                                |                                                       | `implied_prob_share`                | 本馬隱含勝率佔同場總隱含勝率的比重。                |
|                                |                                                       | `draw_zscore_in_race`               | 排位檔位的同場 Z-Score。                            |
| **`OddsMarketGenerator`**      | 賠率與大眾市場預期。                                  | `odds_implied_prob`                 | 獨贏賠率換算的市場隱含勝率 (1 / 賠率)。             |
|                                |                                                       | `is_market_favorite`                | 是否為市場焦點大熱門 (賠率 <= 3.0)。                |
|                                |                                                       | `odds_race_zscore`                  | 賠率在同場賽事的 Z-Score。                          |
| **`InteractionGenerator`**     | 高級特徵交叉與權重組合 (強制排在 Pipeline 最後執行)。 | `weight_to_horse_body_ratio`        | 負磅佔馬匹體重的比例。                              |
|                                |                                                       | `horse_jockey_win_rate_interaction` | 馬匹勝率與騎師勝率的乘積。                          |
|                                |                                                       | `win_odds_inv`                      | 隱含勝率 (與前述重複之保護特徵)。                   |
|                                |                                                       | `odds_vs_history_win_rate_gap`      | 市場隱含勝率與歷史實際勝率的落差 (尋找 Value Bet)。 |
|                                |                                                       | `draw_speed_interaction`            | 檔位與速度的交叉乘積。                              |
|                                |                                                       | `rating_x_rank_weight`              | 評分優勢與負重排名交叉。                            |
|                                |                                                       | `delta_x_rank`                      | 負重變化與排名的交叉。                              |

---

## 🛠️ 4. 工具與安全防護庫 (Utilities & Leakage Guard)

在 `features/utils/` 下封裝了支撐整個 Pipeline 的計算核心：

* **`LeakageGuard` (`leak_guard.py`)**：特徵工程的「防火牆」。強制攔截包含 NaN Primary Key 的輸出、自動偵測特徵與 Target 相關係數 (Threshold: 0.90) 以警示是否發生穿越未來的 Data Leakage。
* **`BayesianSmoother` (`smoother.py`)**：計算滾動平均與平滑勝率的核心庫。內部強制採用 `groupby.shift(1)` 徹底隔離當場比賽數據，並採用擴展窗口與貝氏平滑，處理新馬與小樣本問題。
* **`RaceScaler` (`scale.py`)**：專責計算 Race-Level 的上下文特徵，包含同場 Z-Score、同場差額 (Diff from Mean) 與同場相對排名。
* **`SpeedTimeCalculator` (`time_calc.py`)**：將各場地的路程與完成時間，標準化折算為 M/S (公尺/秒) 或是統一基準路程時間。
* **`TrackEncoder` (`track_bias.py`)**：處理香港賽事獨特的 C+3、AWT 等場地偏移編碼。
