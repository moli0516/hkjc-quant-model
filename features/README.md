# 🐎 HKJC Quant: Feature Engineering Pipeline (量化特徵工程管線)

這是一個專為香港賽馬（HKJC）量化預測設計的高效、防洩漏（Leakage-Proof）、且具備極高擴充性的特徵工程管線。本模組負責將原始的賽事與馬匹數據，轉化為可用於機器學習模型的高品質特徵矩陣。

本專案的核心設計理念為：**「絕對的時間隔離（防數據洩漏）、零碎片化的記憶體管理，以及隨插即用的動態擴充架構。」**

---

## 📂 專案目錄結構 (Project Structure)

```text
features/
├── generators/             # 特徵生成器模組 (熱插拔設計)
│   ├── __init__.py         # 動態載入與排程邏輯
│   └── example_generator.py# 測試用假資料生成器範例
├── utils/                  # 核心工具與防護庫
│   ├── __init__.py
│   ├── leak_guard.py       # Data Leakage 防火牆
│   ├── scale.py            # 同場數據標準化 (Z-score, Ranking)
│   ├── smoother.py         # 嚴格防洩漏的貝氏平滑與滾動統計
│   ├── time_calc.py        # 速度與時間標準化計算
│   └── track_bias.py       # 跑道與檔位偏差編碼
├── __init__.py
├── base_target.py          # 基礎骨架與目標變數建立
└── feature_pipeline.py     # 特徵工程主幹道 (Pipeline)
```

---

## 🏗️ 1. 核心工作流程 (Pipeline Workflow)

整個特徵工程的工作流程由 `BaseTargetBuilder` 與 `FeaturesPipeline` 兩個核心類別協同完成。

### Step 1: 基礎骨架與目標變數建立 (`BaseTargetBuilder`)

- **資料清洗與對齊**：接收 Raw Data，統一欄位命名，過濾無效名次（如 WV, DNF, PU）及空值，並支援平馬 (如 '1 DH') 名次正則提取。
- **嚴格時間排序防護**：強制將數據按 `race_date`, `race_id`, `horse_id` 升冪排序，建立防止未來數據洩漏的第一道防線。
- **動態時序遮罩 (Temporal Guard)**：嚴格比對馬匹的 `import_date` 與 `race_date`。若抵港日晚於賽事日，強制將其遮蔽為 `NaT`，防止「未來資訊」穿越。
- **Target 生成**：統一生成三種機器學習預測標籤：
  - `target_win` (獨贏，二元分類)
  - `target_place` (位置，前三名，二元分類)
  - `target_rank_score` (名次倒數分數，用於排序學習)

### Step 2: 進入主幹道 (`FeaturesPipeline`)

- **動態掃描與載入**：實例化 Pipeline 時，自動掃描 `generators/` 目錄下的所有模組，並依照 `EXECUTION_ORDER` 派發工作。
- **記憶體與效能優化 (Zero-Fragmentation)**：
  - 自動將新生成的 `float64` 特徵降維轉型為 `float32`，大幅降低記憶體消耗。
  - 將每次迴圈產出的純特徵暫存於 List，最後使用 `pd.concat` 一次性水平併合（Zero-copy），避免 Pandas DataFrame 不斷 append 造成的記憶體碎片化 (Fragmentation)。
  - 過濾重複欄位，確保資料不被異常覆寫。
- **索引對齊與驗證**：產出前強制恢復傳入時的原始 Index，並通過 `LeakageGuard` 檢驗。

---

## 🔌 2. 熱插拔特徵生成器機制 (Hot-Swappable Generators)

系統無需在 Pipeline 中手動 `import` 或註冊任何特徵腳本。所有 Generator 均採用「熱插拔 (Plugin)」架構 (`generators/__init__.py`)。

1. **動態模組探索**：自動掃描 `generators` 目錄下所有不以 `_` 開頭的 Python 檔案。
2. **類別反射 (Reflection)**：尋找繼承或命名以 `Generator` 結尾的 Class。
3. **執行序控制 (Execution Order)**：依照類別內的 `EXECUTION_ORDER` 常數（預設 500）進行排序，確保具相依性的特徵依序計算。

> **💡 開發者指南**：
> 若要新增特徵，只需在 `generators/` 目錄下建立 `.py` 檔案，編寫包含 `generate(self, df)` 方法且設定好 `EXECUTION_ORDER` 的類別，下次執行 Pipeline 時將會**自動載入並套用**。

---

## 🛠️ 3. 核心工具與安全防護庫 (Utilities & Leakage Guard)

`features/utils/` 目錄封裝了支撐整個 Pipeline 的計算核心，所有工具皆經過嚴格的「防穿越」設計：

### 🛡️ `LeakageGuard` (leak_guard.py)

特徵工程的「防火牆」。

- **`check_future_leakage`**: 自動偵測特徵與 Target 的相關係數 (Threshold: 0.90)，若異常過高會發出警示，防止漏寫 `.shift(1)` 導致穿越未來的 Data Leakage。
- **`assert_no_null_keys`**: 強制攔截包含 NaN Primary Key (`race_id`, `horse_id`) 的異常資料輸出。

### 📊 `BayesianSmoother` (smoother.py)

計算滾動平均與平滑勝率的核心庫。

- **自動時間排序**: 內部強制進行 `sort_values` 確保時間序列正確。
- **嚴格隔離當場數據**: 強制採用 `groupby.shift(1)` 徹底隔離當場比賽數據，避免拿「當下結果」預測「當下」。
- **貝氏平滑**: `calc_global_smooth_rate` 與 `calc_rolling_smooth_rate` 採用貝氏先驗機率平滑，穩定新馬與小樣本問題。

### ⚖️ `RaceScaler` (scale.py)

專責計算同場賽事（Race-Level）的上下文特徵（Contextual Features）：

- **`race_z_score`**: 計算欄位在同場賽事中的 Z-Score (如：同場速度優勢)。
- **`race_diff_from_mean`**: 計算與同場平均值的差額 (如：負磅差異)。
- **`race_rank`**: 計算數值在同場賽事中的相對排名 (如：賠率排名)。

### ⏱️ `SpeedTimeCalculator` (time_calc.py)

處理時間與路程的特徵轉換：

- **`calc_speed_mps`**: 計算平均每秒跑多少米 (m/s)，用於跨路程比較。
- **`normalize_time_by_distance`**: 將不同路程的完賽時間按比例折算至標準路程（預設 1200 米），便於模型對比實力。

### 🏟️ `TrackEncoder` (track_bias.py)

- **`categorize_course_type`**: 處理跑道字串清洗，將其統一歸類為 `TURF` (草地) 或 `AWT` (全天候/泥地)。
- **`create_track_draw_combo`**: 建立「場地+跑道+檔位」組合特徵，捕捉特定賽道的檔位偏差。

---

## 🚀 4. 快速啟動 (Quick Start)

### 透過 DBManager 自動載入並生成特徵

```python
from features.base_target import BaseTargetBuilder
from features.feature_pipeline import FeaturesPipeline

# 1. 自動從資料庫取得合併數據並建立骨架
df_base = BaseTargetBuilder.build_from_sqlite(db_path="path/to/database.sqlite")

# 2. 初始化管線
pipeline = FeaturesPipeline(key_cols=["race_id", "horse_id"])

# 3. 執行特徵工程
df_features = pipeline.run(df_base)

print(df_features.head())
```
