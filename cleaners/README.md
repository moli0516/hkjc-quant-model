# 🧹 Cleaners Module (`cleaners/`)

歡迎來到 HKJC 賽馬量化系統的 **數據清洗與轉型層（Data Cleaning & Transformation Layer）**。

本模組為純 Python 業務邏輯層，主要職責是將爬蟲層抓取的原始數據（Raw JSON / Staging DB Tables）進行結構展平、異常值與缺失值處置、數據型態強制轉型（Type Casting）及結構正規化（Normalization），為下游的特徵工程（Feature Pipeline）與量化模型訓練提供高品質且一致的 Clean Data。

> ⚠️ **模組定位聲明**：
>
> `cleaners/` 是一個**純數據清洗與轉換模組**，內部不包含任何可執行的主程式碼、獨立腳本或 CLI 工具。本模組不提供直接終端機執行的接口，請透過 Python API 匯入並整合至資料管道或服務層中使用。

## 📁 目錄結構

```
cleaners/
├── __init__.py              # 模組對外 API 導出與包初始化
├── cleaner_pipeline.py      # CleaningPipeline (清洗管道總控與調度器)
├── horses_cleaner.py        # HorseCleaner (馬匹歷史、評分與血統數據清洗)
├── races_cleaner.py         # RaceCleaner (賽事基礎資訊、賽果與排位數據清洗)
└── sectional_cleaner.py     # SectionalCleaner (分段時間、走勢與衝刺速度清洗)
```

## 🔄 數據處理流向與架構

數據從原始爬蟲落地層流轉至標準化 Clean Data 的整體架構如下：

**程式碼片段**

```
flowchart TD
    A[Raw Data / Staging Area\n(JSON / Raw DB Tables)] --> B[CleaningPipeline]
  
    subgraph cleaners/ 模組內部邏輯
        B --> C[RaceCleaner]
        B --> D[SectionalCleaner]
        B --> E[HorseCleaner]
      
        C -->|賽事、檔位、賠率| F[Data Quality Check & Type Casting]
        D -->|分段時間、走勢| F
        E -->|馬匹評分、血統、傷患| F
    end
  
    F --> G[Cleaned Data Store\n(Production DB / Parquet Storage)]
    G --> H[Downstream: Feature Engineering & Model Training]
```

## ⚙️ 核心模組職責說明


| **檔案名稱**               | **核心類別**       | **職責與數據處理邏輯**                                                                                                                                        |
| -------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`cleaner_pipeline.py`**  | `CleaningPipeline` | **頂層調度器**：統一串聯與調度各個具體的 Cleaner。負責管控全量/增量清洗任務的依賴關係與執行順序，提供單一入口對外暴露 API。                                   |
| **`races_cleaner.py`**     | `RaceCleaner`      | **賽果與排位清洗**：處理賽事日期、場次、跑道條件（草地/泥地、C+3 賽道等）、賽程距離、名次、獨贏賠率（WV Odds）、檔位（Draw）、實際負磅與馬伕/練馬師字串清洗。 |
| **`sectional_cleaner.py`** | `SectionalCleaner` | **分段數據清洗**：解析並展平分段時間（Sectional Times）、各分段位置走勢（如`4-4-2-1`）、末段衝刺速度轉換與分段秒數之異常值修正（如棄權/退出馬匹處理）。       |
| **`horses_cleaner.py`**    | `HorseCleaner`     | **馬匹個體數據清洗**：處理馬匹烙號、歷史出賽紀錄、評分變動（Rating History）、進口類別、父系/母系血統數據以及傷患/退役紀錄之正規化。                          |

## 🛠️ 數據品質與清洗規則 (Data Quality Rules)

為了確保量化模型的數據穩定度，`cleaners/` 模組嚴格執行以下清洗規範：

### 1. 欄位型態強制轉型 (Type Casting)

* **數值型態**：所有賠率、名次、負磅、體重等欄位統一轉型為 `float64` 或 `Int64`（可空整數）。
* **時間型態**：賽事日期與分段時間強制解析為 `datetime64[ns]` 或標準 `HH:MM:SS.mm` 格式。

### 2. 缺失值與極端值處置 (Missing Values & Outliers)

* **退出/棄權馬匹**：名次標記為 `WV` (Withdrawn) 或 `FE` (Fell) 者，將賽果時間補值為 `NaN`，並生成對應的標記 flag（如 `is_withdrawn=True`），避免污染模型訓練集。
* **分段走勢解析**：將原始字串（如 `"3 3 1"`）拆解為各分段獨立的整數欄位（`pos_sec1`, `pos_sec2`, ...），無法解析者填入默認空值。

### 3. 字串正規化 (Text Normalization)

* **標準化欄位命名**：統一採用 `snake_case`（如 `horse_code`, `race_date`, `win_odds`）。
* **去除噪訊**：清洗馬名與騎練姓名中的多餘空格、特殊符號及全半形字元轉換。

## 💻 程式碼集成與 API 使用範例

本模組設計為高內聚、低耦合的 Python 類別庫，可以在 Feature Engineering Pipeline 或其他數據服務中直接 import 使用。

### 範例一：使用 `CleaningPipeline` 執行完整清洗流程

**Python**

```
from cleaners.cleaner_pipeline import CleaningPipeline

# 1. 初始化清洗管道（可帶入資料庫連線設定或配置參數）
pipeline = CleaningPipeline(config={
    "strict_mode": True,
    "drop_invalid_races": True
})

# 2. 傳入 Raw Data (Dataframe 或 Dict 結構)
raw_race_payload = {
    "race_date": "2026-03-22",
    "venue": "ST",
    "races_data": [...] # Raw json records
}

# 3. 執行全量清洗
cleaned_dataset = pipeline.run_pipeline(raw_data=raw_race_payload)

# 4. 取得清洗後的各數據集
cleaned_races_df = cleaned_dataset["races"]
cleaned_sectionals_df = cleaned_dataset["sectionals"]

print(f"Successfully processed {len(cleaned_races_df)} race records.")
```

### 範例二：單獨使用 `SectionalCleaner` 處理分段時間

**Python**

```
import pandas as pd
from cleaners.sectional_cleaner import SectionalCleaner

# 假設從 Staging DB 讀取原始分段資料
raw_sectional_df = pd.DataFrame([
    {"race_id": "20260322_ST_01", "horse_code": "E123", "sec_time_str": "23.41 - 22.80 - 23.15", "position_str": "4 3 1"},
    {"race_id": "20260322_ST_01", "horse_code": "G456", "sec_time_str": "23.80 - 22.95 - 24.10", "position_str": "2 2 4"}
])

# 實例化單一 Cleaner 並執行轉換
sectional_cleaner = SectionalCleaner()
cleaned_sectional_df = sectional_cleaner.clean(raw_sectional_df)

# 查看處理結果
print(cleaned_sectional_df[["horse_code", "sec1_time", "sec2_time", "sec3_time", "last_400m_time"]])
```

### 範例三：單獨使用 `HorseCleaner` 清洗馬匹歷史評分

**Python**

```
from cleaners.horses_cleaner import HorseCleaner

horse_cleaner = HorseCleaner()

# 清洗馬匹歷史評分與體重數據
raw_horse_profile = {
    "horse_code": "J102",
    "raw_ratings": [{"date": "2025-09-01", "rating": "60"}, {"date": "2026-01-10", "rating": "68"}]
}

cleaned_horse_data = horse_cleaner.process_rating_history(raw_horse_profile)
```

## 🧪 單元測試與開發維護

新增或修改清洗邏輯時，請確保遵守以下原則：

1. **冪等性 (Idempotency)**：確保相同的 Raw Data 輸入經過 Cleaner 處理後，永遠輸出相同的 Clean Data。
2. **無副作用 (Pure Functions)**：Cleaner 類別內部方法應盡可能保持純粹，避免在清洗過程中直接修改傳入的原物件（使用 `.copy()` 進行操作）。
3. **類型提示 (Type Hints)**：新增的函數與方法必須包含 Type Hints（如 `pd.DataFrame -> pd.DataFrame`）。
