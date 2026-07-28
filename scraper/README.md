# HKJC 賽馬量化模型系統 (HKJC Quant Model)

# 🛠️ HKJC Data Scraper Module (`scraper`)

`scraper` 模組是 `hkjc-quant-model` 量化數據系統的核心**資料擷取層 (Data Ingestion Layer)**。本模組負責從香港賽馬會 (HKJC) 官方網站高效率、穩定且結構化地擷取賽事日曆、詳細賽果、分段時間與跑法、馬匹歷史評分及詳細背景數據，為下游的特徵工程與量化模型提供乾淨且高品質的數據源。

## 📌 模組簡介 (Overview)

在賽馬量化分析中，資料的完整性與即時性直接決定模型的預測品質。`scraper` 模組採用非同步 (AsyncIO) 與模組化架構設計，兼具高吞吐量與低偶合度。

### 核心特色

* **分離式雙管線設計 (Dual Pipeline Architecture)**：將「賽事賽果/分段時間 (Race Pipeline)」與「馬匹個體資料 (Horse Pipeline)」拆分為獨立運作的 Pipeline，支援增量更新與彈性排程。
* **高併發異步爬取 (AsyncIO / aiohttp)**：基於 Python 非同步 I/O 引擎，大幅降低網路 I/O 阻塞時間，實現高效率的全量與增量數據同步。
* **Pipeline 與 Parser 完全解耦**：網路請求、異步調度與數據解析 (HTML/JSON) 職責明確分離，提高代碼可維護性與單元測試便利性。
* **統一數據持久化與快取 (DataManager)**：封裝數據儲存與快取機制，支援高效能 Parquet 列式儲存與資料庫寫入，減少重複發起網路請求。
* **事件驅動鉤子 (Webhook Hooks)**：提供生命週期事件 Hook，方便在資料擷取完成後自動觸發 downstream 的資料清洗與特徵建構。

## 🏗️ 架構與設計模式 (Architecture & Design)

本模組嚴格遵循 **Pipeline & Parser 雙層架構** 設計：

```
                    ┌─────────────────────────┐
                    │   HKJC Official Web     │
                    └────────────┬────────────┘
                                 │ HTTP (aiohttp)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline Layer                          │
│         ( race_pipeline.py  /  horse_pipeline.py )               │
│  - Concurrency Control (Semaphore)                              │
│  - Exponential Backoff & Retry                                  │
│  - Request Interceptors & Jitter Delay                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ Raw HTML / Response Data
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Parser Layer                           │
│                      ( scraper/parser/* )                       │
│  - Pure Functions (No I/O Side Effects)                         │
│  - BeautifulSoup4 / RegEx / JSON Extraction                     │
│  - Data Normalization & Validation                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ Clean Structured Dict / DataFrame
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Data Manager                           │
│                       ( data_manager.py )                       │
│  - Parquet / SQL Persistence                                    │
│  - Local Cache & Duplication Check                              │
└─────────────────────────────────────────────────────────────────┘
```

### 🧠 雙層架構設計優勢

1. **職責分離 (Separation of Concerns)**：Pipeline 只關注網路請求、併發速率與重試機制；Parser 專注於將 HTML/JSON 轉化為結構化數據。當 HKJC 網頁結構異動時，僅需維護對應 Parser，不影響網路調度邏輯。
2. **零副作用解析測試 (Pure Function Parsing)**：Parser 層不包含任何 I/O 操作，可輕鬆針對本地儲存的 HTML 快取檔案進行自動化單元測試。

### 🧩 `parser/` 解析器職責分工


| **解析器檔案**        | **核心功能說明**                                           | **對應 HKJC 資料源**           |
| --------------------- | ---------------------------------------------------------- | ------------------------------ |
| `calander_parser.py`  | 解析賽事日曆，擷取賽期、場次與開跑場地 (沙田 / 跑馬地)     | 賽事日曆頁面                   |
| `result_parser.py`    | 解析每場賽事基礎數據（名次、騎練、負磅、檔位、獨贏賠率等） | 賽果網頁 (`LocalResults`)      |
| `sectional_parser.py` | 解析每匹馬的分段時間 (Sectional Time) 與沿途走位跑法       | 分段時間網頁 (`SectionalTime`) |
| `rating_parser.py`    | 解析馬匹歷史評分變動記錄、班次調整與評分加減分             | 馬匹評分紀錄頁面               |
| `horse_parser.py`     | 解析馬匹詳細資料（烙號、父系/母系、傷患紀錄、進口類別等）  | 馬匹基本資料頁面               |

## 🚀 快速開始 (Quick Start & Usage)

`scraper` 模組設計為可被其他 Python 模組直接導入與調用的純套件。

### 1. 指定日期範圍異步爬取賽果與分段時間

**Python**

```
import asyncio
from scraper.race_pipeline import RacePipeline
from scraper.data_manager import DataManager

async def main():
    # 初始化數據管理器
    data_mgr = DataManager(store_path="./data")
  
    # 建立賽事爬蟲管線，限制最大併發數為 3
    race_pipe = RacePipeline(data_manager=data_mgr, concurrency=3)
  
    # 指定爬取日期清單
    target_dates = ["2024-12-15", "2024-12-18"]
  
    print(f"Starting race pipeline for dates: {target_dates}")
    race_data = await race_pipe.fetch_dates(target_dates)
  
    print(f"Successfully processed {len(race_data)} race records.")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 帶入馬匹 ID 清單異步爬取馬匹詳細資料

**Python**

```
import asyncio
from scraper.horse_pipeline import HorsePipeline
from scraper.data_manager import DataManager

async def main():
    data_mgr = DataManager(store_path="./data")
    horse_pipe = HorsePipeline(data_manager=data_mgr, concurrency=2)
  
    # 馬匹識別碼清單
    horse_ids = ["HK_2022_H123", "HK_2023_J456", "HK_2021_G088"]
  
    print("Starting horse pipeline...")
    horses = await horse_pipe.fetch_horses(horse_ids)
  
    for horse in horses:
        print(f"Horse: {horse.get('name')} | Code: {horse.get('code')} | Total Starts: {horse.get('starts')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 結合 Hook 機制觸發完成事件

**Python**

```
import asyncio
from scraper.race_pipeline import RacePipeline
from scraper.hook import PipelineHook

# 定義自訂 Hook
class QuantDataHook(PipelineHook):
    def on_success(self, date: str, result_count: int):
        print(f"[Hook Notification] Date {date} scraped successfully. {result_count} races saved.")

async def run_with_hook():
    hook = QuantDataHook()
    pipeline = RacePipeline(concurrency=3, hooks=[hook])
    await pipeline.fetch_dates(["2024-12-18"])

if __name__ == "__main__":
    asyncio.run(run_with_hook())
```

## 📂 檔案職責說明 (File Structure & Responsibilities)


| **檔案路徑**        | **模組層級**       | **功能與職責描述**                                                                      |
| ------------------- | ------------------ | --------------------------------------------------------------------------------------- |
| `parser/`           | **Parser Layer**   | HTML/JSON 解析器目錄，包含純函數解析邏輯，無任何網路 I/O Side Effects。                 |
| `data_manager.py`   | **Data Layer**     | 負責數據持久化（Parquet/SQLite/PostgreSQL），管理本地請求快取與增量比對，防範重複爬取。 |
| `hook.py`           | **Event Layer**    | 提供生命週期 Hook 介面（如`on_success`,`on_error`），供外部訂閱爬蟲執行狀態。           |
| `horse_pipeline.py` | **Pipeline Layer** | 馬匹資料異步爬取管線，負責管理馬匹基本資料、歷史評分與傷患紀錄之網路請求與控速。        |
| `race_pipeline.py`  | **Pipeline Layer** | 賽事數據異步爬取管線，負責整合賽事日曆、基礎賽果與分段時間之網路請求與數據對齊。        |
| `__init__.py`       | **Package Entry**  | 套件初始化檔，對外暴露主要的 Pipeline 與 DataManager 類別。                             |

## ⚠️ 爬蟲反制與注意事項 (Caveats & Best Practices)

### 1. 請求頻率與併發控制 (Rate Limiting & Concurrency)

HKJC 官方伺服器配有嚴格的流量監控機制，高頻率請求容易導至 HTTP 403 / 429 錯誤或 TCP 連線強制斷開。

* **併發上限設定**：建議 `RacePipeline` 併發數 (Concurrency) 設定為 **3 \~ 5**，`HorsePipeline` 建議設定為 **2 \~ 3**。
* **隨機延遲 (Jitter Delay)**：Pipeline 內部發起 HTTP 請求前，請確保搭配 0.5s - 1.5s 的隨機延遲，模擬真實使用者行為。

**Python**

```
# 推薦的安全 Pipeline 初始化參數配置
pipeline = RacePipeline(
    concurrency=3,         # 限制同時運行的 Task 數量
    timeout=15.0,          # 請求 Timeout 時間 (秒)
    min_delay=0.5,         # 最小隨機延遲
    max_delay=1.5          # 最大隨機延遲
)
```

### 2. 管線數據相依性 (Pipeline Dependencies)

* **馬匹爬蟲相依於賽果數據**：`horse_pipeline.py` 需依賴 `race_pipeline.py` 的擷取結果。正確的工作流程為：
  1. 先執行 `race_pipeline` 取得最新賽事的賽果與參賽馬匹名單。
  2. 由 `data_manager` 自動提取目前資料庫中尚未記錄或需更新的 `horse_id`。
  3. 將該 `horse_id` 清單輸入至 `horse_pipeline` 執行增量更新。

### 3. 連線重試與 Exponential Backoff

* 當遇上網路波動或暫時性阻擋時， Pipeline 內部應實作 **Exponential Backoff（指數退避重試）** 策略（預設重試 3 次）。
* 若重試失敗，失敗的 URL 與參數應記錄至 `DataManager` 的 Fail Queue 中，以便進行離線修復或重試。
