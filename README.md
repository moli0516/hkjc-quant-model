# HKJC 賽馬數據工程 - Workspace 架構與管道運作文檔

本文件為 **HKJC 賽馬量化數據工程專案** 的核心技術文檔，詳細說明系統中的數據管道 (Data Pipeline) 架構、模組職責、數據依賴性以及 CLI 排程控制邏輯。

## 📂 專案架構與模組職責

專案採用高內聚、低耦合的管道式設計，將數據的**抓取 (Scraping)**、**清洗 (Cleaning)** 與 **持久化 (Database)** 嚴格分離：

**Plaintext**

```
├── database/                   # [db/] 資料庫管理與 ORM 模組
│   ├── db_manager.py           # DBManager: 負責 SQLite/DB 讀寫、狀態檢查與 Pending 狀態管理
│   └── models.py               # ORM 資料庫 Schema 定義
├── scraper/                    # [scraper/] 數據爬蟲模組
│   ├── race_pipeline.py        # RaceScrapingPipeline: 賽果、排位與分段時間爬蟲總控
│   ├── horse_pipeline.py       # HorseScrapingPipeline: 馬匹歷史詳細數據爬蟲總控
│   ├── data_manager.py         # 爬蟲任務調度與網絡請求管理
│   └── parser/                 # HTML / API 數據解析器
├── cleaners/                   # [cleaner/] 數據清洗模組
│   ├── cleaner_pipeline.py     # CleaningPipeline: 清洗管道進入點 (EntryPoint)
│   ├── races_cleaner.py        # 賽果數據清洗與結構化
│   ├── sectional_cleaner.py    # 分段時間數據清洗與補值
│   └── horses_cleaner.py       # 馬匹歷史記錄與評分清洗
└── cli.py                      # 全局命令列入口與自動化腳本 (HKJCCLI)
```

## 🔄 數據運作週期與依賴關係 (Workflow Lifecycle)

整個系統的數據流遵循 **嚴格的四階段依賴關係**。由於馬匹歷史資料（Step 3）需要以賽果資料庫中已存在的馬匹編號 (`horse_id`) 為基準進行增量抓取，因此必須嚴格按照以下順序執行：

**程式碼片段**

```
graph TD
    A[Step 1: 賽果與分段時間爬蟲<br/>RaceScrapingPipeline] --> B[Step 2: 賽果數據清洗與入庫<br/>CleaningPipeline action='race_sectional']
    B --> C{前置檢查<br/>DBManager.has_race_results}
    C -- 成功 (存在 race_results) --> D[Step 3: 馬匹歷史資料爬蟲<br/>HorseScrapingPipeline]
    C -- 失敗 (資料庫無賽果) --> E[🛑 阻斷執行，提示需先完成 Step 1 & 2]
    D --> F[Step 4: 馬匹數據清洗與更新<br/>CleaningPipeline action='horse']
```

### 階段詳細說明

1. **Step 1: 賽果與分段時間爬蟲 (`run_race_scraper`)**
   * **核心模組**：`scraper.race_pipeline.RaceScrapingPipeline`
   * **職責**：根據指定日期區間 (`start_date` 至 `end_date`)，爬取 HKJC 官方賽果頁面、分段時間 (Sectional Time) 與排位數據，並將 Raw HTML/JSON 暫存。
2. **Step 2: 賽果與分段時間數據清洗 (`run_race_cleaner`)**
   * **核心模組**：`cleaners.cleaner_pipeline.CleaningPipeline(action="race_sectional")`
   * **職責**：將原始賽果與分段時間數據進行結構化清洗、類型轉換（如將字串時間轉為浮點數秒數），並持久化寫入資料庫（建立或更新 `races` 及 `sectional_times` 表格）。
3. **Step 3: 馬匹歷史數據爬蟲 (`run_horse_scraper`)**
   * **核心模組**：`scraper.horse_pipeline.HorseScrapingPipeline`
   * **安全機制**：執行前強制觸發 `DBManager.has_race_results()`。若未獲得賽果數據，系統會主動拋出 Warning 並中斷執行。
   * **執行邏輯**：
     * 呼叫 `DBManager.get_pending_horse_ids()`，自動對比已入庫賽果中的馬匹與馬匹主表，過濾出需要新增或補抓歷史紀錄的 `horse_id`。
     * 採用 `asyncio.run()` 啟動非同步高併發爬蟲，抓取馬匹的歷史評分變動、過往傷患紀錄與參賽軌跡。
4. **Step 4: 馬匹數據清洗與更新 (`run_horse_cleaner`)**
   * **核心模組**：`cleaners.cleaner_pipeline.CleaningPipeline(action="horse")`
   * **職責**：清洗馬匹詳細歷史數據，更新 `horses` 表格及歷史績效表。

## 💻 CLI 工具操作指南 (`cli.py`)

`cli.py` 為系統的統一進入點，支援 **互動式選單** 與 **命令行參數 (CLI Arguments)** 兩套操作模式。

### 1. 互動式選單模式 (Interactive Menu)

直接執行腳本，系統會自動引導進入 Console 選單：

**Bash**

```
python cli.py
```

**選單功能選單**：

* `[1]`**執行賽果爬蟲** (Race Scraping Pipeline)
* `[2]`**執行賽果清洗** (Race Cleaning Pipeline)
* `[3]`**執行馬匹爬蟲** (Horse Scraping Pipeline - 內建 DB 前置檢查)
* `[4]`**執行馬匹清洗** (Horse Cleaning Pipeline)
* `[5]`**⚡ 一鍵全套執行** (Run All Pipelines sequentially: 1 ➔ 2 ➔ 3 ➔ 4)
* `[0]`**退出系統**

### 2. 命令行參數模式 (Command Line Mode)

適用於 CronJob、Airflow 或 CI/CD 批次自動化排程：

* **爬取指定日期範圍的賽果**：
  **Bash**

  ```
  python cli.py --scrape-races --start-date 2024-01-01 --end-date 2024-12-31
  ```
* **清洗賽果與分段時間數據**：
  **Bash**

  ```
  python cli.py --clean-races
  ```
* **抓取未完成/待更新的馬匹資料**：
  **Bash**

  ```
  python cli.py --scrape-horses
  ```
* **清洗馬匹歷史資料**：
  **Bash**

  ```
  python cli.py --clean-horses
  ```
* **⚡ 執行完整的 ETL 全套自動化流程**：
  **Bash**

  ```
  python cli.py --all --start-date 2024-01-01 --end-date 2024-12-31
  ```
