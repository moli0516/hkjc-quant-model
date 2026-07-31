# 🏇 HKJC Racing Quant: End-to-End ETL & Machine Learning Pipeline

本專案是一個針對香港賽馬（HKJC）打造的端到端量化數據工程與機器學習預測系統。系統涵蓋了從網路爬蟲獲取原始數據、數據清洗正規化、特徵工程，一直到模型訓練與賽果推論的完整管線。

---

## 🚀 系統安裝指南

為了順利運行本系統的各項數據管線與機器學習模型，請遵循以下步驟建立環境。

### 環境準備與依賴安裝

1. **複製專案庫**：請先將本專案複製到本地端環境。
2. **建立虛擬環境**：建議使用 Python 3.10+ 建立獨立的虛擬環境（例如使用 `python -m venv .venv`），以避免套件衝突。
3. **安裝依賴套件**：專案根目錄下附有 `requirements.txt`，請透過以下指令安裝所有必需的套件：
```bash
pip install -r requirements.txt

```


4. **目錄結構初始化**：系統依賴 `config/settings.json` 定義的路徑運作，執行管線時會自動在指定路徑建立 `data/` 與 `hkjc_racing.db` 等基礎設施。



---

## 🔄 輸入、處理與輸出的生命週期 (Data Lifecycle)

本系統的數據生命週期經過嚴格的解耦與結構化設計，分為以下三個主要階段：

### 1. 輸入階段 (Input - Data Collection)

* 系統透過 `scraper` 模組向下游發起請求，收集賽果、分段時間與馬匹詳細資料。


* 收集到的原始資料會以 JSON 格式儲存於 `config/settings.json` 定義的目錄中（例如 `data/raw_json/races` 與 `data/raw_json/horses`）。



### 2. 處理階段 (Processing - ETL & Feature Engineering)

* **數據清洗 (Data Cleaning)**：`CleaningPipeline` 負責將零碎的 JSON 檔案載入，並透過 `races_cleaner`、`horses_cleaner` 等模組進行正規化、型別轉換與異常值處理。清洗後的資料會透過 SQLAlchemy 寫入 SQLite 資料庫（`hkjc_racing.db`）中的關聯式資料表（如 `races`, `race_results`, `horses`）。


* **特徵工程 (Feature Engineering)**：`FeaturesPipeline` 會從資料庫載入合併後的完整歷史賽事數據。系統嚴格依照時間與賽事順序進行排序，防止未來數據洩漏（Data Leakage）。隨後動態掛載各類生成器（Generators），計算滾動統計、歷史勝率、相對賠率、同場 Z-score 等特徵。最終計算結果會被覆蓋寫入資料庫的 `feature_matrix` 表格中。



### 3. 輸出階段 (Output - Modeling & Inference)

* **模型訓練與尋優**：`ModelPipeline` 會提取 `feature_matrix` 與標籤，訓練機器學習模型（預設為 `xgb_ranker`）。系統亦整合了 Optuna 進行自動超參數尋優，優化目標為 `top1_win_rate` 等排名指標。


* **推論預測 (Inference)**：使用者指定預測日期後，系統會載入當日特徵數據交由訓練好的模型進行推論，最終輸出包含馬匹 ID、預測排名（`pred_rank`）與預測分數（`pred_score`）的決策報表。



---

## 🛠️ 統一入口程式：`cli.py` 使用指南

`cli.py` 是整個專案的統一操作入口，封裝了所有 ETL、特徵工程與模型訓練的底層呼叫。系統提供兩種主要的操作模式：直接執行的互動式 UI，以及透過引數（Arguments）驅動的命令列模式。

### 方法 1：互動式 UI (Interactive Menu)

當您直接執行 `cli.py` 且不帶任何命令列參數時，系統會啟動互動式選單介面。

**使用用法：**

```bash
python cli.py

```

**樣本輸出與互動流程：**

```text
==================================================
🏇  HKJC 賽馬數據工程與機器學習模型 CLI 工具
==================================================
1. 執行賽果和分段時間爬蟲
2. 進行賽果和分段時間數據清洗
3. 執行馬匹資料爬蟲 (需要先有賽果資料庫)
4. 進行馬匹資料數據清洗
5. ⚙️  生成量化特徵矩陣 (Features Pipeline)
6. 🤖 訓練賽馬預測模型 (Model Pipeline)
T. 🎯 Optuna 自動尋優超參數 (Model Tuning)
7. 🔮 執行賽事勝率預測 (Inference)
8. ⚡ 執行一鍵全套 ETL + 特徵工程 + 模型訓練 (1 ➔ 6)
R. 🔄 熱重載所有模組與腳本 (Reload Modules)
0. 退出系統
==================================================
請選擇要執行的功能 (0-8 / T / R，或按 Ctrl+C 退出): 8

🔄 開始一鍵執行全套 Pipeline (從爬蟲到模型訓練)...
🚀 [Step 1] 開始執行：賽果與分段時間爬蟲...
...

```

* **熱重載功能**：在互動模式中輸入 `R`，可以在不關閉程式的情況下，動態重新載入所有專案模組與 Pipeline，適合開發階段除錯使用。



---

### 方法 2：命令列參數模式 (CLI Arguments)

透過傳入不同的 Argument，可以將任務整合至排程系統（如 Cron 或 Airflow）中自動執行。

#### 支援的各類參數 (Arguments) 與用途：

| 參數名稱 | 類型 | 預設值 | 用途說明 |
| --- | --- | --- | --- |
| `--scrape-races` | Flag | 無 | 啟動爬蟲模組，抓取賽事基本資料與分段時間。

 |
| `--clean-races` | Flag | 無 | 執行清洗模組，處理賽事與分段數據並寫入資料庫。

 |
| `--scrape-horses` | Flag | 無 | 針對資料庫中缺乏資料的馬匹，發起馬匹資料爬蟲任務。

 |
| `--clean-horses` | Flag | 無 | 清洗馬匹原始 JSON 資料並更新至資料庫的 `horses` 表。

 |
| `--generate-features` | Flag | 無 | 讀取資料庫進行全量特徵工程計算，並產生 `feature_matrix`。

 |
| `--train-model` | Flag | 無 | 基於當前資料庫特徵，啟動模型訓練管線。

 |
| `--tune-model` | Flag | 無 | 執行 Optuna 尋優任務，自動搜尋最佳超參數。

 |
| `--predict` | Flag | 無 | 執行未來或指定日期的賽事推論預測。

 |
| `--all` | Flag | 無 | 一鍵依序執行全套流程：爬蟲賽果 ➔ 清洗賽果 ➔ 爬蟲馬匹 ➔ 清洗馬匹 ➔ 特徵工程 ➔ 模型訓練。

 |
| `--start-date` | Option | `None` | 指定賽事爬蟲或預測推論的起始日期，格式為 `YYYY-MM-DD`。

 |
| `--end-date` | Option | `None` | 指定賽事爬蟲的結束日期，格式為 `YYYY-MM-DD`。

 |
| `--model-type` | Option | `xgb_ranker` | 選擇要訓練或尋優的模型類型，如 `xgb_ranker`。

 |
| `--n-trials` | Option | `30` | 指定 Optuna 超參數尋優的搜尋試驗次數。

 |

#### 使用範例與樣本輸出

**範例 A：執行特定日期的賽事預測推論**

```bash
python cli.py --predict --start-date 2026-07-16

```

**預期輸出：**

```text
🔮 [Step 7] 開始執行：賽事勝率預測推論 (Model Inference)...
📥 正在載入推論資料 (日期條件: 2026-07-16)...

📊 賽事預測排名結果 (Top 10 範例)：
race_id             horse_id  pred_score  pred_rank  placing
2026-07-16_ST_1     A123      1.845       1          1
2026-07-16_ST_1     B456      1.210       2          3
2026-07-16_ST_1     C789      0.542       3          2
...
✅ 推論計算完成！

```

**範例 B：自動化更新全套資料庫與訓練**

```bash
python cli.py --all --model-type xgb_ranker

```

**預期輸出：**

```text
🚀 [Step 1] 開始執行：賽果與分段時間爬蟲...
✅ 賽果與分段時間爬蟲完成！

🧹 [Step 2] 開始執行：賽果與分段時間數據清洗...
✅ 賽果與分段時間數據清洗完成，已寫入資料庫！

🐎 [Step 3] 檢查資料庫狀態以進行馬匹資料爬蟲...
📊 找到 15 匹需要更新的馬匹資料。
✅ 馬匹資料爬蟲完成！

🧹 [Step 4] 開始執行：馬匹資料數據清洗...
✅ 馬匹資料數據清洗完成，已更新至資料庫！

⚙️ [Step 5] 開始執行：全量量化特徵工程 Pipeline...
📥 正在從資料庫載入全量賽事歷史數據...
✅ 全量特徵工程計算完成！已成功寫入 25430 筆數據至資料庫。

🤖 [Step 6] 開始執行：量化模型訓練與評估 (Model Pipeline)...
🎯 使用模型架構: XGB_RANKER
✅ 模型訓練與驗證完成！
📊 評估指標詳細結果:
   ├─ top1_win_rate: 0.2845
   ├─ ndcg_score: 0.8231

```