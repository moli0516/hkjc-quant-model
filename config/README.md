# ⚙️ HKJC Quant: Configuration & Settings Management (系統配置管理中心)

本模組為整個 HKJC Quant 量化交易與預測系統的配置中樞。透過集中式的 JSON 設定檔與單例（Singleton）導向的 `Settings` 封裝，實現了「環境解耦、動態熱切換、路徑自動解析，以及實驗特徵管制」的核心目標。

本模組的核心設計理念為：**「中央集權式配置、絕對路徑自動補齊（pathlib 封裝）、零硬編碼（Zero Hardcoding），以及實驗版本的可追溯性。」**

---

## 🏗️ 1. 核心工作流程 (Configuration Workflow)

整個配置管理模組由 `settings.py` 中的 `Settings` 類別統一維護。系統啟動與讀取設定的完整生命週期如下：

```text
┌────────────────────────┐
│  .active_config 指針  │
└───────────┬────────────┘
            │ 1. 讀取 active 指針 (預設 settings.json)
            ▼
┌────────────────────────┐
│  settings.example.json │ (或當前指定的 .json 檔案)
└───────────┬────────────┘
            │ 2. 開啟並解析 JSON 結構化資料
            ▼
┌────────────────────────┐
│  Settings 實例化解析   │
└───────────┬────────────┘
            │ 3. 結合 project root 自動轉化為 pathlib.Path
            ▼
┌────────────────────────┐
│  對外提供強型別 @property │ ──> 給 Data Loader / Features / Model 呼叫
└────────────────────────┘

```

### Step 1: 狀態指針解析 (Active Config Resolution)

* **自動定位專案根目錄：** `Settings` 類別以 `settings.py` 自身的檔案位置 (`pathlib.Path(__file__)`) 為基準，向上退兩層 (`parent.parent`) 自動鎖定專案根目錄 (`root_dir`)，完全擺脫跨作業系統或執行路徑（CWD）不同產生的路徑報錯。
* **讀取持久化指針 (`.active_config`)：** 系統優先檢查 `config/.active_config` 檔案。若該檔案存在，則讀取其中記載的 `.json` 檔名（例如 `settings.json` 或 `settings.example.json`）；若不存在或讀取失敗，則預設回退至 `settings.json`。

### Step 2: 記憶體載入與快取 (In-Memory Caching)

* 將解析出的 JSON 檔案讀入記憶體中的字典結構 (`self._data`)。
* 透過 Singleton 式的實例 `settings = Settings()` 對外提供全域統一的配置存取點。

### Step 3: 動態路徑自動映射 (Path Auto-Resolution)

* 所有的路徑配置在 JSON 中均只需以**相對路徑**儲存。
* 當透過 `Settings` 的 `@property` 存取路徑時（如 `settings.raw_json_dir`），系統會自動將其與 `self.root_dir` 進行拼合，回傳強型別的 `pathlib.Path` 物件，確保下游模組無需再手動拼接字串。

### Step 4: 進程內熱切換與持久化 (`switch_config`)

* 支援在程式執行期間（例如在網格搜尋、A/B Testing 或模型回測時）隨時呼叫 `settings.switch_config("settings.example.json")`。
* 此動作會立即重載記憶體中的 `_data` 字典，並同步將新的設定檔名寫回 `config/.active_config`，確保下次系統重啟時自動維持最後切換的設定狀態。

---

## 🔌 2. 熱切換與動態載入機制 (Dynamic Switch Mechanism)

本模組無需重啟 Python 進程即可動態更新全域參數，並提供安全的錯誤防護：

```python
from config.settings import settings

# 1. 取得當前使用的特徵與超參數
print(settings.active_features)
print(settings.default_params)

# 2. 在實驗過程中動態切換至不同的設定檔（如基準測試設定）
settings.switch_config("settings.example.json")

# 3. 此時所有 @property 均已更新為 settings.example.json 的數值
print(settings.target)  # 輸出: 'is_place'

```

---

## 📋 3. 配置屬性清單與說明 (`settings.example.json` Catalog)

以下為 `settings.example.json` 檔案中所有配置欄位的結構化說明，以及其在 `Settings` 類別中對應的 `@property` 屬性與實務用途：

### 🛠️ 1. 模型超參數與目標設定 (Model Hyperparameters & Targets)


| JSON 鍵名                    | 對應`@property`           | 資料型態 | 用途與說明                                                                 |
| ---------------------------- | ------------------------- | -------- | -------------------------------------------------------------------------- |
| **`target`**                 | `settings.target`         | `str`    | 當前模型訓練的主目標標籤（如`"is_place"` 代表位置，`"is_win"` 代表獨贏）。 |
| **`default_params`**         | `settings.default_params` | `dict`   | XGBoost / LightGBM 等 Ranking 模型之預設超參數字典。                       |
| ↳`objective`                | -                         | `str`    | 學習目標函數，預設採用排名優化`rank:ndcg`。                                |
| ↳`max_depth`                | -                         | `int`    | 樹模型的最大深度（預設`6`），控制模型複雜度。                              |
| ↳`learning_rate`            | -                         | `float`  | 學習率 / 步長（預設`0.05`）。                                              |
| ↳`n_estimators`             | -                         | `int`    | 弱學習器（樹）的最大迭代次數（預設`500`）。                                |
| ↳`early_stopping_rounds`    | -                         | `int`    | 驗證集指標未提升時的提早終止輪數（預設`50`）。                             |
| ↳`subsample`                | -                         | `float`  | 訓練每棵樹時的樣本採樣比例（預設`0.8`）。                                  |
| ↳`colsample_bytree`         | -                         | `float`  | 訓練每棵樹時的特徵採樣比例（預設`0.8`）。                                  |
| ↳`reg_alpha` / `reg_lambda` | -                         | `float`  | L1 (Lasso) 與 L2 (Ridge) 正則化懲罰項係數。                                |
| ↳`tree_method`              | -                         | `str`    | 樹生成演算法，`"hist"` 代表直方圖加速，能大幅提升大數據訓練速度。          |

### 🧮 2. 貝氏平滑與領域知識映射 (Smoothing & Domain Maps)


| JSON 鍵名              | 對應`@property`             | 資料型態 | 用途與說明                                                                                                                      |
| ---------------------- | --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **`smoothing_params`** | `settings.smoothing_alphas` | `dict`   | 用於計算小樣本或高噪聲實體（如騎師、馬匹）平滑勝率時的貝氏先驗權重（Dirichlet Prior）。                                         |
| ↳`default_alpha`      | -                           | `int`    | 未指定實體時的預設平滑 Alpha 值（預設`20`）。                                                                                   |
| ↳`alphas`             | `settings.smoothing_alphas` | `dict`   | 各細分實體（如`jockey`, `trainer`, `draw`, `horse_track` 等）專屬的平滑權重係數。Alpha 越大代表越不相信小樣本，越傾向全域平均。 |
| **`rating_map`**       | *(內部映射表)*              | `dict`   | 班次（Class 1-5）轉化為基準評分（Rating Score）的數字映射表。例如`"1": 100` 代表一班賽事基準分為 100 分。                       |
| **`track_bias_map`**   | *(內部映射表)*              | `dict`   | 跑道（A, B, C, C+3 移欄）的偏差加權係數，用於計算跑道偏差特徵。                                                                 |

### 📁 3. 檔案系統與目錄架構 (Paths Management)

所有路徑屬性均會由 `Settings` 自動拼合為絕對路徑 (`pathlib.Path`)：


| JSON 鍵名                          | 對應`@property`                   | 回傳型態 | 用途與說明                                                    |
| ---------------------------------- | --------------------------------- | -------- | ------------------------------------------------------------- |
| **`paths.raw_json_dir`**           | `settings.raw_json_dir`           | `Path`   | 原始未處理 JSON 檔案根目錄（`data/raw_json`）。               |
| **`paths.raw_races_json_dir`**     | `settings.raw_races_json_dir`     | `Path`   | 原始賽事 JSON 子目錄（`data/raw_json/races`）。               |
| **`paths.raw_sectional_json_dir`** | `settings.raw_sectional_json_dir` | `Path`   | 原始分段時間 JSON 子目錄（`data/raw_json/sectional`）。       |
| **`paths.raw_horses_json_dir`**    | `settings.raw_horses_json_dir`    | `Path`   | 原始馬匹歷史 JSON 子目錄（`data/raw_json/horses`）。          |
| **`paths.flattened_json_dir`**     | `settings.flattened_json_dir`     | `Path`   | 扁平化清洗後 JSON 之儲存目錄（`data/cleaned_json/flatten`）。 |
| **`paths.normalized_json_dir`**    | `settings.normalized_json_dir`    | `Path`   | 正規化（Relational Normalized）JSON 儲存目錄。                |
| **`paths.horses_sub_dir`**         | `settings.horses_dir`             | `Path`   | 正規化後馬匹資料集目錄。                                      |
| **`paths.races_sub_dir`**          | `settings.races_dir`              | `Path`   | 正規化後賽事資料集目錄。                                      |
| **`paths.rating_json_path`**       | `settings.rating_path`            | `Path`   | 馬匹評分歷史 JSON 檔路徑。                                    |
| **`paths.features_parquet_path`**  | `settings.features_parquet_path`  | `Path`   | **最終特徵矩陣**（Feature Matrix）輸出之 Parquet 檔案路徑。   |
| **`paths.today_rc_json_path`**     | `settings.today_rc_path`          | `Path`   | 即時/當日排位表（Race Card）JSON 路徑。                       |

### 📊 4. 特徵工程管制清單 (Feature Pipeline Governance)

本系統透過中央設定檔精細控制特徵管線的灌入與黑名單，嚴格避免資料洩漏（Data Leakage）與共線性：


| JSON 鍵名                | 對應`@property`               | 資料型態    | 用途與說明                                                                                                     |
| ------------------------ | ----------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------- |
| **`active_features`**    | `settings.active_features`    | `list[str]` | **當前模型訓練生效特徵**。模型只會讀取此列表中定義的欄位進行 Fitting。                                         |
| **`base_features`**      | `settings.base_features`      | `list[str]` | 基準特徵集合（Baseline Features），用於進行特徵篩選或消融實驗（Ablation Study）時的對照組。                    |
| **`candidate_features`** | `settings.candidate_features` | `list[str]` | 候選特徵清單。研發中的新特徵會先在此觀察，經評估有效後才移入`base_features` 或 `active_features`。             |
| **`banned_features`**    | *(內部黑名單)*                | `list[str]` | **禁用的特徵黑名單**。凡包含過度擬合風險、未過濾之未來數據（Future Leakage）或驗證無效的特徵均會在此強制剔除。 |
| **`PHYSICAL_FEATURES`**  | *(物理特徵清單)*              | `list[str]` | 預留之馬匹物理生理指標特徵清單（如體重變動率、負重比等）。                                                     |

### 📥 5. 資料載入器配置 (`data_loader` Settings)


| JSON 鍵名            | 對應`@property`               | 資料型態    | 用途與說明                                                                                  |
| -------------------- | ----------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| **`data_loader`**    | `settings.data_loader_config` | `dict`      | 數據載入器與 Dataset Builder 的核心欄位元資料（Metadata）。                                 |
| ↳`id_cols`          | `settings.id_cols`            | `list[str]` | 主鍵與識別欄位（`["race_id", "horse_id", "horse_name"]`），訓練時會自動隔離不作為 Feature。 |
| ↳`target_cols`      | `settings.target_cols`        | `list[str]` | 預測標籤欄位集合（`["placing", "is_win", "is_top3"]`）。                                    |
| ↳`eval_cols`        | `settings.eval_cols`          | `list[str]` | 回測評估與策略計算所需的上下文欄位（如賠率`win_odds`、檔位 `draw`、騎師 `jockey` 等）。     |
| ↳`categorical_cols` | `settings.categorical_cols`   | `list[str]` | 類別型特徵欄位（如烙號前綴`brand_prefix`、賽道類型 `course_type`）。                        |

---

## 🛠️ 4. 開發者指南與最佳實踐 (Best Practices)

### 1. 如何建立全新的實驗配置？

若要進行新一輪的模型實驗或特徵篩選，**請勿直接修改 `settings.example.json**`：

1. 複製 `settings.example.json` 並命名為 `settings_exp_v1.json`。
2. 在新檔案中修改 `active_features` 或模型超參數 `default_params`。
3. 在程式進入點中呼叫 `settings.switch_config("settings_exp_v1.json")`。

### 2. 新增特徵時的配置規範

當在 `features/generators/` 中撰寫了全新的特徵生成器後：

1. 先將新欄位填寫至 `settings.example.json` 的 `candidate_features` 中。
2. 透過 Feature Selection 腳本驗證其 Feature Importance 與 Feature Interaction。
3. 確定對 NDCG / Top-1 準確率有提升後，再將其拉入 `active_features` 進行正規訓練。
