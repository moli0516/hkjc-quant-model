# 🐎 HKJC Quant: Learning-to-Rank & Modeling Pipeline (量化模型與排序管線)

這是專為香港賽馬（HKJC）量化預測設計的高效、嚴格防洩漏（Zero-Leakage）、基於 Learning-to-Rank (LTR) 架構的模型訓練與推論管線。本模組負責將特徵工程產出的特徵矩陣（Feature Matrix），透過時間序列切分、動態模型工廠、Automated Hyperparameter Tuning (Optuna) 與專業賽馬排序指標，轉化為具備實盤指導價值的預測模型與下注評分。

本模組的核心設計理念為：**「嚴格的臨場與賽後防洩漏（Strict Anti-Leakage）、基於群組的排序優化（Group-based Ranking）、無縫可擴充的模型註冊機制，以及極致的底層 C++ 執行緒防護。」**

---

## 🏗️ 1. 核心工作流程 (Pipeline Workflow)

整個模型管線由 `ModelPipeline` 統籌，協同 `RaceDataLoader`、`TimeSeriesSplitter`、`ModelRegistry` 與各模型 Wrapper 運作。完整資料流如下：

```
 SQLite / DB Matrix
        │
        ▼
[ RaceDataLoader ] ──► 強制過濾賽後結果/賠率洩漏 ──► Category/Float32 記憶體對齊
        │
        ▼
[ TimeSeriesSplitter ] ──► 依 Temporal Cutoff (val_days) 切分 ──► 生成 Ranking Groups
        │
        ▼
[ ModelRegistry ] ──► 動態反射實例化 (如 'xgb_ranker') ──► 轉交由 BaseModel Wrapper 處置
        │
        ▼
[ Model Fit & Optuna ] ──► Early Stopping & Group Pairwise/NDCG 尋優 ──► 特徵重要性排行
        │
        ▼
[ RankingMetrics ] ──► Top-1 冠中率 / Top-3 上名率 / NDCG@5 評估 ──► 產出 Race Rank

```

### Step 1: 數據加載與雙重防洩漏過濾 (`RaceDataLoader`)

* **資料整合：** 從 DBManager 撈取特徵矩陣與賽果數據，自動由名次（`placing`）衍生二元標籤（`is_win`, `is_top3`）與排序相關性分數（`relevance_score`）。
* **無條件黑名單（Post-race & Market Isolation）：** 徹底剔除「賽後方可取得」的數據（如：當場完賽時間 Z 號、末腳速度、走位變化）以及「所有賠率與市場熱門度欄位」（如 `win_odds`, `implied_prob_share`），確保模型僅依據純粹的基本面與歷史數據進行獨立研判。
* **C++ 防爆轉型與對齊：** 將類別型欄位補齊空值後轉為 `category`，並將數值型欄位強制統一為 `float32`，防止 XGBoost/LightGBM 底層 C++ 引擎因 Category 索引含 `float`/`NaN` 而直接崩潰。
* **記憶體快取 (In-Memory Caching)：** 使用 `_cache` 機制暫存預處理後的 DataFrame 與特徵欄位，大幅提升超參數尋優（Optuna Tuning）時重複載入資料的效能。

### Step 2: 時間序列驗證集切分 (`TimeSeriesSplitter`)

* **時間軸隔離：** 計算歷史資料的最大日期，往前推指定天數（如 `val_days=30`）劃分訓練集與驗證集，嚴禁使用 K-Fold 等會造成未來的資料穿越回過去的交叉驗證。
* **Ranking Group 自動計算：** 對切分後的資料強制按照 `race_date` 與 `race_id` 排序，並精確計算 Learning-to-Rank 演算法所需的 `groups` 陣列（即每場賽事的實際出賽馬匹數），且確保 `sum(groups) == len(df)`。

### Step 3: 動態模型工廠實例化與訓練 (`ModelRegistry` & `XGBRankerWrapper`)

* **工廠派發：** 透過 `ModelRegistry.create("xgb_ranker")` 動態載入模型。
* **排序目標函數 (LTR Objective)：** 底層採用 `rank:ndcg` 或 `rank:pairwise`，針對每場賽事的馬匹相對名次做最適優化。
* **驗證與 Early Stopping：** 傳入驗證集 `eval_set` 與 `eval_groups`，觸發 early stopping，防止模型對歷史賽事過度擬合。

### Step 4: 指標評估與特徵權重解析 (`RankingMetrics` & Importance)

* **賽馬專用指標：** 針對驗證集預估分數，計算 **Top-1 冠中率**、**Top-3 上名率** 及 **NDCG@5**。
* **權重排行榜日誌：** 自動解開 Wrapper 提取原生模型的 `feature_importances_` 或 Gain 分數，列印出特徵權重排行榜。

---

## 🧩 2. 設計模式與架構 (Design Patterns & Architecture)

本模組採用了多種軟體設計模式，以達到高耦合度分離與高擴充性：

### 1. 工廠與註冊器模式 (Factory & Registry Pattern)

使用 `@ModelRegistry.register("model_name")` 裝飾器，可以在新增模型時自動將其註冊至工廠集中管理。解耦了模型調用者（`ModelPipeline`）與模型實作類別之間的強關聯。

### 2. 策略與包裝器模式 (Strategy & Wrapper Pattern)

所有模型均繼承自 `BaseModel` 抽象基底類別，實作統一的 `fit()`, `predict()`, `save()`, `load()` 介面。`XGBRankerWrapper` 將複雜的 XGBoost 底層 C++ 互動、數據轉型與異常捕獲封裝在內部，對外提供極簡一致的調用體驗。

### 3. 自動化超參數尋優 (Automated Hyperparameter Tuning)

`OptunaTuner` 封裝了對 `ModelPipeline` 的呼叫，具備 MedianPruner 剪枝功能。調參過程中，每一輪 Trial 均會經歷嚴格的時間序列切分與評估，從而搜尋出最適的 Learning Rate、樹深度與 L1/L2 正則化參數。

---

## 📋 3. 核心組件與 API 目錄 (Core Modules Catalog)


| 模組檔案                       | 主要類別 / API       | 職責與設計亮點                                                                                                             |
| ------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **`base_model.py`**            | `BaseModel`          | **抽象基底類別**：定義規範的 `fit`, `predict`, `save`, `load` 介面，支援模型的 joblib 序列化與跨平台復原。                 |
| **`data_loader.py`**           | `RaceDataLoader`     | **資料加載器**：無賠率/無當場賽果的強防禦加載。自動處理 `category`/`float32` 型態、計算 Ranking Groups，並內建記憶體快取。 |
| **`model_pipeline.py`**        | `ModelPipeline`      | **管線統籌核心**：串連 DataLoader、TimeSplitter、Registry、Metrics 與 Optuna，提供訓練、尋优與推論的一站式調用入口。       |
| **`registry.py`**              | `ModelRegistry`      | **模型註冊工廠**：提供動態類別註冊與反射建立介面，實現模組熱插拔。                                                       |
| **`validation/time_split.py`** | `TimeSeriesSplitter` | **時間切分器**：按日期推進切分，計算 Group 陣列，嚴格防範時間軸穿越（Temporal Data Leakage）。                             |
| **`wrappers/xgb_wrapper.py`**  | `XGBRankerWrapper`   | **XGBRanker 封裝器**：實現 LTR 排序模型的訓練與推論，包含 C++ XGBoostError 捕獲、類別特徵對齊與 Early Stopping。           |
| **`hyperopt/optuna_tuner.py`** | `OptunaTuner`        | **自動調参器**：基於 Optuna 框架的超參數搜尋，支援 MedianPruner 剪枝與異常試驗降級。                                       |
| **`metrics/ranking.py`**       | `RankingMetrics`     | **賽馬排序評估器**：計算指定 $K$ 值下的 Top-K Win Rate、Top-K Place Rate，以及基於賽事群組的 Mean NDCG@K。                 |

---

## 📊 4. 排序指標與評估體系 (Ranking Evaluation Metrics)

傳統分類（AUC、LogLoss）或迴歸指標（MSE）無法真實反映賽馬下注的實務情境。本管線採用三項專為 Learning-to-Rank 量身打造的評估指標：

1. **Top-1 Win Rate (獨贏/冠中率)：**

$$
\text{Top-1 Win Rate} = \frac{\text{模型預測第一名中實際跑出冠軍的場數}}{\text{總評估賽事場數}}
$$

直接衡量模型選出冠軍馬（Top Pick）的精確度。
2. **Top-K Place Rate (位置/上名率)：**
當 $K=3$ 時，計算模型預測前三名的馬匹中，實際跑入前三名的平均比例，用以評估位置（Place）下注策略的命中率。
3. **Mean NDCG@K (Normalized Discounted Cumulative Gain)：**

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

考慮到了名次排列的先後順序權重（冠軍相關性得分高，亞軍次之）。透過計算跨所有賽事的平均 NDCG，能全面評估模型對整場馬匹名次次序的排序能力。

---

## 🛡️ 5. 防洩漏與極致穩定性機制 (Data Leakage Guard & Production Safety)

### 1. 雙重特徵過濾網絡 (Forbidden Columns Firewall)

`RaceDataLoader` 與 `ModelPipeline` 設定了雙層防禦機制，會無條件過濾掉以下欄位：

* **賽後結果數據 (Post-race Leakage)：** `finish_time_sec`, `finish_time_race_z`, `last_400m_speed_z`, `speed_mps_last_sectional`, `sectional_time_last`, `position_gain_first_to_last` 等。
* **市場與臨場賠率數據 (Market Leakage)：** `win_odds`, `win_odds_race_z`, `odds_implied_prob`, `is_market_favorite`, `implied_prob_share` 等。

### 2. C++ 引擎轉型防爆 (Robust Categorical Alignment)

在多數 GBDT 框架（如 XGBoost `enable_categorical=True`）中，若類別欄位包含 `float` 的 `NaN` 或是未對齊的類別，會觸發底層 C++ Core Dump。
`XGBRankerWrapper` 在 `_preprocess_features` 中會：

* 訓練時：記錄各特徵的精確 `dtype` 與 `categories`。
* 推論時：強制將輸入資料映射至訓練時的類別結構，缺失類別自動歸類為未知，確保線上推論零崩潰。

---

## 💡 6. 開發者指南：如何新增自訂模型 (Developer Guide)

若要新增一個全新的排序模型（例如 LightGBM Ranker），只需遵循以下三個步驟：

### Step 1: 繼承 `BaseModel` 並使用裝飾器註冊

在 `models/wrappers/lgb_wrapper.py` 中編寫類別：

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from models.registry import ModelRegistry

@ModelRegistry.register("lgb_ranker")
class LGBMRankerWrapper(BaseModel):
    def __init__(self, model_params: dict = None):
        super().__init__(model_params)
        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.03,
            "random_state": 42
        }
        if self.model_params:
            default_params.update(self.model_params)
        self.model = lgb.LGBMRanker(**default_params)

    def fit(self, train_df, feature_cols, target_col, groups=None, eval_set=None, eval_groups=None, **kwargs):
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
      
        fit_params = {"group": groups}
        if eval_set:
            val_df, val_features, val_target = eval_set
            fit_params["eval_set"] = [(val_df[val_features], val_df[val_target])]
            fit_params["eval_group"] = [eval_groups]
          
        self.model.fit(X_train, y_train, **fit_params, **kwargs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.feature_cols])

```

### Step 2: 在 `ModelPipeline` 中設定搜尋空間 (可選)

若需支援自動調參，在 `models/model_pipeline.py` 的 `_get_default_search_space` 方法中加入對應的 Optuna 參數空間即可。

### Step 3: 直接調用管線執行訓練

新模型會自動被 `ModelRegistry` 識別，隨後即可直接執行：

```python
from models.model_pipeline import ModelPipeline
from database.db_manager import DBManager

db_mgr = DBManager()
pipeline = ModelPipeline(db_manager=db_mgr)

# 直接透過名稱動態調用全新模型！
model, metrics = pipeline.run_train_pipeline(
    model_name="lgb_ranker",
    val_days=30
)

```
