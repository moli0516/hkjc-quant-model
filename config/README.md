# ⚙️ HKJC Quant: Configuration & Settings Management

The `config/` module serves as the central orchestration hub for the HKJC quantitative trading and prediction pipeline. Utilizing centralized JSON configuration files paired with a singleton-based `Settings` wrapper, this module accomplishes key architectural goals: **environment decoupling, dynamic in-process hot-swapping, automated path resolution, and feature-flag governance**.

Core architectural principles: **"Centralized governance, explicit pathlib resolution, zero hardcoding, and strict experiment version control."**

---

## 🏗️ 1. Configuration Workflow

All configuration management is maintained by the `Settings` class within `config/settings.py`. The full lifecycle for initialization, loading, and dynamic switching is structured as follows:

```text
┌────────────────────────┐
│  .active_config Pointer│
└───────────┬────────────┘
            │ 1. Resolve active pointer (default: settings.json)
            ▼
┌────────────────────────┐
│  settings.example.json │ (or selected target .json)
└───────────┬────────────┘
            │ 2. Read & parse structured JSON
            ▼
┌────────────────────────┐
│  Settings Instance     │
└───────────┬────────────┘
            │ 3. Resolve absolute paths relative to project root
            ▼
┌────────────────────────┐
│ Strongly-typed Property│ ──> Exposed to DataLoader / Features / Model Pipelines
└────────────────────────┘

```

### Step 1: Active Configuration Resolution

* **Automatic Root Resolution:** The `Settings` class identifies the project root (`root_dir`) relative to its file path (`pathlib.Path(__file__).parent.parent`). This eliminates cross-platform operating system errors or directory mismatch issues when executing from different Working Directories (CWD).
* **Persisted Pointer Resolution (`.active_config`):** The system inspects `config/.active_config`. If this pointer file exists, it loads the specified file name (e.g., `settings.json` or `settings.example.json`). If missing or invalid, it gracefully falls back to `settings.json`.

### Step 2: In-Memory Caching & Singleton Model

* Loads parsed JSON data into an internal dictionary (`self._data`).
* Exposes a global singleton instance (`settings = Settings()`) to ensure consistent configuration access across all pipeline threads.

### Step 3: Dynamic Path Auto-Resolution

* Directory paths inside JSON definitions are stored as relative path strings.
* Accessing any path property via `@property` (e.g., `settings.raw_json_dir`) automatically joins the relative path with `self.root_dir` and returns a strongly-typed `pathlib.Path` instance.

### Step 4: In-Process Hot-Swapping & Persistence (`switch_config`)

* Supports runtime re-configuration (e.g., during hyperparameter optimization, A/B testing, or financial backtesting) via `settings.switch_config("settings.example.json")`.
* Reloads the in-memory dictionary `_data` and updates `config/.active_config` to persist the setting selection across system restarts.

---

## 🔌 2. Dynamic Switch Mechanism

The active configuration can be switched without restarting the Python interpreter while preserving full safety and path integrity:

```python
from config.settings import settings

# 1. Access active parameters and features
print(settings.active_features)
print(settings.default_params)

# 2. Hot-swap configuration dynamically during experiments
settings.switch_config("settings.example.json")

# 3. Properties reflect the updated target file automatically
print(settings.target)  # Returns value from target JSON file

```

---

## 📋 3. Configuration Attribute Catalog (`settings.example.json`)

Detailed descriptions of configuration schema fields in `settings.example.json` and their corresponding `@property` bindings in `settings.py`:

### 🛠️ 1. Model Hyperparameters & Objectives


| JSON Key                     | Corresponding`@property`      | Data Type | Description & Primary Usage                                        |
| ---------------------------- | ----------------------------- | --------- | ------------------------------------------------------------------ |
| **`target`**                 | `settings.target`<br>         | `str`     | Target label for supervised training (`"is_place"`, `"is_win"`).   |
| **`default_params`**         | `settings.default_params`<br> | `dict`    | Hyperparameters for Gradient Boosted Decision Tree (GBDT) rankers. |
| ↳`objective`                | -                             | `str`     | Ranking objective function (`"rank:ndcg"`, `"lambdarank"`).        |
| ↳`max_depth`                | -                             | `int`     | Maximum tree depth (e.g.,`6`) controlling model complexity.        |
| ↳`learning_rate`            | -                             | `float`   | Shrinkage rate applied to update steps (e.g.,`0.05`).              |
| ↳`n_estimators`             | -                             | `int`     | Maximum number of boosting iterations / decision trees.            |
| ↳`early_stopping_rounds`    | -                             | `int`     | Early stopping patience threshold based on validation metrics.     |
| ↳`subsample`                | -                             | `float`   | Subsample ratio of the training instances per tree.                |
| ↳`colsample_bytree`         | -                             | `float`   | Feature subsampling ratio per tree build.                          |
| ↳`reg_alpha` / `reg_lambda` | -                             | `float`   | L1 (Lasso) and L2 (Ridge) regularization weights.                  |
| ↳`tree_method`              | -                             | `str`     | Tree construction algorithm (`"hist"` for histogram acceleration). |

### 🧮 2. Smoothing & Domain Knowledge Mapping


| JSON Key               | Corresponding`@property`        | Data Type | Description & Primary Usage                                            |
| ---------------------- | ------------------------------- | --------- | ---------------------------------------------------------------------- |
| **`smoothing_params`** | `settings.smoothing_alphas`<br> | `dict`    | Dirichlet prior weights used in`BayesianSmoother` calculations.        |
| ↳`default_alpha`      | -                               | `int`     | Default smoothing alpha when specific entity overrides are absent.     |
| ↳`alphas`             | `settings.smoothing_alphas`<br> | `dict`    | Entity-specific smoothing weights (`jockey`, `trainer`, `draw`, etc.). |
| **`rating_map`**       | *(Internal Mapping)*            | `dict`    | Baseline rating points by HKJC Race Class (Class 1–5).                |
| **`track_bias_map`**   | *(Internal Mapping)*            | `dict`    | Weighting factors reflecting track rail positions (A, B, C, C+3).      |

### 📁 3. System Paths & Storage Directory Architecture

All file paths are automatically compiled into absolute `pathlib.Path` objects relative to project root:


| JSON Key                 | Corresponding`@property`    | Return Type | Usage & Path Target                                               |
| ------------------------ | --------------------------- | ----------- | ----------------------------------------------------------------- |
| **`paths.raw_json_dir`** | `settings.raw_json_dir`<br> | `Path`      | Root directory for raw web scraper JSON output (`data/raw_json`). |

|
| **`paths.raw_races_json_dir`** | `settings.raw_races_json_dir`<br> | `Path` | Subdirectory for raw race result JSONs (`data/raw_json/races`).

|
| **`paths.raw_sectional_json_dir`** | `settings.raw_sectional_json_dir`<br> | `Path` | Subdirectory for raw sectional timing JSONs (`data/raw_json/sectional`).

|
| **`paths.raw_horses_json_dir`** | `settings.raw_horses_json_dir`<br> | `Path` | Subdirectory for raw horse profile JSONs (`data/raw_json/horses`).

|
| **`paths.flattened_json_dir`** | `settings.flattened_json_dir`<br> | `Path` | Cleaned flattened JSON staging directory (`data/cleaned_json/flatten`).

|
| **`paths.normalized_json_dir`** | `settings.normalized_json_dir`<br> | `Path` | Relational normalized JSON staging directory.

|
| **`paths.horses_sub_dir`** | `settings.horses_dir`<br> | `Path` | Subdirectory for normalized horse entities.

|
| **`paths.races_sub_dir`** | `settings.races_dir`<br> | `Path` | Subdirectory for normalized race entities.

|
| **`paths.rating_json_path`** | `settings.rating_path`<br> | `Path` | Historical horse rating master JSON path.

|
| **`paths.features_parquet_path`** | `settings.features_parquet_path`<br> | `Path` | Destination file path for generated feature matrix binaries.

|
| **`paths.today_rc_json_path`** | `settings.today_rc_path`<br> | `Path` | Real-time race card staging JSON path.

|

### 📊 4. Feature Pipeline Governance


| JSON Key              | Corresponding`@property`       | Data Type   | Description & Usage                                                                  |
| --------------------- | ------------------------------ | ----------- | ------------------------------------------------------------------------------------ |
| **`active_features`** | `settings.active_features`<br> | `list[str]` | **Active Features List**. Features loaded by `RaceDataLoader` during model training. |

|
| **`base_features`** | `settings.base_features`<br> | `list[str]` | Baseline feature set used for ablation studies and benchmarking.

|
| **`candidate_features`** | `settings.candidate_features`<br> | `list[str]` | Staging list for newly designed features undergoing validation.

|
| **`banned_features`** | `settings.banned_features`<br> | `list[str]` | **Strict Feature Exclusion List**. Enforces exclusion of post-race data, odds during model fitting, or leaking targets.

|

### 📥 5. DataLoader & Dataset Configuration (`data_loader`)


| JSON Key          | Corresponding`@property`          | Data Type | Description & Usage                     |
| ----------------- | --------------------------------- | --------- | --------------------------------------- |
| **`data_loader`** | `settings.data_loader_config`<br> | `dict`    | Metadata for`RaceDataLoader` execution. |

|
| ↳`id_cols` | `settings.id_cols`<br> | `list[str]` | Key identifiers (`["race_id", "horse_id", "horse_name"]`) isolated during model training.

|
| ↳`target_cols` | `settings.target_cols`<br> | `list[str]` | Supervised targets (`["placing", "is_win", "is_top3"]`).

|
| ↳`eval_cols` | `settings.eval_cols`<br> | `list[str]` | Financial and context evaluation variables (`["win_odds", "draw", "jockey", "trainer", "date"]`).

|
| ↳`categorical_cols` | `settings.categorical_cols`<br> | `list[str]` | Categorical features explicitly cast to `category` dtype for GBDT engines.

|

---

## 🛠️ 4. Developer Guidelines & Best Practices

### 1. Creating New Experiment Configurations

To modify hyperparameters, feature selections, or evaluation criteria, **do not directly alter `settings.example.json**`:

1. Duplicate `settings.example.json` and save it with an experiment tag (e.g., `settings_exp_v1.json`).
2. Update parameters, feature lists, or targets within the new configuration file.
3. Switch active settings programmatically or via CLI:

```python
settings.switch_config("settings_exp_v1.json")

```

### 2. Feature Onboarding Workflow

When introducing a new feature generator in `features/generators/`:

1. Add the output feature column names to `candidate_features` in `settings.example.json`.
2. Validate feature interaction and importance using feature selection tools.
3. Once proven to improve ranking metrics (NDCG / Top-1 Win Rate) without triggering data leakage, migrate the columns to `active_features` for production training runs.
