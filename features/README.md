# 🐎 HKJC Quant: Feature Engineering Pipeline (`features/`)

An asynchronous, leakage-proof, and highly modular feature engineering pipeline engineered specifically for Hong Kong Jockey Club (HKJC) quantitative racing models. This module transforms raw race results, sectional times, trackwork, and barrier trials into high-quality tabular feature matrices for machine learning estimators.

The core design principle is **"Absolute Temporal Isolation (Zero-Data-Leakage), Zero-Fragmentation Memory Management, and Plug-and-Play Dynamic Extension."**

---

## 📂 Module Architecture (`features/`)

```text
features/
├── generators/             # Hot-swappable feature generators (Plugin architecture)
│   ├── __init__.py         # Dynamic module discovery and execution scheduling
│   ├── _example_generator.py
│   ├── body_weight_recovery.py
│   ├── class_performance.py
│   ├── context_relative.py
│   ├── horse_profile.py
│   ├── horse_rolling.py
│   ├── human_sire.py
│   ├── injury_rest.py
│   ├── interaction.py
│   ├── jockey_trainer_alpha.py
│   ├── jockey_trainer_synergy.py
│   ├── jt_recent_form.py
│   ├── odds_market.py
│   ├── pace_strategy.py
│   ├── rating_class.py
│   ├── ratinn_momentum.py
│   ├── sectional_brust.py
│   ├── sectional_speed.py
│   ├── speed_feature.py
│   ├── synergy_fitness.py
│   ├── track_distance.py
│   ├── trackwork_feature.py
│   └── trail_feature.py
├── utils/                  # Core calculation tools and security guard libraries
│   ├── __init__.py
│   ├── leak_guard.py       # Data Leakage firewall
│   ├── scale.py            # Race-level contextual standardization (Z-score, Ranking)
│   ├── smoother.py         # Leak-proof Bayesian smoothing and rolling statistics
│   ├── time_calc.py        # Speed and normalized time calculations
│   └── track_bias.py       # Course, track type, and draw bias encoding
├── __init__.py
├── base_target.py          # Data skeleton builder and machine learning targets
└── feature_pipeline.py     # Main feature engineering orchestration pipeline

```

---

## 🏗️ 1. Core Workflow Pipeline

The execution flow is orchestrated collaboratively by `BaseTargetBuilder` and `FeaturesPipeline`.

### Step 1: Data Skeleton & Target Generation (`BaseTargetBuilder`)

* **Data Cleaning & Alignment**: Receives raw records, unifies column nomenclature, filters invalid placings (e.g., WV, DNF, PU), and extracts regular expressions for dead-heat conditions (e.g., `'1 DH'`).
* **Strict Temporal Sorting**: Forces records to sort ascending by `race_date`, `race_id`, and `horse_id` to establish the first defense line against look-ahead bias.
* **Temporal Guard**: Compares horse registry import dates (`import_date`) against `race_date`. If import occurs post-race, it masks the value as `NaT` to prevent future information leakage.
* **Target Formulation**: Unifies three canonical machine learning prediction targets:
* `target_win` (binary win classification)
* `target_place` (binary top-3 finish classification)
* `target_rank_score` (inverse ranking score for learning-to-rank objectives)

### Step 2: Orchestration Pipeline (`FeaturesPipeline`)

* **Dynamic Discovery**: Automatically scans the `generators/` directory upon initialization and dispatches execution based on `EXECUTION_ORDER`.
* **Memory Optimization (Zero-Fragmentation)**:
* Downcasts newly generated `float64` features to `float32` to minimize memory overhead.
* Stores intermediate feature arrays inside a list and executes a single horizontal concatenation (`pd.concat`) via zero-copy semantics, avoiding DataFrame append memory fragmentation.
* Filters duplicate column tokens to prevent unexpected overrides.
* **Index Alignment & Validation**: Restores original DataFrame indexing prior to output and runs validation checks via `LeakageGuard`.

---

## 🔌 2. Hot-Swappable Feature Generators

The pipeline requires no manual imports or rigid registration scripts. All generators adhere to a plug-and-play architecture (`generators/__init__.py`).

1. **Dynamic Module Exploration**: Scans all non-private `.py` files inside the `generators` directory.
2. **Reflection**: Locates classes ending with the `Generator` suffix.
3. **Execution Sequencing**: Sorts instances based on the internal `EXECUTION_ORDER` constant (defaulting to `500`), ensuring dependent features compute sequentially.

> **💡 Developer Guide**:
> To introduce new features, create a `.py` file under `generators/`, define a class with a `generate(self, df)` method, assign an `EXECUTION_ORDER`, and the pipeline will **automatically load and execute it**.

---

## 🛠️ 3. Core Utilities & Leakage Guard

The `features/utils/` modules enforce strict time isolation across calculations:

### 🛡️ `LeakageGuard` (`leak_guard.py`)

The feature engineering firewall.

* **`check_future_leakage`**: Automatically monitors feature-to-target correlation coefficients (threshold: `0.90`), flagging unshifted historical values.
* **`assert_no_null_keys`**: Intercepts structural failures involving missing primary keys (`race_id`, `horse_id`).

### 📊 `BayesianSmoother` (`smoother.py`)

Computes rolling averages and smoothed win rates.

* **Automatic Sorting**: Enforces internal `sort_values` to safeguard chronological integrity.
* **Strict In-Race Isolation**: Enforces `groupby.shift(1)` to permanently exclude current-race outcomes.
* **Bayesian Smoothing**: Employs prior probability shrinkage (`calc_global_smooth_rate` and `calc_rolling_smooth_rate`) to stabilize sample variance for debutants or low-frequency observations.

### ⚖️ `RaceScaler` (`scale.py`)

Computes race-level contextual features:

* **`race_z_score`**: Computes intra-race Z-scores (e.g., speed advantage relative to field).
* **`race_diff_from_mean`**: Computes deviations from field averages (e.g., weight differentials).
* **`race_rank`**: Computes intra-race ordinal ranks.

### ⏱️ `SpeedTimeCalculator` (`time_calc.py`)

Transforms distance and time parameters:

* **`calc_speed_mps`**: Evaluates meters per second (m/s) across variable distances.
* **`normalize_time_by_distance`**: Proportional time scaling to a standard distance (default 1200m).

### 🏟️ `TrackEncoder` (`track_bias.py`)

* **`categorize_course_type`**: Sanitizes surface strings into `TURF` or `AWT` (All Weather Track).
* **`create_track_draw_combo`**: Encapsulates venue-course-draw combinations to capture specific spatial biases.

---

## 🚀 4. Quick Start

Execute feature extraction directly from the database manager:

```python
from features.base_target import BaseTargetBuilder
from features.feature_pipeline import FeaturesPipeline

# 1. Load merged historical database records and construct data skeleton
df_base = BaseTargetBuilder.build_from_sqlite(db_path="database/hkjc_racing.db")

# 2. Initialize orchestration pipeline
pipeline = FeaturesPipeline(key_cols=["race_id", "horse_id"])

# 3. Execute feature engineering matrix generation
df_features = pipeline.run(df_base)

print(df_features.head())

```
