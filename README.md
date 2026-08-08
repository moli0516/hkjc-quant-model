# 🏇 HKJC Racing Quant: End-to-End ETL & Quantitative Machine Learning Pipeline

An industrial-grade, end-to-end quantitative data engineering and machine learning system tailored specifically for Hong Kong Jockey Club (HKJC) horse racing analytics. The platform spans the entire data lifecycle: asynchronous web scraping, clean relational database storage, temporal leakage-proof feature engineering, out-of-time (OOT) model training, Optuna hyperparameter optimization, and real-world financial backtesting via Kelly Criterion betting strategies.

---

## 🏗️ System Architecture & Codebase Map

The project adheres to a modular design separating scraping, cleaning, feature generation, model execution, and evaluation:

```text
./
├── cleaners/                    # Data Cleaning & Schema Normalization Pipeline
│   ├── cleaner_pipeline.py      # Main Cleaning Manager
│   ├── horses_cleaner.py        # Horse Profiles Parser & Normalizer
│   ├── races_cleaner.py         # Race & Results Cleaner
│   ├── sectional_cleaner.py     # Sectional Times & Positions Cleaner
│   ├── trackwork_cleaner.py     # Trackwork & Morning Gallops Cleaner
│   └── trails_cleaner.py        # Barrier Trials Metadata & Results Cleaner
├── config/                      # Configuration Management
│   ├── .active_config           # Persistent pointer to the active settings JSON
│   ├── settings.py              # Dynamic Settings Proxy Loader
│   ├── settings.json            # Base system & XGBoost hyperparameters
│   └── settings_roi.json        # ROI-optimized financial settings profile
├── database/                    # Relational Database Infrastructure
│   ├── models/                  # SQLAlchemy ORM Data Models
│   └── db_manager.py            # SQLite Connection Manager, Indexing, & Queries
├── features/                    # Feature Engineering Architecture
│   ├── generators/              # Dynamic Feature Generator Plugins
│   │   ├── _example_generator.py# Dummy Data Generator for CI/CD Testing
│   │   ├── body_weight_recovery.py
│   │   ├── class_performance.py
│   │   ├── context_relative.py  # Race-level z-scores, odds ranks & weights
│   │   ├── horse_profile.py     # Horse tenure & HK residency features
│   │   ├── horse_rolling.py     # Shifted rolling performance metrics
│   │   ├── human_sire.py        # Jockey/Trainer/Sire historical statistics
│   │   ├── injury_rest.py       # Rest days & layoff indicator features
│   │   ├── interaction.py       # Cross-feature interactions & ratios
│   │   ├── jockey_trainer_alpha.py
│   │   ├── jockey_trainer_synergy.py
│   │   ├── jt_recent_form.py
│   │   ├── odds_market.py       # Market implied probability & favorite flags
│   │   ├── pace_strategy.py     # Early pace & front-runner competition
│   │   ├── rating_class.py      # Rating class changes & drops
│   │   ├── ratinn_momentum.py   # Rating momentum & race-level advantages
│   │   ├── sectional_brust.py   # Late sectional burst ratios
│   │   ├── sectional_speed.py   # Sectional speeds & position gains
│   │   ├── speed_feature.py     # Speed z-scores & pace expenditure
│   │   ├── synergy_fitness.py   # Jockey switch & human-horse pairing stats
│   │   ├── track_distance.py    # Track/Distance specific win rates
│   │   ├── trackwork_feature.py # Trackwork activity & fast-work counts
│   │   └── trail_feature.py     # Barrier trial forms, pass flags & comments
│   ├── utils/                   # Feature Utilities
│   │   ├── leak_guard.py        # Automated Data Leakage Validation
│   │   ├── scale.py             # Race-level Scalers & Z-score Calculators
│   │   ├── smoother.py          # Bayesian Smoothers & Shifted Rolling Stats
│   │   ├── time_calc.py         # Speed (m/s) & Distance Normalization
│   │   └── track_bias.py        # Track Type & Draw Category Encoders
│   ├── base_target.py           # Base DataFrame Skeleton & Target Generator
│   └── feature_pipeline.py      # Sequential Zero-Copy Feature Engine
├── models/                      # Machine Learning & Quantitative Modeling
│   ├── hyperopt/                # Hyperparameter Tuning Engine
│   │   └── optuna_tuner.py      # Optuna Objective Wrappers & Tuner
│   ├── metrics/                 # Quantitative & Financial Evaluation
│   │   ├── calibration.py       # Odds-Aware Logistic Probability Calibrator
│   │   ├── finance.py           # Kelly Criterion, EV, & ROI Backtesting Engine
│   │   └── ranking.py           # Top-K Win Rates & Mean NDCG@K Metrics
│   ├── validation/              # Strict Temporal Validation & Audit
│   │   ├── micro_tracing.py     # Horse-level Micro-Tracer for Leakage Audit
│   │   └── time_split.py        # Out-Of-Time (OOT) Time-Series Splitter
│   ├── wrappers/                # Model Architecture Wrappers
│   │   └── xgb_wrapper.py       # XGBRanker Wrapper
│   ├── base_model.py            # Abstract Base Model Class
│   ├── data_loader.py           # Strict No-Odds Data Loader Engine
│   ├── model_pipeline.py        # End-to-End Model Execution & Inference Pipeline
│   └── registry.py              # Dynamic Model Factory & Registry
├── scraper/                     # Asynchronous Web Crawling System
│   ├── parser/                  # Selectolax Fast HTML/JSON Parsers
│   ├── hook.py                  # Aiohttp Async Session & Request Hook
│   ├── race_pipeline.py         # Async Race Results Crawler
│   ├── horse_pipeline.py        # Async Horse Profiles Crawler
│   ├── trackwork_pipeline.py    # Async Trackwork JSON API Crawler
│   └── trail_pipeline.py        # Async Barrier Trials Crawler
└── cli.py                       # Unified CLI Command-Line Interface

```

---

## ⚡ Installation & Quick Start

### 1. Requirements & Setup

Ensure you have Python 3.10 or higher installed. Clone the repository and initialize a virtual environment:

```bash
# Clone the repository
git clone <repository_url>
cd hkjc-racing-quant

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Infrastructure Initialization

The system automatically resolves data and database paths specified in `config/settings.json`. Executing any command via `cli.py` will automatically initialize directory structures (`data/raw_json/`, `data/cleaned_json/`) and build the SQLite relational database schema (`hkjc_racing.db`).

---

## 🔄 End-to-End Data Lifecycle

Data moves strictly through three fully decoupled stages:

```text
[ Raw Web / API ] 
       │
       ▼ (Scraper Engine: aiohttp + selectolax)
[ Raw JSON Storage ] ──> (data/raw_json/races, horses, trackworks, trails)
       │
       ▼ (Cleaners & DBManager: Pandas + SQLAlchemy)
[ SQLite Relational DB ] ──> (Tables: races, race_results, horses, race_trackwork, trials)
       │
       ▼ (BaseTargetBuilder + FeaturesPipeline: Bayesian Smoother + Shift Guard)
[ feature_matrix Table ] ──> (Strictly time-sorted, zero future leakage)
       │
       ▼ (ModelPipeline + XGBRanker: OOT Time Split)
[ Trained Model & Predictions ] ──> (NDCG@K, Top-1 Win Rate, Kelly ROI Backtest)

```

1. **Ingestion (Raw Crawling)**:
   The `scraper` module asynchronously fetches race cards, results, sectional times, horse profiles, morning trackwork, and barrier trials from HKJC servers, storing them as raw JSON documents.
2. **ETL & Normalization**:
   The `cleaners` module parses raw JSON files, performs type conversion, handles non-finishing codes (`WV`, `DNF`, `DISQ`), extracts numerical margins, and persists structured records into SQLite tables via SQLAlchemy ORM.
3. **Feature Engineering**:
   `FeaturesPipeline` executes dynamic feature generators ordered by `EXECUTION_ORDER`. All historical calculations rely on Bayesian smoothing and explicit `.shift(1)` operations to prevent post-race data leakage. The output is persisted into the `feature_matrix` database table.
4. **Model Training & Financial Backtesting**:
   `RaceDataLoader` streams the feature matrix while strictly stripping out odds features during training to prevent the ranking model from over-fitting to market consensus. Model predictions are evaluated using ranking metrics (`NDCG@K`, `Top-1 Win Rate`) and passed to `FinanceMetrics` for real-world Expected Value (EV) and Kelly Criterion backtesting.

---

## 🛠️ Command-Line Interface (`cli.py`) Usage

`cli.py` is the mainentry point for executing scrapers, cleaning pipelines, feature generation, hyperparameter tuning, and inference.

### Mode 1: Interactive Menu (Zero Arguments)

Launch the interactive text menu by executing `cli.py` without flags:

```bash
python cli.py

```

```text
==================================================
🏇 HKJC Racing Quant Engine & Machine Learning CLI (Config: settings.json)
==================================================
1.  Run Race & Sectional Scraper
2.  Run Race & Sectional Data Cleaner
3.  Run Horse Profile Scraper (Requires Race DB)
4.  Run Horse Profile Data Cleaner
5.  Run Trackwork Scraper
6.  Run Trackwork Data Cleaner
7.  Run Barrier Trial Scraper
8.  Run Barrier Trial Data Cleaner
9.  ⚙️  Generate Quant Feature Matrix (Features Pipeline)
10. 🤖 Train Racing Model (Model Pipeline)
11. 🎯 Optuna Hyperparameter Optimization (Model Tuning)
12. 🔮 Run Race Inference / Predictions
13. ⚡ Run One-Key Full Pipeline (Steps 1 ➔ 10)
14. ⚙️  Switch Active Settings Configuration
15. 🔄 Hot-Reload All Modules & Generators
0.  Exit
==================================================
Please select an option (0-15):

```

* **Hot-Reload (`15`)**: Dynamically reloads Python modules and feature generators in memory without restarting the CLI session.

---

### Mode 2: Automated CLI Arguments

Integrate tasks into scheduling systems (e.g., Airflow, Cron) by passing explicit arguments:

#### Command-Line Arguments Reference:


| Argument         | Type | Default | Description                                        |
| ---------------- | ---- | ------- | -------------------------------------------------- |
| `--scrape-races` | Flag | `False` | Triggers race results and sectional times crawler. ||
| `--clean-races` | Flag | `False` | Cleans raw race/sectional JSON files into SQLite.|
| `--scrape-horses` | Flag | `False` | Crawls missing horse profiles based on DB gaps.|
| `--clean-horses` | Flag | `False` | Cleans horse profile JSONs and updates `horses` table.|
| `--scrape-trackwork` | Flag | `False` | Crawls morning trackwork records for date ranges. |
| `--clean-trackwork` | Flag | `False` | Cleans raw trackwork JSONs into `race_trackwork`. |
| `--scrape-trails` | Flag | `False` | Crawls barrier trial results and metadata. |
| `--clean-trails` | Flag | `False` | Cleans trial JSONs into `trials` and `trial_results`. |
| `--generate-features` | Flag | `False` | Generates full time-sorted feature matrix into SQLite.|
| `--train-model` | Flag | `False` | Trains specified model model with OOT validation.|
| `--tune-model` | Flag | `False` | Runs Optuna hyperparameter optimization.|
| `--predict` | Flag | `False` | Performs inference for upcoming or target race dates.|
| `--all` | Flag | `False` | Runs complete end-to-end pipeline from scraping to training.|
| `--config` | Option | `None` | Switches active configuration JSON (e.g., `settings_roi.json`). |
| `--start-year` | Option | Current | Start year for date-range crawlers (YYYY).|
| `--end-year` | Option | Current | End year for date-range crawlers (YYYY).|
| `--model-type` | Option | `xgb_ranker` | Ranking architecture (`xgb_ranker`, `lgb_ranker`).|
| `--n-trials` | Option | `30` | Number of Optuna hyperopt trials.|

---

### CLI Command Examples

**1. Run Complete Ingestion, Feature Pipeline, & Model Training for 2025–2026:**

```bash
python cli.py --all --start-year 2025 --end-year 2026 --model-type xgb_ranker

```

**2. Generate Feature Matrix & Perform Hyperparameter Optimization:**

```bash
python cli.py --generate-features --tune-model --n-trials 50 --model-type xgb_ranker

```

**3. Run Inference for Upcoming Races using Custom ROI Settings Profile:**

```bash
python cli.py --config settings_roi.json --predict

```

---

## 🔒 Quantitative Integrity & Leakage Prevention Rules

1. **No Post-Race Data Leakage**:
   Current-race finishing statistics (e.g., `finish_time_sec`, `sec1_time`..`sec6_time`, `margin_len`, `placing`) are strictly isolated. All historical aggregations (running position averages, speed ratios, win rates) apply an explicit `.shift(1)` step via `BayesianSmoother`.
2. **Train/Val vs. Financial Pipeline Separation**:
   Model training datasets generated by `RaceDataLoader` strictly ban market odds features (`win_odds`, `implied_prob_share`, `odds_race_zscore`) to force the ranker to learn fundamental racing mechanics rather than market consensus. Market odds are attached exclusively during financial evaluation (`FinanceMetrics`) to compute Expected Value (EV) and fractional Kelly Criterion bet sizes.
3. **Micro-Tracing Leakage Audit**:
   `ModelPipeline` automatically triggers `HorseMicroTracer` validation prior to training, auditing row-by-row time-series progression for individual horses to verify that no race $N$ outcome leaks into race $N$ features.
