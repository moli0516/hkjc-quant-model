
# 🏇 HKJC Racing Quant: End-to-End ETL & Quantitative Machine Learning Pipeline

An industrial-grade, end-to-end quantitative data engineering and machine learning system tailored for Hong Kong Jockey Club (HKJC) horse racing analytics. The platform spans the full data lifecycle: asynchronous web scraping, relational storage, temporal leakage-proof feature engineering, out-of-time (OOT) ranking models, Optuna hyperparameter search, expanding-window **walk-forward** evaluation, and fixed-stake research backtests against an explicit **market-favorite baseline**.

This repository is a **research and engineering** system. It is not a tip sheet or live betting bot. Empirical walk-forward results on the current feature set show **no stable positive unit-stake win ROI** under the tested policies. Selective agreement with the tote favorite (optionally filtered by barrier-trial flags) can **reduce losses and drawdown** relative to always backing the model top pick—that is a relative finding, not a claim of economic edge.

---

## 🏗️ System Architecture & Codebase Map

The project uses a modular layout: scrape → clean → features → model → evaluate.

```text
./
├── cleaners/                       # Data cleaning & schema normalization
│   ├── cleaner_pipeline.py
│   ├── horses_cleaner.py
│   ├── races_cleaner.py
│   ├── sectional_cleaner.py
│   ├── trackwork_cleaner.py
│   └── trails_cleaner.py
├── config/                         # Configuration management
│   ├── .active_config
│   ├── settings.py
│   ├── settings.example.json
│   ├── settings.json
│   └── settings_roi.json
├── database/                       # Relational infrastructure
│   ├── models/                     # SQLAlchemy ORM
│   └── db_manager.py               # Connection, indexes, merged race load
├── features/                       # Feature engineering
│   ├── generators/                 # Plugin generators (EXECUTION_ORDER)
│   ├── utils/                      # leak_guard, smoother, scale, time_calc, …
│   ├── base_target.py
│   └── feature_pipeline.py
├── models/                         # ML, validation, quantitative evaluation
│   ├── evaluation/
│   │   ├── rules/                  # BettingRule registry (M0/A0/C1/…)
│   │   ├── walk_forward.py         # Expanding-window walk-forward
│   │   ├── diagnostics.py          # Market baseline, strata, fold stability
│   │   ├── betting.py              # Fixed-stake rule execution (run_many)
│   │   ├── prediction_store.py     # Cold-store predictions (parquet + meta)
│   │   ├── baselines.py
│   │   └── metrics_ext.py
│   ├── hyperopt/
│   │   └── optuna_tuner.py
│   ├── metrics/
│   │   ├── calibration.py
│   │   ├── finance.py              # EV / ROI / optional Kelly-style summaries
│   │   └── ranking.py
│   ├── validation/
│   │   ├── micro_tracing.py
│   │   └── time_split.py
│   ├── wrappers/
│   │   └── xgb_wrapper.py
│   ├── base_model.py
│   ├── data_loader.py
│   ├── model_pipeline.py
│   └── registry.py
├── scraper/                        # Async crawling
│   ├── parser/
│   ├── hook.py
│   ├── race_pipeline.py
│   ├── horse_pipeline.py
│   ├── trackwork_pipeline.py
│   └── trail_pipeline.py
├── cli.py                          # Unified CLI
└── start.bat                       # Windows launcher (process-level reload)
```

Modeling-layer narrative: see [`models/README.md`](models/README.md) when present.

---

## ⚡ Installation & Quick Start

### 1. Requirements & Setup

Python 3.10+ recommended.

```bash
git clone <repository_url>
cd hkjc-quant-model   # or your local folder name

python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Infrastructure Initialization

Paths are resolved from `config/settings.json` (see `settings.example.json`). Running `cli.py` initializes data directories and the SQLite schema as needed. Do not commit local `*.db`, full `data/`, or secret-bearing settings (see `.gitignore`).

---

## 🔄 End-to-End Data Lifecycle

```text
[ HKJC Web / API ]
       │
       ▼  scraper (aiohttp + selectolax)
[ Raw JSON ]  ── data/raw_json/…
       │
       ▼  cleaners + DBManager
[ SQLite ]  ── races, race_results, sectionals, horses, trackwork, trails, trail_results
       │
       ▼  load_all_merged_race_data (time-safe trial/trackwork context)
[ Race panel ]
       │
       ▼  FeaturesPipeline (Bayesian smoothers + shift guards)
[ feature_matrix ]
       │
       ▼  RaceDataLoader (odds / results banned from X at fit)
[ Train / eval frames ]
       │
       ▼  ModelPipeline + XGBRanker
[ OOT train / Optuna / inference ]
       │
       ▼  WalkForwardEvaluator
[ Predictions + fold_id ]
       ├── RankingMetrics          model vs market top-K
       ├── WalkForwardDiagnostics  odds profile, strata, trial residual, fold stability
       ├── BetEvaluator + rules/   fixed-stake policies → report["rules"]
       └── PredictionStore         data/predictions/<run_id>/  (offline re-eval)
```

1. **Ingestion** — Async crawl of results, sectionals, horses, trackwork, barrier trials → raw JSON.  
2. **ETL** — Cleaners normalize types, non-finishers, margins; persist via SQLAlchemy.  
3. **Features** — Generators run in `EXECUTION_ORDER` with explicit anti-leakage shifts; output → `feature_matrix`.  
4. **Model** — Ranker trains without contemporaneous odds in `feature_cols`.  
5. **Evaluation** — Calendar OOT and expanding walk-forward; always report **market baseline**; research rules use **unit stakes** for comparability. Optional Kelly-style sizing lives in `FinanceMetrics` for methodology experiments, not as a claim of live edge.

---

## 🛠️ Command-Line Interface (`cli.py`)

### Mode 1: Interactive menu

```bash
python cli.py
# or: start.bat
```

Typical menu (labels may vary slightly by build):

| Step | Action |
|------|--------|
| 1–2 | Race & sectional scrape / clean |
| 3–4 | Horse scrape / clean |
| 5–6 | Trackwork scrape / clean |
| 7–8 | Barrier trial scrape / clean |
| 9 | Feature matrix pipeline |
| 10 | Train model (OOT window) |
| 11 | Optuna hyperparameter search |
| 12 | Inference / predictions |
| 13 | One-key pipeline (when wired) |
| **14** | Walk-forward evaluation (+ diagnosis, optional prediction dump) |
| **15** | Offline evaluation from cold-stored predictions |
| **R** | Hot-reload project modules |
| **S** | Switch active settings JSON |
| 0 | Exit |

After editing `models/`, prefer a **full process restart** (`start.bat` **R** or relaunch) so offline evaluation does not use stale bytecode.

### Mode 2: CLI flags (scheduling / automation)

| Argument | Description |
|----------|-------------|
| `--scrape-races` / `--clean-races` | Race & sectional crawl / clean |
| `--scrape-horses` / `--clean-horses` | Horse profiles |
| `--scrape-trackwork` / `--clean-trackwork` | Morning trackwork |
| `--scrape-trails` / `--clean-trails` | Barrier trials |
| `--generate-features` | Full feature matrix |
| `--train-model` | OOT train |
| `--tune-model` | Optuna search |
| `--predict` | Inference |
| `--walk-forward` | Expanding walk-forward (when exposed) |
| `--eval-store` | Offline eval from `PredictionStore` |
| `--all` | End-to-end through train (as implemented) |
| `--config` | Switch settings profile |
| `--start-year` / `--end-year` | Crawl year range |
| `--model-type` | Default `xgb_ranker` |
| `--n-trials` | Optuna trials |

Examples:

```bash
python cli.py --all --start-year 2025 --end-year 2026 --model-type xgb_ranker
python cli.py --generate-features --tune-model --n-trials 50
python cli.py --config settings_roi.json --predict
```

### Library-style

```python
from database.db_manager import DBManager
from models.model_pipeline import ModelPipeline

pipe = ModelPipeline(db_manager=DBManager())
model, metrics = pipe.run_train_pipeline(model_name="xgb_ranker", val_days=90)

report = pipe.run_walk_forward_evaluation(
    model_name="xgb_ranker",
    min_train_days=730,
    step_days=30,
    run_diagnosis=True,
)

report = pipe.run_offline_evaluation(run_id="wf_YYYYMMDD_HHMMSS")
# report["ranking"], report["rules"], report["diagnosis"], …
```

---

## 📏 Evaluation layer (research)

| Component | Role |
|-----------|------|
| `WalkForwardEvaluator` | Expanding train window; step test origin by `step_days` |
| `RankingMetrics` | Model top-1 / top-3 vs market favorite |
| `WalkForwardDiagnostics` | M vs A, odds bins, agree/disagree strata, trial residual, **fold stability** |
| `BetEvaluator` + `evaluation/rules/` | Registered fixed-stake policies (`run_many` → `report["rules"]`) |
| `PredictionStore` | Parquet predictions + meta for offline re-runs without retraining |

### Research rules (`rule_id`)

| ID | Concept |
|----|---------|
| **M0** | Market rank == 1 |
| **A0** | Model rank == 1 |
| **B0** | Overlay / value-style filter on scores |
| **C1** | Same pick (model ∩ market top-1) **and** strong barrier trial |
| **C2** | Same pick **and** non-strong trial (contrast) |
| **C3** | Same pick **and** fresh trial window |
| **C4** | Same pick **and** not fresh (contrast) |
| **D0** | Model rank 1 **and** market rank ≤ 2 |
| **E0** | Same pick **and** strong **and** fresh |

Unit stakes only. ROI and drawdown are methodology scores under HKJC-style win pricing—not staking advice.

**Illustrative baseline** (research log): model top-1 hit ≈ 0.25 vs market ≈ 0.31; no tested rule showed stable positive ROI. **C1** improved hit and reduced loss vs **A0** with strong fold-level support; stacking **E0** raised hit further but did not improve ROI over **C1**.

---

## 🔒 Quantitative Integrity & Leakage Prevention

1. **No post-race leakage in features**  
   Current-race finish stats (`finish_time_sec`, sectionals, `margin_len`, `placing`, …) are not used as covariates for the same race. Historical aggregates use Bayesian smoothers and explicit `.shift(1)` (and related guards) in `features/`.

2. **Train vs financial / market separation**  
   `RaceDataLoader` excludes banned columns and live odds fields from training `feature_cols` so the ranker is not trained to copy the tote. `win_odds` / `market_rank` attach at **evaluation** for baselines and unit-stake (or optional Kelly-style) research summaries.

3. **Micro-tracing audit**  
   Optional `HorseMicroTracer` checks that rolling features do not absorb the same race’s outcome for the same horse before train.

4. **Time-safe external context**  
   Barrier-trial and trackwork joins in `load_all_merged_race_data` enforce work/trial **before** race day (normalized date formats).

5. **Honest metrics**  
   Report ranking quality **and** fixed-stake ROI / max drawdown; keep market baselines side-by-side.

---

## 🎯 Design goals (summary)

1. End-to-end path with one CLI entry.  
2. Temporal integrity by default (OOT + walk-forward).  
3. Market baseline on every serious evaluation.  
4. Reproducible experiments via cold-stored predictions.  
5. Modular research rules (`rule_id` registry).  
6. Transparent reporting of negative and relative results.

---

## 📦 Scope

**In scope:** reproducible historical research on HKJC-structured data; leakage-aware features; OOT / walk-forward; market-baseline diagnostics; modular policy experiments; academic packaging of methodology (including negative results).

**Out of scope:** live order routing; real-money bankroll products; guaranteed returns; publishing bet-level tip streams.

A large engineering surface does not imply economic edge.

---

## License & responsibility

Intended for education, methodology development, and reproducible quantitative research on HKJC historical data. Users are responsible for compliance with applicable law and HKJC terms when accessing data or placing any wagers.
