# models/

Machine Learning, Temporal Validation & Quantitative Evaluation Layer

This package is the modeling and evaluation core of the HKJC Racing Quant pipeline. It turns the leakage-controlled `feature_matrix` into ranking models, enforces strict out-of-time (OOT) protocols, and measures performance against an explicit **market-favorite baseline**—not only against internal ranking scores.

Training never sees contemporaneous odds or race results as features. Odds are attached only at evaluation time for baseline comparison and fixed-stake research backtests. Betting helpers are **abstract research utilities**, not live wagering advice.

---

## 🎯 Design Goals

1. **Train ranking models** (default: XGBoost `rank:ndcg`) without target or odds leakage.
2. **Validate in time**, via calendar OOT splits and expanding-window **walk-forward**.
3. **Always report model vs market** (favorite by win odds / `market_rank`).
4. **Diagnose failures** (odds profile of model top picks, agree/disagree with the favorite, optional barrier-trial residuals).
5. **Separate learning from betting**: rankers optimize relevance; financial summaries use fixed unit stakes for comparability.

Empirical walk-forward results on the current feature set show **no stable positive win ROI** under the tested policies. Selective agreement with the market (and weak barrier-trial filters) can **reduce losses and drawdown**; they are not presented as an economic edge.

---

## 🏗️ Module Map

```text
models/
├── base_model.py              # Abstract model interface
├── data_loader.py             # Feature matrix loader (odds stripped for training)
├── model_pipeline.py          # Train / tune / infer / walk-forward orchestration
├── registry.py                # Model name → factory
├── evaluation/                # Out-of-time evaluation & research backtests
│   ├── walk_forward.py        # Expanding-window walk-forward evaluator
│   ├── diagnostics.py         # Market baseline, odds bins, strata, trial residual
│   ├── betting.py             # Fixed-stake rule backtests (research only)
│   ├── baselines.py           # Market ranking baseline helpers
│   └── metrics_ext.py         # Extra ranking utilities
├── metrics/
│   ├── ranking.py             # Top-K hit rates, mean NDCG@K
│   ├── finance.py             # ROI / EV / optional Kelly-style summaries
│   └── calibration.py         # Score calibration utilities
├── validation/
│   ├── time_split.py          # Calendar-day OOT splitter
│   └── micro_tracing.py       # Per-horse shift / leakage audit
├── hyperopt/
│   └── optuna_tuner.py        # Optuna search wrapper
├── wrappers/
│   └── xgb_wrapper.py         # XGBRanker adapter
└── tests/                     # Unit tests for loaders, splits, ranking, registry
```

For a fuller narrative of the evaluation layer, see also the methodology notes in this directory when published alongside the root project README.

---

## 🔄 Modeling Lifecycle

```text
[ feature_matrix ] 
       │
       ▼  RaceDataLoader (ban contemporaneous odds / results from X)
[ Training Frame + feature_cols ]
       │
       ▼  TimeSeriesSplitter / WalkForwardEvaluator
[ Train folds (past only) → Test folds (future only) ]
       │
       ▼  ModelRegistry + XGBRankerWrapper
[ pred_score / model_rank ]
       │
       ├──► RankingMetrics     (top1, top3, NDCG@K)
       ├──► Market baseline    (market_rank from win_odds)
       ├──► WalkForwardDiagnostics
       └──► BetEvaluator       (fixed-stake research rules A/B/C/D)
```

1. **Load**: `RaceDataLoader` builds the frame from SQLite; training feature lists exclude banned columns (`settings.banned_features`) and live odds fields.
2. **Audit (optional but recommended)**: `HorseMicroTracer` checks that rolling features do not absorb same-race outcomes.
3. **Fit**: Rankers consume relevance labels derived from `placing` (e.g. graded top-3 relevance)—objective design is explicit and swappable.
4. **Score**: Inference assigns `pred_score` and `model_rank` within each `race_id`.
5. **Evaluate**: Walk-forward aggregates ranking quality **and** unit-stake PnL; diagnostics explain *where* the model diverges from the tote favorite.

---

## 🚀 Usage

### Train (single OOT window)

```python
from database.db_manager import DBManager
from models.model_pipeline import ModelPipeline

pipe = ModelPipeline(db_manager=DBManager())
model, metrics = pipe.run_train_pipeline(
    model_name="xgb_ranker",
    val_days=90,
)
```

### Walk-forward evaluation (+ optional diagnosis)

```python
report = pipe.run_walk_forward_evaluation(
    model_name="xgb_ranker",
    min_train_days=730,
    step_days=30,
    overlay_threshold=1.15,
    run_diagnosis=True,
)

# report["ranking"]     → model_top1 vs market_top1, coverage
# report["rule_a"|…]    → fixed-stake summaries
# report["predictions"] → long panel for custom analysis
# report["diagnosis"]   → diagnostic tables (if enabled)
```

### Diagnosis only (reuse existing predictions)

```python
from models.evaluation.diagnostics import WalkForwardDiagnostics

WalkForwardDiagnostics(stake=1.0).run(report["predictions"])
```

### CLI (project root)

```bash
python cli.py --train-model --model-type xgb_ranker
python cli.py --walk-forward
python cli.py --tune-model --n-trials 30 --model-type xgb_ranker
```

Interactive menu entries mirror these flows (train, Optuna tune, walk-forward when wired in `cli.py`).

---

## 📏 Evaluation Philosophy


| Principle          | Implementation                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| Temporal integrity | Train on past races only; test on later folds / windows                                          |
| Market baseline    | Favorite by`win_odds` → `market_rank`; always report side-by-side                               |
| No odds in**X**    | Odds columns may exist on the evaluation frame; they are stripped from`feature_cols` at fit time |
| Leakage discipline | `.shift(1)` in features upstream; micro-tracing before train                                     |
| Honest metrics     | Hit rate**and** unit-stake ROI / max drawdown—not accuracy alone                                |

### Walk-forward

`WalkForwardEvaluator` uses an **expanding** training window and advances the test origin by `step_days` after `min_train_days` of history. Each fold refits the model and scores only the forward segment—closer to a deployment timeline than a single hold-out.

### Diagnostics (research)

`WalkForwardDiagnostics` produces four standard views:

1. **Rule M vs A** — always back market rank 1 vs always back model rank 1
2. **Odds profile** of model rank 1 (bins, hit rate vs rough implied probability)
3. **Strata** — model vs market by favorite-odds band; **agree vs disagree** with the favorite
4. **Trial residual** — coarse barrier-trial flags inside a chosen odds band

These tools explain structure; they do not authorize live staking.

---

## 💰 Research Backtest Rules (`evaluation/betting.py`)

All rules use **fixed unit stakes** so policies remain comparable.


| Rule  | Condition (conceptual)                                  | Role                                    |
| ----- | ------------------------------------------------------- | --------------------------------------- |
| **A** | `model_rank == 1`                                       | Naive “always trust the model” policy |
| **B** | Overlay / value filter on model scores                  | Stress-test probability calibration     |
| **C** | Model rank 1**and** market rank 1 (optional trial flag) | Agreement filter with the favorite      |
| **D** | Model rank 1**and** `market_rank <= 2`                  | Softer market filter                    |

ROI and drawdown under these rules are **research scores** under HKJC-style win pricing and takeout. They are not a tip sheet and not a claim of profitable edge.

---

## 🔒 Integrity Rules (Modeling Layer)

1. **Banned training inputs**: Contemporaneous `placing`, margins, sectional finish times, and market odds features are excluded from `feature_cols` during fit.
2. **Labels vs features**: Relevance labels may be derived from `placing` for supervised ranking; those labels are targets, never covariates.
3. **Evaluation-only odds**: `win_odds` / `market_rank` join the panel for baselines and unit-stake PnL after scores are produced.
4. **Micro-tracing**: Optional pre-train audit verifies rolling statistics do not absorb the current race’s outcome for the same horse.

Upstream feature generators (`features/`) remain the first line of defense (Bayesian smoothers + explicit shifts). This package enforces the same contract at load and evaluation time.

---

## 🧪 Tests

```bash
pytest models/tests -q
```

Coverage focuses on data loading contracts, time splits, ranking metrics, registry construction, and wrapper fit/predict interfaces.

---

## 📦 Related Packages


| Package     | Role                                                          |
| ----------- | ------------------------------------------------------------- |
| `features/` | Leakage-aware feature generators &`FeaturesPipeline`          |
| `database/` | ORM models, indexes,`load_all_merged_race_data`               |
| `config/`   | `settings.json`, banned feature lists, model defaults         |
| `cli.py`    | Unified entry for scrape → features → train → walk-forward |

---

## ⚠️ Scope & Research Note

**In scope**

- Reproducible OOT / walk-forward evaluation
- Market-baseline comparison and diagnostic reporting
- Abstract, fixed-stake policy definitions for methodology research

**Out of scope**

- Live betting execution, bankroll management for real money, or guaranteed returns
- Publishing full bet-level prediction dumps as a public “system tip” product
- Treating takeout-heavy win markets as easy alpha

A complete engineering stack does not imply economic edge. When sharing results, report **market baselines alongside model metrics**, and prefer summary tables over actionable bet lists.

---

## License / Use

Intended for education, methodology development, and reproducible quantitative research on HKJC historical data. Users are responsible for compliance with local law and HKJC terms when accessing data or placing wagers.

```

```
