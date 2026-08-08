# 🛠️ HKJC Data Scraper Module (`scraper`)

The `scraper` module serves as the primary **Data Ingestion Layer** for the `hkjc-quant-model` system. It is engineered to perform high-throughput, structured, and resilient asynchronous data extraction from the Hong Kong Jockey Club (HKJC) official portal. It covers race calendars, detailed race results, sectional times with running positions, horse profiles, trackwork, and barrier trial records, feeding clean, raw historical contexts directly into downstream cleaning pipelines and database storage.

---

## 📌 Architectural Overview

In quantitative horse racing analysis, missing data or leakage directly degrades ranker predictive capacity. The `scraper` package implements an **AsyncIO-based Decoupled Pipeline & Parser Architecture**, ensuring high concurrency, rate control, and clean separation of concerns.

### Core Features

* **Modular Multi-Pipeline Architecture**: Independent AsyncIO pipelines (`RaceScrapingPipeline`, `HorseScrapingPipeline`, `TrackworkScrapingPipeline`, `TrailScrapingPipeline`) support both batch backfills and incremental updates.
* **High-Concurrency Non-Blocking I/O**: Built on `aiohttp` and non-blocking event loops, dramatically minimizing network I/O wait times.
* **Pure Parser Layer**: Network fetching (`Hook`) and DOM parsing (`selectolax`) are completely decoupled. Parsers are side-effect-free pure functions, allowing fast unit testing on cached offline raw HTML/JSON.
* **Unified Persistence (`DataManager`)**: Handles asynchronous file persistence and local deduplication checks before firing remote HTTP requests.
* **Strict Error Context & Exception Tracing**: Deep exception loggers intercept network, HTTP, or parser failures and log exact line numbers without breaking batch event loops.

---

## 🏗️ Architecture & Design Patterns

```
                                ┌─────────────────────────┐
                                │   HKJC Official Web     │
                                └────────────┬────────────┘
                                             │ HTTP (aiohttp via Hook)
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                Pipeline Layer                                          │
│  ( race_pipeline.py / horse_pipeline.py / trackwork_pipeline.py / trail_pipeline.py )  │
│  - Concurrency Control (asyncio.Semaphore)                                             │
│  - Local Deduplication Checks (DataManager)                                            │
│  - Exception Isolation & Deep Tracing                                                  │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                        │ Raw HTML Trees / JSON Responses
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 Parser Layer                                    │
│                            ( scraper/parser/* )                                 │
│  - Pure Functions (No I/O Side Effects)                                         │
│  - selectolax HTML Parser / RegEx / JSON Extraction                             │
│  - Data Schema Normalization                                                    │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ Clean Standardized Dictionaries
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  Data Manager                                   │
│                              ( data_manager.py )                                │
│  - Asynchronous Persistence (`aiofiles`)                                        │
│  - Structured Parquet / JSON Storage (`data/raw_json/`)                         │
└─────────────────────────────────────────────────────────────────────────────────┘

```

### 🧩 `parser/` Module Division of Responsibilities


| Parser File          | Target Data Source | Responsibility Description                                                  |
| -------------------- | ------------------ | --------------------------------------------------------------------------- |
| `calander_parser.py` | Fixture Calendar   | Extracts all valid race dates across specified years/months asynchronously. |

|
| `result_parser.py` | Local Results (`localresults`) | Extracts race-level metadata (venue, track, distance) and horse-level outcome records (placing, jockey, trainer, weight, odds).

|
| `sectional_parser.py` | Sectional Time (`displaysectionaltime`) | Parses sub-sectional split times, running positions, and margins behind.

|
| `horse_parser.py` | Horse Profile (`horse`) | Extracts background profiles (origin, age, color, sex, import date, dam, sire, damsire, total stakes).

|
| `rating_parser.py` | Horse Rating Index | Extracts current horse ratings and class ratings.

|
| `trackwork_parser.py` | Trackwork API (`TrackworkOneDayRecords`) | Parses daily trackwork JSON records (fast gallops, trot, swim, rider details).

|
| `trail_parser.py` | Barrier Trials (`btresult`) | Parses trial group metadata, track types, finish times, sectional split times, and performance comments.

|

---

## 🚀 Quick Start & Usage

The scraper can be driven via the command-line interface (`cli.py`) or imported directly as a Python package.

### 1. Execute Race & Sectional Scraper (CLI)

```bash
# Interactively scrape races for specified years
python cli.py --scrape-races --start-year 2025 --end-year 2026

# Perform cleaning and write parsed results to SQLite database
python cli.py --clean-races

```

### 2. Programmatic Asynchronous Ingestion

#### Fetching Race Results & Sectionals

```python
import asyncio
from scraper.race_pipeline import RaceScrapingPipeline

async def main():
    # Initialize pipeline with a max concurrency limit of 10 days
    pipeline = RaceScrapingPipeline(max_concurrent_days=10)
  
    # Run async scraping for target years
    print("Starting race and sectional scraping...")
    await pipeline.run(years=[2025, 2026])

if __name__ == "__main__":
    asyncio.run(main())

```

#### Fetching Pending Horse Profiles

```python
import asyncio
from database.db_manager import DBManager
from scraper.horse_pipeline import HorseScrapingPipeline

async def main():
    db = DBManager()
  
    # Query database for horse IDs present in race_results but missing in horses table
    pending_ids = db.get_pending_horse_ids()
    print(f"Found {len(pending_ids)} pending horse profiles to fetch.")

    if pending_ids:
        pipeline = HorseScrapingPipeline(max_concurrent=10)
        await pipeline.run(pending_ids)

if __name__ == "__main__":
    asyncio.run(main())

```

#### Fetching Trackwork & Barrier Trial Data

```python
import asyncio
from scraper.trackwork_pipeline import TrackworkScrapingPipeline
from scraper.trail_pipeline import TrailScrapingPipeline

async def main():
    # Scrape trackwork records for 2026
    trackwork_pipe = TrackworkScrapingPipeline(max_concurrent_days=5)
    await trackwork_pipe.run(start_year=2026, end_year=2026)

    # Scrape barrier trial records for 2026
    trail_pipe = TrailScrapingPipeline(max_concurrent_days=10)
    await trail_pipe.run(start_year=2026, end_year=2026)

if __name__ == "__main__":
    asyncio.run(main())

```

---

## 📂 File Directory & Responsibilities

```text
scraper/
├── parser/
│   ├── __init__.py
│   ├── calander_parser.py     # HTML Parser: Fixture race calendars
│   ├── horse_parser.py        # HTML Parser: Horse profiles & pedigrees
│   ├── rating_parser.py       # HTML Parser: Horse ratings
│   ├── result_parser.py       # HTML Parser: Detailed race results
│   ├── sectional_parser.py    # HTML Parser: Sectional times & running positions
│   ├── trackwork_parser.py    # JSON Parser: Daily trackwork records
│   └── trail_parser.py        # HTML Parser: Barrier trial group & horse details
├── __init__.py
├── data_manager.py            # Local disk cache & asynchronous file IO
├── hook.py                    # Low-level HTTP Client Session wrapper (aiohttp + selectolax)
├── horse_pipeline.py          # Async pipeline for horse profile ingestion
├── race_pipeline.py           # Async pipeline for race results & sectional ingestion
├── trackwork_pipeline.py      # Async pipeline for daily trackwork JSON ingestion
└── trail_pipeline.py          # Async pipeline for barrier trial ingestion

```

---

## ⚠️ Anti-Scraping Policies & Best Practices

1. **Concurrency Rate Control (`Semaphore`)**:

* Official HKJC servers monitor request frequency. Exceeding rate limits triggers temporary IP bans or `HTTP 403 / 429` errors.
* Recommended Semaphore settings:
* `RaceScrapingPipeline`: **10–20** concurrent days.
* `HorseScrapingPipeline`: **5–10** concurrent workers.
* `TrackworkScrapingPipeline`: **5** concurrent days.

2. **Jitter Delay Simulation**:

* The underlying `Hook._fetch()` automatically injects randomized delays (`0.5s ~ 1.8s`) per request using `asyncio.sleep()`, effectively mimicking human browsing patterns.

3. **Pipeline Dependency Order**:
   To ensure complete database integrity, run the pipelines in the strict sequential order established by the system CLI:
4. **`RaceScrapingPipeline`** $\rightarrow$ Raw Race & Sectional JSON
5. **`RaceCleaner` / `SectionalCleaner**` $\rightarrow$ Populate `races` and `race_results` tables in SQLite
6. **`HorseScrapingPipeline`** $\rightarrow$ Extract pending `horse_id`s from DB and scrape profiles
7. **`HorseCleaner`** $\rightarrow$ Populate `horses` table in SQLite
8. **`TrackworkScrapingPipeline` / `TrailScrapingPipeline**` $\rightarrow$ Scrape exercise context
9. **`TrackworkCleaner` / `TrailsCleaner**` $\rightarrow$ Populate `race_trackwork`, `trials`, and `trial_results`
