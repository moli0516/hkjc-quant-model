# 🐎 HKJC Quant: Database Data Dictionary (`database/`)

This data dictionary defines the core relational database schema for the Hong Kong Jockey Club (HKJC) quantitative racing system. The database aggregates raw ingestion files into normalized relational tables optimized for historical performance querying, feature engineering pipelines, and machine learning inference.

---

## 📋 Table Summary

1. [`races`](#1-races-race-metadata) - Core race fixture metadata (Date, Venue, Class, Distance, Track conditions).
2. [`race_results`](#2-race_results-race-results--horse-performance) - Finishing results, jockeys, trainers, weights, odds, and final timing.
3. [`race_sectionals`](#3-race_sectionals-sectional-times--running-positions) - In-race running positions and sectional split times[cite: 6].
4. [`horses`](#4-horses-horse-registry-profiles) - Comprehensive equine registry profiles, import records, and rating histories.
5. [`race_trackwork`](#5-race_trackwork-morning-trackwork-records) - Daily morning trackwork, gallop trials, and training logs.
6. [`trials`](#6-trials-barrier-trial-metadata) - Barrier trial session metadata and split times.
7. [`trial_results`](#7-trial_results-barrier-trial-horse-results) - Individual equine performance outcomes during barrier trials.

---

## 1. `races` (Race Metadata)

Records essential metadata for each race fixture, including date, venue, class, distance, and track configuration[cite: 6].


| Column Name           | Data Type | Primary Key | Nullable | Description                         | Example / Notes                                |
| :-------------------- | :-------- | :---------: | :------: | :---------------------------------- | :--------------------------------------------- |
| **`race_id`**         | `TEXT`    | Primary Key |    No    | Unique race identifier              | `2024-01-01_ST_1` (`{date}_{venue}_{race_no}`) |
| **`date`**            | `TEXT`    |      -      |    No    | Race meeting date                   | `2024-01-01` (Format: YYYY-MM-DD)              |
| **`venue`**           | `TEXT`    |      -      |    No    | Racecourse venue                    | `ST` (Sha Tin) / `HV` (Happy Valley)           |
| **`race_no`**         | `INTEGER` |      -      |    No    | Race number within the fixture      | `1`, `2`, `3` ...                              |
| **`race_class`**      | `TEXT`    |      -      |   Yes   | Race class classification           | `1`, `2`, `3`, `4`, `5`, `G1`, `G2`, `G3`      |
| **`distance`**        | `INTEGER` |      -      |   Yes   | Race distance in meters             | `1200`, `1400`, `1600`, `2000`                 |
| **`track_condition`** | `TEXT`    |      -      |   Yes   | Track surface condition description | `Good`, `Firm`, `Yielding`, `Slow`             |
| **`track_texture`**   | `TEXT`    |      -      |   Yes   | Track texture / surface type        | `TURF`, `AWT` (All Weather Track)              |
| **`track_type`**      | `TEXT`    |      -      |   Yes   | Course variant or layout direction  | `A`, `B`, `C+3`, `N/A` (for AWT)               |

---

## 2. `race_results` (Race Results & Horse Performance)

Captures individual horse performances, final placings, jockey/trainer allocations, carried weights, odds, and finish times[cite: 6].


| Column Name           | Data Type | Foreign Key | Nullable | Description                             | Example / Notes                              |
| :-------------------- | :-------- | :----------: | :------: | :-------------------------------------- | :------------------------------------------- |
| **`result_id`**       | `INTEGER` | Primary Key |    No    | Auto-incremented primary key            | `1`, `2`, `3` ...                            |
| **`race_id`**         | `TEXT`    | FK (`races`) |    No    | Corresponding race identifier           | `2024-01-01_ST_1`                            |
| **`horse_id`**        | `TEXT`    |      -      |   Yes   | Equine branding / registration code     | `E123`, `G045`                               |
| **`horse_name`**      | `TEXT`    |      -      |    No    | Registered equine name                  | `Golden Sixty`                               |
| **`placing`**         | `INTEGER` |      -      |   Yes   | Final finishing order position          | `1`, `2`, `3` ... (Null for scratchings/DNF) |
| **`draw`**            | `INTEGER` |      -      |   Yes   | Starting barrier gate position          | `1` ~ `14`                                   |
| **`jockey`**          | `TEXT`    |      -      |   Yes   | Assigned jockey name                    | `Z. Purton`, `K. C. Leung`                   |
| **`trainer`**         | `TEXT`    |      -      |   Yes   | Licensed stable trainer name            | `F. C. Lor`, `A. S. Cruz`                    |
| **`actual_weight`**   | `FLOAT`   |      -      |   Yes   | Actual carried weight (lbs)             | `123.0`, `133.0`                             |
| **`declared_weight`** | `FLOAT`   |      -      |   Yes   | Body weight declared at race eve        | `1120.0`                                     |
| **`win_odds`**        | `FLOAT`   |      -      |   Yes   | Final Pari-Mutuel Win odds              | `2.5`, `15.0`                                |
| **`finish_time_sec`** | `FLOAT`   |      -      |   Yes   | Total finish time in seconds            | `70.12` (equivalent to 1:10.12)              |
| **`margin_len`**      | `FLOAT`   |      -      |   Yes   | Margin behind winner (in horse lengths) | `0.0` (Winner), `0.25` (Neck), `1.5`         |
| **`rating`**          | `INTEGER` |      -      |   Yes   | Official handicap rating at race time   | `60.0`, `85.0`                               |

---

## 3. `race_sectionals` (Sectional Times & Running Positions)

Logs sectional running positions, incremental split times, and margin deficits during race progression[cite: 6].


| Column Name              | Data Type | Foreign Key | Nullable | Description                           | Example / Notes                |
| :----------------------- | :-------- | :----------: | :------: | :------------------------------------ | :----------------------------- |
| **`sec_id`**             | `INTEGER` | Primary Key |    No    | Auto-incremented primary key          | `1`, `2`, `3` ...              |
| **`race_id`**            | `TEXT`    | FK (`races`) |    No    | Corresponding race identifier         | `2024-01-01_ST_1`              |
| **`horse_id`**           | `TEXT`    |      -      |   Yes   | Equine branding / registration code   | `E123`                         |
| **`horse_name`**         | `TEXT`    |      -      |    No    | Registered equine name                | `Golden Sixty`                 |
| **`section_no`**         | `INTEGER` |      -      |    No    | Sectional checkpoint index (1-based)  | `1`, `2`, `3`, `4`             |
| **`position`**           | `INTEGER` |      -      |   Yes   | Running position at checkpoint        | `1` (Leader), `5` (Midfield)   |
| **`sectional_time_sec`** | `FLOAT`   |      -      |   Yes   | Elapsed time for the specific section | `22.88`, `23.15`               |
| **`margin_behind`**      | `TEXT`    |      -      |   Yes   | Distance deficit behind leader/prev   | `1-1/2` (Length string format) |

---

## 4. `horses` (Horse Registry Profiles)

Maintains master equine metadata, pedigree structures, import milestones, and career stakes summaries.


| Column Name          | Data Type | Primary Key | Nullable | Description                            | Example / Notes            |
| :------------------- | :-------- | :---------: | :------: | :------------------------------------- | :------------------------- |
| **`horse_code`**     | `TEXT`    | Primary Key |    No    | Unique equine registration code        | `E123`, `G045`             |
| **`origin`**         | `TEXT`    |      -      |   Yes   | Country of origin                      | `AUS`, `NZ`, `IRE`         |
| **`age`**            | `INTEGER` |      -      |   Yes   | Current equine age                     | `3`, `4`, `5`              |
| **`color`**          | `TEXT`    |      -      |   Yes   | Coat color description                 | `Bay`, `Chestnut`          |
| **`sex`**            | `TEXT`    |      -      |   Yes   | Equine sex / gelding status            | `Gelding`, `Colt`, `Horse` |
| **`import_type`**    | `TEXT`    |      -      |   Yes   | Import classification category         | `PP`, `ISG`                |
| **`season_stakes`**  | `FLOAT`   |      -      |   Yes   | Cumulative prize money won this season | `2650450.0`                |
| **`total_stakes`**   | `FLOAT`   |      -      |   Yes   | Lifetime cumulative prize money        | `120000000.0`              |
| **`wins`**           | `INTEGER` |      -      |   Yes   | Total career wins                      | `15`                       |
| **`seconds`**        | `INTEGER` |      -      |   Yes   | Total career second placings           | `3`                        |
| **`thirds`**         | `INTEGER` |      -      |   Yes   | Total career third placings            | `2`                        |
| **`total_runs`**     | `INTEGER` |      -      |   Yes   | Total career starts                    | `25`                       |
| **`import_date`**    | `DATE`    |      -      |   Yes   | Date imported into Hong Kong           | `2023-05-12`               |
| **`trainer`**        | `TEXT`    |      -      |   Yes   | Current assigned stable trainer        | `C. S. Shum`               |
| **`current_rating`** | `INTEGER` |      -      |   Yes   | Current official handicap rating       | `131`                      |
| **`sire`**           | `TEXT`    |      -      |   Yes   | Sire (paternal bloodline)              | `Medaglia d'Oro`           |
| **`dam`**            | `TEXT`    |      -      |   Yes   | Dam (maternal bloodline)               | `Gaudeamus`                |

---

## 5. `race_trackwork` (Morning Trackwork Records)

Logs daily morning workouts, fast gallops, trial gallops, and exercise observations.


| Column Name           | Data Type | Primary Key | Nullable | Description                          | Example / Notes                        |
| :-------------------- | :-------- | :---------: | :------: | :----------------------------------- | :------------------------------------- |
| **`trackwork_id`**    | `INTEGER` | Primary Key |    No    | Auto-incremented primary key         | `1`, `2`, `3` ...                      |
| **`horse_id`**        | `TEXT`    |      -      |    No    | Equine branding code                 | `E123`                                 |
| **`work_date`**       | `TEXT`    |      -      |    No    | Morning workout calendar date        | `2026-05-15`                           |
| **`work_type`**       | `TEXT`    |      -      |   Yes   | Workout category type                | `Fast Work`, `Barrier Trial`, `Pacing` |
| **`distance`**        | `INTEGER` |      -      |   Yes   | Distance covered in workout (meters) | `1200`, `800`                          |
| **`finish_time_sec`** | `FLOAT`   |      -      |   Yes   | Total elapsed workout time (seconds) | `75.4`                                 |
| **`rider`**           | `TEXT`    |      -      |   Yes   | Exercise rider or jockey name        | `Assistant Trainer`, `Z. Purton`       |
| **`remark`**          | `TEXT`    |      -      |   Yes   | Trackwork descriptive comment        | `Moved smoothly through trial`         |

---

## 6. `trials` (Barrier Trial Metadata)

Stores metadata for official barrier trial heats conducted at Sha Tin or Conghua.


| Column Name           | Data Type | Primary Key | Nullable | Description                         | Example / Notes                        |
| :-------------------- | :-------- | :---------: | :------: | :---------------------------------- | :------------------------------------- |
| **`trial_id`**        | `TEXT`    | Primary Key |    No    | Unique trial session identifier     | `2026-05-10_G1` (`{date}_G{group_no}`) |
| **`date`**            | `TEXT`    |      -      |    No    | Barrier trial date                  | `2026-05-10`                           |
| **`group_no`**        | `INTEGER` |      -      |    No    | Heat / Group number                 | `1`, `2`, `3`                          |
| **`venue`**           | `TEXT`    |      -      |   Yes   | Trial training venue                | `Sha Tin`, `Conghua`                   |
| **`track_type`**      | `TEXT`    |      -      |   Yes   | Surface type                        | `Turf`, `All Weather Track`            |
| **`distance`**        | `INTEGER` |      -      |   Yes   | Trial heat distance (meters)        | `1000`, `1200`                         |
| **`finish_time_sec`** | `FLOAT`   |      -      |   Yes   | Overall heat winning time (seconds) | `58.2`                                 |

---

## 7. `trial_results` (Barrier Trial Horse Results)

Records individual participant performances, trial placings, and performance assessments during barrier trials.


| Column Name               | Data Type |  Foreign Key  | Nullable | Description                         | Example / Notes                      |
| :------------------------ | :-------- | :-----------: | :------: | :---------------------------------- | :----------------------------------- |
| **`result_id`**           | `INTEGER` |  Primary Key  |    No    | Auto-incremented primary key        | `1`, `2`, `3` ...                    |
| **`trial_id`**            | `TEXT`    | FK (`trials`) |    No    | Corresponding trial identifier      | `2026-05-10_G1`                      |
| **`horse_id`**            | `TEXT`    |       -       |    No    | Equine registration code            | `E123`                               |
| **`placing`**             | `INTEGER` |       -       |   Yes   | Trial heat finishing position       | `1`, `2`, `3`                        |
| **`margin_len`**          | `FLOAT`   |       -       |   Yes   | Margin behind heat winner (lengths) | `0.0`, `1.5`, `3.75`                 |
| **`result_remark`**       | `TEXT`    |       -       |   Yes   | Official qualification status       | `Pass` (`及格`), `Fail` (`不及格`)   |
| **`performance_comment`** | `TEXT`    |       -       |   Yes   | Evaluator observation notes         | `Finished strongly, full of running` |

---

## 🔗 Entity Relationship Architecture

```text
 [ races ] (1)
    ├───< (N) [ race_results ]     (Linked via race_id)
    ├───< (N) [ race_sectionals ]  (Linked via race_id)
  
 [ horses ] (1)
    └───< (N) [ race_results ]     (Linked via horse_code = horse_id)
    └───< (N) [ race_trackwork ]   (Linked via horse_code = horse_id)

 [ trials ] (1)
    └───< (N) [ trial_results ]    (Linked via trial_id)
```


```
