# HKJC Quant Model — 研究日記

**專案：** hkjc-quant-model
**作者：** Moli
**基線 run：** `wf_20260810_205259`
**資料範圍：** 賽事約 2020–2026；試閘約 2019–2026
**最後更新：** 2026-08-11

---

## 1. 研究目標（本階段）

在**不重訓**的前提下，用 walk-forward 預測做：

1. 模型排序 vs 市場大熱的能力對照
2. 固定注碼下注規則的 ROI / hit / drawdown
3. 試閘相關過濾是否構成**相對**改善（非宣稱正期望）
4. 規則與評估流程模組化，便於可復現實驗

**非目標：** 本階段不追求上線實盤；不把負 ROI 規則包裝成「已驗證 alpha」。

---

## 2. 方法摘要

### 2.1 Walk-forward

- 模型：`xgb_ranker`（既有超參）
- 訓練窗：expanding；`min_train_days=730`，`step_days=30`
- 特徵：排除賠率類 banned features
- 標籤：名次相關 ranking

### 2.2 冷儲存

- 路徑：`data/predictions/wf_20260810_205259`
- 內容：全量 OOS predictions（約 47,488 runners / 3,917 races）
- 後續規則與診斷一律走 **CLI 選單 15 離線評估**，避免重訓造成不可比。

### 2.3 規則架構（Registry）

- 介面：`BettingRule.select()`
- 目錄：`RuleRegistry` + `rule_id` 穩定契約
- 執行：`BetEvaluator.run_many()` → `report["rules"]`
- 報告順序：`DEFAULT_REPORT_RULE_IDS`（與規則 class 分離）

原則：**註冊 ≠ 發表**；新假說加 class + 可選進報告列表，不覆蓋舊規則。

### 2.4 代號表


| ID | 名稱                       | 條件（概念）                    |
| -- | -------------------------- | ------------------------------- |
| M0 | market_top1                | market_rank == 1                |
| A0 | model_top1                 | model_rank == 1                 |
| B0 | overlay_value              | model_prob 相對 market 超閾值等 |
| C0 | same_pick                  | model 與 market 同為第 1        |
| C1 | same_pick_strong_trial     | C0 + is_strong_trial == 1       |
| C2 | same_pick_weak_trial       | C0 + is_strong_trial == 0       |
| C3 | same_pick_fresh_trial      | C0 + is_fresh_trial_7_28d == 1  |
| C4 | same_pick_not_fresh_trial  | C0 + not fresh                  |
| D0 | model_top1_market_top2     | model_rank==1 且 market_rank≤2 |
| E0 | same_pick_strong_and_fresh | C0 + strong + fresh             |

---

## 3. 基線結果（run `wf_20260810_205259`）

### 3.1 排序能力


| 指標      | Model  | Market     |
| --------- | ------ | ---------- |
| Top-1 hit | 0.2484 | **0.3071** |
| Top-3 hit | 0.4493 | **0.5042** |
| n_races   | 3917   | 3917       |
| n_runners | 47488  | 47488      |

**結論：** 純排序上模型未勝市場大熱。

### 3.2 主要規則（flat stake = 1）


| ID | n_bets | hit_rate   | ROI         | max_dd |
| -- | ------ | ---------- | ----------- | ------ |
| M0 | 3917   | 0.3071     | −0.160     | −634  |
| A0 | 3917   | 0.2484     | −0.180     | −711  |
| B0 | 2212   | 0.0547     | −0.166     | −466  |
| C1 | 1341   | 0.3803     | **−0.107** | −151  |
| C2 | 583    | 0.3293     | −0.124     | −77   |
| C3 | 863    | 0.3859     | −0.109     | −94   |
| C4 | 1061   | 0.3478     | −0.115     | −136  |
| D0 | 2729   | 0.3144     | −0.127     | −353  |
| E0 | 644    | **0.3975** | −0.111     | −72   |

**共同結論：** 全部規則 ROI 為負；市場抽水下無穩定正 edge。

---

## 4. 假說與實驗記錄

### H1 — 同選 + 強試閘 相對 A0 是否跨 fold 更穩？

- **Treatment：** 等價 C1（model_rank==1 ∧ market_rank==1 ∧ strong_trial）
- **Control：** A0
- **門檻：** 有效 fold 需 n_A≥50、n_C≥20；支援比例 ≥ 2/3
- **結果：** `decision: support`
  - n_valid_folds = 42
  - frac (hit_C > hit_A) ≈ **97.6%**
  - overall hit A/C/M ≈ 0.248 / **0.380** / 0.307
  - overall ROI A/C/M ≈ −0.180 / **−0.107** / −0.160

**解讀：** C1 相對 A0 在 hit 上跨 fold 穩定較優，且少輸；**不是**正 ROI 驗證。

### H2（探索）— Fresh trial


| 對照     | 結果                                       |
| -------- | ------------------------------------------ |
| C3 vs C4 | fresh 優於 not-fresh（hit 0.386 vs 0.348） |
| C3 vs C1 | 接近；C3 未明顯贏 C1 的 ROI                |

### H3（探索）— 條件交集 E0

- E0 = strong ∧ fresh（在同選下）
- hit 升至 ≈0.398，n 降至 644
- ROI ≈ −0.111，**未優於 C1（−0.107）**

**解讀：** 堆疊試閘條件提高 hit、減少注數，但未改善資金效率；停止在同一維度繼續 AND。

---

## 5. 診斷摘要（同一 run）

1. **大熱 vs 模型第一：** 大熱 hit/ROI 皆優於盲信 model top1。
2. **模型第一賠率畫像：** 低賠率段 hit 較高但仍難覆蓋；高賠率段崩壞。
3. **同意／不同意大熱：** same_pick 時 model top1 = market top1（定義使然）；disagree 時模型明顯較差。
4. **試閘 residual（fav odds 約 2–5）：** strong / close / fresh 旗標為 1 時 win_rate 略高於 0，屬弱方向訊號，非單獨可交易 edge。
5. 在同選大熱家族中，強試閘（C1）、fresh（C3）、兩者交集（E0）相對盲信模型第一（A0）均通過 fold 穩定性門檻；以有效 fold 數與 ROI 綜合，C1 仍為首選用法描述。E0 提高 hit、減少注數，但未改善 ROI。

---

## 6. 工程進度（本階段完成）

- [X]  Walk-forward 評估器（OOP）
- [X]  Predictions 冷儲存 + 離線評估（CLI 15）
- [X]  Rules 模組：`base` / `definitions` / `registry` + `run_many`
- [X]  報告只依賴 `report["rules"]` + `default_report_ids()`
- [X]  H1 穩定性診斷輸出
- [X]  C3 / C4 / E0 註冊與離線復現

**已知小差異：** B0 注數與更早版本（約 2689）不完全一致（現約 2212），不影響 C 族結論；待有空對齊 `RuleB0` 與舊 `rule_b`。

---

## 7. 結論（截至 2026-08-11）

1. **模型排序 < 市場大熱**（本設定、本樣本）。
2. **無穩定正 ROI 規則**；最佳敘事是「相對少輸的用法」，不是 alpha 產品化。
3. **相對最佳用法描述：** **C1（同選大熱 + 強試閘）** — 相對 A0 跨 fold 穩定、ROI 最好之一。
4. **Fresh（C3）** 優於 not-fresh（C4），與 C1 接近，未取代 C1。
5. **E0 交集** 提高 hit、降低 n，ROI 無增益 → 不採納為主規則。
6. 實驗流程（冷儲存 + registry）已可支撐後續假說，無需每次重訓。

---

## 8. 下一步（候選，未執行）


| 優先 | 項目         | 說明                                                |
| ---- | ------------ | --------------------------------------------------- |
| P1   | 停刷試閘 AND | 改正交維度或收工寫短報告                            |
| P2   | H1 泛化      | treatment/control 改為`rule_id` 參數（C3 vs A0 等） |
| P3   | 賠率帶 × C1 | 預先登記假說，避免掃描                              |
| P4   | 對齊 B0      | 與歷史 rule_b 注數一致                              |
| P5   | 學術敘事     | Track A CV 提案與本量化實驗嚴格分軌                 |

---

## 9. 復現指令

```text
# 互動
CLI 選單 15 → run_id: wf_20260810_205259

# 或
python cli.py --eval-store --run-id wf_20260810_205259
```

修改規則後：**重開 Python 行程**（避免舊 bytecode），再跑離線評估。

---

## 10. 變更日誌


| 日期       | 記事                                                               |
| ---------- | ------------------------------------------------------------------ |
| 2026-08-10 | WF 完成並冷儲存`wf_20260810_205259`；診斷 + H1 support             |
| 2026-08-10 | Rules registry 模組化；offline`report["rules"]` 接通               |
| 2026-08-11 | 新增 C3/C4（fresh 對照）、E0（strong∧fresh）；確認堆疊無 ROI 增益 |
| 2026-08-11 | 本日記初稿                                                         |

---

*本日記記錄的是研究過程與負結果，可公開學術討論；實盤下注決策不在此文件範圍。*
