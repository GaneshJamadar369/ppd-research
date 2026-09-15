# Model Comparison — PPD Detection

## Current: Short-form screener (`phq9_perinatal.csv`)

7,008 unique rows (file had 7,000 exact duplicates), 64.8% labelled depressed (PHQ-9 ≥ 10).
Item subsets and cut-offs chosen on the training split; all numbers below are on a held-out 20% test split.

| Rank | Model | Questions | AUC | 95% CI | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| 🏆 **1 — DOMINANT** | **4 items: Sleep + Concentration + Slowness + Self-harm (score ≥ 5)** | **4** | **0.860** | **[0.841, 0.881]** | 0.728 | 0.846 | 0.804 |
| 2 | 3 items + sociodemographics (GBM) | 3 | 0.819 | [0.796, 0.841] | 0.727 | 0.763 | 0.783 |
| 3 | 3 items: Sleep + Slowness + Self-harm (score ≥ 4) | 3 | 0.818 | [0.796, 0.841] | 0.655 | 0.840 | 0.752 |
| 4 | 2 items: Slowness + Self-harm (score ≥ 3) | 2 | 0.755 | [0.730, 0.780] | 0.580 | 0.836 | 0.695 |
| 5 | PHQ-2 standard (Interest + Down, score ≥ 3) | 2 | 0.719 | [0.692, 0.746] | 0.708 | 0.603 | 0.736 |
| 6 | Sociodemographics only (no questions) | 0 | 0.561 | [0.527, 0.593] | 0.567 | 0.543 | 0.625 |

**Winner: 4-item sum score, AUC 0.860.** Asks less than half the PHQ-9 and needs no model — just add four answers.
Adding sociodemographics to 3 items adds nothing (0.819 vs 0.818).

**Why not use all 9 items?** The label is defined as the 9-item sum ≥ 10, so a 9-item model is 100% by arithmetic and meaningless.

**Caution on this dataset:**
- A 64.8% depression rate is far above typical perinatal prevalence (~10–20%).
- In real clinical data the standard PHQ-2 (Interest + Down) is usually the strongest pair; here it ranks 5th.

Both suggest the data may not reflect a real clinical population. Validate on independent data (e.g. PRAMS) before trusting the item selection.

---

## Previous: Kaggle `post natal data.csv` — discarded

1,168 rows collapse to 248 distinct questionnaires; best honest AUC 0.730 (Random Forest). See `results/baseline_improved_summary.csv`.

## Files

- `phq9_perinatal.csv` — current dataset
- `ppd_short_form.py` — short-form comparison above → `results/short_form_comparison.csv`
- `post natal data.csv`, `ppd_preprocessing.py`, `ppd_improved_baselines.py` — previous dataset
