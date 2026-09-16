# Model Comparison — PPD Detection

## Current: NHANES — real patients, labs + questionnaire on the same woman

8 NHANES cycles pooled (2005–2023). Target is PHQ-9 ≥ 10; **every predictor comes from a different
instrument** (demographics, reproductive history, blood work, BMI, sleep), so the label is not a
function of the features. 5-fold CV, bootstrap CI.

| Rank | Population | Model | n | Prevalence | AUC | 95% CI | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| 🏆 **1 — DOMINANT** | **All women 18–49** | **Gradient Boosting** | **11,104** | 11.8% | **0.713** | **[0.699, 0.728]** | 0.591 | 0.734 |
| 2 | All women 18–49 | Random Forest | 11,104 | 11.8% | 0.711 | [0.697, 0.726] | 0.564 | 0.751 |
| 3 | All women 18–49 | Logistic Regression | 11,104 | 11.8% | 0.666 | [0.651, 0.681] | 0.517 | 0.722 |
| 4 | All women 18–49 | Baseline | 11,104 | 11.8% | 0.500 | — | — | — |
| **1 (perinatal)** | **Pregnant or ≤24m postpartum** | **Random Forest** | **1,467** | 8.8% | **0.661** | [0.616, 0.708] | 0.814 | 0.458 |
| 2 | Perinatal only | Logistic Regression | 1,467 | 8.8% | 0.656 | [0.605, 0.701] | 0.783 | 0.506 |
| 3 | Perinatal only | Gradient Boosting | 1,467 | 8.8% | 0.645 | [0.601, 0.690] | 0.752 | 0.506 |

**Why this is the real one:** prevalence 8.8–11.8% matches clinical reality (the 14k survey set claimed 64.8%).
Cohort sizes: 11,104 reproductive-age women; 1,467 perinatal; 510 within 12 months postpartum.
Lab coverage in the perinatal group — haemoglobin 1,396, BMI 1,465, CRP 410.

**Honest limits:** AUC 0.66–0.71 is what non-symptom predictors genuinely achieve; published PPD risk
models sit at 0.70–0.80. The perinatal subgroup has ~129 cases, so its CI is wide. Survey weights are
not applied (fine for prediction, not for national prevalence estimates).

---

## Previous datasets

**14k survey set** (`phq9_perinatal.csv`) — label is exactly `sum(9 PHQ items) ≥ 10`, half the rows are
duplicates. Only legitimate use is short-form derivation: best 4 items (Sleep + Concentration + Slowness +
Self-harm) reach AUC 0.860 vs 0.719 for standard PHQ-2. See `results/short_form_comparison.csv`.

**Kaggle set** (`post natal data.csv`) — 1,168 rows collapse to 248 distinct questionnaires; best honest
AUC 0.730. See `results/baseline_improved_summary.csv`.

## Files

- `nhanes_build.py` → downloads NHANES, writes `nhanes_perinatal.csv`
- `nhanes_model.py` → the table above → `results/nhanes_model_comparison.csv`
- `ppd_short_form.py`, `ppd_improved_baselines.py`, `ppd_preprocessing.py` — earlier datasets
