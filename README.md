# Postpartum Depression — Structured Branch Baselines & Data-Quality Audit

Classical ML baselines (Logistic Regression, SVM, Random Forest, ANN) on the widely cited
Kaggle **"PostPartum Depression"** survey (`post natal data.csv`, 1,503 responses), together
with a data-quality audit of that benchmark.

The audit is the main result: **the ~99% accuracies reported for this dataset are an artifact
of duplicate leakage**, and the dataset does not behave like 1,503 independent respondents.

---

## Headline findings

### 1. The benchmark has an effective sample size of ~248, not 1,503

The 9 items admit 14,580 possible response combinations. Drawing 1,168 independent
respondents from the dataset's own marginal distributions gives:

| | Observed | Simulated (independent respondents) |
|---|---|---|
| Unique response patterns | **248** | 1,080 |
| Patterns occurring exactly once | **1** | 999 |
| Max repeats of a single pattern | **33** | ~3 |

Independent sampling predicts ~999 singleton patterns; the file contains **one**. Inter-item
association is far too weak (Cramér's V mostly ≈0.2, max 0.51) to explain a gap that size.
The data is consistent with a small pool of templates resampled to inflate the row count.

### 2. Duplicate leakage inflates scores, and the inflation scales with model capacity

Identical data and models; only the cross-validation split differs.

| Model | ΔAUC | ΔF1 | ΔAccuracy |
|---|---|---|---|
| Logistic Regression | 0.063 | 0.044 | 0.025 |
| SVM (RBF) | 0.106 | **0.525** | 0.267 |
| Random Forest | 0.107 | 0.269 | 0.202 |
| ANN (MLP) | **0.178** | 0.291 | 0.235 |

Logistic regression barely moves because it cannot memorise. The apparent superiority of the
nonlinear models under a standard split is memorisation, not signal — so **published model
rankings on this dataset do not survive a leakage-controlled protocol**, and the flexible
models (including the ANN) are the ones most affected.

The confusion matrices make it concrete: the RBF-SVM goes from **4 false negatives to 318**.

### 3. Symptom–risk relationships are clinically inverted

| Item | Attempt rate by response level |
|---|---|
| Feeling sad or tearful | **No = 0.61**, Sometimes = 0.20, Yes = 0.32 |
| Problems concentrating | No = 0.45, **Often = 0.15**, Yes = 0.62 |
| Feeling of guilt | No = 0.47, **Maybe = 0.17**, Yes = 0.60 |

Respondents reporting *no* sadness show a ~3× higher attempt rate than those reporting
"sometimes." Five of eight symptom items are non-monotone in severity, and the middle option
is protective in four of them — a construct-validity failure, not a subtle interaction. Two
items (`Feeling anxious`, `Overeating or loss of appetite`) have no association with the
target at all (p ≈ 0.73/0.75, Cramér's V = 0.000).

**Recommendation:** treat this dataset as an audit object, not as training data for a
deployed screening model.

---

## Results

Target is `Suicide attempt` (Yes/No), the PPD severity proxy used in prior work;
`Not interested to say` (n=335) is dropped as non-response, leaving 1,168 rows at a 39.3%
positive rate.

**Protocol B — leakage-controlled grouped 5-fold CV (the defensible numbers):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | **0.784 ± 0.052** | 0.748 | 0.686 | **0.714** | 0.890 |
| SVM (RBF) | 0.724 ± 0.024 | **0.971** | 0.307 | 0.464 | **0.891** |
| Logistic Regression | 0.729 ± 0.067 | 0.669 | 0.653 | 0.651 | 0.778 |
| ANN (MLP) | 0.727 ± 0.048 | 0.654 | 0.674 | 0.661 | 0.808 |

Protocol A (standard stratified CV) reproduces the literature-style figures — SVM 0.991
accuracy, AUC 0.997 — and is reported alongside for comparison only.

At the default 0.5 threshold the SVM is unusable as a screener (recall 0.307, missing 318 of
459 at-risk mothers). Tuning the operating point on training folds only lifts recall to 0.862
and F1 to 0.745. **Operating-point choice matters more here than model choice.**

Full tables, ROC curves, confusion matrices and EDA figures are in [`results/`](results/),
with the preprocessing audit in [`results/preprocessing_report.md`](results/preprocessing_report.md)
and the model tables in [`results/comparison_table.md`](results/comparison_table.md).

---

## Method

**Preprocessing** ([`ppd_preprocessing.py`](ppd_preprocessing.py))

- Drops `Timestamp` (a submission clock — keeping it lets a model memorise respondents).
- Canonicalises the survey's inconsistent option wording. Notably `Not at all` and `No` in the
  appetite item are the same answer under two form labels, and are merged.
- Builds a binary target and drops non-response.
- Engineers a `Symptom severity score` (EPDS-style 0–14 composite), which ranks as the single
  most important feature.
- Offers three encodings: `onehot` (default — the middle levels matter non-monotonically, so
  no ordering is imposed), `ordinal`, and `hybrid`.
- Imputation and scaling are **not** applied here; they sit inside the modelling pipeline so
  they are fitted on training folds only and cannot leak.

**Evaluation** ([`ppd_model_comparison.py`](ppd_model_comparison.py))

Nested cross-validation — outer 5-fold scoring, inner 3-fold grid search on ROC-AUC — under
two protocols:

- **Protocol A**: standard stratified 5-fold CV. Comparable to published work; duplicated
  respondents span train and test, so it is optimistically biased.
- **Protocol B**: `StratifiedGroupKFold` grouped on the unique response pattern, so identical
  questionnaires never appear on both sides of a split. This is the generalisation estimate.

Reported metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, MCC, and pooled
out-of-fold confusion matrices.

---

## Usage

```bash
pip install -r requirements.txt

python ppd_preprocessing.py                      # data-quality audit + EDA figures
python ppd_model_comparison.py                   # model comparison (default: onehot)
python ppd_model_comparison.py --encoding hybrid # ordinal codes + one-hot
```

All artefacts are written to `results/`.

---

## Data

`post natal data.csv` — Kaggle "PostPartum Depression" survey (Mosaraf, 2023): 1,503
self-administered Google Form responses, 9 categorical symptom items plus an age bracket.

## Limitations

- `Suicide attempt` is a **self-reported severity proxy, not a clinical diagnosis**, and it is
  predicted from symptoms co-reported in the same questionnaire. This is concurrent screening,
  not anticipatory prediction.
- The independence test in finding (1) assumes inter-item independence. Real items are
  correlated, which reduces the expected unique-pattern count — but the observed association
  is far too weak to account for 999 expected singletons versus 1 observed.
- At an effective n ≈ 248, deep models are not justified on this branch; the ANN results here
  support that.
