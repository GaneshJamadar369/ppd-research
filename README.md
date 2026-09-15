# Postpartum Depression — Structured Branch Baselines & Data-Quality Audit

Classical ML baselines (Logistic Regression, SVM, Random Forest, ANN, plus five learners suited
to small categorical data and a pre-specified ensemble — 13 models in total) on the widely
cited Kaggle **"PostPartum Depression"** survey (`post natal data.csv`, 1,503 responses),
together with a data-quality audit of that benchmark.

The audit is the main result: **the ~99% accuracies reported for this dataset are inflated
twice over** — once by duplicate leakage, once by scoring duplicated rows instead of distinct
questionnaires — and the corrected ceiling, which thirteen models and extensive tuning cannot
cross, is **ROC-AUC ≈ 0.73**.

---

## Headline findings

### 1. The benchmark has an effective sample size of 109–248, not 1,503

The 9 items admit 14,580 possible response combinations. The 1,168 analysed rows contain only
**248 distinct response patterns**, and two independent tests reject that this arose from
independent sampling: simulating independent respondents from the data's own marginals
predicts ~999 patterns occurring exactly once; the file contains **one**. A second,
assumption-free test that grants the observed pattern frequencies entirely and asks only
whether the repeat *counts* look like random sampling still rejects (39 expected singletons
vs. 1 observed).

Accounting for how unevenly those 248 patterns repeat (Kish's design effect = 10.68) puts the
**effective sample size at 109** — the standard survey-methodology answer, and more
conservative than the raw pattern count. The duplication is not label-neutral either: patterns
positive for the outcome are repeated less (mean 3.9×) than negative ones (mean 5.4×), shifting
row-level prevalence (39.3%) away from pattern-level prevalence (47.2%) at p = 0.030 — so
row-weighted statistics are systematically biased, not just noisier.

### 2. Scores are inflated twice over, and thirteen models cannot recover the difference

| Stage | What it fixes | ROC-AUC (Random Forest) |
|---|---|---|
| Standard CV (literature as reported) | nothing | **0.998** |
| Grouped CV, row-weighted | duplicate leakage | **0.895–0.901** |
| Grouped CV, **pattern-weighted** | leakage + row-weighting | **0.703–0.730** |

Identical data and models; only the protocol differs. The pattern-weighted figure is
corroborated three ways (a direct evaluation, an independent learning-curve experiment, and a
from-scratch nested-tuning pass) and all three land in the same 0.66–0.73 band.

**We then tried to beat it.** Thirteen models — the four originals, the same four families with
wide randomised search tuned directly on the pattern-weighted metric, five learners suited to
small categorical data (categorical naive Bayes, extra trees, categorical gradient boosting,
Hamming-kNN, interaction logistic regression), and a pre-specified soft-voting ensemble — were
evaluated under identical, paired, leakage- and weighting-controlled splits. **None beat a
plain, small-grid Random Forest** (Holm-adjusted q = 1.000 for every original-vs-improved
pair). The ceiling is a property of the data, not of modelling effort.

### 3. Only one of nine items survives as a defensible predictor

Each item was tested under six corrections of increasing rigor. Naively, 7 of 9 look
significant; a design-effect correction alone drops that to 5; requiring the association to
hold on distinct patterns (no duplicate weight at all) drops it to **one: Irritable towards
baby & partner**.

This is corroborated by: four items having their *lowest*-risk level in the middle of the
response scale rather than at "no symptom"; a composite severity score (sum of all eight
symptom items) scoring **AUC 0.491** on distinct patterns — chance; and negative scale
reliability (**Cronbach's α = −0.22** on distinct patterns, ω = 0.03, KMO = 0.51). A
nine-encoding, four-training-strategy robustness check confirms this isn't a preprocessing
artefact — nothing beats plain one-hot at a Holm-corrected significance level.

**Recommendation:** treat this dataset as an audit object, not as training data for a deployed
screening model.

---

## Results

Target is `Suicide attempt` (Yes/No), the PPD severity proxy used in prior work;
`Not interested to say` (n=335) is dropped as non-response, leaving 1,168 rows at a 39.3%
positive rate.

**Corrected reference result — leakage- and weighting-controlled, pattern-weighted:**

| Model | AUC | 95% CI | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **Random Forest (original, small grid)** | **0.730** | **[0.673, 0.795]** | 0.633 | 0.635 | 0.630 |
| Random Forest (improved, wide search) | 0.722 | [0.668, 0.790] | 0.696 | 0.920 | 0.560 |
| Extra Trees | 0.717 | [0.659, 0.784] | 0.700 | 0.946 | 0.555 |
| SVM (original) | 0.705 | [0.649, 0.771] | 0.645 | 0.709 | 0.592 |
| Soft-vote ensemble (5 members) | 0.703 | [0.644, 0.769] | 0.681 | 0.923 | 0.540 |
| Logistic Regression (original) | 0.652 | [0.586, 0.723] | 0.568 | 0.521 | 0.626 |

No configuration in the table above differs significantly from the top row (Nadeau–Bengio
corrected, Holm-adjusted). At a fixed 0.5 threshold, recall varies from 0.52–0.71 across models
for reasons more attributable to score-distribution shape than discriminative quality —
choosing the threshold inside the training fold instead raises recall to 0.85–0.95 at a
precision cost, for **every** model, changing recall far more than switching models does.

Full tables, robustness ladders, encoding/imputation benchmarks, and every figure are in
[`results/`](results/); the audit narrative is in
[`paper/audit_section.md`](paper/audit_section.md) (LaTeX: `paper/audit_section.tex`).

---

## Method

**Preprocessing** ([`ppd_preprocessing.py`](ppd_preprocessing.py))
Drops `Timestamp`; canonicalises inconsistent option wording (`Not at all` merged with `No` in
the appetite item); builds the binary target; engineers an EPDS-style composite severity score
(which turns out to carry no signal — see Finding 3); offers `onehot` / `ordinal` / `hybrid`
encodings. Imputation and scaling are fitted inside training folds only.

**Statistical audit** ([`ppd_statistics.py`](ppd_statistics.py))
Column profiles, Wilson CIs, univariate association (naive / Rao–Scott / cluster-permutation /
distinct-pattern χ², bias-corrected Cramér's V, mutual information, Information Value),
Cochran–Armitage trend tests, a cluster-robust multivariable logistic model, inter-item
dependence (polychoric correlation, GVIF), scale reliability (Cronbach/ordinal α, ω, KMO,
Bartlett, Horn's parallel analysis), multiple correspondence analysis, missingness forensics
(Little's MCAR test), and the duplication-vs-label analysis (Kish design effect, permutation
test on prevalence shift).

**Preprocessing robustness** ([`ppd_preprocessing_experiments.py`](ppd_preprocessing_experiments.py))
Benchmarks imputation methods (mode, marginal sampling, KNN-Hamming, MICE-logistic, MICE-RF,
iterative MCA) by recovery accuracy on masked *distinct* questionnaires; benchmarks 9 encodings
and 4 training strategies against plain one-hot under grouped CV with Nadeau–Bengio corrected
paired tests.

**Baseline evaluation** ([`ppd_model_comparison.py`](ppd_model_comparison.py),
[`ppd_evaluation.py`](ppd_evaluation.py), [`ppd_audit.py`](ppd_audit.py))
Nested cross-validation under two/three protocols — standard stratified CV (leaky), grouped CV
row-weighted (leakage-controlled only), and grouped CV pattern-weighted (leakage- and
weighting-controlled) — with cluster-bootstrap confidence intervals, calibration, and
decision-curve analysis.

**Improved baselines** ([`ppd_improved_baselines.py`](ppd_improved_baselines.py))
Thirteen models sharing identical paired outer splits: the four original baselines reproduced
exactly; the same four families with wide randomised search tuned directly on pattern-weighted
AUC via a grouping-aware inner scorer; five learners suited to small categorical data; and a
soft-voting ensemble with members fixed before any result was seen. Every hyperparameter and
decision threshold is selected inside the training fold. Differences are tested with the
Nadeau–Bengio corrected resampled t-test, Holm-adjusted. Per-model results are cached in
`results/baseline_cache/`, so an interrupted run resumes.

---

## Usage

```bash
pip install -r requirements.txt

python ppd_preprocessing.py                        # data-quality audit + EDA figures
python ppd_statistics.py                            # full statistical audit
python ppd_preprocessing_experiments.py              # imputation / encoding robustness
python ppd_model_comparison.py                       # baseline comparison (standard + grouped)
python ppd_evaluation.py --repeats 3                 # + calibration, decision curves, CIs
python ppd_audit.py                                  # leakage-inflation figures + learning curve
python ppd_improved_baselines.py                     # 13-model ceiling (resumable; slow, ~30–60 min)
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
- The independence test behind the effective-*n* claim assumes inter-item independence and so
  overstates expected diversity; a second, assumption-free permutation test does not rely on
  that assumption and is the claim we rest on. We do not have provenance information for the
  file and do not assert *how* the duplication arose.
- Kish's design effect corrects for the *consequence* of the clustering, not its cause.
- With 244–248 clusters, the most conservative significance tests in the robustness ladder are
  themselves low-powered; absence of significance there is suggestive, not proof of no effect,
  for items beyond the two that fail even the naive test.
- The 13-model comparison is broad but not exhaustive; the ensemble's members were
  pre-specified to avoid post-hoc selection, at the cost of not searching the full ensemble
  space.
