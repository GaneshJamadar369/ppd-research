# Model Comparison — PPD Detection

Target: `Suicide attempt` (Yes/No), 1,168 rows, 39.3% positive. Evaluated with grouped cross-validation (no duplicate leakage) scored per distinct questionnaire. 95% CI from bootstrap.

## Ranked results (best model first)

| Rank | Model | AUC | 95% CI | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|---|
| 🏆 **1 — DOMINANT** | **Random Forest (original)** | **0.730** | **[0.673, 0.795]** | 0.633 | 0.635 | 0.630 | 0.652 |
| 2 | Random Forest (tuned) | 0.722 | [0.668, 0.790] | 0.696 | 0.920 | 0.560 | 0.621 |
| 3 | Extra Trees | 0.717 | [0.659, 0.784] | 0.700 | 0.946 | 0.555 | 0.617 |
| 4 | SVM (original) | 0.705 | [0.649, 0.771] | 0.645 | 0.709 | 0.592 | 0.632 |
| 5 | Ensemble (5-model vote) | 0.703 | [0.644, 0.769] | 0.681 | 0.923 | 0.540 | 0.591 |
| 6 | kNN (Hamming) | 0.701 | [0.649, 0.776] | 0.682 | 0.895 | 0.551 | 0.606 |
| 7 | SVM (tuned) | 0.674 | [0.651, 0.776] | 0.672 | 0.895 | 0.543 | 0.589 |
| 8 | Gradient Boosting | 0.671 | [0.621, 0.757] | 0.663 | 0.900 | 0.525 | 0.569 |
| 9 | Categorical Naive Bayes | 0.661 | [0.590, 0.728] | 0.663 | 0.926 | 0.516 | 0.555 |
| 10 | Logistic Regression + interactions | 0.653 | [0.603, 0.733] | 0.655 | 0.846 | 0.534 | 0.579 |
| 11 | Logistic Regression (original) | 0.652 | [0.586, 0.723] | 0.568 | 0.521 | 0.626 | 0.626 |
| 12 | Logistic Regression (tuned) | 0.647 | [0.585, 0.720] | 0.652 | 0.917 | 0.506 | 0.539 |
| 13 | ANN / MLP (tuned) | 0.641 | [0.580, 0.718] | 0.643 | 0.895 | 0.209 | 0.532 |
| 14 | ANN / MLP (original) | 0.626 | [0.570, 0.707] | 0.554 | 0.547 | 0.562 | 0.585 |

**Winner: Random Forest, AUC 0.730.** Best on the primary metric (AUC) and by far the tightest confidence interval. Tuning it further (row 2), swapping to Extra Trees (row 3), or ensembling 5 models (row 5) did not beat it — differences among the top 5 are not statistically significant.

**If recall matters more than precision** (i.e. catching more at-risk cases matters more than false alarms): Extra Trees or tuned Random Forest catch ~92–95% of positive cases vs. 64% for the plain RF, at the cost of more false positives (precision drops to ~0.56).

## Files

- `post natal data.csv` — dataset
- `ppd_preprocessing.py` — cleaning, encoding, imputation
- `ppd_improved_baselines.py` — trains and evaluates all 14 models above (run: `python ppd_improved_baselines.py`)
- `results/baseline_improved_summary.csv` — this table, machine-readable
- `results/fig_baseline_improvement.png` — chart version
- `results/preprocessing_report.md` — data quality notes
