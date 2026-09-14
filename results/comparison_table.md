# PPD Classifier Comparison — `post natal data.csv`

**Target:** `Suicide attempt` (Yes = 1, No = 0); `Not interested to say` responses dropped.  
**Samples:** 1168 | **Positive rate:** 0.393 | **Encoding:** `onehot` (+ engineered `Symptom severity score`)  
**Preprocessing:** see `preprocessing_report.md` — category canonicalisation is applied up front; imputation, encoding and scaling are fitted inside each training fold only.  
**Evaluation:** nested CV — outer 5-fold scoring, inner 3-fold grid search on ROC-AUC. Mean ± SD across outer folds; confusion-matrix counts are pooled out-of-fold predictions.

### Protocol A — standard stratified 5-fold CV

_Comparable to published results on this dataset. These 1168 rows contain only 248 unique response patterns, so near-identical rows fall on both sides of the split and these scores are optimistically biased._

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | MCC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9863 ± 0.010 | 0.9787 ± 0.020 | 0.9870 ± 0.012 | 0.9827 ± 0.013 | 0.9976 ± 0.002 | 0.9859 ± 0.013 | 0.9715 ± 0.021 | 699 | 10 | 6 | 453 |
| SVM (RBF) | 0.9914 ± 0.009 | 0.9872 ± 0.017 | 0.9913 ± 0.012 | 0.9892 ± 0.011 | 0.9974 ± 0.003 | 0.9915 ± 0.012 | 0.9822 ± 0.019 | 703 | 6 | 4 | 455 |
| ANN (MLP) | 0.9615 ± 0.027 | 0.9469 ± 0.042 | 0.9565 ± 0.028 | 0.9516 ± 0.034 | 0.9863 ± 0.012 | 0.9647 ± 0.029 | 0.9198 ± 0.057 | 684 | 25 | 20 | 439 |
| Logistic Regression | 0.7543 ± 0.008 | 0.6806 ± 0.031 | 0.7166 ± 0.075 | 0.6951 ± 0.020 | 0.8407 ± 0.018 | 0.7785 ± 0.055 | 0.4938 ± 0.017 | 552 | 157 | 130 | 329 |


### Protocol B — leakage-controlled grouped 5-fold CV

_All rows sharing an identical questionnaire response pattern are kept in the same fold. This is the honest generalisation estimate and the one to report as the primary result._

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | MCC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | 0.7243 ± 0.024 | 0.9708 ± 0.041 | 0.3068 ± 0.060 | 0.4640 ± 0.073 | 0.8913 ± 0.013 | 0.9943 ± 0.008 | 0.4448 ± 0.062 | 705 | 4 | 318 | 141 |
| Random Forest | 0.7842 ± 0.052 | 0.7475 ± 0.076 | 0.6858 ± 0.078 | 0.7137 ± 0.066 | 0.8902 ± 0.028 | 0.8477 ± 0.055 | 0.5440 ± 0.109 | 601 | 108 | 144 | 315 |
| ANN (MLP) | 0.7270 ± 0.048 | 0.6543 ± 0.071 | 0.6736 ± 0.059 | 0.6605 ± 0.044 | 0.8079 ± 0.045 | 0.7617 ± 0.085 | 0.4361 ± 0.085 | 540 | 169 | 150 | 309 |
| Logistic Regression | 0.7294 ± 0.067 | 0.6688 ± 0.111 | 0.6528 ± 0.151 | 0.6508 ± 0.096 | 0.7778 ± 0.053 | 0.7783 ± 0.119 | 0.4408 ± 0.140 | 552 | 157 | 159 | 300 |


### Protocol B (supplementary) — decision threshold tuned for F1

_Same grouped folds, but the operating point is selected on the training folds instead of fixed at 0.5. Reported because a screening instrument should not trade away recall: at 0.5 the RBF-SVM reaches high precision while missing most positive cases. `Threshold` is the mean cut-off chosen across outer folds._

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | MCC | Threshold | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.7782 ± 0.043 | 0.6785 ± 0.056 | 0.8403 ± 0.102 | 0.7476 ± 0.051 | 0.8902 ± 0.028 | 0.7376 ± 0.073 | 0.5699 ± 0.093 | 0.36 | 523 | 186 | 73 | 386 |
| SVM (RBF) | 0.7697 ± 0.041 | 0.6610 ± 0.045 | 0.8621 ± 0.108 | 0.7448 ± 0.050 | 0.8913 ± 0.013 | 0.7094 ± 0.074 | 0.5638 ± 0.087 | 0.43 | 503 | 206 | 63 | 396 |
| ANN (MLP) | 0.7142 ± 0.070 | 0.6285 ± 0.100 | 0.7489 ± 0.106 | 0.6752 ± 0.053 | 0.8079 ± 0.045 | 0.6908 ± 0.155 | 0.4425 ± 0.109 | 0.42 | 490 | 219 | 115 | 344 |
| Logistic Regression | 0.6636 ± 0.071 | 0.6008 ± 0.096 | 0.6784 ± 0.272 | 0.5948 ± 0.112 | 0.7778 ± 0.053 | 0.6529 ± 0.261 | 0.3640 ± 0.059 | 0.43 | 463 | 246 | 147 | 312 |
