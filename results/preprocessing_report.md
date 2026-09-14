# Data Preprocessing Report — `post natal data.csv`

## 1. Source
- Rows: **1503**, columns retained: **9** features + 1 target (`Timestamp` dropped — a submission clock, not a clinical variable).
- Target `Suicide attempt`: No = 709, Yes = 459, Not interested to say = 335

## 2. Data quality audit
| Check | Raw | After preprocessing |
|---|---|---|
| Rows | 1503 | 1168 |
| Missing cells | 27 (0.18%) | 27 — imputed per fold |
| Duplicate full rows | 1177 | 919 |
| Duplicate feature rows | 1180 | 920 |
| Unique response patterns | 323 | 248 |
| Patterns with conflicting labels | 3 | 1 |

**Missing values by column (raw):** `Irritable towards baby & partner` = 6, `Problems concentrating or making decision` = 12, `Feeling of guilt` = 9

> The duplicate count is the headline issue: 1168 analysed rows carry only 248 distinct questionnaire answers, and only 1 of those patterns disagrees on the label. A random train/test split therefore tests on rows it has already seen, which is why published accuracies on this dataset sit near 100%. The modelling script reports a grouped-CV protocol alongside the standard one for this reason.

## 3. Category canonicalisation
The survey form used different option wording per item. Responses were mapped onto one severity vocabulary with an explicit ordinal code:

| Item | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| Feeling sad or Tearful | No | Sometimes | Yes |
| Irritable towards baby & partner | No | Sometimes | Yes |
| Trouble sleeping at night | No | Two or more days a week | Yes |
| Problems concentrating or making decision | No | Often | Yes |
| Overeating or loss of appetite | No | Yes | — |
| Feeling anxious | No | Yes | — |
| Feeling of guilt | No | Maybe | Yes |
| Problems of bonding with baby | No | Sometimes | Yes |

- `Overeating or loss of appetite`: **'Not at all' and 'No' were merged** — they are the same response recorded under two labels (319 + 841 rows).
- `Age` is ordinal: 25-30 < 30-35 < 35-40 < 40-45 < 45-50.

## 4. Target construction
- `Suicide attempt` → binary: Yes = 1, No = 0.
- **335** 'Not interested to say' responses dropped as non-response rather than treated as a third class.
- Final: **1168** rows, 459 positive / 709 negative (positive rate 39.3%) — mild imbalance, handled with `class_weight` in the model grids rather than resampling.

## 5. Feature engineering & encoding
- **`Symptom severity score`**: sum of the 8 ordinal symptom codes on a 0–14 scale (EPDS-style composite), rescaled for rows with a missing item.
- Three interchangeable encodings are available: `onehot` (default — imposes no ordering, and the middle levels turn out to matter non-monotonically), `ordinal` (compact severity codes), and `hybrid` (both).
- **Imputation and scaling are not applied here.** They sit inside the modelling pipeline so they are fitted on training folds only and cannot leak.

## 6. Feature–target association

| Feature | Chi² | dof | p-value | Cramér's V | Positive-rate range |
|---|---|---|---|---|---|
| Problems concentrating or making decision | 182.15 | 2 | 2.8e-40 | 0.395 | 0.15–0.62 |
| Feeling sad or Tearful | 141.79 | 2 | 1.63e-31 | 0.346 | 0.20–0.61 |
| Feeling of guilt | 132.04 | 2 | 2.13e-29 | 0.335 | 0.17–0.60 |
| Irritable towards baby & partner | 120.57 | 2 | 6.57e-27 | 0.320 | 0.25–0.59 |
| Trouble sleeping at night | 76.59 | 2 | 2.33e-17 | 0.253 | 0.23–0.52 |
| Problems of bonding with baby | 37.27 | 2 | 8.08e-09 | 0.174 | 0.27–0.47 |
| Age | 15.68 | 4 | 0.00348 | 0.100 | 0.28–0.47 |
| Overeating or loss of appetite | 0.10 | 1 | 0.753 | 0.000 | 0.38–0.40 |
| Feeling anxious | 0.12 | 1 | 0.733 | 0.000 | 0.39–0.40 |

Plots: `eda_target_by_feature.png`, `eda_class_and_score.png`, `eda_feature_association.png`.
