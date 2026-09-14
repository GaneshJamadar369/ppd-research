"""
Data preprocessing for the PPD structured survey (post natal data.csv).

Run standalone to produce the data-quality / EDA report:

    python ppd_preprocessing.py

or import `prepare()` from the model comparison script.

Preprocessing stages
--------------------
1.  Load, drop the free-running `Timestamp` column (a survey submission clock, not a
    clinical variable — keeping it would let a model memorise respondents).
2.  Data-quality audit: missingness per column, duplicate rows, duplicate response
    patterns, label-conflicting patterns, class balance, category cardinality.
3.  Category canonicalisation. The Google-Form instrument used inconsistent option
    wording across items; responses are mapped onto one severity vocabulary:
      - 'Not at all' and 'No' in the appetite item are the same response, recorded
        under two labels by the form. They are merged.
      - 'Sometimes' / 'Maybe' / 'Often' / 'Two or more days a week' all occupy the
        intermediate severity slot of their respective items.
4.  Target construction: `Suicide attempt` -> binary (Yes = 1, No = 0). The
    'Not interested to say' responses are non-response, not a class, and are dropped.
5.  Encoding. Each item gets an ordinal severity code AND a one-hot expansion; the
    encoder chosen at model time decides which representation is used. A composite
    `Symptom severity score` (sum of the ordinal symptom codes, EPDS-style) is
    engineered as an extra feature.
6.  Imputation and scaling are deliberately NOT done here — they live inside the
    modelling pipeline so they are fitted on training folds only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA = HERE / "post natal data.csv"
OUT = HERE / "results"

TARGET = "Suicide attempt"
POSITIVE_LABEL, NEGATIVE_LABEL, NON_RESPONSE = "Yes", "No", "Not interested to say"
DROP_COLS = ["Timestamp"]

AGE_COL = "Age"
AGE_ORDER = ["25-30", "30-35", "35-40", "40-45", "45-50"]

# Raw response -> (canonical label, ordinal severity code) for every symptom item.
SEVERITY_MAPS: dict[str, dict[str, tuple[str, int]]] = {
    "Feeling sad or Tearful": {
        "No": ("No", 0), "Sometimes": ("Sometimes", 1), "Yes": ("Yes", 2),
    },
    "Irritable towards baby & partner": {
        "No": ("No", 0), "Sometimes": ("Sometimes", 1), "Yes": ("Yes", 2),
    },
    "Trouble sleeping at night": {
        "No": ("No", 0),
        "Two or more days a week": ("Two or more days a week", 1),
        "Yes": ("Yes", 2),
    },
    "Problems concentrating or making decision": {
        "No": ("No", 0), "Often": ("Often", 1), "Yes": ("Yes", 2),
    },
    # 'Not at all' and 'No' are the same answer under two form labels -> merged.
    "Overeating or loss of appetite": {
        "Not at all": ("No", 0), "No": ("No", 0), "Yes": ("Yes", 1),
    },
    "Feeling anxious": {"No": ("No", 0), "Yes": ("Yes", 1)},
    "Feeling of guilt": {"No": ("No", 0), "Maybe": ("Maybe", 1), "Yes": ("Yes", 2)},
    "Problems of bonding with baby": {
        "No": ("No", 0), "Sometimes": ("Sometimes", 1), "Yes": ("Yes", 2),
    },
}
SYMPTOM_COLS = list(SEVERITY_MAPS)
ORD_SUFFIX = "__ord"
SCORE_COL = "Symptom severity score"


# ----------------------------------------------------------------- 1-2. load, audit
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def audit(df: pd.DataFrame) -> dict:
    features = [c for c in df.columns if c != TARGET]
    # astype(str) first: the ordinal columns are nullable Float64, whose NA cannot be
    # filled with a string sentinel. Casting renders missing as a consistent '<NA>'.
    patterns = df[features].astype(str).agg("|".join, axis=1)
    conflicts = df.groupby(patterns.to_numpy())[TARGET].nunique()

    # Count missingness on the source columns only: after normalisation each item also
    # carries a derived `__ord` copy (and the composite score), which would double-count.
    source = [c for c in df.columns if not c.endswith(ORD_SUFFIX) and c != SCORE_COL]
    missing = df[source].isna().sum()
    return {
        "n_rows": len(df),
        "n_features": len(features),
        "missing_by_column": missing[missing > 0].to_dict(),
        "missing_total": int(missing.sum()),
        "missing_pct": float(missing.sum() / df[source].size * 100),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df[features].duplicated().sum()),
        "unique_patterns": int(patterns.nunique()),
        "label_conflicting_patterns": int((conflicts > 1).sum()),
        "target_counts": df[TARGET].value_counts().to_dict(),
        "cardinality": {c: int(df[c].nunique()) for c in features},
    }


# -------------------------------------------------- 3-5. canonicalise, label, encode
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, unify the severity vocabulary, add ordinal codes + composite."""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype("string").str.strip()

    unmapped: dict[str, set] = {}
    for col, mapping in SEVERITY_MAPS.items():
        seen = set(out[col].dropna().unique()) - set(mapping)
        if seen:
            unmapped[col] = seen
        out[col + ORD_SUFFIX] = out[col].map(lambda v: mapping.get(v, (None, None))[1])
        out[col] = out[col].map(lambda v: mapping.get(v, (v, None))[0])
    if unmapped:
        raise ValueError(f"Unmapped response values, extend SEVERITY_MAPS: {unmapped}")

    age_codes = {lvl: i for i, lvl in enumerate(AGE_ORDER)}
    unseen_age = set(out[AGE_COL].dropna().unique()) - set(age_codes)
    if unseen_age:
        raise ValueError(f"Unexpected age brackets: {unseen_age}")
    out[AGE_COL + ORD_SUFFIX] = out[AGE_COL].map(age_codes)

    ord_cols = [c + ORD_SUFFIX for c in SYMPTOM_COLS]
    # numpy-native dtypes: sklearn's imputers cannot evaluate pandas' nullable pd.NA.
    out = out.astype({c: "float64" for c in ord_cols + [AGE_COL + ORD_SUFFIX]})
    for col in [AGE_COL] + SYMPTOM_COLS + ([TARGET] if TARGET in out else []):
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)

    # Composite severity: row sum over available items, rescaled to the full item set
    # so partially-missing rows stay on the same 0-14 scale.
    present = out[ord_cols].notna().sum(axis=1)
    max_scale = sum(max(m.values(), key=lambda t: t[1])[1] for m in SEVERITY_MAPS.values())
    out[SCORE_COL] = (out[ord_cols].sum(axis=1) / present.replace(0, np.nan)) * len(ord_cols)
    out[SCORE_COL] = out[SCORE_COL].clip(upper=max_scale).astype("float64")
    return out


def make_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    keep = df[TARGET].isin([POSITIVE_LABEL, NEGATIVE_LABEL])
    dropped = int((~keep).sum())
    kept = df[keep].reset_index(drop=True)
    y = (kept[TARGET] == POSITIVE_LABEL).astype(int).rename("ppd_risk")
    return kept.drop(columns=[TARGET]), y, dropped


def feature_columns(mode: str) -> tuple[list[str], list[str], list[str]]:
    """-> (nominal one-hot columns, ordinal numeric columns, engineered numeric columns)"""
    nominal = [AGE_COL] + SYMPTOM_COLS
    ordinal = [AGE_COL + ORD_SUFFIX] + [c + ORD_SUFFIX for c in SYMPTOM_COLS]
    if mode == "onehot":
        return nominal, [], [SCORE_COL]
    if mode == "ordinal":
        return [], ordinal, [SCORE_COL]
    if mode == "hybrid":
        return nominal, ordinal, [SCORE_COL]
    raise ValueError(f"Unknown encoding mode: {mode!r}")


def build_preprocessor(mode: str = "onehot", scale: bool = False) -> Pipeline:
    """Fold-safe encoder: imputation is fitted inside each training fold, never globally."""
    from sklearn.preprocessing import StandardScaler

    nominal, ordinal, engineered = feature_columns(mode)
    blocks = []
    if nominal:
        blocks.append((
            "nominal",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            nominal,
        ))
    if ordinal:
        blocks.append((
            "ordinal",
            Pipeline([("impute", SimpleImputer(strategy="median"))]),
            ordinal,
        ))
    if engineered:
        blocks.append((
            "engineered",
            Pipeline([("impute", SimpleImputer(strategy="median"))]),
            engineered,
        ))

    steps = [("encode", ColumnTransformer(blocks))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def prepare(verbose: bool = True) -> dict:
    """Full preprocessing run -> features, target, grouping key, and the audit reports."""
    raw = load_raw()
    before = audit(raw)
    clean = normalise(raw)
    X, y, dropped = make_target(clean)
    after = audit(X.assign(**{TARGET: np.where(y == 1, POSITIVE_LABEL, NEGATIVE_LABEL)}))

    pattern_cols = [AGE_COL] + SYMPTOM_COLS
    groups = X[pattern_cols].astype(str).agg("|".join, axis=1).factorize()[0]

    if verbose:
        print(f"Raw rows                    : {before['n_rows']}")
        print(f"Dropped '{NON_RESPONSE}'    : {dropped}")
        print(f"Analysed rows               : {len(X)}")
        print(f"Missing cells               : {before['missing_total']} "
              f"({before['missing_pct']:.2f}%) -> fold-wise imputation")
        print(f"Merged categories           : 'Not at all' -> 'No' "
              f"(Overeating or loss of appetite)")
        print(f"Class balance               : {dict(y.value_counts())} "
              f"(positive rate {y.mean():.3f})")
        print(f"Duplicate feature rows      : {after['duplicate_feature_rows']} "
              f"/ {after['n_rows']}")
        print(f"Unique response patterns    : {after['unique_patterns']} "
              f"(label-conflicting: {after['label_conflicting_patterns']})")

    return {"X": X, "y": y, "groups": groups, "audit_raw": before, "audit_clean": after,
            "dropped_non_response": dropped}


# ------------------------------------------------------------------- EDA / reporting
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def association_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in [AGE_COL] + SYMPTOM_COLS:
        tab = pd.crosstab(X[col], y)
        chi2, p, dof, _ = chi2_contingency(tab)
        rows.append({
            "Feature": col,
            "Chi2": chi2,
            "dof": dof,
            "p-value": p,
            "Cramer's V": cramers_v(X[col], y),
            "Positive rate range": f"{tab.div(tab.sum(1), axis=0)[1].min():.2f}"
                                   f"–{tab.div(tab.sum(1), axis=0)[1].max():.2f}",
        })
    return pd.DataFrame(rows).sort_values("Cramer's V", ascending=False).reset_index(drop=True)


def level_order(col: str) -> list[str]:
    """Canonical response levels for a column, ordered by severity (not alphabetically)."""
    if col == AGE_COL:
        return AGE_ORDER
    seen: dict[str, int] = {}
    for canon, code in SEVERITY_MAPS[col].values():
        seen.setdefault(canon, code)
    return sorted(seen, key=seen.get)


def eda_plots(X: pd.DataFrame, y: pd.Series) -> None:
    cols = [AGE_COL] + SYMPTOM_COLS
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5))
    for ax, col in zip(axes.ravel(), cols):
        tab = pd.crosstab(X[col], y, normalize="index") * 100
        tab = tab.reindex(index=level_order(col)).reindex(columns=[0, 1], fill_value=0)
        tab.plot(kind="bar", stacked=True, ax=ax, color=["#8fb8de", "#c1443c"],
                 legend=False, width=0.72)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("% of respondents")
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#8fb8de", "#c1443c")]
    fig.legend(handles, ["No attempt", "Attempt"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Target rate by response level, ordered by severity "
                 "(after category canonicalisation)")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / "eda_target_by_feature.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = y.value_counts().sort_index()
    axes[0].bar(["No (0)", "Yes (1)"], counts.values, color=["#8fb8de", "#c1443c"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[0].set_title(f"Class balance (positive rate {y.mean():.1%})")
    axes[0].set_ylabel("Respondents")

    for label, colour, name in [(0, "#8fb8de", "No attempt"), (1, "#c1443c", "Attempt")]:
        axes[1].hist(X.loc[y == label, SCORE_COL].astype(float), bins=15, alpha=0.65,
                     color=colour, label=name)
    axes[1].set_title("Engineered composite symptom severity score")
    axes[1].set_xlabel("Score (0–14)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "eda_class_and_score.png", dpi=200)
    plt.close(fig)

    assoc = [AGE_COL] + SYMPTOM_COLS
    mat = pd.DataFrame(
        [[cramers_v(X[a], X[b]) for b in assoc] for a in assoc], index=assoc, columns=assoc
    )
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(assoc)), [c[:22] for c in assoc], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(assoc)), [c[:22] for c in assoc], fontsize=8)
    for i in range(len(assoc)):
        for j in range(len(assoc)):
            ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="w")
    fig.colorbar(im, label="Cramér's V")
    ax.set_title("Inter-feature association (Cramér's V)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_feature_association.png", dpi=200)
    plt.close(fig)


def write_report(prep: dict) -> None:
    X, y = prep["X"], prep["y"]
    before, after = prep["audit_raw"], prep["audit_clean"]
    assoc = association_table(X, y)
    assoc.to_csv(OUT / "feature_association.csv", index=False)

    miss = before["missing_by_column"]
    lines = [
        "# Data Preprocessing Report — `post natal data.csv`",
        "",
        "## 1. Source",
        f"- Rows: **{before['n_rows']}**, columns retained: **{before['n_features']}** "
        "features + 1 target (`Timestamp` dropped — a submission clock, not a clinical "
        "variable).",
        f"- Target `{TARGET}`: " + ", ".join(f"{k} = {v}" for k, v in before["target_counts"].items()),
        "",
        "## 2. Data quality audit",
        "| Check | Raw | After preprocessing |",
        "|---|---|---|",
        f"| Rows | {before['n_rows']} | {after['n_rows']} |",
        f"| Missing cells | {before['missing_total']} ({before['missing_pct']:.2f}%) | "
        f"{after['missing_total']} — imputed per fold |",
        f"| Duplicate full rows | {before['duplicate_rows']} | {after['duplicate_rows']} |",
        f"| Duplicate feature rows | {before['duplicate_feature_rows']} | "
        f"{after['duplicate_feature_rows']} |",
        f"| Unique response patterns | {before['unique_patterns']} | {after['unique_patterns']} |",
        f"| Patterns with conflicting labels | {before['label_conflicting_patterns']} | "
        f"{after['label_conflicting_patterns']} |",
        "",
        "**Missing values by column (raw):** "
        + (", ".join(f"`{k}` = {v}" for k, v in miss.items()) if miss else "none"),
        "",
        f"> The duplicate count is the headline issue: {after['n_rows']} analysed rows carry "
        f"only {after['unique_patterns']} distinct questionnaire answers, and only "
        f"{after['label_conflicting_patterns']} of those patterns "
        f"{'disagrees' if after['label_conflicting_patterns'] == 1 else 'disagree'} on the "
        "label. A random train/test split therefore tests on rows it has already seen, "
        "which is why "
        "published accuracies on this dataset sit near 100%. The modelling script reports a "
        "grouped-CV protocol alongside the standard one for this reason.",
        "",
        "## 3. Category canonicalisation",
        "The survey form used different option wording per item. Responses were mapped onto "
        "one severity vocabulary with an explicit ordinal code:",
        "",
        "| Item | Level 0 | Level 1 | Level 2 |",
        "|---|---|---|---|",
    ]
    for col, mapping in SEVERITY_MAPS.items():
        by_code: dict[int, list[str]] = {}
        for raw_v, (canon, code) in mapping.items():
            by_code.setdefault(code, [])
            if canon not in by_code[code]:
                by_code[code].append(canon)
        cells = [", ".join(by_code.get(i, ["—"])) for i in (0, 1, 2)]
        lines.append(f"| {col} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "- `Overeating or loss of appetite`: **'Not at all' and 'No' were merged** — they are "
        "the same response recorded under two labels (319 + 841 rows).",
        "- `Age` is ordinal: " + " < ".join(AGE_ORDER) + ".",
        "",
        "## 4. Target construction",
        f"- `{TARGET}` → binary: Yes = 1, No = 0.",
        f"- **{prep['dropped_non_response']}** '{NON_RESPONSE}' responses dropped as "
        "non-response rather than treated as a third class.",
        f"- Final: **{len(y)}** rows, {int((y == 1).sum())} positive / "
        f"{int((y == 0).sum())} negative (positive rate {y.mean():.1%}) — mild imbalance, "
        "handled with `class_weight` in the model grids rather than resampling.",
        "",
        "## 5. Feature engineering & encoding",
        f"- **`{SCORE_COL}`**: sum of the 8 ordinal symptom codes on a 0–14 scale "
        "(EPDS-style composite), rescaled for rows with a missing item.",
        "- Three interchangeable encodings are available: `onehot` (default — imposes no "
        "ordering, and the middle levels turn out to matter non-monotonically), `ordinal` "
        "(compact severity codes), and `hybrid` (both).",
        "- **Imputation and scaling are not applied here.** They sit inside the modelling "
        "pipeline so they are fitted on training folds only and cannot leak.",
        "",
        "## 6. Feature–target association",
        "",
        "| Feature | Chi² | dof | p-value | Cramér's V | Positive-rate range |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in assoc.iterrows():
        v = r["Cramer's V"]
        lines.append(
            f"| {r['Feature']} | {r['Chi2']:.2f} | {int(r['dof'])} | {r['p-value']:.3g} | "
            f"{v:.3f} | {r['Positive rate range']} |"
        )
    lines += [
        "",
        "Plots: `eda_target_by_feature.png`, `eda_class_and_score.png`, "
        "`eda_feature_association.png`.",
        "",
    ]
    (OUT / "preprocessing_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prep = prepare()
    eda_plots(prep["X"], prep["y"])
    write_report(prep)
    print("\nFeature–target association (Cramér's V):")
    print(association_table(prep["X"], prep["y"]).to_string(index=False))
    print(f"\nReport written to {OUT / 'preprocessing_report.md'}")


if __name__ == "__main__":
    main()
