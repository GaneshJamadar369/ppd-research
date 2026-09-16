"""
Build a real perinatal depression dataset from NHANES — per-person, free, no application.

Why: PRAMS published tables are aggregate (one row per state), so no patient-level model can be
fitted on them. NHANES releases individual records, and crucially carries lab work and the PHQ-9
on the *same woman* — the "clinical reports + questionnaire on one person" that the survey
datasets lacked.

Target is PHQ-9 >= 10, scored from the nine DPQ items, while every predictor comes from a
different instrument (demographics, reproductive history, blood work). The label is therefore
never a function of the features, which is the flaw that sank the two survey datasets.

    python nhanes_build.py          # downloads (cached) and writes nhanes_perinatal.csv
"""

import os
from pathlib import Path

import pandas as pd
import requests

CACHE = Path("nhanes_cache")
OUT = "nhanes_perinatal.csv"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suf}.xpt"

# NHANES release folders and their cycle suffix
CYCLES = {"2005": "D", "2007": "E", "2009": "F", "2011": "G",
          "2013": "H", "2015": "I", "2017": "J", "2021": "L"}

# file -> columns to keep (SEQN is the person id and always kept)
FILES = {
    "DEMO": ["RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "DMDMARTL", "INDFMPIR", "RIDEXPRG"],
    "RHQ":  ["RHQ131", "RHD143", "RHQ197", "RHQ200", "RHD180"],
    "CBC":  ["LBXHGB", "LBXRBCSI", "LBXWBCSI", "LBXPLTSI"],
    "HSCRP": ["LBXHSCRP"],
    "BMX":  ["BMXBMI"],
    "SLQ":  ["SLD010H", "SLD012"],
}
DPQ_ITEMS = [f"DPQ{i:03d}" for i in (10, 20, 30, 40, 50, 60, 70, 80, 90)]


def fetch(year, suf, name):
    """Download one NHANES file, cached on disk. Returns None when the cycle lacks it."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}_{suf}.xpt"
    if not path.exists():
        r = requests.get(BASE.format(year=year, name=name, suf=suf), timeout=120)
        if r.status_code != 200 or len(r.content) < 5000:
            return None
        path.write_bytes(r.content)
    try:
        return pd.read_sas(path)
    except Exception:
        return None


def build_cycle(year, suf):
    dpq = fetch(year, suf, "DPQ")
    if dpq is None:
        return None
    items = [c for c in DPQ_ITEMS if c in dpq.columns]
    if len(items) < 9:
        return None
    # 7 = refused, 9 = don't know -> missing
    dpq[items] = dpq[items].where(dpq[items] <= 3)
    df = dpq[["SEQN"] + items].copy()

    for name, cols in FILES.items():
        f = fetch(year, suf, name)
        if f is None:
            continue
        keep = ["SEQN"] + [c for c in cols if c in f.columns]
        df = df.merge(f[keep], on="SEQN", how="left")

    df["cycle"] = year
    return df


def main():
    frames = [f for f in (build_cycle(y, s) for y, s in CYCLES.items()) if f is not None]
    d = pd.concat(frames, ignore_index=True)

    items = [c for c in DPQ_ITEMS if c in d.columns]
    d["phq9"] = d[items].sum(axis=1, min_count=9)      # require all 9 answered
    d["depressed"] = (d["phq9"] >= 10).astype("Int64").where(d["phq9"].notna())

    women = (d.RIAGENDR == 2) & d.RIDAGEYR.between(18, 49) & d.depressed.notna()
    d = d[women].copy()

    # RHQ197 = months since last live birth; 99/999 are refused/don't know
    d["months_pp"] = d["RHQ197"].where(d["RHQ197"] <= 96)
    d["postpartum_12m"] = d["months_pp"] <= 12
    d["postpartum_24m"] = d["months_pp"] <= 24
    d["pregnant_now"] = (d["RIDEXPRG"] == 1)
    d["perinatal"] = d["postpartum_24m"] | d["pregnant_now"]

    d.to_csv(OUT, index=False)
    print(f"cycles pooled: {len(frames)}  ->  {OUT}")
    print(f"women 18-49 with complete PHQ-9 : {len(d):5d}  ({d.depressed.mean():.1%} depressed)")
    print(f"  pregnant now                  : {d.pregnant_now.sum():5d}")
    print(f"  <=12 months postpartum        : {d.postpartum_12m.sum():5d}"
          f"  ({d.loc[d.postpartum_12m,'depressed'].mean():.1%} depressed)")
    print(f"  <=24 months postpartum        : {d.postpartum_24m.sum():5d}"
          f"  ({d.loc[d.postpartum_24m,'depressed'].mean():.1%} depressed)")
    print(f"  perinatal (pregnant or <=24m) : {d.perinatal.sum():5d}"
          f"  ({d.loc[d.perinatal,'depressed'].mean():.1%} depressed)")
    lab = d.loc[d.perinatal, [c for c in ('LBXHGB', 'LBXHSCRP', 'BMXBMI') if c in d.columns]]
    print("  lab coverage in perinatal     :", {c: int(lab[c].notna().sum()) for c in lab.columns})


if __name__ == "__main__":
    main()
