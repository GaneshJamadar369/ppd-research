# PPD Detection — Model Experiments

Kaggle "PostPartum Depression" survey (1,503 responses) → binary classifier for `Suicide attempt`.

**Results: see [RESULTS.md](RESULTS.md).** Best model: Random Forest, AUC 0.730.

## Run it

```bash
pip install -r requirements.txt
python ppd_improved_baselines.py
```

## Files

- `post natal data.csv` — dataset
- `ppd_preprocessing.py` — cleaning / encoding / imputation (imported by the script below)
- `ppd_improved_baselines.py` — trains + evaluates 14 models, outputs to `results/`
