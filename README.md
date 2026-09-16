# PPD Detection — Model Experiments

**Results: see [RESULTS.md](RESULTS.md).** Best model: Gradient Boosting on pooled NHANES, AUC 0.713.

## Run

```bash
pip install -r requirements.txt
python nhanes_build.py    # downloads NHANES (free, no application) -> nhanes_perinatal.csv
python nhanes_model.py    # depression risk models
```
