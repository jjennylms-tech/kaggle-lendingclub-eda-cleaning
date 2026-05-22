# LendingClub Credit Risk Pipeline

A clean, end-to-end machine-learning pipeline that predicts **loan default** using the public LendingClub dataset. The focus is a reproducible, well-documented credit-risk workflow — not maximum AUC.

The model predicts default using only information available **at loan origination**, mimicking a real accept/reject decision.

## Approach

- **Models:** logistic regression (interpretable baseline) + XGBoost (primary).
- **Validation:** two evaluation splits — a random split (optimistic) and a **time-based / out-of-time** split (train on older loans, test on newer), to measure real-world generalisation.
- **Tuning:** Optuna with StratifiedKFold cross-validation.
- **Interpretation:** SHAP (global + per-loan explanations).
- **Imbalance (~20% default):** headline metrics are AUC and PR-AUC, not accuracy.

## Data

The data is **not included** in this repo (it's large and gitignored). Download it from Kaggle and place it in `data/lendingclub/`:

- Kaggle dataset: `husainsb/lendingclub-issued-loans`
- Expected files in `data/lendingclub/`:
  - `lc_loan.csv` — loans issued 2007–2015 (training period)
  - `lc_2016_2017.csv` — loans issued 2016–2017 (out-of-time test period)
  - `us-state-codes.csv` — state-code lookup (display only)
  - `LCDataDictionary.xlsx` — column definitions (reference)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then open the notebook and run top to bottom:

```bash
jupyter notebook notebooks/02_correct.ipynb
```

## Project structure

```
.
├── notebooks/
│   └── 02_correct.ipynb     # the end-to-end pipeline
├── src/                     # reusable utilities (placeholder)
├── data/lendingclub/        # raw data — gitignored, download from Kaggle
├── requirements.txt
└── README.md
```

## Pipeline sections (in the notebook)

1. **Problem Framing** — objective, target definition, success criteria.
2. **Data Understanding** — inventory, schema/date checks, EDA (default rate by grade, term, DTI, income, time), missingness analysis, correlation.
3. **Data Preparation** — target creation, leakage audit & column categorisation, missing-value treatment, cleaning, train/test split strategy.
4. **Modelling** — logistic-regression baseline, XGBoost baseline, Optuna tuning.
5. **Evaluation & Interpretation** — final test performance (random + time-based), stress-test by slices, SHAP.
6. **Conclusions.**
