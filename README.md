## Review Request

Main notebook: 02_correct.ipynb

Feedback requested on:
1. 12-month default window
2. Optuna setup
3. AUC/Gini performance

# LendingClub Credit Risk Pipeline

A clean, end-to-end machine-learning pipeline that predicts **loan default** using the public LendingClub dataset. The focus is a reproducible, well-documented credit-risk workflow, not maximum AUC.

The model predicts default using only information available **at loan origination**, mimicking a real accept/reject decision.

## Approach

- **Model:** XGBoost (gradient-boosted trees), chosen for tabular data and native handling of missing values.
- **Target:** a binary 12-month default flag. A loan is bad if it reaches a terminal bad status (Charged Off / Default) within 12 months of origination, and good if it survives the first 12 months. Loans without a full 12-month window observed are excluded.
- **Leakage discipline:** an explicit keep/drop audit removes any field populated after origination (payments, recoveries, and similar), plus LendingClub's own grade and interest rate.
- **Validation:** two evaluation splits, a random split (optimistic, same era) and a time-based / out-of-time split (train on older loans, test on newer ones), to measure real-world generalisation.
- **Tuning:** Optuna with StratifiedKFold cross-validation inside the training set, with the test set untouched until final scoring.
- **Interpretation:** SHAP (global drivers + per-loan explanations).
- **Imbalance (~5% default):** headline metrics are AUC, PR-AUC, and Gini, not accuracy.

## Data

The data is **not included** in this repo (it is large and gitignored). Download it from Kaggle and place it in `data/lendingclub/`:

- Kaggle dataset: `husainsb/lendingclub-issued-loans`
- Expected files in `data/lendingclub/`:
  - `lc_loan.csv` — loans issued 2007 to 2015 (training period)
  - `lc_2016_2017.csv` — loans issued 2016 to 2017 (out-of-time test period)
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
├── data/lendingclub/        # raw data, gitignored, download from Kaggle
├── requirements.txt
└── README.md
```

## Pipeline sections (in the notebook)

1. **Problem Framing** — objective, 12-month target definition, why it matters.
2. **Data Understanding** — file inventory, schema and date checks, EDA (default rate by grade, term, DTI, income, and over time), missingness analysis, feature scope / leakage audit, distributions and correlation.
3. **Data Preparation** — data cleaning, feature engineering (numeric conversions, one-hot encoding, target-encoding setup), and the random + out-of-time split strategy.
4. **Modelling** — untuned XGBoost baseline, then hyperparameter tuning with Optuna.
5. **Evaluation & Interpretation** — final test performance (random vs out-of-time, untuned vs tuned) and SHAP interpretation (global + local).
6. **Conclusion** — findings, limitations, and what carries forward to the wider credit-risk work.
