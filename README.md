# LendWise — Predicting Loan Sanction Amounts

Given an applicant's personal and financial profile, predict **whether a loan will be sanctioned
at all** and, if so, **how much**.

## Problem

Estimating loan eligibility is slow and manual, and applicants rarely get an upfront figure. This
project builds a data-driven estimator over 30,000 historical US loan applications.

Roughly **27% of applicants are sanctioned nothing**. A single regressor cannot express a
rejection, so the system is deliberately built in two stages:

| Stage | Model | Question |
|---|---|---|
| 1 | Gradient Boosting **classifier** | Will anything be sanctioned? |
| 2 | Gradient Boosting **regressor** | If so, how much? |

Final prediction = `stage1(x) × stage2(x)`, which is evaluated end-to-end on all applicants
including rejections.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── NOTEBOOK_REVIEW.md        # correctness audit of the original notebook
├── src/
│   ├── LendWise.ipynb        # the project: cleaning → EDA → modelling → evaluation
│   └── data/
│       ├── train.csv         # source dataset (30,000 rows, 24 columns)
│       └── preprocessed.csv  # written by the cleaning section
└── rubric/                   # assignment brief and grading rubric
```

## Setup

Requires **Python 3.11+** (developed and verified on 3.14.4).

```bash
git clone <repo-url> && cd FAI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name lendwise --display-name "LendWise"
```

## Running

```bash
jupyter lab src/LendWise.ipynb
```

Select the **LendWise** kernel, then *Kernel → Restart Kernel and Run All Cells*.

The notebook reads `data/train.csv` **relative to `src/`**, so launch Jupyter from the repo root
(or `cd src` first). It runs top to bottom with no manual steps and no `pip install` cells.

**Expected runtime: 15–25 minutes**, dominated by two `GridSearchCV` sweeps (~360 fits each) and
the SHAP explanations. If you only need the results, read the saved outputs — they are committed
with the notebook.

> **Note on TensorFlow.** The neural network is implemented in **PyTorch**, not Keras.
> TensorFlow publishes no wheel for Python 3.14, so `pip install tensorflow` fails outright on
> this interpreter. PyTorch covers the same architecture.

## Method

**Cleaning** — `-999` sentinels converted to `NaN`; rows with a missing target dropped;
`Property Age` dropped after an audit showed it to be a byte-for-byte duplicate of
`Income (USD)`; feature outliers (|z| > 3) smoothed for the EDA copy only, with the **target left
untouched**.

**Leakage control** — imputation, scaling and one-hot encoding all live inside a
`ColumnTransformer` fitted on the training split only, once per cross-validation fold. Nothing
is fitted on the full dataset. Row-wise transforms (`log1p`, `sqrt`) are applied before the split
because they learn nothing from the data.

**Encoding** — nominal categories are one-hot encoded rather than label encoded, so the linear
and PCA models are not handicapped by an invented ordering over `Profession` and `Location`.

**Models compared** (all on the same preprocessor, same split, same metric): Gradient Boosting,
Random Forest, Linear Regression, PCA + Linear Regression, and a PyTorch neural network — each
benchmarked against trivial baselines, including OLS on `Loan Amount Request` alone.

**Evaluation** — RMSE / MAE / R² for the regressors; accuracy, precision, recall, F1, ROC-AUC and
a confusion matrix for the classifier; SHAP for feature attribution; learning curves for
data-sufficiency; end-to-end scoring of the combined system on all applicants.

## Dataset

`src/data/train.csv` — 30,000 loan applications, 24 columns (demographics, income, credit score,
property details). Target: `Loan Sanction Amount (USD)`.

## Known limitations

See §9 of the notebook. In brief: the target is close to a linear function of
`Loan Amount Request`, so baselines are strong; residuals are heteroscedastic; the split is
random rather than chronological; and `Gender` is currently a model input, which would require a
disparate-impact review (and most likely removal) before any real credit decisioning use.
