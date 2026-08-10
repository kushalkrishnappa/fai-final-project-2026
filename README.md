# Loan Sanction Amount Estimation

## Objective

A bank rarely sanctions the exact amount an applicant asks for. The applicant learns
this weeks into the process, once the paperwork is already filed.

This project estimates that figure up front, from details the applicant already knows.
Income, credit score, existing loan expenses and property price. It is an early
estimate, not a lending decision.

Everything runs from `src/LoanSanctionAmountEstimation.ipynb`, which produces:

1. A model that predicts the sanctioned amount in US dollars for one application, and
   whether anything is sanctioned at all.
2. A score for it on applications held out from training, measured with both stages
   chained together.
3. A record of what the raw file held and which parts of it were unusable.

## Setup

Built and run on Python 3.14.4. The pins in `requirements.txt` were resolved against that
interpreter, so it is the version to reach for if anything fails to install.

**1. Create an environment and install the dependencies.**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyterlab ipywidgets
```

`requirements.txt` pins the nine packages the analysis imports. `jupyterlab` is what runs the
notebook and `ipywidgets` supplies the progress bar SHAP looks for, warning when it is absent.
Neither is used by the analysis itself, which is why they are installed separately.

**2. Check the data is in place.**

`src/data/train.csv` is committed to the repository, so there is nothing to download. If it is
missing, take `train.csv` from the Kaggle dataset linked under [Data](#data) and put it back at
that path.

**3. Run the notebook from `src/`.**

```bash
cd src
jupyter lab LoanSanctionAmountEstimation.ipynb
```

The working directory matters. `DATA` is `data/train.csv` and `MODELS` is `../models`, both
written relative to `src/`, so launching from the repository root makes the first read fail.
Once it opens, **Run → Restart Kernel and Run All Cells**.

**4. Expect the first run to be much slower than later ones.**

`models/` holds the fitted estimators and is excluded by `.gitignore`, since the two random
forests alone come to 190 MB. A fresh clone therefore starts with no cache, and `fit_or_load`
trains each model and writes it to `models/` the first time regardless of the `RETRAIN` flag.
That means four grid searches, every candidate cross validated over five folds, plus a 50 epoch
PyTorch loop that is never cached. Every run after that prints `loading cached ...` instead and
finishes in a few minutes.

`RANDOM_STATE` is pinned to 42 throughout, so a rerun reproduces the figures quoted in the
narration. Setting `RETRAIN = True` in the setup cell forces the searches to run again and
overwrite the cache.

## Approach

![The two stage design](report/figures/two_stage.png)

The target column stacks two questions. Is anything sanctioned, and if yes, how much?

26.8 percent of applications are refused and carry a sanctioned amount of exactly 0.
That 0 is a rejection, not a very small loan. One regressor over the whole column has
to aim at 0 for a quarter of the rows and at a large amount for the rest, so it lands
in between and fits neither. Refused applicants get small positive amounts, and the
approved predictions get pulled down.

Two stages avoid that. Each one answers a single question, and the two chain at
prediction time. Whatever stage 1 marks as refused is reported as 0 and never reaches
the regressor.

## Stage 1, is anything sanctioned

A binary classifier over every application, scored on F1 so that refusals cannot be
ignored. Only what it approves reaches stage 2.

- **Dummy (most frequent)**, always predicts approved. This is the bar to clear.
- **Logistic regression**, tuned over the regularisation strength `C`.
- **Random forest**, tuned over tree depth and minimum leaf size.
- **Gradient boosting**, tuned over learning rate and tree depth.

## Stage 2, how much

A regressor for the sanctioned amount in USD, fitted and scored on approved rows only.

- **Dummy (mean)**, always predicts the mean sanctioned amount. The bar to clear here.
- **Linear regression**, untuned. The simple model the others have to beat.
- **PCA then linear regression**, the same fit on 10 principal components.
- **Random forest**, tuned over tree count, depth, leaf size and the feature sampling
  rule.
- **Gradient boosting**, tuned over learning rate, depth, tree count and subsample
  fraction.
- **Neural network**, a PyTorch MLP with dropout and batch normalisation, on a
  standardised target.

## Data

Kaggle, "Predict Loan Amount Data":
<https://www.kaggle.com/datasets/phileinsophos/predict-loan-amount-data>

The file used is `train.csv`, kept at `src/data/train.csv`. 30,000 applications in 24
columns, with `Loan Sanction Amount (USD)` as the target.
