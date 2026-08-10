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
