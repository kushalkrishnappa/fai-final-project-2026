# Loan Sanction Amount Estimation

## Objective

A bank almost never sanctions the exact figure an applicant asks for, and the applicant
usually finds out only weeks into the process. This project
(`src/LoanSanctionAmountEstimation.ipynb`) estimates that figure up front from details
the applicant already knows, such as income, credit score, existing loan expenses and
property price. It is an early estimate, not a lending decision.

1. A model that takes a single application and predicts the sanctioned amount in US
   dollars, together with whether any amount is sanctioned at all.
2. An honest score for it, measured on applications it never saw during training and
   reported end to end rather than on whichever subset flatters each stage.
3. A record of what the raw file contained and which parts of it were unusable.

## Approach

![The two stage design](report/figures/two_stage.png)

The target column stacks two different questions: is anything sanctioned at all, and if
so how much. 26.8 percent of applications are refused outright and their sanctioned
amount is exactly 0, which records a rejection rather than a very small loan. A single
regressor fitted over the whole column has to aim at 0 for a quarter of the rows and at
a large amount for the rest, so it settles in between and serves neither, predicting
small positive amounts for refused applicants while pulling the approved predictions
down. Splitting the problem gives each stage one question to answer. At prediction time
they chain: an application stage 1 marks as refused is reported as 0 and never reaches
the regressor.

## Stage 1, is anything sanctioned at all

A binary classifier on every application, scored on F1 so refusals cannot be ignored.
Only applications it approves reach stage 2.

- **Dummy (most frequent)**, always predicts approved, the bar a real model has to clear.
- **Logistic regression**, tuned over the regularisation strength `C`.
- **Random forest**, tuned over tree depth and minimum leaf size.
- **Gradient boosting**, tuned over learning rate and tree depth.

## Stage 2, how much

A regressor for the sanctioned amount in USD, fitted and scored on approved rows only.

- **Dummy (mean)**, always predicts the mean sanctioned amount, the bar to clear.
- **Linear regression**, untuned, the simple model the rest have to justify improving on.
- **PCA then linear regression**, the same fit on 10 principal components.
- **Random forest**, tuned over tree count, depth, leaf size and the feature sampling rule.
- **Gradient boosting**, tuned over learning rate, depth, tree count and subsample fraction.
- **Neural network**, a PyTorch MLP with dropout and batch normalisation on a standardised
  target.

## Data

Kaggle, "Predict Loan Amount Data":
<https://www.kaggle.com/datasets/phileinsophos/predict-loan-amount-data>. The file used
is `train.csv`, kept at `src/data/train.csv`: 30,000 applications in 24 columns, target
`Loan Sanction Amount (USD)`.
