# SBA Approved Loan Amount Prediction Proposal

## Goal

Replace the current property loan sanction dataset with the SBA National loan dataset and predict `GrAppv`, the gross amount of loan approved by the bank. This remains a regression project, so the overall notebook structure can stay familiar, but the preprocessing schema should be rewritten around business-loan fields instead of borrower/property fields.

Sources: [SBA National dataset on Kaggle](https://www.kaggle.com/datasets/maddraf/sbanational-csv), [SBA 7(a) and 504 data dictionary](https://legacy.sba.gov/document/support--7a-504-loan-data-dictionary).

## Column Comparison

| Current column | Closest SBA column | Change needed |
|---|---|---|
| `Customer ID` | `LoanNr_ChkDgt` | Use as identifier/index, not a feature. |
| `Name` | `Name` | Drop. Borrower/business name is high-cardinality and not useful for a clean model. |
| `Gender` | none | Drop. SBA has business loans, not personal applicant demographics. |
| `Age` | none | Drop. |
| `Income (USD)` | none direct | Drop or approximate business capacity from `NoEmp`, `CreateJob`, `RetainedJob`, `NAICS`, `State`. |
| `Income Stability` | `NewExist`, `Term`, `LowDoc` | Replace with business/new-loan indicators. |
| `Profession` | `NAICS` | Replace with industry sector from first two NAICS digits. |
| `Type of Employment` | none direct | Drop. |
| `Location` | `City`, `State`, `Zip` | Replace with business location. Prefer `State` and maybe ZIP prefix. |
| `Loan Amount Request (USD)` | none direct | No true requested amount. Do not substitute `GrAppv`, since it is the target. |
| `Current Loan Expenses (USD)` | none | Drop. |
| `Expense Type 1` | none | Drop. |
| `Expense Type 2` | none | Drop. |
| `Dependents` | none | Drop. |
| `Credit Score` | none | Drop. SBA National does not include borrower credit score. |
| `No. of Defaults` | `MIS_Status`, `ChgOffPrinGr` | Do not use for `GrAppv`; these are post-loan performance leakage. |
| `Has Active Credit Card` | `RevLineCr` | Replace with revolving credit line flag. |
| `Property ID` | none | Drop. |
| `Property Age` | none | Drop. |
| `Property Type` | none | Drop. |
| `Property Location` | `State`, `City`, `Zip` | Replace with business geography. |
| `Co-Applicant` | none | Drop. |
| `Property Price` | none | Drop. |
| `Loan Sanction Amount (USD)` | `GrAppv` | New target. |

Verdict: this is not a drop-in dataset replacement. It is still a loan amount regression problem, so the modeling/evaluation skeleton can remain, but the preprocessing feature lists, target selection, leakage handling, and encoders should change.

## SBA Feature Plan

Use `GrAppv` as the target.

Keep as model features:

| SBA column | Treatment |
|---|---|
| `Term` | Numeric. |
| `NoEmp`, `CreateJob`, `RetainedJob` | Numeric business-size/employment features. |
| `NewExist`, `UrbanRural`, `RevLineCr`, `LowDoc` | Categorical/boolean after code cleanup. |
| `NAICS` | Convert to string, derive `NAICS_sector = first 2 digits`, optionally keep full code as categorical. |
| `City`, `State`, `Zip`, `Bank`, `BankState` | Categorical. Use state directly; use top-N or frequency encoding for high-cardinality city/bank/zip. |
| `ApprovalDate`, `ApprovalFY`, `DisbursementDate` | Parse dates; derive year/month. Use cautiously if doing time-based validation. |
| `FranchiseCode` | Convert to `is_franchise` flag where `00000`/`00001` mean no franchise. |

Drop from training for `GrAppv`:

| SBA column | Reason |
|---|---|
| `LoanNr_ChkDgt`, `Name` | Identifier/name. |
| `SBA_Appv` | Strong leakage because it is calculated from the approved amount. |
| `DisbursementGross`, `BalanceGross` | Occur after or near approval; likely leakage. |
| `MIS_Status`, `ChgOffDate`, `ChgOffPrinGr` | Future loan performance; leakage for approval amount. |

## Preprocessing Changes

1. Replace hardcoded current-dataset column lists with SBA-specific `numeric_features`, `categorical_features`, `date_features`, `drop_features`, and `target = "GrAppv"`.
2. Stop using `df_processed.iloc[:, :-1]` and `df_processed.iloc[:, -1]`; explicitly use `X = df.drop(columns=[target])` and `y = df[target]`.
3. Parse currency strings in `GrAppv`, `SBA_Appv`, `DisbursementGross`, `BalanceGross`, and `ChgOffPrinGr` by removing `$`, commas, and whitespace.
4. Normalize messy categorical codes, especially `RevLineCr` and `LowDoc`, into clean values such as `Y`, `N`, and `Unknown`.
5. Parse dates and derive compact features like `approval_year`, `approval_month`, and `disbursement_year`.
6. Replace `LabelEncoder` with a `ColumnTransformer`:
   - numeric: median imputation, optional scaling for linear models
   - categorical: most-frequent imputation plus `OneHotEncoder(handle_unknown="ignore")`
7. Train on `log1p(GrAppv)` using `TransformedTargetRegressor` or manual log/expm1 conversion because approved amounts are positive and skewed.
8. Keep outlier handling light. Prefer log transform and reporting loan-size slices over deleting large SBA loans with a Z-score rule.

## Model Changes

Keep or adapt:

| Current model | SBA version |
|---|---|
| Linear Regression | Replace with `Ridge` or `ElasticNet` as the linear baseline. |
| Random Forest Regressor | Keep. Useful nonlinear baseline. |
| Gradient Boosting Regressor | Replace or supplement with `HistGradientBoostingRegressor`. |
| Keras regression model | Optional. Keep only if time allows after stronger tabular models are working. |
| PCA + Linear Regression | Drop. Not very useful after one-hot/categorical feature engineering. |

Add:

| New model | Why |
|---|---|
| `XGBRegressor` or `LGBMRegressor` | Strong tabular regression model; likely top performer. |
| `CatBoostRegressor` | Excellent fit for many categorical fields like bank, state, NAICS, and city. |
| `DummyRegressor` | Simple baseline using mean/median target. |

Recommended final model set: `DummyRegressor`, `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `XGBRegressor` or `LGBMRegressor`, and `CatBoostRegressor`.

## Evaluation Changes

Use regression metrics on the original dollar scale:

| Metric | Purpose |
|---|---|
| MAE | Easy business interpretation: average dollar error. |
| RMSE | Penalizes large approval-amount mistakes. |
| R2 | Overall explained variance. |
| RMSLE | Good for skewed money targets when using log transform. |

Validation:

1. Start with an 80/20 random split for comparability with the current notebook.
2. Add a time-based split using `ApprovalFY`, for example train on older years and test on the latest years.
3. Report error by slices: `State`, `NAICS_sector`, `NewExist`, `UrbanRural`, and loan-size bands.

## Minimal Notebook Rewrite

The smallest clean rewrite is:

1. Change input file and target:
   - from `src/data/train.csv` and `Loan Sanction Amount (USD)`
   - to `src/data/SBAnational.csv` and `GrAppv`
2. Replace current manual preprocessing with an SBA preprocessing function.
3. Replace label encoding with sklearn pipelines.
4. Replace `LinearRegression` with `Ridge`.
5. Add XGBoost/LightGBM and CatBoost model sections.
6. Replace the current accuracy-style comparison with MAE/RMSE/R2/RMSLE plus a small slice-error table.

This is a medium rewrite, not a full project restart. The EDA, train/test split, model comparison table, and regression framing can stay. The column engineering and leakage rules are the main parts that must change.
