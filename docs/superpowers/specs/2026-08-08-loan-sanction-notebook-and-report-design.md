# Loan Sanction Amount Estimation: notebook rebuild and report

Date: 2026-08-08
Status: approved, ready for implementation planning

## 1. Context

`src/LendWise.ipynb` was executed end to end on the pytorch kernel
(`~/.pyenv/versions/pytorch/bin/python`, Python 3.14.4). All 98 code cells ran. Two failed hard and
three passed while producing wrong values. A rubric review against
`rubric/ProjectReportAndSubmissionDetails.pdf` estimated roughly 58/100.

Defects found, all verified against the data or the notebook's own stored output:

| ID | Defect | Evidence |
|----|--------|----------|
| D1 | `tensorflow` import fails | No cp314 wheel exists. Cell 160 `ModuleNotFoundError` |
| D2 | Cell 163 cascade | `NameError: history is not defined` |
| D3 | Cells 162, 165, 167 silently report Gradient Boosting results under a Neural Network heading | Cell 162 printed `9028.976...`, identical to cell 120's GB RMSE |
| D4 | `max_features='auto'` removed in sklearn 1.3, so 180 of 540 GB fits scored NaN | All `auto` fits logged `total time= 0.0s` in both the original and the fresh run |
| D5 | `Property Age` duplicates `Income (USD)` | Identical to the cent in all 25,150 rows where both are non-null |
| D6 | `/365.25` "convert Property Age to years" is meaningless | The column holds USD |
| D7 | Leakage: imputation, outlier replacement and label encoding all fitted before the split | Cells 20, 23, 114 precede cell 116 |
| D8 | `LabelEncoder` on nominal categories feeds ordinal meaning to Linear Regression and PCA | Cell 114 |
| D9 | Mean substitution for z-score outliers produces `Dependents = 2.23` | Verified in processed data |
| D10 | Report's Random Forest "RMSE 3854.27" is the MAE | Cell 136 prints RMSE 9141.60 then MAE 3854.27 |
| D11 | Report's Gradient Boosting "MSE 0.25" is wrong, should be about 0.05 | `RMSE^2/var(y_test)` with y_test std 40271 reproduces every other reported R2 to 4 dp |
| D12 | Cell 121 states RMSE 9157.37 and 8864.41 in the same markdown cell | Cell 121 |
| D13 | PCA claim "3 to 4 components suffice" contradicted by its own output | Cumulative variance 0.366 at 3, 0.423 at 4, 0.741 at 10 |
| D14 | "Not overfitting" claim unsupported by the figure shown | Only CV validation MSE is plotted, no training curve |
| D15 | 26.8% of rows dropped silently | 7,865 of 29,322 have sanction 0, and rejection rate varies by group |
| D16 | Dead code presented as the imputation strategy | `KNNImputer`, `knn_num_cols`, `mean_num_cols`, `knn_cat` never used |
| D17 | Variable reuse creates order dependence | `best_model` and `y_pred` reassigned across models |
| D18 | Notebook never run top to bottom | Execution counts out of order, cells 118, 120, 126 hold no output |
| D19 | Global `warnings.filterwarnings("ignore")` hid D4 | Cell 1 |

## 2. Goals

Produce `src/LoanSanctionAmountEstimation.ipynb` and a report that satisfy the CS5100 rubric, with
every defect above either fixed or explicitly disclosed as a limitation.

## 3. Non-goals

- Modifying `src/LendWise.ipynb`. It stays as the original record.
- Any git commit. The user asked for none.
- A deployed application. A Streamlit demo was suggested during review and is out of scope here.

## 4. Decisions

| Decision | Choice |
|---|---|
| Fix depth | Full methodology fix |
| Zero sanction rows | Two-stage model, classifier then regressor |
| References | Web searched and verified before citing |
| Report format | `report/Report.md` built to `report/Report.pdf` with pandoc and xelatex |
| Run time | Trimmed grids plus joblib caching to `models/` |
| Authorship | Kushal Krishnappa, Deepak, Rohit, Dips. Work split assigned, see section 7.1 |
| Course and title | CS5100. Project titled "Loan Sanction Amount Estimation" |
| Voice | Impersonal. No "we", no "I", no em-dash, no serial comma, IELTS 7 register |
| Diagrams | Mermaid, single source, rendered to PNG for the PDF. See section 11 |
| EDA | Performed on the training split only, after the split |
| Encoding | One shared one-hot encoder for every model, replacing `LabelEncoder` |
| Outliers | Percentile clipping at 1 and 99, fitted on train, continuous columns only |

## 5. Notebook architecture

Two models chained, since stage 1 decides whether stage 2 applies.

```
all 29,322 applicants
        |
   STAGE 1 classifier: is anything sanctioned?
        |
  no ---+--- yes
   |          |
 predict 0   STAGE 2 regressor: how much?
        |
  END TO END score across all test rows
```

Block layout:

| Block | Contents |
|---|---|
| 0 | Title, problem, goal, dataset, metric, references |
| 1 | Config: `RANDOM_STATE = 42`, `RETRAIN = False`, paths |
| 2 | Load, convert `-999` to NaN, drop the duplicated column |
| 3 | Data quality audit, including the duplicate detection shown as evidence |
| 4 | Train and test split |
| 5 | EDA on the training split, univariate then multivariate |
| 6 | Preprocessing pipeline definition |
| 7 | Stage 1: `DummyClassifier` baseline, then tuned classifiers |
| 8 | Stage 2: `DummyRegressor` baseline, Linear, PCA, Random Forest, Gradient Boosting, PyTorch MLP |
| 9 | End to end evaluation on all test rows |
| 10 | SHAP for both stages |
| 11 | Results table, conclusion, limitations |

### 5.1 Preprocessing

A single `ColumnTransformer` inside a `Pipeline`, fitted on training folds only:

- numeric: percentile clip fitted on train, then `SimpleImputer(strategy="median")`, then `StandardScaler`
- categorical: `SimpleImputer(strategy="most_frequent")`, then `OneHotEncoder(handle_unknown="ignore")`

Applied before the split, because these are schema fixes and not fitted statistics:
`-999` to NaN, dropping `Property Age`, dropping `Name`, `Customer ID` and `Property ID`.

`Type of Employment` is kept, unlike the original which dropped it. It is a genuine categorical
feature and the one-hot encoder plus most-frequent imputation handles its missingness, so there is
no reason to discard it.

Discrete columns `Dependents` and `No. of Defaults` are excluded from clipping, which is what
prevents D9.

The split between what runs before and after the train and test split is deliberate. Block 3 is a
schema audit only, meaning missingness counts, sentinel detection and duplicate column detection,
and it runs on the full data because it informs no modelling choice. Block 5 is EDA proper, meaning
distributions, correlations and relationships with the target, and it runs on the training split
only because those do drive modelling choices.

### 5.2 Stage 1 classifier

Target: `Loan Sanction Amount (USD) > 0`. Class balance is roughly 73 to 27, so metrics must not
rely on accuracy alone. Reported: accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
The rubric names accuracy, F1 and a confusion matrix explicitly.

Candidates: `DummyClassifier(strategy="most_frequent")` as the baseline, then `LogisticRegression`,
`RandomForestClassifier` and `GradientBoostingClassifier`. Model selection scores on F1, since
accuracy would reward always predicting approval.

### 5.3 Stage 2 regressor

Trained on approved rows only. Reported for every model on the original USD scale: RMSE, MAE and R2.
No mixing of scaled and unscaled figures in one table, which is what produced D10 and D11. Every
search selects on `neg_root_mean_squared_error`, so the models are compared on one criterion. The
original used a custom RMSE scorer for Gradient Boosting and `neg_mean_squared_error` for Random
Forest, which meant the two were never selected under the same rule.

Trimmed grids are centred on the region that won the exhaustive searches already run, Gradient
Boosting at `learning_rate 0.05, max_depth 5, max_features None, n_estimators 100, subsample 0.8`
and Random Forest at `max_depth None, max_features None, min_samples_leaf 2, min_samples_split 2,
n_estimators 100`, and keep one step either side of each winner so the search still means something:

- Gradient Boosting: `learning_rate [0.05, 0.1]`, `max_depth [3, 5]`, `n_estimators [100, 200]`,
  `subsample [0.8, 1.0]`, giving 16 candidates and 80 fits, down from 108 and 540.
- Random Forest: `n_estimators [100, 300]`, `max_depth [None, 15]`, `min_samples_leaf [1, 2]`,
  `max_features [None, "sqrt"]`, giving 16 candidates and 80 fits, down from 108 and 540.

`max_features='auto'` is gone and `error_score='raise'` is set, so a bad grid fails loudly instead
of scoring NaN. That pair is what fixes D4.

The PyTorch MLP keeps the original topology and matches Keras defaults that differ in torch: Glorot
uniform init, `BatchNorm1d(eps=1e-3, momentum=0.01)`, `Adam(eps=1e-7)`, and the L2 penalty applied
to only the three originally regularised layers rather than through `weight_decay`. A standalone
run of this port produced R2 0.857, MSE 0.147, MAE 0.291 and RMSE 15233, against the original
Keras 0.853, 0.151, 0.297 and 15441.

### 5.4 End to end evaluation

Across all test rows including rejections: if stage 1 predicts no, the prediction is 0, otherwise it
is the stage 2 output. Reported as RMSE, MAE and R2 over the full test set. This is the headline
number and it has no counterpart in the old work.

### 5.5 Caching

Each fitted estimator is written to `models/` with joblib. A cell loads a saved model when present
and retrains only when missing or when `RETRAIN = True`. First run about 12 to 15 minutes, later
runs under a minute.

## 6. Markdown narration

Every code cell is preceded by a markdown cell giving the reason for the step. Every cell producing
a figure or a number is followed by a short reading of what it shows.

Hard rule: no number appears in any markdown cell until the notebook has printed it. Narration is
written with markers, the notebook is executed, then the real values are filled in. D12 came from
breaking this rule.

Textbook definitions are replaced by the reason for the choice in this specific problem.

## 7. Report

`report/Report.md`, about 8 pages, built to PDF.

| Section | Contents | Pages |
|---|---|---|
| Title | Title, four names, CS5100, date | 0.5 |
| 1 Introduction | Problem, motivation, goals, 2 citations | 1 |
| 2 Background | Related work with citations, plain explanation of methods | 1.5 |
| 3 Methodology | Tools, dataset link, data quality findings, pipeline, two-stage design | 2 |
| 4 Results | Stage 1 table and confusion matrix, stage 2 table, end to end score, SHAP | 2 |
| 5 Discussion | Interpretation, limitations, improvements | 1 |
| 6 Conclusion | Achievements, lessons learned, individual contributions | 0.5 |
| 7 References | Verified, each tied to a claim | 0.5 |

### 7.1 Individual contributions

Four roughly equal blocks of work, one per member. This section of the report states the split
plainly and does not claim anything beyond it.

| Member | Owns |
|---|---|
| Kushal Krishnappa | Data quality audit and the preprocessing pipeline, including sentinel handling, the duplicate column finding, clipping and encoding |
| Deepak | Exploratory data analysis and all figures, univariate through to correlation analysis |
| Rohit | Stage 1 classifier, threshold and metric choice, and the end to end evaluation |
| Dips | Stage 2 regressors including the PyTorch network, plus SHAP interpretation |

Report writing and final review are shared across all four.

### 7.2 Figures

Roughly seven figures exported from the executed notebook to `report/figures/`: missingness,
correlation heatmap, target distribution, confusion matrix, predicted against actual, SHAP summary
and the two-stage flow diagram.

The report is standalone and does not reference the earlier version. Corrected figures are simply
presented as the results. The duplicated column is reported as an audit finding, since a marker who
checks the data will see it.

## 8. File layout

```
src/LoanSanctionAmountEstimation.ipynb   new
src/LendWise.ipynb                       untouched
models/                                  cached estimators
report/Report.md, Report.pdf, figures/
requirements.txt
docs/superpowers/specs/2026-08-08-loan-sanction-notebook-and-report-design.md
```

## 9. Verification criteria

1. Restart and Run All completes with zero error outputs and execution counts 1..N in order.
2. Every number quoted in notebook markdown and in the report matches a printed output.
3. No `LabelEncoder` on nominal columns, no fitted transformer touching the test set.
4. `Dependents` holds whole numbers only.
5. Stage 1, stage 2 and end to end metrics all present.
6. Every reference resolves to a real paper.
7. Report builds to PDF and lands within 5 to 10 pages.

## 10. Diagrams

All diagrams are written in mermaid and kept in one place, `report/diagrams/*.mmd`. Nothing is drawn
twice, so the notebook and the report cannot drift apart.

- Notebook: the mermaid source goes straight into a markdown cell. JupyterLab 4.6 renders it live,
  and so does GitHub if the repository is viewed there.
- Report: `mmdc` renders the same `.mmd` file to `report/figures/*.png`, and the markdown embeds the
  PNG. Pandoc and xelatex have no mermaid support of their own, which is why the render step exists.

Toolchain, installed and verified on 2026-08-08:

```
npm install -g @mermaid-js/mermaid-cli          # mmdc 11.16.0
mmdc -i x.mmd -o x.png -p pptr.json -b white -s 3
```

`pptr.json` points puppeteer at the Google Chrome already on the machine, so no second Chromium is
downloaded:

```json
{"executablePath":"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","args":["--no-sandbox"]}
```

Two practical notes from the test render. Flowcharts default to `TD` and come out very tall, which
overflows a PDF page, so report diagrams use `LR` where the shape allows. Image embeds also need an
explicit width such as `{width=70%}`, otherwise xelatex warns that the float is too large.

Diagrams planned: the two-stage flow, and the preprocessing pipeline.

## 11. Open items

- Full surnames for Deepak, Rohit and Dips, if the title page needs them. First names are used until
  they are supplied.
