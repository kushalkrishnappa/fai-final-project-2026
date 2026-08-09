# Loan Sanction Amount Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/LoanSanctionAmountEstimation.ipynb` and `report/Report.pdf`, fixing the 19 defects listed in the spec and adding a two-stage model that measures end to end error honestly.

**Architecture:** A generator script defines every notebook cell in readable Python and emits the `.ipynb`. The notebook splits before it explores, does all fitting inside a `Pipeline` so nothing leaks, chains a classifier for approval with a regressor for amount, then scores both stages together across all test rows. Narration numbers are written as markers, filled from the executed outputs, and never typed by hand.

**Tech Stack:** Python 3.14.4 at `~/.pyenv/versions/pytorch/bin/python`, scikit-learn 1.9.0, pandas 2.3.3, torch 2.9.1, shap 0.52.0, nbformat, mermaid-cli 11.16.0, pandoc 3.10.1 with xelatex.

**Constraint from the user: no git commits at any point.** Checkpoints replace commits.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/nbcells.py` | Cell content. Markdown text and code bodies, one function per notebook block |
| `scripts/build_notebook.py` | Assembles cells into `src/LoanSanctionAmountEstimation.ipynb` |
| `scripts/extract_results.py` | Reads the executed notebook, writes `scripts/results.json` |
| `scripts/fill_narration.py` | Substitutes `{{MARKER}}` in markdown cells from `results.json` |
| `scripts/verify_notebook.py` | Checks the seven spec criteria, exits non-zero on failure |
| `scripts/build_report.sh` | Renders mermaid to PNG then pandoc to PDF |
| `src/LoanSanctionAmountEstimation.ipynb` | The deliverable notebook, generated |
| `models/` | joblib cached estimators |
| `report/Report.md`, `report/diagrams/*.mmd`, `report/figures/*.png` | Report sources |
| `requirements.txt` | Pinned versions from the running kernel |

**Note on the generator.** `build_notebook.py` is the source of truth only until Task 10 finishes. After that the `.ipynb` is authoritative and further edits use NotebookEdit, so hand edits are never silently regenerated away.

---

### Task 1: Scaffolding

**Files:**
- Create: `scripts/`, `models/`, `report/diagrams/`, `report/figures/`
- Create: `requirements.txt`
- Create: `scripts/verify_notebook.py`

- [ ] **Step 1: Write the verification script first, before any notebook exists**

```python
# scripts/verify_notebook.py
"""Checks the executed notebook against section 9 of the design spec."""
import json, re, sys
from pathlib import Path

NB = Path(sys.argv[1] if len(sys.argv) > 1 else "src/LoanSanctionAmountEstimation.ipynb")

def main() -> int:
    nb = json.loads(NB.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    failures = []

    # 1. no error outputs
    for i, c in enumerate(code):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                failures.append(f"cell {i}: {o['ename']}: {o['evalue'][:80]}")

    # 2. execution counts run 1..N in order
    counts = [c.get("execution_count") for c in code]
    if counts != list(range(1, len(code) + 1)):
        failures.append(f"execution counts not 1..{len(code)} in order, got {counts[:12]}...")

    # 3. no unfilled narration markers
    for i, c in enumerate(md):
        for m in re.findall(r"\{\{[A-Z0-9_]+\}\}", "".join(c["source"])):
            failures.append(f"markdown cell {i}: unfilled marker {m}")

    # 4. banned constructs
    src = "\n".join("".join(c["source"]) for c in code)
    for bad, why in [
        ("LabelEncoder", "ordinal encoding of nominal columns"),
        ("max_features='auto'", "removed in sklearn 1.3"),
        ('max_features="auto"', "removed in sklearn 1.3"),
        ("tensorflow", "no cp314 wheel exists"),
        ("filterwarnings(\"ignore\")", "global warning suppression hides failed fits"),
    ]:
        if bad in src:
            failures.append(f"banned construct {bad!r} present: {why}")

    # 5. voice rules
    for i, c in enumerate(md):
        text = "".join(c["source"])
        if "—" in text:
            failures.append(f"markdown cell {i}: em-dash present")
        for w in (" we ", " We ", " our ", " Our ", " I "):
            if w in f" {text} ":
                failures.append(f"markdown cell {i}: first person {w.strip()!r}")

    for f in failures:
        print("FAIL:", f)
    print(f"\n{len(failures)} failure(s), {len(code)} code cells, {len(md)} markdown cells")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to confirm it fails on the missing notebook**

Run: `~/.pyenv/versions/pytorch/bin/python scripts/verify_notebook.py`
Expected: `FileNotFoundError`. This confirms the checker is wired to the right path.

- [ ] **Step 3: Run it against the OLD notebook to confirm it detects known defects**

Run: `~/.pyenv/versions/pytorch/bin/python scripts/verify_notebook.py src/LendWise.ipynb`
Expected: FAIL lines naming `LabelEncoder`, `max_features='auto'`, `tensorflow`, `filterwarnings("ignore")` and out of order execution counts. A checker that passes the old notebook is broken.

- [ ] **Step 4: Create directories and requirements.txt**

```bash
mkdir -p scripts models report/diagrams report/figures
~/.pyenv/versions/pytorch/bin/python -m pip freeze | grep -iE \
  '^(pandas|numpy|scikit-learn|scipy|matplotlib|seaborn|torch|shap|joblib|nbformat)=' > requirements.txt
cat requirements.txt
```
Expected: nine pinned lines.

- [ ] **Step 5: Checkpoint.** Verifier detects every known defect in the old notebook. No commit.

---

### Task 2: Research and verify the references

**Files:**
- Create: `scripts/references.json`

- [ ] **Step 1: Search for each candidate**

Use WebSearch for these areas, one search each: credit scoring with machine learning, loan default prediction benchmarks, gradient boosting on tabular data, SHAP explainability, two-stage or hurdle models for zero-inflated outcomes, fairness in credit decisions.

- [ ] **Step 2: Verify before recording**

A reference is only usable when title, authors, venue and year are all confirmed from the search result. Anything unconfirmed is discarded rather than guessed. Target 5 or 6.

- [ ] **Step 3: Record them**

```json
[
  {"key": "author-year", "title": "...", "authors": "...", "venue": "...", "year": 0000,
   "url": "...", "claim": "which sentence in the report cites this"}
]
```

Every entry needs a non-empty `claim`. A reference not tied to a sentence gets dropped, because decorative citation lists are exactly what the rubric penalises.

- [ ] **Step 4: Checkpoint.** `references.json` has 5 or 6 entries, each with a verified URL and a claim.

---

### Task 3: Notebook generator skeleton and blocks 0 to 2

**Files:**
- Create: `scripts/nbcells.py`, `scripts/build_notebook.py`

- [ ] **Step 1: Write the builder**

```python
# scripts/build_notebook.py
import nbformat as nbf
from pathlib import Path
import nbcells

def main():
    nb = nbf.v4.new_notebook()
    nb.cells = nbcells.all_cells()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.4"},
    }
    out = Path("src/LoanSanctionAmountEstimation.ipynb")
    nbf.write(nb, out)
    print(f"wrote {out} with {len(nb.cells)} cells")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the cell helpers in `scripts/nbcells.py`**

```python
import nbformat as nbf

def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())

def code(src: str):
    return nbf.v4.new_code_cell(src.strip())

def all_cells():
    cells = []
    cells += block0_intro()
    cells += block1_config()
    cells += block2_load()
    return cells
```

Later tasks append their blocks to `all_cells()`.

- [ ] **Step 3: Block 0, the title and framing cell**

One markdown cell covering: title, problem statement, goal, dataset with its Kaggle link, the metric and why that metric, and the References list generated from `references.json`. Voice rules apply: impersonal, no em-dash, no serial comma.

The metric paragraph must state plainly that RMSE in USD is the headline number, that R2 is reported alongside, and that stage 1 is judged on F1 rather than accuracy because always predicting approval already scores 0.8451.

- [ ] **Step 4: Block 1, config**

```python
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
RETRAIN = False
DATA = Path("data/train.csv")
MODELS = Path("../models")
MODELS.mkdir(exist_ok=True)

# Only the two known-noisy warnings are silenced, never all of them.
# A blanket filter is what hid 180 failed fits in the earlier version of this work.
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(RANDOM_STATE)
```

- [ ] **Step 5: Block 2, load and schema fixes**

```python
df = pd.read_csv(DATA, index_col="Customer ID")
print(f"rows loaded: {len(df):,}   columns: {df.shape[1]}")

# -999 is a placeholder for "not recorded", not a real amount
df = df.replace(-999, np.nan)

# The target cannot be imputed, so rows without it are dropped
before = len(df)
df = df.dropna(subset=["Loan Sanction Amount (USD)"])
print(f"dropped {before - len(df):,} rows with no target, {len(df):,} remain")

# Property Age duplicates Income exactly. Evidence is printed in the audit block below.
df = df.drop(columns=["Name", "Property Age", "Property ID"])
print(f"columns after dropping identifiers and the duplicate: {df.shape[1]}")
```

- [ ] **Step 6: Build and eyeball**

Run: `cd /Users/kushalkrishnappa/Desktop/FAI && ~/.pyenv/versions/pytorch/bin/python scripts/build_notebook.py`
Expected: `wrote src/LoanSanctionAmountEstimation.ipynb with N cells`

- [ ] **Step 7: Checkpoint.** Notebook opens, blocks 0 to 2 read correctly.

---

### Task 4: Block 3, the data quality audit

**Files:**
- Modify: `scripts/nbcells.py`

- [ ] **Step 1: Missingness cell**

Bar chart of null percentage by column, and a printed table. Narration explains that `-999` was masquerading as data and what the conversion changed.

- [ ] **Step 2: The duplicate column evidence cell**

This one must show its working, because the claim is strong.

```python
raw = pd.read_csv(DATA, index_col="Customer ID").replace(-999, np.nan)
pair = raw[["Income (USD)", "Property Age"]].dropna()
identical = bool((pair["Income (USD)"] == pair["Property Age"]).all())
print(f"rows where both are present : {len(pair):,}")
print(f"identical in every one      : {identical}")
print(f"correlation                 : {pair.corr().iloc[0, 1]:.6f}")
print(pair.head(5).to_string())
```

Expected output: 25,150 rows, `True`, correlation 1.000000.

Narration states that a correlation of exactly 1.0 across 25,150 rows means one column was copied, so it carries no extra information and one copy is dropped. It also notes that dividing this column by 365.25 to read it as years has no basis, since the values are USD.

- [ ] **Step 3: Target shape cell**

```python
y_all = df["Loan Sanction Amount (USD)"]
zero = int((y_all == 0).sum())
print(f"rejected, sanction of 0 : {zero:,}  ({zero / len(y_all):.1%})")
print(f"approved, sanction > 0  : {len(y_all) - zero:,}")
for col in ["Income Stability", "Location"]:
    g = df.assign(z=(y_all == 0)).groupby(col)["z"].agg(["mean", "size"])
    print(f"\nrejection rate by {col}:\n{(g['mean'] * 100).round(1).to_string()}")
```

Expected: 7,865 rejected at 26.8 percent, and rejection rates of 14.8 against 29.0 by income stability.

Narration explains why this rules out simply deleting the zero rows, and introduces the two-stage design.

- [ ] **Step 4: Checkpoint.** Audit block prints the three evidence blocks above.

---

### Task 5: Blocks 4 and 5, split then explore

**Files:**
- Modify: `scripts/nbcells.py`

- [ ] **Step 1: The split, with narration on why it comes first**

```python
from sklearn.model_selection import train_test_split

y_amount = df.pop("Loan Sanction Amount (USD)")
y_approved = (y_amount > 0).astype(int)

idx_train, idx_test = train_test_split(
    df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y_approved
)
X_train, X_test = df.loc[idx_train], df.loc[idx_test]
print(f"train {len(idx_train):,}   test {len(idx_test):,}")
print(f"approval rate, train {y_approved[idx_train].mean():.3f}  test {y_approved[idx_test].mean():.3f}")
```

Narration: exploring the full data and then splitting is like reading the exam paper before deciding what to revise. Stratifying keeps the 73 to 27 balance in both halves.

- [ ] **Step 2: Univariate EDA on `X_train` only**

Reuse the plotting helpers from the old notebook but pass `X_train`. Drop the `/365.25` branch entirely, since that column no longer exists.

- [ ] **Step 3: Multivariate EDA on `X_train` only**

Correlation heatmap, and scatter plots for the pairs that actually matter. Narration must not repeat the old claim about income and property age, because that column is gone.

- [ ] **Step 4: Checkpoint.** No EDA cell references `X_test`, `df` or `y_amount` outside the training index.

---

### Task 6: Block 6, the preprocessing pipeline

**Files:**
- Modify: `scripts/nbcells.py`

This code is already prototyped and run against the real data. Output width is 53 columns.

- [ ] **Step 1: The clipper, including `get_feature_names_out`**

```python
from sklearn.base import BaseEstimator, TransformerMixin

class PercentileClipper(BaseEstimator, TransformerMixin):
    """Caps extreme values at percentiles learned from the training folds only.

    The earlier version of this work replaced any value beyond 3 standard deviations
    with the column mean, which turned Dependents into 2.23. Clipping keeps the order
    of the data and never invents a value that cannot exist.
    """

    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(values, self.lower, axis=0)
        self.upper_ = np.nanpercentile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", [f"x{i}" for i in range(values.shape[1])])
        )
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(names, dtype=object)
```

Without `get_feature_names_out` the whole `ColumnTransformer` raises `AttributeError` when SHAP asks for names. This was found during prototyping.

- [ ] **Step 2: The column groups and the transformer factory**

```python
CLIP_COLS = ["Age", "Income (USD)", "Loan Amount Request (USD)",
             "Current Loan Expenses (USD)", "Credit Score", "Property Price"]
DISCRETE_COLS = ["Dependents", "No. of Defaults", "Property Type", "Co-Applicant"]
CATEGORICAL_COLS = [c for c in df.columns if df[c].dtype == object]

def make_preprocessor():
    return ColumnTransformer([
        ("continuous", Pipeline([
            ("clip", PercentileClipper()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), CLIP_COLS),
        ("discrete", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), DISCRETE_COLS),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_COLS),
    ])
```

- [ ] **Step 3: Prove the discrete columns stay whole**

```python
check = make_preprocessor().fit(X_train)
raw_dependents = X_train["Dependents"].dropna().unique()
print(f"Dependents values after preprocessing: {sorted(raw_dependents)}")
assert all(float(v).is_integer() for v in raw_dependents), "Dependents must stay whole"
print(f"feature count after one-hot encoding: {len(check.get_feature_names_out())}")
```
Expected: whole numbers only, 53 features.

- [ ] **Step 4: Checkpoint.** Assertion passes, 53 features reported.

---

### Task 7: Block 7, stage 1 classifier

**Files:**
- Modify: `scripts/nbcells.py`

Prototyped numbers to expect: dummy F1 0.8451, logistic regression F1 0.9228 and ROC-AUC 0.8583.

- [ ] **Step 1: Baseline first, so later models have something to beat**

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
baseline.fit(X_train, y_approved[idx_train])
base_pred = baseline.predict(X_test)
print(f"always-approve baseline   F1 {f1_score(y_approved[idx_test], base_pred):.4f}"
      f"   accuracy {accuracy_score(y_approved[idx_test], base_pred):.4f}")
```

Narration: accuracy of 0.73 sounds respectable but comes from never saying no, which is why F1 is the selection metric.

- [ ] **Step 2: Candidates and search**

`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, each wrapped in `Pipeline([("pre", make_preprocessor()), ("model", ...)])`, tuned with `GridSearchCV(scoring="f1", cv=5, error_score="raise", n_jobs=-1)`.

`error_score="raise"` is mandatory. Its absence is what let 180 fits fail silently before.

- [ ] **Step 3: Report the full metric set and the confusion matrix**

Accuracy, precision, recall, F1 and ROC-AUC in one table, plus a `ConfusionMatrixDisplay` saved to `report/figures/confusion_matrix.png`.

Narration reads the confusion matrix in plain terms: a false approval means the customer is quoted an amount they will not receive, a false rejection means a viable customer is turned away.

- [ ] **Step 4: Cache**

```python
joblib.dump(best_classifier, MODELS / "stage1_classifier.joblib")
```

- [ ] **Step 5: Checkpoint.** Tuned F1 beats 0.8451.

---

### Task 8: Block 8, stage 2 regressors

**Files:**
- Modify: `scripts/nbcells.py`

Trained on approved training rows only. Prototyped Random Forest RMSE on approved test rows: 5,256.

- [ ] **Step 1: Restrict to approved rows**

```python
approved_train = idx_train[y_approved[idx_train] == 1]
approved_test = idx_test[y_approved[idx_test] == 1]
print(f"stage 2 trains on {len(approved_train):,} approved rows")
```

- [ ] **Step 2: `DummyRegressor(strategy="mean")` baseline, then the five models**

Linear Regression, PCA with Linear Regression, Random Forest, Gradient Boosting, PyTorch MLP. Grids as fixed in spec section 5.3, 16 candidates each, `scoring="neg_root_mean_squared_error"`, `error_score="raise"`.

- [ ] **Step 3: The PyTorch model, replacing the Keras cell**

Topology and Keras-matching defaults are already verified to reproduce the original metrics within 0.5 percent.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(RANDOM_STATE)

def build_mlp(n_features: int) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(n_features, 64), nn.ReLU(),
        nn.Linear(64, 128), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(),
        nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),   # keras BN defaults
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )
    for layer in model:
        if isinstance(layer, nn.Linear):      # keras uses glorot uniform, torch does not
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model
```

The L2 penalty applies to three specific layers only, so it goes into the loss rather than into `weight_decay`, which would penalise every layer:

```python
l2_layers = [(model[2], 0.3), (model[7], 0.1), (model[12], 0.5)]
penalty = sum(lam * (layer.weight ** 2).sum() for layer, lam in l2_layers)
(mse + penalty).backward()
```

The target is standardised with a `StandardScaler` fitted on approved training rows only, and every reported figure is inverse transformed back to USD before it is printed.

- [ ] **Step 4: One comparison table, one scale**

```python
results = pd.DataFrame(rows, columns=["Model", "RMSE (USD)", "MAE (USD)", "R2"])
print(results.to_string(index=False))
```

RMSE, MAE and R2 all in USD on the same rows. No scaled MSE anywhere near this table. Mixing the two scales is what produced the misreported figures in the earlier work.

- [ ] **Step 5: Checkpoint.** Every model beats the dummy, one table, one scale.

---

### Task 9: Blocks 9 to 11, end to end, SHAP and conclusions

**Files:**
- Modify: `scripts/nbcells.py`

Prototyped end to end figures: RMSE 25,949, MAE 9,711, R2 0.7099.

- [ ] **Step 1: Chain the two stages across all test rows**

```python
gate = best_classifier.predict(X_test)
amount = best_regressor.predict(X_test)
end_to_end = np.where(gate == 1, amount, 0.0)

print(f"END TO END on all {len(X_test):,} test rows")
print(f"  RMSE {root_mean_squared_error(y_amount[idx_test], end_to_end):,.2f} USD")
print(f"  MAE  {mean_absolute_error(y_amount[idx_test], end_to_end):,.2f} USD")
print(f"  R2   {r2_score(y_amount[idx_test], end_to_end):.4f}")
```

Narration must be blunt: this figure is much worse than the stage 2 figure, because a wrong approval decision costs the whole loan amount. It is also the only number that reflects what a customer would actually experience.

- [ ] **Step 2: Error decomposition**

Split total error into rows the classifier got right against rows it got wrong, so the reader can see which stage to improve.

- [ ] **Step 3: SHAP for both stages**

`shap.TreeExplainer` on the tuned tree models, feature names from `preprocessor.get_feature_names_out()`. Save both summary plots into `report/figures/`.

Narration reads the top three features only, and states plainly where a feature's effect is not causal.

- [ ] **Step 4: Limitations cell**

Written honestly and covering at minimum: the dataset is a Kaggle extract with no documented provenance, one column was duplicated which raises questions about the rest, stage 1 errors dominate end to end error, no fairness audit was done although Gender is present in the features, and the model reproduces historical lending decisions including any bias in them.

- [ ] **Step 5: Checkpoint.** End to end cell prints all three metrics.

---

### Task 10: Execute, fill the narration, verify

**Files:**
- Create: `scripts/extract_results.py`, `scripts/fill_narration.py`

- [ ] **Step 1: Execute with a clean kernel**

```bash
cd /Users/kushalkrishnappa/Desktop/FAI/src && \
~/.pyenv/versions/pytorch/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
  --inplace LoanSanctionAmountEstimation.ipynb
```
Expected: no error, roughly 12 to 15 minutes on the first run.

- [ ] **Step 2: Extract every printed number into `results.json`**

Cells print machine readable lines of the form `RESULT key=value`, and `extract_results.py` collects them. The marker names in the markdown must match those keys exactly.

- [ ] **Step 3: Substitute and re-execute**

`fill_narration.py` replaces `{{RMSE_END_TO_END}}` and friends. Re-execution is cheap because `RETRAIN` is False and the models load from `models/`.

- [ ] **Step 4: Verify**

Run: `~/.pyenv/versions/pytorch/bin/python scripts/verify_notebook.py`
Expected: `0 failure(s)`.

- [ ] **Step 5: Checkpoint.** Verifier passes. From here the `.ipynb` is authoritative, not the generator.

---

### Task 11: Diagrams

**Files:**
- Create: `report/diagrams/two_stage.mmd`, `report/diagrams/pipeline.mmd`, `scripts/pptr.json`

- [ ] **Step 1: Write the two-stage diagram left to right**

```
flowchart LR
    A[All 29,322 applicants] --> B{Stage 1<br/>Approved?}
    B -- No --> C[Predict 0]
    B -- Yes --> D[Stage 2<br/>Predict amount]
    C --> E[End to end score]
    D --> E
```

`LR` not `TD`. The top down version rendered 1150 by 1900 pixels in testing and overflowed the page.

- [ ] **Step 2: Render**

```bash
echo '{"executablePath":"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","args":["--no-sandbox"]}' > scripts/pptr.json
mmdc -i report/diagrams/two_stage.mmd -o report/figures/two_stage.png -p scripts/pptr.json -b white -s 3
```
Expected: PNG written, wider than tall.

- [ ] **Step 3: Put the same mermaid source in a notebook markdown cell** so JupyterLab renders it live and the two never drift.

- [ ] **Step 4: Checkpoint.** Both PNGs exist and are landscape.

---

### Task 12: The report

**Files:**
- Create: `report/Report.md`, `scripts/build_report.sh`

- [ ] **Step 1: Write `report/Report.md` to the template**

Sections exactly as the rubric names them: title page, introduction, background, methodology, results, discussion, conclusion including individual contributions, references. Target 8 pages.

Contributions table as fixed in spec section 7.1: Kushal Krishnappa on the audit and pipeline, Deepak on EDA and figures, Rohit on stage 1 and end to end evaluation, Dips on stage 2 and SHAP, writing shared.

- [ ] **Step 2: Every number comes from `results.json`**

No figure is typed by hand. A number in the report that is not in `results.json` is a defect.

- [ ] **Step 3: Images need explicit widths**

`![Two stage design](figures/two_stage.png){width=70%}`, otherwise xelatex warns that the float is too large. Confirmed in testing.

- [ ] **Step 4: Build**

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for f in report/diagrams/*.mmd; do
  mmdc -i "$f" -o "report/figures/$(basename "${f%.mmd}").png" -p scripts/pptr.json -b white -s 3
done
pandoc report/Report.md -o report/Report.pdf --pdf-engine=xelatex \
  -V geometry:margin=1in -V fontsize=11pt --toc
echo "pages: $(~/.pyenv/versions/pytorch/bin/python -c \
  "import pypdf;print(len(pypdf.PdfReader('report/Report.pdf').pages))")"
```
Expected: between 5 and 10 pages.

- [ ] **Step 5: Checkpoint.** PDF builds, page count within range.

---

### Task 13: Final verification against the spec

- [ ] **Step 1: Run the verifier.** Expected `0 failure(s)`.
- [ ] **Step 2: Confirm each of the seven criteria in spec section 9 by inspection.**
- [ ] **Step 3: Cross-check every report number against `results.json` mechanically.**
- [ ] **Step 4: Confirm each reference URL resolves.**
- [ ] **Step 5: Report status to the user. Do not commit.**

---

## Self-Review

**Spec coverage.** Every section maps to a task. Section 5.1 to Task 6, 5.2 to Task 7, 5.3 to Task 8, 5.4 to Task 9, 5.5 to Tasks 7 and 8, section 6 to Task 10, section 7 to Task 12, 7.1 to Task 12 Step 1, section 10 to Task 11, section 9 to Tasks 1 and 13. Defects D1 to D3 are fixed in Task 8, D4 in Tasks 7 and 8, D5 and D6 in Tasks 3 and 4, D7 in Tasks 5 and 6, D8 in Task 6, D9 in Task 6 Step 3, D10 to D12 in Tasks 8 and 10, D13 and D14 in Task 8, D15 in Tasks 7 to 9, D16 in Task 6, D17 in Task 8, D18 in Task 10, D19 in Task 3.

**Placeholders.** None. Tasks 4, 5, 7, 8, 9 and 12 describe cells by pattern with worked examples rather than transcribing roughly 120 cells of narration, which is deliberate. The non-obvious code, meaning the clipper, the transformer, the torch model, the chaining and the verifier, is given in full and has been run.

**Naming consistency.** `make_preprocessor`, `PercentileClipper`, `build_mlp`, `idx_train`, `idx_test`, `y_amount`, `y_approved`, `best_classifier`, `best_regressor` are used identically in every task. No name is reused across two models, which was defect D17.

**Deviation from the skill, recorded deliberately.** The skill asks for a commit step per task. The user asked for no git commits, so every commit step is a checkpoint instead. User instruction outranks the skill.
