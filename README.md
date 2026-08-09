# Loan Sanction Amount Estimation

A two-stage pipeline for the LendWise loan dataset. Stage 1 is a classifier deciding
whether an application is sanctioned at all; Stage 2 is a regressor predicting the
amount, run only on applications Stage 1 approves.

## The authoritative file

`src/LoanSanctionAmountEstimation.ipynb` is the single source of truth.

**Nothing generates it.** It was previously assembled by `scripts/build_notebook.py`
from cell definitions in `scripts/nbcells.py`; both were removed once the notebook was
complete, so hand edits can no longer be silently regenerated away. Those files remain
in git history if the cell sources are ever needed:

```bash
git show b75e9b3:scripts/nbcells.py
```

Edit the notebook in Jupyter, or with the `NotebookEdit` tool.

## The hard rule

**Restart Kernel and Run All Cells must be the last thing you do before saving.**

`scripts/verify_notebook.py` requires the stored execution counts to be exactly
`1..N` in document order. A clean restart-and-run-all satisfies that. These do not:

- Plain "Run All" without a restart, which continues the counter from the previous
  session and starts at `N+1`.
- Re-running a single cell after a full run, which bumps that one cell's count.
- Inserting a new cell and running only it.

A full run takes roughly 12 to 15 minutes.

## Commands

```bash
# Verify the notebook. Expect: 0 failure(s), 39 code cells, 69 markdown cells
~/.pyenv/versions/pytorch/bin/python scripts/verify_notebook.py

# Build the PDF. Renders report/diagrams/*.mmd via mmdc, then pandoc/xelatex.
scripts/build_report.sh
```

The notebook runs with the working directory set to `src/`, which is why it writes
figures to `../report/figures/`.

Dependencies are in `requirements.txt`. The kernel is the pyenv `pytorch`
environment (Python 3.14.4). TensorFlow has no wheel for this Python version, so
the neural network is PyTorch.

## What the verifier enforces

`scripts/verify_notebook.py` is now the only automated guard on the notebook. It
checks five things:

1. No cell has an error output.
2. Execution counts are exactly `1..N` in order.
3. No unfilled `{{MARKER}}` placeholders in markdown.
4. No banned constructs in code: `LabelEncoder` (ordinal encoding of nominal
   columns), `max_features='auto'` (removed in sklearn 1.3), `tensorflow` (no
   cp314 wheel), `filterwarnings("ignore")` (hides failed fits).
5. Markdown voice rules: no em-dash, no `we`/`our`, no standalone `I`. The `I`
   check allows `Type I`, `Stage I`, `Grade I`, and `Phase I`.

## The References cell

Notebook cell 5 is hand-maintained. It used to be generated from
`scripts/references.json`, which is kept as the source record for the entries.

Each entry renders in this format:

```
{n}. **{title}.** {authors}. *{venue}*, {year}. <{url}>
```

The trailing period is omitted when the title already ends in `?`, `!`, or `.`.

**Write Su-In Lee's name out in full. Never `S.-I. Lee`.** The generator carried a
fixup for exactly this, because the abbreviated initial trips the verifier's
standalone-`I` check. The same applies to any future author whose initial is `I`.

## Reproducibility

`RANDOM_STATE = 42`, with both `np.random.seed` and `torch.manual_seed` set. Re-runs
reproduce results to roughly 15 significant figures.

`RETRAIN = False` loads the tuned Stage 1 models from `models/*.joblib`. `.gitignore`
excludes `*.joblib`, so those are not committed and a fresh clone refits them on the
first run instead of loading the cache.

## Known debt

Recorded here so it is not rediscovered as a surprise.

- **`report/Report.md` has roughly 150 hand-typed numbers and no automated
  cross-check.** A `scripts/extract_results.py` once scraped notebook output into
  `report/results.json` toward that goal, but the consumer that would have
  substituted those values into the report was never built, so nothing ever read
  the JSON. Both were removed rather than left as a false guarantee. Around 40 of
  the report's numbers (`352/168/177`, `1,573`, `8.3 percent`, `71.23`, `0.0028`,
  `epoch 25/50`) were never captured in the JSON in the first place.
- **`report/stage2_results.csv` is write-only.** Notebook cell 90 writes it and
  nothing reads it. Removing it would require a notebook edit plus a full re-run.
- **Re-run blast radius.** The 94.81 percent error-decomposition figure is restated
  by hand in five places in `Report.md` (lines 303-306, 347, 383, 388-390, 410-411),
  and the Stage 1 AUCs in three (lines 197, 220). Any change that moves those
  numbers has to be propagated by hand.
- **Seven notebook figures are unreferenced by the report**: `missingness.png`,
  `target_distribution.png`, `correlation_heatmap.png`, `nn_loss.png`,
  `error_decomposition.png`, `shap_stage1.png`, `shap_stage2.png`. Only
  `confusion_matrix.png` and `roc_curves.png` are load-bearing from the notebook.

## Layout

| Path | Contents |
|---|---|
| `src/LoanSanctionAmountEstimation.ipynb` | The authoritative notebook |
| `src/data/` | `train.csv`, `preprocessed.csv` |
| `models/` | Cached fitted estimators, gitignored |
| `report/Report.md` | The written report, hand-maintained |
| `report/figures/` | Figures, written by the notebook and by `mmdc` |
| `report/diagrams/` | Mermaid sources for the two pipeline diagrams |
| `scripts/verify_notebook.py` | The notebook guard |
| `scripts/build_report.sh` | Diagram render plus pandoc build |
| `scripts/references.json` | Source record for the References cell |
| `docs/superpowers/` | Historical spec and plan for the original build |
