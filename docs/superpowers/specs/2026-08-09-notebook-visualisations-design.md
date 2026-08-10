# Four evaluation charts for the loan sanction notebook

## Problem

`src/LoanSanctionAmountEstimation.ipynb` runs to 114 cells and reads as a wall of prose.
Only 14 of its 57 code cells render a figure, and the passages carrying the most weight
are the ones with no picture at all. Three model comparisons and the end to end chaining
result are printed as tables and paragraphs, so a reader has to hold five numbers in their
head to see a gap that a chart would state at a glance.

Twenty one candidates were reviewed. Four were selected. The existing prose stays exactly
as written: this work adds figures, it does not trim narration.

## Scope

Four new code cells, inserted after the cell that already computes their data. No markdown
cell is added, edited or removed. No existing code cell is modified.

| Chart | Inserted after | Data already in scope |
|---|---|---|
| Stage 1 model by metric | cell 59 | `stage1_metrics` |
| Stage 2 six model comparison | cell 93 | `stage2_results`, `best_name_reg` |
| Gate flow | cell 100 | `gate`, `y_test_cls` |
| End to end degradation | the gate flow cell | `end_to_end`, `end_to_end_perfect`, `y_test_true`, `pred_gb` |

Each figure saves to `../report/figures/` at `dpi=150`, matching every existing plot cell,
so `report/Report.md` can pick them up later.

## Colour

The notebook's existing figures use matplotlib's `tab10`. That palette fails the
colourblind separation check: `tab:green` against `tab:orange` measures OKLab ΔE 0.7 under
protanopia, and cell 64's ROC panel puts exactly that pair side by side. Fixing the
existing figures was considered and deliberately deferred, since this task is additive.
The existing 14 charts keep `tab10` untouched.

The four new charts use palettes validated with `scripts/validate_palette.js` from the
dataviz skill:

- **Four categorical series** (stage 1 models): `#2a78d6` blue, `#eb6834` orange,
  `#1baf7a` aqua, `#4a3aa7` violet. Passes all-pairs in light mode, worst CVD ΔE 9.2,
  worst normal-vision ΔE 16.3. The aqua slot sits below 3:1 against the surface, so the
  relief rule applies; cell 59 prints the full metric table directly above the chart,
  which satisfies it.
- **Two categorical series** (RMSE against MAE): `#2a78d6`, `#eb6834`. All checks pass.
- **Ordinal three step** (degradation): `#86b6ef`, `#3987e5`, `#184f95`. Monotone
  lightness, single hue, light end clears the surface at 2.06:1.
- **Status pair** (gate correct against gate wrong): `#0ca30c` and `#d03b3b`, fixed status
  tokens. Every ribbon is direct-labelled so colour never carries meaning alone.

## The charts

**Stage 1, four models across five metrics.** Vertical grouped bars, one group per metric,
one colour per model. All five metrics are unitless and bounded by 0 and 1, so a single
axis is correct. Only the F1 group is direct-labelled, since F1 is the stated selection
metric; the rest are read off the axis and the printed table.

**Stage 2, six models.** Two panels sharing a model ordering. The left panel holds RMSE and
MAE as horizontal grouped bars in dollars; the right holds R2. These are split because
dollars and R2 cannot share one axis without inventing a relationship. Horizontal bars
accommodate the long model names without rotated labels. The winner by RMSE is marked.

**Gate flow.** All 5,865 test rows enter on the left and fan into four outcome buckets,
ribbon width proportional to row count. Counts are 4,290 correctly approved, 486 falsely
approved, 1,087 correctly refused and 2 falsely rejected. That last bucket is 0.03 percent
of the test set and would render thinner than a pixel, so segment heights are floored at a
small minimum and the figure states that the floor is in effect. The chart reports where
rows go; the error decomposition in cell 103 immediately after reports where error comes
from, and the two are complementary rather than duplicative.

**End to end degradation.** Two panels, because the three numbers do not share a
denominator. The left panel carries stage 2 alone at $5,137.77, scored on 4,292 approved
rows, on its own axis. The right panel carries the two figures that are genuinely
comparable, both over all 5,865 test rows: a perfect gate at $4,395.12 against the real
gate at $22,814.62, annotated with the 5.2x ratio.

Note that the perfect gate figure is *lower* than stage 2 alone. A perfect gate adds 1,089
refused rows predicted at exactly 0 with no error, which dilutes the mean. The split panels
exist precisely so that dip cannot be misread as an improvement.

## Constraints

- `scripts/verify_notebook.py` requires execution counts to run 1..N in order, so inserting
  cells forces a full clean re-run of the notebook. `RETRAIN = False` and the cached
  `models/*.joblib` mean the grid searches reload rather than refit; the PyTorch MLP and
  the two SHAP explainers are the slow parts.
- The same script bans em-dashes and first person in markdown. No markdown is added here,
  so the rule is not engaged, but it stays true of the file.
- `requirements.txt` pins matplotlib 3.11 and seaborn 0.13. No new dependency is needed;
  the ribbon diagram is drawn with `matplotlib.path` Bezier curves.

## Verification

1. Re-run the notebook top to bottom.
2. `python scripts/verify_notebook.py` reports 0 failures.
3. All four PNGs exist under `report/figures/`.
4. Inspect the four rendered figures for label collisions and overflow.
