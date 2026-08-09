# scripts/build_notebook.py
"""Builds src/LoanSanctionAmountEstimation.ipynb from the cell definitions in nbcells.py.

Run from the repository root:
    ~/.pyenv/versions/pytorch/bin/python scripts/build_notebook.py
"""
import sys
from pathlib import Path

import nbformat as nbf

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))  # so the sibling nbcells module imports from any cwd

import nbcells  # noqa: E402


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = nbcells.all_cells()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.4"},
    }
    out = ROOT / "src" / "LoanSanctionAmountEstimation.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out.relative_to(ROOT)} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
