# scripts/extract_results.py
"""Pulls every reported metric out of the executed notebook's cell outputs and
writes report/results.json, so the written report never retypes a number by hand.

The notebook is treated as the single source of truth. Every value below is
found with a regular expression anchored on the exact label the notebook
prints, matched against the concatenated stdout of its code cells. If a label
cannot be found, this script raises rather than writing a guess or a
placeholder, since a silently wrong number is worse than a script that stops.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "src" / "LoanSanctionAmountEstimation.ipynb"
OUT_PATH = ROOT / "report" / "results.json"


class ExtractionError(RuntimeError):
    """Raised when an expected value cannot be found in the notebook's output.

    Deliberately fatal. A missing match here must stop the script, not fall
    back to a placeholder that later gets mistaken for a real result.
    """


def notebook_stdout(nb: dict) -> str:
    """Concatenates the stdout/execute_result text of every code cell, in order.

    Warnings and errors printed to stderr are left out on purpose, since they
    are not part of the numbers this report needs and can contain text that
    would otherwise confuse a label-based search.
    """
    parts = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                parts.append("".join(out.get("text", [])))
            elif out.get("output_type") == "execute_result":
                data = out.get("data", {})
                if "text/plain" in data:
                    parts.append("".join(data["text/plain"]))
    return "\n".join(parts)


def find(pattern: str, text: str, *, flags=0, what: str = "") -> "re.Match":
    m = re.search(pattern, text, flags)
    if m is None:
        raise ExtractionError(
            f"could not find {what or pattern!r} in the notebook's output. "
            f"Pattern used: {pattern!r}"
        )
    return m


def as_int(s: str) -> int:
    return int(s.replace(",", ""))


def as_float(s: str) -> float:
    return float(s.replace(",", ""))


# --------------------------------------------------------------------------- #
# Section extractors
# --------------------------------------------------------------------------- #


def extract_dataset(text: str) -> dict:
    m = find(r"rows loaded: ([\d,]+)\s+columns: (\d+)", text, what="raw rows/columns")
    raw_rows, raw_columns = as_int(m.group(1)), as_int(m.group(2))

    m = find(
        r"dropped ([\d,]+) rows with no target, ([\d,]+) remain",
        text,
        what="rows dropped for missing target",
    )
    dropped_rows_no_target, rows_after_dropping_target = as_int(m.group(1)), as_int(m.group(2))

    m = find(
        r"columns after dropping identifiers and the duplicate: (\d+)",
        text,
        what="column count after cleanup",
    )
    columns_after_cleanup = as_int(m.group(1))

    m = find(
        r"rows where both are present\s*:\s*([\d,]+)",
        text,
        what="duplicate column evidence row count",
    )
    duplicate_evidence_rows = as_int(m.group(1))

    m = find(
        r"correlation\s*:\s*([\d.]+)",
        text,
        what="duplicate column correlation",
    )
    duplicate_correlation = as_float(m.group(1))

    m = find(
        r"rejected, sanction of 0 : ([\d,]+)\s+\(([\d.]+)%\)",
        text,
        what="refused count and percentage",
    )
    refused_count, refused_pct = as_int(m.group(1)), as_float(m.group(2))

    m = find(
        r"approved, sanction > 0\s+: ([\d,]+)",
        text,
        what="approved count",
    )
    approved_count = as_int(m.group(1))

    return {
        "raw_rows": raw_rows,
        "raw_columns": raw_columns,
        "dropped_rows_no_target": dropped_rows_no_target,
        "rows_after_dropping_target": rows_after_dropping_target,
        "columns_after_cleanup": columns_after_cleanup,
        "refused_count": refused_count,
        "refused_pct": refused_pct,
        "approved_count": approved_count,
        "duplicate_evidence_rows": duplicate_evidence_rows,
        "duplicate_correlation": duplicate_correlation,
    }


def extract_split(text: str) -> dict:
    m = find(
        r"^train ([\d,]+)\s+test ([\d,]+)$",
        text,
        flags=re.MULTILINE,
        what="train/test row counts",
    )
    train_rows, test_rows = as_int(m.group(1)), as_int(m.group(2))

    m = find(
        r"approval rate, train ([\d.]+)\s+test ([\d.]+)",
        text,
        what="approval rate by split",
    )
    approval_rate_train, approval_rate_test = as_float(m.group(1)), as_float(m.group(2))

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "approval_rate_train": approval_rate_train,
        "approval_rate_test": approval_rate_test,
    }


def extract_features(text: str) -> dict:
    m = find(
        r"feature count after one-hot encoding: (\d+)",
        text,
        what="total feature count, all training rows",
    )
    total_all_training_rows = as_int(m.group(1))

    m = find(
        r"features after preprocessing, approved rows only: (\d+)",
        text,
        what="feature count, approved rows only",
    )
    approved_rows_only = as_int(m.group(1))

    return {
        "total_all_training_rows": total_all_training_rows,
        "approved_rows_only": approved_rows_only,
    }


_STAGE1_MODELS = [
    "baseline (always approve)",
    "logistic regression",
    "random forest",
    "gradient boosting",
]


def extract_stage1(text: str) -> dict:
    metrics = {}
    for name in _STAGE1_MODELS:
        pattern = (
            rf"^{re.escape(name)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
        )
        m = find(pattern, text, flags=re.MULTILINE, what=f"stage 1 metrics row for {name!r}")
        metrics[name] = {
            "accuracy": as_float(m.group(1)),
            "precision": as_float(m.group(2)),
            "recall": as_float(m.group(3)),
            "f1": as_float(m.group(4)),
            "roc_auc": as_float(m.group(5)),
        }

    m = find(r"winner by F1: (.+)", text, what="stage 1 winner")
    winner = m.group(1).strip()

    tn = as_int(find(r"correctly refused \(true negative\)\s*:\s*(\d+)", text, what="TN").group(1))
    fp = as_int(find(r"false approval \(false positive\)\s*:\s*(\d+)", text, what="FP").group(1))
    fn = as_int(find(r"false rejection \(false negative\)\s*:\s*(\d+)", text, what="FN").group(1))
    tp = as_int(find(r"correctly approved \(true positive\)\s*:\s*(\d+)", text, what="TP").group(1))

    return {
        "metrics": metrics,
        "winner": winner,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


_STAGE2_MODELS = [
    "Dummy (mean)",
    "Linear Regression",
    "PCA + Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "Neural Network (PyTorch MLP)",
]


def extract_stage2(text: str) -> dict:
    comparison = {}
    for display_name in _STAGE2_MODELS:
        pattern = rf"^\s*{re.escape(display_name)}\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$"
        m = find(
            pattern, text, flags=re.MULTILINE, what=f"stage 2 comparison row for {display_name!r}"
        )
        comparison[display_name] = {
            "rmse": as_float(m.group(1)),
            "mae": as_float(m.group(2)),
            "r2": as_float(m.group(3)),
        }

    m = find(r"winner by RMSE: (.+)", text, what="stage 2 winner")
    winner = m.group(1).strip()

    m = find(
        r"cumulative explained variance, 10 components: ([\d.]+)",
        text,
        what="PCA cumulative explained variance",
    )
    pca_cumulative_explained_variance = as_float(m.group(1))

    m = find(
        r"requested amount alone\s+RMSE ([\d,.]+)\s+R2 ([\d.]+)",
        text,
        what="requested-amount-alone RMSE/R2",
    )
    requested_amount_alone = {"rmse": as_float(m.group(1)), "r2": as_float(m.group(2))}

    return {
        "comparison": comparison,
        "winner": winner,
        "pca_cumulative_explained_variance": pca_cumulative_explained_variance,
        "requested_amount_alone": requested_amount_alone,
    }


def extract_endtoend(text: str) -> dict:
    m = find(
        r"END TO END across all [\d,]+ test rows\s+"
        r"RMSE ([\d,.]+) USD\s+MAE\s+([\d,.]+) USD\s+R2\s+([\d.]+)",
        text,
        what="end to end, predicted gate",
    )
    predicted_gate = {
        "rmse": as_float(m.group(1)),
        "mae": as_float(m.group(2)),
        "r2": as_float(m.group(3)),
    }

    m = find(
        r"END TO END WITH A PERFECT GATE.*?"
        r"RMSE ([\d,.]+) USD\s+MAE\s+([\d,.]+) USD\s+R2\s+([\d.]+)",
        text,
        flags=re.DOTALL,
        what="end to end, perfect gate",
    )
    perfect_gate = {
        "rmse": as_float(m.group(1)),
        "mae": as_float(m.group(2)),
        "r2": as_float(m.group(3)),
    }

    return {"predicted_gate": predicted_gate, "perfect_gate": perfect_gate}


_ERROR_GROUPS = ["falsely approved", "correctly approved", "falsely rejected", "correctly refused"]


def extract_error_decomposition(text: str) -> dict:
    decomposition = {}
    for group in _ERROR_GROUPS:
        pattern = rf"^{re.escape(group)}\s+(\d+)\s+([\d.]+e[+-]\d+)\s+([\d.]+)\s*$"
        m = find(pattern, text, flags=re.MULTILINE, what=f"error decomposition row for {group!r}")
        decomposition[group] = {
            "count": as_int(m.group(1)),
            "total_squared_error": float(m.group(2)),
            "share_of_total_error_pct": as_float(m.group(3)),
        }
    return decomposition


def _parse_feature_lines(block: str) -> list:
    features = []
    for line in block.strip("\n").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.*\S)\s+([\d.]+)$", line)
        if m is None:
            raise ExtractionError(f"could not parse a SHAP feature line: {line!r}")
        features.append({"feature": m.group(1), "value": as_float(m.group(2))})
    return features


def extract_shap(text: str) -> dict:
    m = find(
        r"stage 1 SHAP sample: (\d+) of ([\d,]+) test rows\n(.*?)\n\n"
        r"top three combined share of total mean \|SHAP\|: ([\d.]+)%",
        text,
        flags=re.DOTALL,
        what="stage 1 SHAP block",
    )
    stage1 = {
        "sample_size": as_int(m.group(1)),
        "total_rows": as_int(m.group(2)),
        "top_features": _parse_feature_lines(m.group(3)),
        "top_three_share_pct": as_float(m.group(4)),
    }

    m = find(
        r"stage 2 SHAP sample: (\d+) of ([\d,]+) approved test rows\n(.*?)\n\n"
        r"top feature alone, share of total mean \|SHAP\|: ([\d.]+)%\n"
        r"top three combined share of total mean \|SHAP\|: ([\d.]+)%",
        text,
        flags=re.DOTALL,
        what="stage 2 SHAP block",
    )
    stage2 = {
        "sample_size": as_int(m.group(1)),
        "total_rows": as_int(m.group(2)),
        "top_features": _parse_feature_lines(m.group(3)),
        "top_feature_alone_share_pct": as_float(m.group(4)),
        "top_three_share_pct": as_float(m.group(5)),
    }

    return {"stage1": stage1, "stage2": stage2}


def main() -> int:
    if not NB_PATH.exists():
        raise ExtractionError(f"executed notebook not found at {NB_PATH}")

    nb = json.loads(NB_PATH.read_text())
    text = notebook_stdout(nb)
    if not text.strip():
        raise ExtractionError(
            "the notebook has no stdout in any code cell. It must be executed "
            "before results can be extracted."
        )

    results = {
        "dataset": extract_dataset(text),
        "split": extract_split(text),
        "features": extract_features(text),
        "stage1": extract_stage1(text),
        "stage2": extract_stage2(text),
        "endtoend": extract_endtoend(text),
        "error_decomposition": extract_error_decomposition(text),
        "shap": extract_shap(text),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExtractionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
