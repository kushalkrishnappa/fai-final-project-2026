# scripts/verify_notebook.py
"""Checks the executed notebook against section 9 of the design spec."""
import json, re, sys
from pathlib import Path

NB = Path(sys.argv[1] if len(sys.argv) > 1 else "src/LoanSanctionAmountEstimation.ipynb")

FIRST_PERSON_I = re.compile(r"(?<!Type )(?<!Stage )(?<!Grade )(?<!Phase )\bI\b")

def main() -> int:
    if not NB.exists():
        print(f"FAIL: notebook not found: {NB}")
        return 1

    nb = json.loads(NB.read_text())
    code = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    md = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
    failures = []

    # 1. no error outputs
    for idx, c in code:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                failures.append(f"cell {idx}: {o['ename']}: {o['evalue'][:80]}")

    # 2. execution counts run 1..N in order
    counts = [c.get("execution_count") for _, c in code]
    if counts != list(range(1, len(code) + 1)):
        shown = counts[:12]
        suffix = "..." if len(counts) > 12 else ""
        failures.append(f"execution counts not 1..{len(code)} in order, got {shown}{suffix}")

    # 3. no unfilled narration markers
    for idx, c in md:
        for m in re.findall(r"\{\{[A-Z0-9_]+\}\}", "".join(c["source"])):
            failures.append(f"markdown cell {idx}: unfilled marker {m}")

    # 4. banned constructs
    src = "\n".join("".join(c["source"]) for _, c in code)
    for label, pattern, why in [
        ("LabelEncoder", re.escape("LabelEncoder"), "ordinal encoding of nominal columns"),
        ("max_features='auto'", r"max_features['\"]?\s*[:=]\s*(?:\[[^\]]*['\"]auto['\"][^\]]*\]|['\"]auto['\"])", "removed in sklearn 1.3"),
        ("tensorflow", re.escape("tensorflow"), "no cp314 wheel exists"),
        ('filterwarnings("ignore")', re.escape('filterwarnings("ignore")'), "global warning suppression hides failed fits"),
    ]:
        if re.search(pattern, src):
            failures.append(f"banned construct {label!r} present: {why}")

    # 5. voice rules
    for idx, c in md:
        text = "".join(c["source"])
        if "—" in text:
            failures.append(f"markdown cell {idx}: em-dash present")
        for w in (" we ", " We ", " our ", " Our "):
            if w in f" {text} ":
                failures.append(f"markdown cell {idx}: first person {w.strip()!r}")
        if FIRST_PERSON_I.search(text):
            failures.append(f"markdown cell {idx}: first person 'I'")

    for f in failures:
        print("FAIL:", f)
    print(f"\n{len(failures)} failure(s), {len(code)} code cells, {len(md)} markdown cells")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
