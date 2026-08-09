#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for f in report/diagrams/*.mmd; do
  mmdc -i "$f" -o "report/figures/$(basename "${f%.mmd}").png" -p scripts/pptr.json -b white -s 3
done
cd report
pandoc Report.md -o Report.pdf --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt
cd ..
~/.pyenv/versions/pytorch/bin/python -c "import pypdf;print('pages:',len(pypdf.PdfReader('report/Report.pdf').pages))"
