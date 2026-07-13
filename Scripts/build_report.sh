#!/usr/bin/env bash
# Build report.tex using latexmk if available, otherwise fallback to pdflatex
set -euo pipefail
cd "$(dirname "$0")/.."/Report || exit 1

if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode report.tex
else
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
fi

echo "Build finished: $(pwd)/report.pdf"
