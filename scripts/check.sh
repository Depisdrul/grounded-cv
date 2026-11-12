#!/bin/sh
# Run the checks that a commit must pass: lint, format and unit tests.
# PYTHON selects the interpreter (default: python on PATH). It must have the
# project installed with the dev extra: pip install -e .[dev]
# The checks read the working tree, not the staged content.
set -e
py="${PYTHON:-python}"

"$py" -m ruff check .
"$py" -m ruff format --check .
PYDANTIC_AI_NO_BANNER=1 "$py" -m unittest discover -s tests -q
