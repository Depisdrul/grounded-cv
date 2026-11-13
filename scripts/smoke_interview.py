"""Run a short interview with scripted answers, to check that a model works.

The check: the model asks questions, calls save_fact, and stores facts with
an actor and a status. The answers are synthetic. The facts go to a
temporary file and are printed at the end.

Usage (from the repository root):
    python scripts/smoke_interview.py ollama:qwen3.5:4b
    python scripts/smoke_interview.py google:gemini-2.5-flash

An Ollama model needs OLLAMA_BASE_URL, for example
http://localhost:11434/v1. A Gemini model needs GEMINI_API_KEY; this script
loads `.env`.
"""

import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from grounded_cv.interviewer import run_interview
from grounded_cv.store import FactStore

ANSWERS = [
    'I was a data engineer at Example Freight Ltd from 2021 to 2023.',
    'I wrote the scripts that moved the customer database to the cloud. '
    'The platform team moved the servers.',
    'We planned a mobile app for Q3, but it was cancelled before any work.',
]


def answer_from_script(question: str) -> str:
    """Print the question and return the next scripted answer."""
    print(f'Q: {question}')
    reply = ANSWERS.pop(0) if ANSWERS else "I don't know."
    print(f'A: {reply}\n')
    return reply


def main(model: str) -> None:
    load_dotenv()
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FactStore(Path(tmp_dir) / 'facts.json')
        start = time.monotonic()
        result = run_interview(
            model, store, answer_from_script, max_questions=4
        )
        print(f'End: {result.end_reason}, {time.monotonic() - start:.0f} s')
        for fact in store.all():
            print(fact.model_dump_json())


if __name__ == '__main__':
    main(sys.argv[1])
