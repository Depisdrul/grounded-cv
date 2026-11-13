"""Fact store: a JSON file that holds the facts of one interview."""

import os
import threading
import time
from pathlib import Path

from pydantic import TypeAdapter

from grounded_cv.models import Fact, NewFact

_FACTS = TypeAdapter(list[Fact])

_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_SECONDS = 0.05


class FactStore:
    """Facts kept in memory and written to `path` after each change.

    The store never deletes a fact, so the next ID is the fact count plus one.
    `add` is thread-safe: PydanticAI runs the tool calls of one model
    response in parallel threads.
    """

    def __init__(self, path: Path) -> None:
        """Open the store and load the facts in `path`, if the file exists.

        Args:
            path: The JSON file that holds the facts of one interview. The
                store creates it at the first `add`. Its folder must exist.
        """
        self.path = path
        self._lock = threading.Lock()
        self._facts: list[Fact] = []
        if path.exists():
            self._facts = _FACTS.validate_json(path.read_bytes())

    def add(self, new_fact: NewFact) -> Fact:
        """Give the fact the next ID, write the file, then store it in memory.

        Returns:
            The stored fact, with its new ID.

        Raises:
            PermissionError: The file write failed. The store and the file
                keep the facts they had before the call.
        """
        with self._lock:
            fact = Fact(id=f'F{len(self._facts) + 1}', **new_fact.model_dump())
            facts = [*self._facts, fact]
            self._write(facts)
            self._facts = facts
        return fact

    def get(self, fact_id: str) -> Fact | None:
        """Return the fact with this ID, or None if there is none."""
        return next((fact for fact in self._facts if fact.id == fact_id), None)

    def all(self) -> list[Fact]:
        """Return a copy of all facts, in the order of storage."""
        return list(self._facts)

    def _write(self, facts: list[Fact]) -> None:
        # Write a temporary file, then replace: a crash during the write
        # leaves the previous file intact.
        tmp_path = self.path.with_suffix(self.path.suffix + '.tmp')
        tmp_path.write_bytes(_FACTS.dump_json(facts, indent=2))
        _replace_with_retry(tmp_path, self.path)


def _replace_with_retry(src: Path, dst: Path) -> None:
    """Replace `dst` with `src`, and retry while Windows denies access.

    On Windows, os.replace raises PermissionError while another process, for
    example a virus scanner, holds `dst` open. The lock lasts milliseconds.

    Raises:
        PermissionError: The last of `_REPLACE_ATTEMPTS` attempts failed.
    """
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_WAIT_SECONDS)
