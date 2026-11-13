"""Fact store: a JSON file that holds the facts of one interview."""

import os
from pathlib import Path

from pydantic import TypeAdapter

from grounded_cv.models import Fact, NewFact

_FACTS = TypeAdapter(list[Fact])


class FactStore:
    """Facts kept in memory and written to `path` after each change.

    The store never deletes a fact, so the next ID is the fact count plus one.
    """

    def __init__(self, path: Path) -> None:
        """Open the store and load the facts in `path`, if the file exists.

        Args:
            path: The JSON file that holds the facts of one interview. The
                store creates it at the first `add`. Its folder must exist.
        """
        self.path = path
        self._facts: list[Fact] = []
        if path.exists():
            self._facts = _FACTS.validate_json(path.read_bytes())

    def add(self, new_fact: NewFact) -> Fact:
        """Give the fact the next ID, store it and write the file.

        Returns:
            The stored fact, with its new ID.
        """
        fact = Fact(id=f'F{len(self._facts) + 1}', **new_fact.model_dump())
        self._facts.append(fact)
        self._write()
        return fact

    def get(self, fact_id: str) -> Fact | None:
        """Return the fact with this ID, or None if there is none."""
        return next((fact for fact in self._facts if fact.id == fact_id), None)

    def all(self) -> list[Fact]:
        """Return a copy of all facts, in the order of storage."""
        return list(self._facts)

    def _write(self) -> None:
        # Write a temporary file, then replace: a crash during the write
        # leaves the previous file intact.
        tmp_path = self.path.with_suffix(self.path.suffix + '.tmp')
        tmp_path.write_bytes(_FACTS.dump_json(self._facts, indent=2))
        os.replace(tmp_path, self.path)
