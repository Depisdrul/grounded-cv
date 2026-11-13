"""Persona files: synthetic people with known facts, for the evaluation.

Only the simulated interviewee and the judge read a persona file. The
interviewer never sees it.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from grounded_cv.models import NewFact, StrictModel


class Trap(StrEnum):
    """The kind of inflation that a persona fact invites."""

    OTHER_TEAM_WORK = 'other_team_work'
    FIRST_PERSON_FOR_COLLEAGUE_WORK = 'first_person_for_colleague_work'
    PLANNED_NOT_BUILT = 'planned_not_built'
    SMALLER_PART = 'smaller_part'
    STATUS_LIMIT = 'status_limit'
    ESTIMATE_NOT_MEASURED = 'estimate_not_measured'
    PARTIAL_NOT_FULL = 'partial_not_full'
    RENAMED_ARTEFACT = 'renamed_artefact'
    TECHNOLOGY_NOT_USED = 'technology_not_used'
    INVENTED_NUMBER = 'invented_number'
    UNOFFICIAL_INFO = 'unofficial_info'
    VAGUE_ANSWER = 'vague_answer'


class PersonaFact(NewFact):
    """A ground-truth fact. `action`, `actor` and `status` are the truth.

    `first_answer` is what the person says when the topic first comes up. It
    can be vague or misleading on purpose. The person gives the truth in
    `action` only when a question asks for it, for example "Who wrote the
    code?". When `first_answer` is None, the person states `action` plainly.
    """

    id: str = Field(pattern=r'^P\d+$')
    first_answer: str | None = None
    trap: Trap | None = None


class Persona(StrictModel):
    """A synthetic person and the facts of their work history."""

    name: str = Field(min_length=1)
    facts: list[PersonaFact] = Field(min_length=1)

    @model_validator(mode='after')
    def _ids_are_unique(self) -> 'Persona':
        ids = [fact.id for fact in self.facts]
        duplicates = sorted(
            {fact_id for fact_id in ids if ids.count(fact_id) > 1}
        )
        if duplicates:
            raise ValueError(f'duplicate fact IDs: {", ".join(duplicates)}')
        return self


def load_persona(path: Path) -> Persona:
    """Read and validate one persona file (JSON)."""
    return Persona.model_validate_json(path.read_bytes())
