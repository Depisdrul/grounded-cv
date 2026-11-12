"""Typed facts and CV bullets."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Actor(StrEnum):
    """Who did the work."""

    ME = 'me'
    TEAM = 'team'
    COLLEAGUE = 'colleague'


class Status(StrEnum):
    """Whether a fact may appear on the CV."""

    CONFIRMED = 'confirmed'
    TO_CONFIRM = 'to_confirm'
    DO_NOT_CLAIM = 'do_not_claim'


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and strips whitespace."""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class NewFact(StrictModel):
    """A fact before the store gives it an ID.

    The interviewer's save_fact tool takes this.
    """

    employer: str = Field(min_length=1)
    role: str = Field(min_length=1)
    period: str = Field(min_length=1)
    action: str = Field(
        min_length=1, description='One sentence: what was done.'
    )
    actor: Actor
    status: Status
    source: str | None = None


class Fact(NewFact):
    """A stored fact. IDs are F1, F2, ... in the order of storage."""

    id: str = Field(min_length=1)


class Bullet(StrictModel):
    """One CV line and the IDs of the facts that support it.

    An empty `fact_ids` list is valid here, so that `check_bullets()` can reject
    the bullet with a stated reason instead of a parse error.
    """

    text: str = Field(min_length=1)
    fact_ids: list[str]
