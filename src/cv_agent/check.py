"""Deterministic grounding check for CV bullets. No LLM is involved."""

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel

from cv_agent.models import Bullet, Fact, Status


class RejectionReason(StrEnum):
    """Why a bullet fails the check."""

    NO_FACT_IDS = 'no_fact_ids'
    UNKNOWN_ID = 'unknown_id'
    NOT_CONFIRMED = 'not_confirmed'


class BulletCheckResult(BaseModel):
    """The result for one bullet.

    `reason` and `fact_id` are None when the bullet passes.
    """

    bullet: Bullet
    passed: bool
    reason: RejectionReason | None = None
    fact_id: str | None = None


def check_bullets(
    bullets: Iterable[Bullet], facts: Iterable[Fact]
) -> list[BulletCheckResult]:
    """Check that each bullet cites at least one fact, all known and confirmed.

    Args:
        bullets: The bullets to check, usually the writer's output.
        facts: All stored facts, of every status. A cited ID that is not
            among them is unknown.

    Returns:
        One result per bullet, in input order. A failing bullet reports its
        first failure only.
    """
    facts_by_id = {fact.id: fact for fact in facts}
    return [_check_one(bullet, facts_by_id) for bullet in bullets]


def _check_one(
    bullet: Bullet, facts_by_id: dict[str, Fact]
) -> BulletCheckResult:
    if not bullet.fact_ids:
        return BulletCheckResult(
            bullet=bullet, passed=False, reason=RejectionReason.NO_FACT_IDS
        )
    for fact_id in bullet.fact_ids:
        fact = facts_by_id.get(fact_id)
        if fact is None:
            return BulletCheckResult(
                bullet=bullet,
                passed=False,
                reason=RejectionReason.UNKNOWN_ID,
                fact_id=fact_id,
            )
        if fact.status is not Status.CONFIRMED:
            return BulletCheckResult(
                bullet=bullet,
                passed=False,
                reason=RejectionReason.NOT_CONFIRMED,
                fact_id=fact_id,
            )
    return BulletCheckResult(bullet=bullet, passed=True)
