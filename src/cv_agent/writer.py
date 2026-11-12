"""Writer agent: writes CV bullets from confirmed facts, citing their IDs."""

from collections.abc import Iterable

from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import Model

from cv_agent.models import Bullet, Fact, Status

INSTRUCTIONS = """\
You write CV bullets from a list of facts. Each fact has an ID.

Rules:
- Each bullet uses only the facts that it cites in `fact_ids`.
  Cite at least one fact.
- Do not add numbers, tools, results or claims that are not in the cited
  facts.
- A fact with actor "team" or "colleague" is not the person's own work.
  A bullet that uses it must name the others, for example
  "With the data team, ...".
- One bullet can combine facts from the same role.
"""

_FACTS = TypeAdapter(list[Fact])

writer = Agent(
    output_type=list[Bullet], instructions=INSTRUCTIONS, name='writer'
)


def write_bullets(model: Model | str, facts: Iterable[Fact]) -> list[Bullet]:
    """Write bullets from the facts with status `confirmed` only.

    Other facts are not sent to the model.

    Args:
        model: A PydanticAI model, or a model name such as
            `google-gla:gemini-2.5-flash`.
        facts: The stored facts, of every status.

    Returns:
        The bullets, each with the IDs of the facts that it cites.
    """
    confirmed = [fact for fact in facts if fact.status is Status.CONFIRMED]
    prompt = 'Facts:\n' + _FACTS.dump_json(confirmed, indent=2).decode()
    return writer.run_sync(prompt, model=model).output
