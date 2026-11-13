"""Writer agent: writes CV bullets from confirmed facts, citing their IDs."""

from collections.abc import Iterable

from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import Model

from grounded_cv.models import Bullet, Fact, Status
from grounded_cv.prompts import load_prompt

INSTRUCTIONS = load_prompt('writer')

_FACTS = TypeAdapter(list[Fact])

writer = Agent(
    output_type=list[Bullet], instructions=INSTRUCTIONS, name='writer'
)


def write_bullets(model: Model | str, facts: Iterable[Fact]) -> list[Bullet]:
    """Write bullets from the facts with status `confirmed` only.

    Other facts are not sent to the model.

    Args:
        model: A PydanticAI model, or a model name such as
            `google:gemini-2.5-flash`.
        facts: The stored facts, of every status.

    Returns:
        The bullets, each with the IDs of the facts that it cites.
    """
    confirmed = [fact for fact in facts if fact.status is Status.CONFIRMED]
    prompt = 'Facts:\n' + _FACTS.dump_json(confirmed, indent=2).decode()
    return writer.run_sync(prompt, model=model).output
