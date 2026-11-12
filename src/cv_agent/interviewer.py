"""Interviewer agent: asks one question at a time and stores each answer as a fact."""

from collections.abc import Callable

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model

from cv_agent.models import NewFact
from cv_agent.store import FactStore

INSTRUCTIONS = """\
You interview a person about their work history, to collect facts for a CV.

How to ask:
- Ask one question at a time, about one topic.
- Start with the employers, roles and periods. Then ask what the person did in each role.
- Ask who did the work when the answer does not say it.

How to store facts:
- After each answer, call save_fact once for each fact in the answer.
- `action` is one sentence in the words of the person. Do not add numbers, tools,
  results or claims that the person did not say.
- `actor`: "me" only when the person says that they did the work. "team" when a group
  did it. "colleague" when another person did it.
- `status`:
  - "confirmed" when the person clearly says that the work was done and who did it.
  - "to_confirm" when the answer is vague about what was done or who did it.
    Do not guess the missing part.
  - "do_not_claim" when the work was planned but not built, or the person says it must
    not go on the CV.
- Store a fact about another team's work with actor "team" or "colleague". Never change
  it to "me".

When the person has nothing more to add, or says "I don't know" to every new topic,
end the interview with Done.
"""


class Question(BaseModel):
    """The next question for the person."""

    text: str


class Done(BaseModel):
    """The interview is complete."""

    reason: str


Answerer = Callable[[str], str]
"""Takes a question and returns the answer: a human at a prompt, or a simulated interviewee."""

interviewer = Agent(
    deps_type=FactStore,
    output_type=[Question, Done],
    instructions=INSTRUCTIONS,
    name="interviewer",
)


@interviewer.tool
def save_fact(ctx: RunContext[FactStore], fact: NewFact) -> str:
    """Store one fact from the last answer. Returns the ID of the stored fact."""
    return f"Saved as {ctx.deps.add(fact).id}."


_LAST_ANSWER_NOTE = "\n\n[Note to the interviewer: this is the last answer. Store its facts, then end.]"


def run_interview(
    model: Model | str, store: FactStore, answer: Answerer, max_questions: int = 30
) -> list[ModelMessage]:
    """Run the interview until the agent returns Done or the person answers `max_questions`.

    After the last allowed answer, the agent runs once more with Done as the only
    output, so that it stores the facts of that answer. The facts go to `store`.
    Returns the full message history, for tracing and evaluation.
    """
    history: list[ModelMessage] = []
    prompt = "Start the interview."
    answers = 0
    while True:
        last_run = answers == max_questions
        result = interviewer.run_sync(
            prompt + _LAST_ANSWER_NOTE if last_run else prompt,
            output_type=Done if last_run else None,
            deps=store,
            model=model,
            message_history=history,
        )
        history = result.all_messages()
        if isinstance(result.output, Done):
            return history
        prompt = answer(result.output.text)
        answers += 1
