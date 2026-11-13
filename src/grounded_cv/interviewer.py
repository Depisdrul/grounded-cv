"""Interviewer agent: asks one question at a time and stores facts."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model

from grounded_cv.models import NewFact
from grounded_cv.prompts import load_prompt
from grounded_cv.store import FactStore

INSTRUCTIONS = load_prompt('interviewer')


class Question(BaseModel):
    """The next question for the person."""

    text: str


class Done(BaseModel):
    """The interview is complete."""

    reason: str


Answerer = Callable[[str], str]
"""Takes a question, returns the answer: a human or a simulated interviewee."""

interviewer = Agent(
    deps_type=FactStore,
    output_type=[Question, Done],
    instructions=INSTRUCTIONS,
    name='interviewer',
)


@interviewer.tool
def save_fact(ctx: RunContext[FactStore], fact: NewFact) -> str:
    """Store one fact from the last answer. Returns the new fact ID."""
    return f'Saved as {ctx.deps.add(fact).id}.'


_LAST_ANSWER_NOTE = (
    '\n\n[Note to the interviewer: this is the last answer. '
    'Store its facts, then end.]'
)


class EndReason(StrEnum):
    """Why the interview ended."""

    AGENT_DONE = 'agent_done'
    QUESTION_LIMIT = 'question_limit'
    NO_NEW_FACTS = 'no_new_facts'


@dataclass(frozen=True)
class InterviewResult:
    """How the interview ended, and the full message history."""

    end_reason: EndReason
    messages: list[ModelMessage]


def run_interview(
    model: Model | str,
    store: FactStore,
    answer: Answerer,
    max_questions: int = 30,
    max_answers_without_facts: int = 3,
) -> InterviewResult:
    """Ask questions and store facts until one of three end conditions.

    The interview ends when:
    - the agent returns Done;
    - the person has given `max_questions` answers. The agent then runs once
      more with Done as the only output, so that it stores the facts of the
      last answer;
    - `max_answers_without_facts` answers in a row add no fact, for example
      "I don't know".

    Args:
        model: A PydanticAI model, or a model name such as
            `google:gemini-3.1-flash-lite`.
        store: Receives the facts that the agent saves.
        answer: Returns the answer to each question.
        max_questions: The most answers that the interview asks for.
        max_answers_without_facts: The most answers in a row that add no
            fact.

    Returns:
        The end reason and the full message history, for tracing and
        evaluation.
    """
    history: list[ModelMessage] = []
    prompt = 'Start the interview.'
    answers = 0
    answers_without_facts = 0
    while True:
        last_run = answers == max_questions
        facts_before = len(store.all())
        result = interviewer.run_sync(
            prompt + _LAST_ANSWER_NOTE if last_run else prompt,
            output_type=Done if last_run else None,
            deps=store,
            model=model,
            message_history=history,
        )
        history = result.all_messages()
        if len(store.all()) > facts_before:
            answers_without_facts = 0
        elif answers > 0:
            answers_without_facts += 1

        if last_run:
            return InterviewResult(EndReason.QUESTION_LIMIT, history)
        if isinstance(result.output, Done):
            return InterviewResult(EndReason.AGENT_DONE, history)
        if answers_without_facts == max_answers_without_facts:
            return InterviewResult(EndReason.NO_NEW_FACTS, history)
        prompt = answer(result.output.text)
        answers += 1
