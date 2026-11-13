"""Run the evaluation on the synthetic personas, with Pydantic Evals.

Usage: python -m grounded_cv.eval --help

Each persona file in `personas/` is one case. A case runs the interview with
a simulated interviewee, writes the bullets, and asks a judge model to score
them. The result of each case goes to `results/` as soon as the case ends.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_evals import Case, Dataset

from grounded_cv.check import BulletCheckResult, check_bullets
from grounded_cv.config import DEFAULT_MODEL
from grounded_cv.interviewer import EndReason, run_interview
from grounded_cv.models import Bullet, Fact, StrictModel
from grounded_cv.persona import Persona, load_persona
from grounded_cv.prompts import load_prompt
from grounded_cv.store import FactStore
from grounded_cv.writer import write_bullets

# Not `__name__`: under `python -m grounded_cv.eval` it is `__main__`, which
# is outside the `grounded_cv` logger that `main` sets to INFO.
logger = logging.getLogger('grounded_cv.eval')

LOCAL_MODEL = 'ollama:qwen3.5-4b-16k'
PROMPT_NAMES = ('interviewer', 'writer', 'interviewee', 'judge')


class FailureType(StrEnum):
    """Why the judge rejects a bullet. prompts/judge.md defines each type."""

    ACTOR = 'actor'
    SCOPE = 'scope'
    STATUS = 'status'
    NUMBER = 'number'
    TECHNOLOGY = 'technology'
    ESTIMATE = 'estimate'


class FactMatch(StrictModel):
    """The persona fact that a stored fact describes, or None."""

    fact_id: str
    persona_fact_id: str | None
    reason: str


class BulletVerdict(StrictModel):
    """The judge's verdict on one bullet. Bullet indexes start at 1."""

    bullet_index: int = Field(ge=1)
    failure_types: list[FailureType]
    reason: str


class JudgeReport(StrictModel):
    """The judge's output: one match per stored fact, one verdict per bullet."""

    matches: list[FactMatch]
    bullets: list[BulletVerdict]


class Turn(StrictModel):
    """One question of the interviewer and the interviewee's answer."""

    question: str
    answer: str


class Scores(StrictModel):
    """The five metrics. A metric is None when its denominator is 0."""

    recall: float | None
    precision: float | None
    attribution: float | None
    inflation: float | None
    grounding: float | None
    inflation_by_type: dict[FailureType, int]


class RunResult(StrictModel):
    """Everything about one case, so that a reader can check the scores.

    `judge` and `scores` are None when the judge step failed; `judge_error`
    then says why. The interview, the facts and the bullets are kept.
    """

    persona: Persona
    product_model: str
    product_thinking_off: bool
    interviewee_model: str
    judge_model: str
    max_questions: int
    end_reason: EndReason
    transcript: list[Turn]
    facts: list[Fact]
    bullets: list[Bullet]
    checks: list[BulletCheckResult]
    judge: JudgeReport | None
    scores: Scores | None
    judge_error: str | None
    prompt_sha256: dict[str, str]


class JudgeOutcome(StrictModel):
    """The judge report and the scores, or the error of the judge step."""

    judge: JudgeReport | None
    scores: Scores | None
    judge_error: str | None


# `thinking=False` does not reach Ollama; this key does, and other providers
# ignore it. See docs/decisions.md, entry 19.
NO_THINKING = OpenAIChatModelSettings(openai_reasoning_effort='none')

# On Ollama, a thinking interviewee fills its context before it answers.
_INTERVIEWEE = Agent(
    instructions=load_prompt('interviewee'), model_settings=NO_THINKING
)
_JUDGE = Agent(output_type=JudgeReport, instructions=load_prompt('judge'))

_INTERVIEWEE_FIELDS = {'employer', 'role', 'period', 'action', 'first_answer'}


def build_interviewee_view(persona: Persona) -> str:
    """Return the persona as compact JSON, without the trap labels.

    The interviewee does not see `id`, `actor`, `status`, `source` or
    `trap`. A label such as "do_not_claim" prompts a model to state the
    truth at once, and the trap no longer tests the interviewer.
    """
    return json.dumps(
        [
            fact.model_dump(mode='json', include=_INTERVIEWEE_FIELDS)
            for fact in persona.facts
        ],
        separators=(',', ':'),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(
    persona: Persona,
    facts: list[Fact],
    bullets: list[Bullet],
    judge: JudgeReport,
) -> Scores:
    """Check the judge's references, then compute the five metrics.

    Raises:
        ValueError: The judge did not give exactly one match per stored
            fact and one verdict per bullet, or named an unknown persona
            fact.
    """
    persona_by_id = {fact.id: fact for fact in persona.facts}
    facts_by_id = {fact.id: fact for fact in facts}
    match_ids = [match.fact_id for match in judge.matches]
    if len(match_ids) != len(set(match_ids)) or set(match_ids) != set(
        facts_by_id
    ):
        raise ValueError('Judge must give exactly one match per stored fact.')
    if any(
        match.persona_fact_id is not None
        and match.persona_fact_id not in persona_by_id
        for match in judge.matches
    ):
        raise ValueError('Judge returned an unknown persona fact ID.')
    indexes = [verdict.bullet_index for verdict in judge.bullets]
    if len(indexes) != len(set(indexes)) or set(indexes) != set(
        range(1, len(bullets) + 1)
    ):
        raise ValueError('Judge must give exactly one verdict per bullet.')

    matches = [
        match for match in judge.matches if match.persona_fact_id is not None
    ]
    correct_actors = sum(
        facts_by_id[match.fact_id].actor
        == persona_by_id[match.persona_fact_id].actor
        for match in matches
    )
    checks = check_bullets(bullets, facts)
    return Scores(
        recall=_rate(
            len({match.persona_fact_id for match in matches}),
            len(persona.facts),
        ),
        precision=_rate(len(matches), len(facts)),
        attribution=_rate(correct_actors, len(matches)),
        inflation=_rate(
            sum(bool(verdict.failure_types) for verdict in judge.bullets),
            len(bullets),
        ),
        grounding=_rate(sum(check.passed for check in checks), len(bullets)),
        inflation_by_type={
            kind: sum(
                kind in verdict.failure_types for verdict in judge.bullets
            )
            for kind in FailureType
        },
    )


def _build_judge_input(
    persona: Persona, facts: list[Fact], bullets: list[Bullet]
) -> str:
    return json.dumps(
        {
            'persona': persona.model_dump(mode='json'),
            'stored_facts': [fact.model_dump(mode='json') for fact in facts],
            'bullets': [
                {'bullet_index': index, **bullet.model_dump(mode='json')}
                for index, bullet in enumerate(bullets, 1)
            ],
        }
    )


def _model_label(model: Model | str) -> str:
    return model if isinstance(model, str) else model.model_name


def run_case(
    persona: Persona,
    product_model: Model | str,
    interviewee_model: Model | str,
    judge_model: Model | str,
    max_questions: int,
    product_thinking_off: bool,
) -> RunResult:
    """Run one interview, write the bullets, and judge them.

    Model arguments are PydanticAI models or names such as
    `ollama:qwen3.5-4b-16k`. The result records each model by name.
    `product_thinking_off` turns off thinking for the interviewer and the
    writer; otherwise they use the default of their model.

    A failed judge step does not raise: the result keeps the interview and
    records the error in `judge_error`. A failed interview or writer step
    raises.
    """
    product_settings = NO_THINKING if product_thinking_off else None
    transcript: list[Turn] = []
    history: list[ModelMessage] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FactStore(Path(tmp_dir) / 'facts.json')
        last_answer_at = time.monotonic()

        def answer(question: str) -> str:
            nonlocal history, last_answer_at
            number = len(transcript) + 1
            logger.info(
                'Q%d (interviewer %.0f s, %d facts stored): %s',
                number,
                time.monotonic() - last_answer_at,
                len(store.all()),
                question,
            )
            prompt = question
            if not history:
                prompt = (
                    f'Your work history:\n{build_interviewee_view(persona)}'
                    f'\n\nQuestion: {question}'
                )
            asked_at = time.monotonic()
            response = _INTERVIEWEE.run_sync(
                prompt, model=interviewee_model, message_history=history
            )
            history = response.all_messages()
            transcript.append(Turn(question=question, answer=response.output))
            last_answer_at = time.monotonic()
            logger.info(
                'A%d (interviewee %.0f s): %s',
                number,
                last_answer_at - asked_at,
                response.output,
            )
            return response.output

        interview = run_interview(
            product_model,
            store,
            answer,
            max_questions=max_questions,
            model_settings=product_settings,
        )
        facts = store.all()
    logger.info(
        'Interview ended (%s) with %d facts. Writing bullets.',
        interview.end_reason,
        len(facts),
    )
    bullets = write_bullets(product_model, facts, product_settings)
    outcome = judge_case(persona, facts, bullets, judge_model)

    return RunResult(
        persona=persona,
        product_model=_model_label(product_model),
        product_thinking_off=product_thinking_off,
        interviewee_model=_model_label(interviewee_model),
        judge_model=_model_label(judge_model),
        max_questions=max_questions,
        end_reason=interview.end_reason,
        transcript=transcript,
        facts=facts,
        bullets=bullets,
        checks=check_bullets(bullets, facts),
        judge=outcome.judge,
        scores=outcome.scores,
        judge_error=outcome.judge_error,
        prompt_sha256={name: _prompt_sha256(name) for name in PROMPT_NAMES},
    )


def _prompt_sha256(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()


def judge_case(
    persona: Persona,
    facts: list[Fact],
    bullets: list[Bullet],
    judge_model: Model | str,
) -> JudgeOutcome:
    """Ask the judge to match the facts and rate the bullets, then score.

    Any failure (quota, invalid output, unknown reference) is returned in
    `judge_error`, not raised, so that the caller keeps the interview.
    """
    logger.info('Judging %d bullets.', len(bullets))
    try:
        judge = _JUDGE.run_sync(
            _build_judge_input(persona, facts, bullets), model=judge_model
        ).output
        scores = score(persona, facts, bullets, judge)
    except Exception as error:
        judge_error = f'{type(error).__name__}: {error}'
        logger.error('Judge step failed: %s', judge_error)
        return JudgeOutcome(judge=None, scores=None, judge_error=judge_error)
    return JudgeOutcome(judge=judge, scores=scores, judge_error=None)


def rejudge(result: RunResult, judge_model: Model | str) -> RunResult:
    """Return a copy of the result, judged again with the current judge prompt.

    The interview, the facts and the bullets stay the same. The copy records
    the new judge model and the hash of the current judge prompt.
    """
    outcome = judge_case(
        result.persona, result.facts, result.bullets, judge_model
    )
    return result.model_copy(
        update={
            'judge_model': _model_label(judge_model),
            'judge': outcome.judge,
            'scores': outcome.scores,
            'judge_error': outcome.judge_error,
            'prompt_sha256': {
                **result.prompt_sha256,
                'judge': _prompt_sha256('judge'),
            },
        }
    )


def save_result(result: RunResult, results_dir: Path, case_name: str) -> Path:
    """Write the result to `<results_dir>/<UTC time>-<case_name>.json`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    path = results_dir / f'{timestamp}-{case_name}.json'
    path.write_text(result.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path


def _rejudge_files(
    paths: list[Path], judge_model: str, results_dir: Path
) -> None:
    """Judge each saved result again and save the copy as a new file."""
    failed = False
    for path in paths:
        result = RunResult.model_validate_json(path.read_text(encoding='utf-8'))
        logger.info('Re-judging %s with %s.', path.name, judge_model)
        new_result = rejudge(result, judge_model)
        logger.info(
            'Saved %s',
            save_result(new_result, results_dir, f'{path.stem}-rejudged'),
        )
        summary = (
            new_result.scores.model_dump_json()
            if new_result.scores
            else 'judge failed'
        )
        sys.stdout.write(f'{path.name}: {summary}\n')
        failed = failed or new_result.judge_error is not None
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate on synthetic personas only.'
    )
    parser.add_argument('--product-model', default=LOCAL_MODEL)
    parser.add_argument('--interviewee-model', default=DEFAULT_MODEL)
    parser.add_argument('--judge-model', default=DEFAULT_MODEL)
    parser.add_argument('--max-questions', type=int, default=30)
    parser.add_argument('--personas', type=Path, default=Path('personas'))
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument(
        '--product-thinking-off',
        action='store_true',
        help='Turn off thinking for the interviewer and the writer.',
    )
    parser.add_argument(
        '--rejudge',
        type=Path,
        nargs='+',
        metavar='RESULT',
        help='Judge saved results again with --judge-model. No interview.',
    )
    args = parser.parse_args()
    if args.max_questions < 1:
        parser.error('--max-questions must be positive')

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('grounded_cv').setLevel(logging.INFO)
    load_dotenv()
    os.environ.setdefault('OLLAMA_BASE_URL', 'http://localhost:11434/v1')

    if args.rejudge:
        _rejudge_files(args.rejudge, args.judge_model, args.results)
        return

    paths = sorted(args.personas.glob('*.json'))
    if not paths:
        parser.error(f'No persona JSON files found in {args.personas}')
    for path in paths:
        load_persona(path)  # Fail before the first model call.
    suffix = '-thinking-off' if args.product_thinking_off else ''

    def run_and_save(path: Path) -> RunResult:
        logger.info(
            'Case %s: product %s (thinking %s), interviewee %s, judge %s, '
            '%d questions.',
            path.stem,
            args.product_model,
            'off' if args.product_thinking_off else 'default',
            args.interviewee_model,
            args.judge_model,
            args.max_questions,
        )
        result = run_case(
            load_persona(path),
            args.product_model,
            args.interviewee_model,
            args.judge_model,
            args.max_questions,
            args.product_thinking_off,
        )
        logger.info(
            'Saved %s',
            save_result(result, args.results, path.stem + suffix),
        )
        return result

    report = Dataset(
        name='synthetic-personas',
        cases=[Case(name=path.stem, inputs=path) for path in paths],
    ).evaluate_sync(run_and_save, max_concurrency=1, progress=False)

    for case in report.cases:
        scores = case.output.scores
        summary = scores.model_dump_json() if scores else 'judge failed'
        sys.stdout.write(f'{case.name}: {summary}\n')
    for failure in report.failures:
        sys.stderr.write(f'{failure.name}: {failure.error_message}\n')
        sys.stderr.write(f'{failure.error_stacktrace}\n')
    if report.failures or any(case.output.judge_error for case in report.cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
