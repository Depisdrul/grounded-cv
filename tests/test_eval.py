import json
import unittest
from unittest import mock

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from grounded_cv.eval import (
    BulletVerdict,
    FactMatch,
    FailureType,
    JudgeReport,
    build_interviewee_view,
    run_case,
    score,
)
from grounded_cv.interviewer import EndReason, InterviewResult
from grounded_cv.models import Actor, Bullet, Fact, NewFact, Status
from grounded_cv.persona import Persona, PersonaFact


def persona_fact(fact_id, action, actor, status):
    return PersonaFact(
        id=fact_id,
        employer='Example Rail Services',
        role='Software developer',
        period='2015-2017',
        action=action,
        actor=actor,
        status=status,
    )


def stored_fact(fact_id, action, actor, status):
    return Fact(
        id=fact_id,
        employer='Example Rail Services',
        role='Software developer',
        period='2015-2017',
        action=action,
        actor=actor,
        status=status,
    )


def fact_match(fact_id, persona_fact_id):
    return FactMatch(
        fact_id=fact_id, persona_fact_id=persona_fact_id, reason='Same work.'
    )


def verdict(bullet_index, failure_types):
    return BulletVerdict(
        bullet_index=bullet_index,
        failure_types=failure_types,
        reason='Checked.',
    )


def fake_interview(question, new_fact, end_reason):
    """Return a run_interview that asks one question and stores one fact."""

    def interview(model, store, answer, max_questions):
        answer(question)
        store.add(new_fact)
        return InterviewResult(end_reason, [])

    return interview


def reply_with_text(text):
    return FunctionModel(
        lambda messages, info: ModelResponse(parts=[TextPart(text)])
    )


def reply_with_judge_report(report):
    def reply(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name, report.model_dump(mode='json')
                )
            ]
        )

    return FunctionModel(reply)


class ScoreTest(unittest.TestCase):
    def test_recall_counts_each_matched_persona_fact_once(self):
        scores = score(
            persona=Persona(
                name='Two facts',
                facts=[
                    persona_fact(
                        'P1',
                        'Wrote the import job.',
                        Actor.ME,
                        Status.CONFIRMED,
                    ),
                    persona_fact(
                        'P2', 'Wrote the tests.', Actor.ME, Status.CONFIRMED
                    ),
                ],
            ),
            facts=[
                stored_fact(
                    'F1', 'Wrote the import job.', Actor.ME, Status.CONFIRMED
                ),
                stored_fact(
                    'F2', 'Built the import job.', Actor.ME, Status.CONFIRMED
                ),
            ],
            bullets=[],
            judge=JudgeReport(
                matches=[fact_match('F1', 'P1'), fact_match('F2', 'P1')],
                bullets=[],
            ),
        )

        self.assertEqual(scores.recall, 0.5)
        self.assertEqual(scores.precision, 1.0)

    def test_precision_counts_an_unmatched_stored_fact_as_invented(self):
        scores = score(
            persona=Persona(
                name='One fact',
                facts=[
                    persona_fact(
                        'P1',
                        'Wrote the import job.',
                        Actor.ME,
                        Status.CONFIRMED,
                    )
                ],
            ),
            facts=[
                stored_fact(
                    'F1', 'Wrote the import job.', Actor.ME, Status.CONFIRMED
                ),
                stored_fact(
                    'F2', 'Designed the platform.', Actor.ME, Status.CONFIRMED
                ),
            ],
            bullets=[],
            judge=JudgeReport(
                matches=[
                    fact_match('F1', 'P1'),
                    fact_match('F2', persona_fact_id=None),
                ],
                bullets=[],
            ),
        )

        self.assertEqual(scores.precision, 0.5)

    def test_attribution_compares_the_actor_of_each_matched_fact(self):
        scores = score(
            persona=Persona(
                name='Team and personal work',
                facts=[
                    persona_fact(
                        'P1',
                        'Moved the database.',
                        Actor.TEAM,
                        Status.CONFIRMED,
                    ),
                    persona_fact(
                        'P2', 'Wrote the tests.', Actor.ME, Status.CONFIRMED
                    ),
                ],
            ),
            facts=[
                stored_fact(
                    'F1', 'Moved the database.', Actor.ME, Status.CONFIRMED
                ),
                stored_fact(
                    'F2', 'Wrote the tests.', Actor.ME, Status.CONFIRMED
                ),
            ],
            bullets=[],
            judge=JudgeReport(
                matches=[fact_match('F1', 'P1'), fact_match('F2', 'P2')],
                bullets=[],
            ),
        )

        self.assertEqual(scores.attribution, 0.5)

    def test_inflation_counts_a_failed_bullet_once_and_each_type(self):
        scores = score(
            persona=Persona(
                name='One fact',
                facts=[
                    persona_fact(
                        'P1',
                        'Moved the database.',
                        Actor.TEAM,
                        Status.CONFIRMED,
                    )
                ],
            ),
            facts=[
                stored_fact(
                    'F1', 'Moved the database.', Actor.TEAM, Status.CONFIRMED
                )
            ],
            bullets=[
                Bullet(text='Led the database move.', fact_ids=['F1']),
                Bullet(
                    text='With the team, moved the database.',
                    fact_ids=['F1'],
                ),
            ],
            judge=JudgeReport(
                matches=[fact_match('F1', 'P1')],
                bullets=[
                    verdict(1, [FailureType.ACTOR, FailureType.SCOPE]),
                    verdict(2, []),
                ],
            ),
        )

        self.assertEqual(scores.inflation, 0.5)
        self.assertEqual(
            scores.inflation_by_type,
            {
                FailureType.ACTOR: 1,
                FailureType.SCOPE: 1,
                FailureType.STATUS: 0,
                FailureType.NUMBER: 0,
                FailureType.TECHNOLOGY: 0,
                FailureType.ESTIMATE: 0,
            },
        )

    def test_grounding_comes_from_the_check_not_from_the_judge(self):
        scores = score(
            persona=Persona(
                name='One fact',
                facts=[
                    persona_fact(
                        'P1',
                        'Wrote the import job.',
                        Actor.ME,
                        Status.CONFIRMED,
                    )
                ],
            ),
            facts=[
                stored_fact(
                    'F1', 'Wrote the import job.', Actor.ME, Status.CONFIRMED
                )
            ],
            bullets=[
                Bullet(text='Wrote the import job.', fact_ids=['F1']),
                Bullet(text='Improved the planning system.', fact_ids=[]),
            ],
            judge=JudgeReport(
                matches=[fact_match('F1', 'P1')],
                bullets=[verdict(1, []), verdict(2, [])],
            ),
        )

        self.assertEqual(scores.grounding, 0.5)

    def test_empty_denominators_give_none_not_a_perfect_score(self):
        scores = score(
            persona=Persona(
                name='One fact',
                facts=[
                    persona_fact(
                        'P1',
                        'Wrote the import job.',
                        Actor.ME,
                        Status.CONFIRMED,
                    )
                ],
            ),
            facts=[],
            bullets=[],
            judge=JudgeReport(matches=[], bullets=[]),
        )

        self.assertEqual(scores.recall, 0.0)
        self.assertIsNone(scores.precision)
        self.assertIsNone(scores.attribution)
        self.assertIsNone(scores.inflation)
        self.assertIsNone(scores.grounding)


class ScoreRejectsJudgeOutputTest(unittest.TestCase):
    def setUp(self):
        self.persona = Persona(
            name='One fact',
            facts=[
                persona_fact(
                    'P1', 'Wrote the import job.', Actor.ME, Status.CONFIRMED
                )
            ],
        )
        self.facts = [
            stored_fact(
                'F1', 'Wrote the import job.', Actor.ME, Status.CONFIRMED
            )
        ]
        self.bullets = [Bullet(text='Wrote the import job.', fact_ids=['F1'])]

    def test_a_missing_match_for_a_stored_fact(self):
        with self.assertRaisesRegex(ValueError, 'one match per stored fact'):
            score(
                persona=self.persona,
                facts=self.facts,
                bullets=self.bullets,
                judge=JudgeReport(matches=[], bullets=[verdict(1, [])]),
            )

    def test_an_unknown_persona_fact_id(self):
        with self.assertRaisesRegex(ValueError, 'unknown persona fact ID'):
            score(
                persona=self.persona,
                facts=self.facts,
                bullets=self.bullets,
                judge=JudgeReport(
                    matches=[fact_match('F1', 'P99')],
                    bullets=[verdict(1, [])],
                ),
            )

    def test_two_verdicts_for_the_same_bullet(self):
        with self.assertRaisesRegex(ValueError, 'one verdict per bullet'):
            score(
                persona=self.persona,
                facts=self.facts,
                bullets=self.bullets,
                judge=JudgeReport(
                    matches=[fact_match('F1', 'P1')],
                    bullets=[verdict(1, []), verdict(1, [])],
                ),
            )


class RunCaseTest(unittest.TestCase):
    def test_records_the_interview_and_the_scores(self):
        with (
            mock.patch(
                'grounded_cv.eval.run_interview',
                autospec=True,
                side_effect=fake_interview(
                    'What did you do?',
                    NewFact(
                        employer='Example Rail Services',
                        role='Software developer',
                        period='2015-2017',
                        action='Wrote the import job.',
                        actor=Actor.ME,
                        status=Status.CONFIRMED,
                    ),
                    EndReason.AGENT_DONE,
                ),
            ),
            mock.patch(
                'grounded_cv.eval.write_bullets',
                autospec=True,
                # F1 is the ID that FactStore gives the first fact.
                return_value=[
                    Bullet(text='Wrote the import job.', fact_ids=['F1'])
                ],
            ),
        ):
            result = run_case(
                Persona(
                    name='One fact',
                    facts=[
                        persona_fact(
                            'P1',
                            'Wrote the import job.',
                            Actor.ME,
                            Status.CONFIRMED,
                        )
                    ],
                ),
                product_model='test:product',
                interviewee_model=reply_with_text('I wrote the import job.'),
                judge_model=reply_with_judge_report(
                    JudgeReport(
                        matches=[fact_match('F1', 'P1')],
                        bullets=[verdict(1, [])],
                    )
                ),
                max_questions=5,
            )

        self.assertEqual(
            [(turn.question, turn.answer) for turn in result.transcript],
            [('What did you do?', 'I wrote the import job.')],
        )
        self.assertEqual(result.end_reason, EndReason.AGENT_DONE)
        self.assertEqual(
            [fact.action for fact in result.facts], ['Wrote the import job.']
        )
        self.assertEqual(result.scores.recall, 1.0)
        self.assertEqual(result.scores.grounding, 1.0)
        self.assertIsNone(result.judge_error)

    def test_a_judge_failure_keeps_the_interview(self):
        with (
            mock.patch(
                'grounded_cv.eval.run_interview',
                autospec=True,
                side_effect=fake_interview(
                    'What did you do?',
                    NewFact(
                        employer='Example Rail Services',
                        role='Software developer',
                        period='2015-2017',
                        action='Wrote the import job.',
                        actor=Actor.ME,
                        status=Status.CONFIRMED,
                    ),
                    EndReason.AGENT_DONE,
                ),
            ),
            mock.patch(
                'grounded_cv.eval.write_bullets',
                autospec=True,
                return_value=[
                    Bullet(text='Wrote the import job.', fact_ids=['F1'])
                ],
            ),
            self.assertLogs('grounded_cv.eval', level='ERROR') as logs,
        ):
            result = run_case(
                Persona(
                    name='One fact',
                    facts=[
                        persona_fact(
                            'P1',
                            'Wrote the import job.',
                            Actor.ME,
                            Status.CONFIRMED,
                        )
                    ],
                ),
                product_model='test:product',
                interviewee_model=reply_with_text('I wrote the import job.'),
                judge_model=reply_with_judge_report(
                    JudgeReport(matches=[], bullets=[verdict(1, [])])
                ),
                max_questions=5,
            )

        self.assertEqual(
            [(turn.question, turn.answer) for turn in result.transcript],
            [('What did you do?', 'I wrote the import job.')],
        )
        self.assertEqual(
            [fact.action for fact in result.facts], ['Wrote the import job.']
        )
        self.assertIsNone(result.scores)
        self.assertEqual(
            result.judge_error,
            'ValueError: Judge must give exactly one match per stored fact.',
        )
        self.assertEqual(
            logs.output,
            [
                'ERROR:grounded_cv.eval:Judge step failed: ValueError: '
                'Judge must give exactly one match per stored fact.'
            ],
        )


class IntervieweeViewTest(unittest.TestCase):
    def test_view_hides_the_labels_that_reveal_traps(self):
        persona = Persona(
            name='Example',
            facts=[
                PersonaFact(
                    id='P7',
                    employer='Example Clinics Group',
                    role='Backend developer',
                    period='2017-2020',
                    action='Designed a retry queue. It was never built.',
                    actor='me',
                    status='do_not_claim',
                    source='design document',
                    first_answer='I designed a retry queue.',
                    trap='planned_not_built',
                )
            ],
        )

        self.assertEqual(
            json.loads(build_interviewee_view(persona)),
            [
                {
                    'employer': 'Example Clinics Group',
                    'role': 'Backend developer',
                    'period': '2017-2020',
                    'action': 'Designed a retry queue. It was never built.',
                    'first_answer': 'I designed a retry queue.',
                }
            ],
        )


if __name__ == '__main__':
    unittest.main()
