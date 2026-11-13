import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from grounded_cv.eval import (
    BulletVerdict,
    FactMatch,
    FailureType,
    JudgeReport,
    build_interviewee_view,
    run_case,
    score,
)
from grounded_cv.interviewer import EndReason
from grounded_cv.models import Bullet, Fact, NewFact
from grounded_cv.persona import Persona, PersonaFact


def fact_fields(action, actor='me', status='confirmed'):
    return {
        'employer': 'Example Company',
        'role': 'Developer',
        'period': '2020-2021',
        'action': action,
        'actor': actor,
        'status': status,
    }


class EvaluationScoresTest(unittest.TestCase):
    def setUp(self):
        self.persona = Persona(
            name='Example',
            facts=[
                PersonaFact(id='P1', **fact_fields('Wrote the import.')),
                PersonaFact(
                    id='P2', **fact_fields('Moved servers.', actor='team')
                ),
                PersonaFact(id='P3', **fact_fields('Wrote tests.')),
            ],
        )
        self.facts = [
            Fact(id='F1', **fact_fields('Wrote the import.')),
            Fact(id='F2', **fact_fields('Moved servers.')),
            Fact(id='F3', **fact_fields('Invented a feature.')),
        ]
        self.bullets = [
            Bullet(text='Wrote the import.', fact_ids=['F1']),
            Bullet(text='Moved servers.', fact_ids=['F2']),
            Bullet(text='No citation.', fact_ids=[]),
        ]
        self.judge = JudgeReport(
            matches=[
                FactMatch(fact_id='F1', persona_fact_id='P1', reason='Same'),
                FactMatch(fact_id='F2', persona_fact_id='P2', reason='Same'),
                FactMatch(fact_id='F3', persona_fact_id=None, reason='New'),
            ],
            bullets=[
                BulletVerdict(bullet_index=1, failure_types=[], reason='OK'),
                BulletVerdict(
                    bullet_index=2,
                    failure_types=[FailureType.ACTOR, FailureType.SCOPE],
                    reason='Team work claimed as personal.',
                ),
                BulletVerdict(
                    bullet_index=3,
                    failure_types=[FailureType.SCOPE],
                    reason='No support.',
                ),
            ],
        )

    def test_metrics_and_failure_counts(self):
        scores = score(self.persona, self.facts, self.bullets, self.judge)
        self.assertAlmostEqual(scores.recall, 2 / 3)
        self.assertAlmostEqual(scores.precision, 2 / 3)
        self.assertEqual(scores.attribution, 0.5)
        self.assertAlmostEqual(scores.inflation, 2 / 3)
        self.assertAlmostEqual(scores.grounding, 2 / 3)
        self.assertEqual(scores.inflation_by_type[FailureType.ACTOR], 1)
        self.assertEqual(scores.inflation_by_type[FailureType.SCOPE], 2)

    def test_empty_denominators_are_not_successes(self):
        scores = score(
            self.persona, [], [], JudgeReport(matches=[], bullets=[])
        )
        self.assertEqual(scores.recall, 0)
        self.assertIsNone(scores.precision)
        self.assertIsNone(scores.attribution)
        self.assertIsNone(scores.inflation)
        self.assertIsNone(scores.grounding)

    def test_duplicate_and_unknown_judge_references_fail(self):
        bad = self.judge.model_copy(deep=True)
        bad.matches[0].persona_fact_id = 'P99'
        with self.assertRaisesRegex(ValueError, 'unknown persona'):
            score(self.persona, self.facts, self.bullets, bad)
        bad = self.judge.model_copy(deep=True)
        bad.bullets[1].bullet_index = 1
        with self.assertRaisesRegex(ValueError, 'one verdict'):
            score(self.persona, self.facts, self.bullets, bad)

    def test_run_case_connects_transcript_facts_writer_and_judge(self):
        answer = SimpleNamespace(
            output='I wrote the import.', all_messages=lambda: []
        )
        report = JudgeReport(
            matches=[
                FactMatch(fact_id='F1', persona_fact_id='P1', reason='Same')
            ],
            bullets=[
                BulletVerdict(bullet_index=1, failure_types=[], reason='OK')
            ],
        )

        def interview(model, store, answerer, max_questions):
            self.assertEqual(answerer('What did you do?'), answer.output)
            store.add(NewFact(**fact_fields('Wrote the import.')))
            return SimpleNamespace(end_reason=EndReason.AGENT_DONE)

        bullet = Bullet(text='Wrote the import.', fact_ids=['F1'])
        with (
            patch('grounded_cv.eval.run_interview', side_effect=interview),
            patch('grounded_cv.eval.write_bullets', return_value=[bullet]),
            patch('grounded_cv.eval._SIMULATOR.run_sync', return_value=answer),
            patch(
                'grounded_cv.eval._JUDGE.run_sync',
                return_value=SimpleNamespace(output=report),
            ),
        ):
            result = run_case(self.persona, 'product', 'simulator', 'judge', 5)

        self.assertEqual(result.transcript[0].answer, answer.output)
        self.assertEqual(result.facts[0].id, 'F1')
        self.assertEqual(result.bullets, [bullet])
        self.assertEqual(result.scores.grounding, 1)
        self.assertEqual(result.scores.recall, 1 / 3)
        self.assertEqual(
            set(result.prompt_sha256),
            {'interviewer', 'writer', 'interviewee', 'judge'},
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
