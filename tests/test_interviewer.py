import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from grounded_cv.interviewer import _LAST_ANSWER_NOTE, EndReason, run_interview
from grounded_cv.models import Actor, Fact, Status
from grounded_cv.store import FactStore


def scripted_model(*responses):
    """A model that returns the given responses in order, one per model call."""
    queue = list(responses)
    return FunctionModel(lambda messages, info: queue.pop(0))


def user_prompts(history):
    return [
        part.content
        for message in history
        for part in getattr(message, 'parts', [])
        if isinstance(part, UserPromptPart)
    ]


def save_answers_with_built_else_ask(messages, info):
    """A model that saves an answer that contains 'built' as a fact.

    The answer text becomes the fact's action. On every other model call,
    the model asks 'What else?'.
    """
    answer_texts = [
        part.content
        for part in messages[-1].parts
        if isinstance(part, UserPromptPart)
    ]
    if answer_texts and 'built' in answer_texts[0]:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'save_fact',
                    {
                        'fact': {
                            'employer': 'Example Freight Ltd',
                            'role': 'Data engineer',
                            'period': '2021-2023',
                            'action': answer_texts[0],
                            'actor': 'me',
                            'status': 'confirmed',
                        }
                    },
                )
            ]
        )
    return ModelResponse(
        parts=[ToolCallPart('final_result_Question', {'text': 'What else?'})]
    )


class RunInterviewTest(unittest.TestCase):
    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.store = FactStore(Path(tmp_dir.name) / 'facts.json')

    def test_save_fact_stores_the_fact_from_the_answer(self):
        model = scripted_model(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'final_result_Question', {'text': 'What did you build?'}
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'save_fact',
                        {
                            'fact': {
                                'employer': 'Example Freight Ltd',
                                'role': 'Data engineer',
                                'period': '2021-2023',
                                'action': 'Moved the servers to the cloud.',
                                'actor': 'team',
                                'status': 'to_confirm',
                                'source': 'interview turn 1',
                            }
                        },
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        'final_result_Done', {'reason': 'No more topics.'}
                    )
                ]
            ),
        )
        questions = []

        def answer(question):
            questions.append(question)
            return 'We moved the servers to the cloud.'

        result = run_interview(
            model,
            self.store,
            answer,
            max_questions=5,
            max_answers_without_facts=3,
        )

        self.assertEqual(result.end_reason, EndReason.AGENT_DONE)
        self.assertEqual(questions, ['What did you build?'])
        self.assertEqual(
            self.store.all(),
            [
                Fact(
                    id='F1',
                    employer='Example Freight Ltd',
                    role='Data engineer',
                    period='2021-2023',
                    action='Moved the servers to the cloud.',
                    actor=Actor.TEAM,
                    status=Status.TO_CONFIRM,
                    source='interview turn 1',
                )
            ],
        )

    def test_last_answer_reaches_the_agent_when_the_question_limit_is_hit(self):
        def ask_until_only_done_is_allowed(messages, info):
            output_tool_names = [tool.name for tool in info.output_tools]
            if 'final_result_Question' in output_tool_names:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'final_result_Question', {'text': 'Anything else?'}
                        )
                    ]
                )
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'reason': 'Limit reached.'})
                ]
            )

        answers = iter(['First answer.', 'Second answer.'])

        result = run_interview(
            FunctionModel(ask_until_only_done_is_allowed),
            self.store,
            lambda question: next(answers),
            max_questions=2,
            max_answers_without_facts=5,
        )

        self.assertEqual(result.end_reason, EndReason.QUESTION_LIMIT)
        self.assertEqual(
            user_prompts(result.messages),
            [
                'Start the interview.',
                'First answer.',
                'Second answer.' + _LAST_ANSWER_NOTE,
            ],
        )

    def test_ends_after_the_answers_in_a_row_without_facts(self):
        answers = iter(["I don't know.", "I don't know."])

        result = run_interview(
            FunctionModel(save_answers_with_built_else_ask),
            self.store,
            lambda question: next(answers),
            max_questions=10,
            max_answers_without_facts=2,
        )

        self.assertEqual(result.end_reason, EndReason.NO_NEW_FACTS)
        self.assertEqual(
            user_prompts(result.messages),
            ['Start the interview.', "I don't know.", "I don't know."],
        )

    def test_an_answer_with_a_fact_restarts_the_count(self):
        answers = iter(
            [
                "I don't know.",
                'I built the export job.',
                "I don't know.",
                "I don't know.",
            ]
        )

        result = run_interview(
            FunctionModel(save_answers_with_built_else_ask),
            self.store,
            lambda question: next(answers),
            max_questions=10,
            max_answers_without_facts=2,
        )

        self.assertEqual(result.end_reason, EndReason.NO_NEW_FACTS)
        self.assertEqual(
            user_prompts(result.messages),
            [
                'Start the interview.',
                "I don't know.",
                'I built the export job.',
                "I don't know.",
                "I don't know.",
            ],
        )
        self.assertEqual(
            [fact.action for fact in self.store.all()],
            ['I built the export job.'],
        )

    def test_sends_the_model_settings_with_every_request(self):
        sent_settings = []

        def ask_once_then_done(messages, info):
            sent_settings.append(info.model_settings)
            if len(sent_settings) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'final_result_Question', {'text': 'What else?'}
                        )
                    ]
                )
            return ModelResponse(
                parts=[ToolCallPart('final_result_Done', {'reason': 'Done.'})]
            )

        run_interview(
            FunctionModel(ask_once_then_done),
            self.store,
            lambda question: 'Nothing else.',
            max_questions=5,
            max_answers_without_facts=3,
            model_settings={'temperature': 0.25},
        )

        self.assertEqual(
            [settings.get('temperature') for settings in sent_settings],
            [0.25, 0.25],
        )


if __name__ == '__main__':
    unittest.main()
