import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from cv_agent.interviewer import _LAST_ANSWER_NOTE, run_interview
from cv_agent.models import Actor, Fact, Status
from cv_agent.store import FactStore


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

        run_interview(model, self.store, answer, max_questions=5)

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
                    ToolCallPart(
                        output_tool_names[0], {'reason': 'Limit reached.'}
                    )
                ]
            )

        answers = iter(['First answer.', 'Second answer.'])

        history = run_interview(
            FunctionModel(ask_until_only_done_is_allowed),
            self.store,
            lambda question: next(answers),
            max_questions=2,
        )

        self.assertEqual(
            user_prompts(history),
            [
                'Start the interview.',
                'First answer.',
                'Second answer.' + _LAST_ANSWER_NOTE,
            ],
        )


if __name__ == '__main__':
    unittest.main()
