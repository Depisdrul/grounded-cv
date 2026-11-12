import json
import unittest

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from cv_agent.models import Actor, Bullet, Fact, Status
from cv_agent.writer import write_bullets


def make_fact(fact_id, status):
    return Fact(
        id=fact_id,
        employer="Example Freight Ltd",
        role="Data engineer",
        period="2021-2023",
        action="Wrote the data export.",
        actor=Actor.ME,
        status=status,
    )


class WriteBulletsTest(unittest.TestCase):
    def test_returns_the_bullets_from_the_model(self):
        def reply_with_one_bullet(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {"response": [{"text": "Built the export job.", "fact_ids": ["F1"]}]},
                    )
                ]
            )

        bullets = write_bullets(FunctionModel(reply_with_one_bullet), [make_fact("F1", Status.CONFIRMED)])

        self.assertEqual(bullets, [Bullet(text="Built the export job.", fact_ids=["F1"])])

    def test_sends_only_confirmed_facts_to_the_model(self):
        user_prompt_texts = []

        def record_prompt(messages, info):
            user_prompt_texts.extend(
                part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)
            )
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": []})])

        confirmed = make_fact("F1", Status.CONFIRMED)
        write_bullets(
            FunctionModel(record_prompt),
            [confirmed, make_fact("F2", Status.TO_CONFIRM), make_fact("F3", Status.DO_NOT_CLAIM)],
        )

        self.assertEqual(len(user_prompt_texts), 1)
        self.assertEqual(
            json.loads(user_prompt_texts[0].removeprefix("Facts:\n")),
            [confirmed.model_dump(mode="json")],
        )


if __name__ == "__main__":
    unittest.main()
