import unittest

from grounded_cv.eval import _INTERVIEWEE_FIELDS, FailureType
from grounded_cv.interviewer import save_fact
from grounded_cv.models import Actor, Bullet, Status
from grounded_cv.prompts import load_prompt


class InterviewerPromptTest(unittest.TestCase):
    def test_names_every_status_value(self):
        for status in Status:
            with self.subTest(status=status):
                self.assertIn(f'"{status}"', load_prompt('interviewer'))

    def test_names_every_actor_value(self):
        for actor in Actor:
            with self.subTest(actor=actor):
                self.assertIn(f'"{actor}"', load_prompt('interviewer'))

    def test_names_the_save_fact_tool(self):
        self.assertIn(save_fact.__name__, load_prompt('interviewer'))


class WriterPromptTest(unittest.TestCase):
    def test_names_every_actor_that_is_not_the_person(self):
        for actor in Actor:
            if actor is Actor.ME:
                continue
            with self.subTest(actor=actor):
                self.assertIn(f'"{actor}"', load_prompt('writer'))

    def test_names_the_fact_ids_field_of_a_bullet(self):
        self.assertIn('fact_ids', Bullet.model_fields)
        self.assertIn('`fact_ids`', load_prompt('writer'))


class JudgePromptTest(unittest.TestCase):
    def test_names_every_failure_type(self):
        for failure_type in FailureType:
            with self.subTest(failure_type=failure_type):
                self.assertIn(f'"{failure_type}"', load_prompt('judge'))


class IntervieweePromptTest(unittest.TestCase):
    def test_names_every_field_of_the_interviewee_view(self):
        for field in sorted(_INTERVIEWEE_FIELDS):
            with self.subTest(field=field):
                self.assertIn(f'`{field}`', load_prompt('interviewee'))


if __name__ == '__main__':
    unittest.main()
