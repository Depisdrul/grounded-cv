import unittest
from pathlib import Path

from pydantic import ValidationError

from grounded_cv.persona import Persona, load_persona

PERSONAS_DIR = Path(__file__).parent.parent / 'personas'


def persona_fact_dict(fact_id):
    return {
        'id': fact_id,
        'employer': 'Example Rail Services',
        'role': 'Software developer',
        'period': '2015-2017',
        'action': 'Wrote the timetable import job.',
        'actor': 'me',
        'status': 'confirmed',
    }


class PersonaFilesTest(unittest.TestCase):
    def test_every_committed_persona_file_is_valid(self):
        paths = sorted(PERSONAS_DIR.glob('*.json'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                load_persona(path)


class PersonaTest(unittest.TestCase):
    def test_duplicate_fact_ids_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, 'duplicate fact IDs: P1'):
            Persona.model_validate(
                {
                    'name': 'Duplicate IDs',
                    'facts': [persona_fact_dict('P1'), persona_fact_dict('P1')],
                }
            )

    def test_a_stored_fact_id_is_rejected(self):
        with self.assertRaises(ValidationError) as caught:
            Persona.model_validate(
                {'name': 'Stored ID', 'facts': [persona_fact_dict('F1')]}
            )

        self.assertEqual(
            [error['loc'] for error in caught.exception.errors()],
            [('facts', 0, 'id')],
        )


if __name__ == '__main__':
    unittest.main()
