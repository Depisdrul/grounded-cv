import unittest

from pydantic import ValidationError

from grounded_cv.models import Bullet, NewFact


def error_locations(validation_error):
    return [error['loc'] for error in validation_error.errors()]


class NewFactTest(unittest.TestCase):
    def test_unknown_actor_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            NewFact(
                employer='Example Freight Ltd',
                role='Data engineer',
                period='2021-2023',
                action='Wrote the data export.',
                actor='manager',
                status='confirmed',
            )

        self.assertEqual(error_locations(ctx.exception), [('actor',)])

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            NewFact(
                employer='Example Freight Ltd',
                role='Data engineer',
                period='2021-2023',
                action='Wrote the data export.',
                actor='me',
                status='maybe',
            )

        self.assertEqual(error_locations(ctx.exception), [('status',)])

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            NewFact(
                employer='Example Freight Ltd',
                role='Data engineer',
                period='2021-2023',
                action='Wrote the data export.',
                actor='me',
                status='confirmed',
                impact='Saved two days a month.',
            )

        self.assertEqual(error_locations(ctx.exception), [('impact',)])


class BulletTest(unittest.TestCase):
    def test_empty_fact_ids_is_accepted_so_the_check_can_reject_it(self):
        self.assertEqual(
            Bullet(text='Wrote the data export.', fact_ids=[]).fact_ids, []
        )


if __name__ == '__main__':
    unittest.main()
