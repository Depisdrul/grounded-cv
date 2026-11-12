import unittest

from cv_agent.check import BulletCheckResult, RejectionReason, check_bullets
from cv_agent.models import Actor, Bullet, Fact, Status


def make_fact(fact_id, status):
    return Fact(
        id=fact_id,
        employer='Example Freight Ltd',
        role='Data engineer',
        period='2021-2023',
        action='Wrote the data export.',
        actor=Actor.ME,
        status=status,
    )


class CheckBulletsTest(unittest.TestCase):
    def test_bullet_citing_a_confirmed_fact_passes(self):
        bullet = Bullet(text='Wrote the data export.', fact_ids=['F1'])

        results = check_bullets([bullet], [make_fact('F1', Status.CONFIRMED)])

        self.assertEqual(
            results,
            [
                BulletCheckResult(
                    bullet=bullet, passed=True, reason=None, fact_id=None
                )
            ],
        )

    def test_bullet_without_fact_ids_fails(self):
        bullet = Bullet(text='Wrote the data export.', fact_ids=[])

        results = check_bullets([bullet], [make_fact('F1', Status.CONFIRMED)])

        self.assertEqual(
            results,
            [
                BulletCheckResult(
                    bullet=bullet,
                    passed=False,
                    reason=RejectionReason.NO_FACT_IDS,
                    fact_id=None,
                )
            ],
        )

    def test_bullet_citing_an_unknown_id_fails_and_names_it(self):
        bullet = Bullet(text='Wrote the data export.', fact_ids=['F1', 'F9'])

        results = check_bullets([bullet], [make_fact('F1', Status.CONFIRMED)])

        self.assertEqual(
            results,
            [
                BulletCheckResult(
                    bullet=bullet,
                    passed=False,
                    reason=RejectionReason.UNKNOWN_ID,
                    fact_id='F9',
                )
            ],
        )

    def test_bullet_citing_an_unconfirmed_fact_fails_and_names_it(self):
        for status in (Status.TO_CONFIRM, Status.DO_NOT_CLAIM):
            with self.subTest(status=status):
                bullet = Bullet(
                    text='Wrote the data export.', fact_ids=['F1', 'F2']
                )

                results = check_bullets(
                    [bullet],
                    [
                        make_fact('F1', Status.CONFIRMED),
                        make_fact('F2', status),
                    ],
                )

                self.assertEqual(
                    results,
                    [
                        BulletCheckResult(
                            bullet=bullet,
                            passed=False,
                            reason=RejectionReason.NOT_CONFIRMED,
                            fact_id='F2',
                        )
                    ],
                )


if __name__ == '__main__':
    unittest.main()
