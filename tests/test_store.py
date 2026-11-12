import tempfile
import unittest
from pathlib import Path

from cv_agent.models import Actor, Fact, NewFact, Status
from cv_agent.store import FactStore


def make_new_fact(action):
    return NewFact(
        employer='Example Freight Ltd',
        role='Data engineer',
        period='2021-2023',
        action=action,
        actor=Actor.ME,
        status=Status.CONFIRMED,
    )


class FactStoreTest(unittest.TestCase):
    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.path = Path(tmp_dir.name) / 'facts.json'

    def test_add_assigns_sequential_ids(self):
        store = FactStore(self.path)

        self.assertEqual(
            store.add(make_new_fact('Wrote the data export.')).id, 'F1'
        )
        self.assertEqual(
            store.add(make_new_fact('Moved the servers to the cloud.')).id, 'F2'
        )

    def test_reopened_store_holds_the_saved_facts(self):
        FactStore(self.path).add(
            NewFact(
                employer='Example Freight Ltd',
                role='Data engineer',
                period='2021-2023',
                action='Moved the servers to the cloud.',
                actor=Actor.TEAM,
                status=Status.TO_CONFIRM,
                source='interview turn 3',
            )
        )

        self.assertEqual(
            FactStore(self.path).all(),
            [
                Fact(
                    id='F1',
                    employer='Example Freight Ltd',
                    role='Data engineer',
                    period='2021-2023',
                    action='Moved the servers to the cloud.',
                    actor=Actor.TEAM,
                    status=Status.TO_CONFIRM,
                    source='interview turn 3',
                )
            ],
        )

    def test_reopened_store_continues_the_ids(self):
        FactStore(self.path).add(make_new_fact('Wrote the data export.'))

        self.assertEqual(
            FactStore(self.path)
            .add(make_new_fact('Moved the servers to the cloud.'))
            .id,
            'F2',
        )

    def test_get_returns_none_for_an_unknown_id(self):
        store = FactStore(self.path)
        store.add(make_new_fact('Wrote the data export.'))

        self.assertIsNone(store.get('F9'))


if __name__ == '__main__':
    unittest.main()
