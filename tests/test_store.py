import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from grounded_cv.models import Actor, Fact, NewFact, Status
from grounded_cv.store import FactStore


def make_new_fact(action):
    return NewFact(
        employer='Example Freight Ltd',
        role='Data engineer',
        period='2021-2023',
        action=action,
        actor=Actor.ME,
        status=Status.CONFIRMED,
    )


# Read before any test patches grounded_cv.store.os.replace, which is the
# same object as os.replace in every module.
REAL_REPLACE = os.replace


def deny_access_then_replace(failure_count):
    """Return an os.replace that denies access `failure_count` times."""
    errors = [PermissionError('Access is denied')] * failure_count

    def replace(src, dst):
        if errors:
            raise errors.pop()
        REAL_REPLACE(src, dst)

    return replace


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

    def test_add_retries_until_the_last_attempt_succeeds(self):
        store = FactStore(self.path)

        with (
            mock.patch('grounded_cv.store._REPLACE_ATTEMPTS', 3),
            mock.patch('grounded_cv.store.time.sleep', autospec=True) as sleep,
            mock.patch(
                'grounded_cv.store.os.replace',
                autospec=True,
                side_effect=deny_access_then_replace(failure_count=2),
            ) as replace,
        ):
            store.add(make_new_fact('Wrote the data export.'))

        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(
            [fact.action for fact in FactStore(self.path).all()],
            ['Wrote the data export.'],
        )

    def test_add_that_fails_leaves_store_and_file_unchanged(self):
        store = FactStore(self.path)
        store.add(make_new_fact('Wrote the data export.'))

        with (
            mock.patch('grounded_cv.store._REPLACE_ATTEMPTS', 3),
            mock.patch('grounded_cv.store.time.sleep', autospec=True) as sleep,
            mock.patch(
                'grounded_cv.store.os.replace',
                autospec=True,
                side_effect=PermissionError('Access is denied'),
            ) as replace,
            self.assertRaises(PermissionError),
        ):
            store.add(make_new_fact('Moved the servers to the cloud.'))

        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(
            [fact.action for fact in store.all()], ['Wrote the data export.']
        )
        self.assertEqual(
            [fact.action for fact in FactStore(self.path).all()],
            ['Wrote the data export.'],
        )
        self.assertEqual(
            store.add(make_new_fact('Moved the servers to the cloud.')).id,
            'F2',
        )

    def test_parallel_adds_get_distinct_ids_and_are_all_written(self):
        store = FactStore(self.path)
        barrier = threading.Barrier(8)

        def add_fact(number):
            barrier.wait()
            store.add(make_new_fact(f'Task {number}.'))

        threads = [
            threading.Thread(target=add_fact, args=(number,))
            for number in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(
            sorted(fact.id for fact in FactStore(self.path).all()),
            sorted(f'F{number}' for number in range(1, 9)),
        )


if __name__ == '__main__':
    unittest.main()
