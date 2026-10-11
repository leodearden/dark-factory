"""Contract tests for ``escalation.mismatch_streak.MismatchStreak``.

The INV-4 storm escape for the γ2 ``project_root`` guard
(``plans/escalation-store-ambiguity-prd.md`` §6.4), driven over a REAL
``EscalationQueue``: a MagicMock cannot witness a fold or a dedupe_count.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import pytest

from escalation.mismatch_streak import DEFAULT_MISMATCH_STREAK_THRESHOLD, MismatchStreak
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.store_identity import StoreIdentity

REFUSAL = 'project_root mismatch: the refusal text the caller saw'


def storm(queue: EscalationQueue) -> list[Escalation]:
    return [e for e in queue.get_pending() if e.category == 'infra_issue' and e.level == 1]


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    path = tmp_path / 'proj'
    path.mkdir()
    return path


@pytest.fixture
def queue(proj: Path) -> EscalationQueue:
    return EscalationQueue(proj / 'esc')


@pytest.fixture
def project_identity(queue: EscalationQueue, proj: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='project', queue_dir=queue.queue_dir, project_id='proj', project_root=proj,
    )


@pytest.fixture
def recon_queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'recon')


@pytest.fixture
def recon_identity(recon_queue: EscalationQueue) -> StoreIdentity:
    return StoreIdentity(
        kind='reconciliation', queue_dir=recon_queue.queue_dir, project_id=None, project_root=None,
    )


@pytest.fixture
def other(tmp_path: Path) -> Path:
    path = tmp_path / 'other'
    path.mkdir()
    return path


def _record_n(streak: MismatchStreak, identity: StoreIdentity, root: str, n: int) -> list[bool]:
    return [streak.record_mismatch(identity, root, REFUSAL) for _ in range(n)]


class TestDefaultThreshold:
    def test_default_is_five(self) -> None:
        assert DEFAULT_MISMATCH_STREAK_THRESHOLD == 5

    def test_fifth_mismatch_files_one_record(
        self, queue: EscalationQueue, project_identity: StoreIdentity, other: Path,
    ) -> None:
        streak = MismatchStreak(queue)

        assert _record_n(streak, project_identity, str(other), 4) == [False] * 4
        assert storm(queue) == []

        assert streak.record_mismatch(project_identity, str(other), REFUSAL) is True
        assert len(storm(queue)) == 1


class TestThresholdValidation:
    @pytest.mark.parametrize('bad', [0, -1, True, 2.5])
    def test_constructor_rejects_non_positive_int(self, queue: EscalationQueue, bad) -> None:
        with pytest.raises(ValueError, match=re.escape(repr(bad))):
            MismatchStreak(queue, threshold=bad)


class TestRecordShape:
    def test_project_store_record(
        self, queue: EscalationQueue, project_identity: StoreIdentity, other: Path,
    ) -> None:
        streak = MismatchStreak(queue, threshold=3)
        _record_n(streak, project_identity, str(other), 3)

        records = storm(queue)
        assert len(records) == 1
        esc = records[0]
        assert esc.category == 'infra_issue'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.agent_role.startswith('harness-')
        assert esc.dedupe_fingerprint
        assert re.fullmatch(r'[A-Za-z0-9._-]+', esc.task_id)
        assert str(other) in esc.summary
        assert str(queue.queue_dir) in esc.summary
        assert 'proj' in esc.summary
        assert REFUSAL in esc.detail

    def test_reconciliation_store_record(
        self, recon_queue: EscalationQueue, recon_identity: StoreIdentity, other: Path,
    ) -> None:
        streak = MismatchStreak(recon_queue, threshold=3)
        _record_n(streak, recon_identity, str(other), 3)

        records = storm(recon_queue)
        assert len(records) == 1
        assert 'reconciliation' in records[0].summary
        assert str(recon_queue.queue_dir) in records[0].summary


class TestReArm:
    def test_second_streak_folds_into_the_open_record(
        self, queue: EscalationQueue, project_identity: StoreIdentity, other: Path,
    ) -> None:
        streak = MismatchStreak(queue, threshold=3)
        _record_n(streak, project_identity, str(other), 3)

        assert _record_n(streak, project_identity, str(other), 2) == [False, False]
        records = storm(queue)
        assert len(records) == 1
        assert records[0].dedupe_count == 0

        assert streak.record_mismatch(project_identity, str(other), REFUSAL) is True
        records = storm(queue)
        assert len(records) == 1
        assert records[0].dedupe_count == 1

    def test_resolved_record_does_not_suppress_the_next_streak(
        self, queue: EscalationQueue, project_identity: StoreIdentity, other: Path,
    ) -> None:
        streak = MismatchStreak(queue, threshold=3)
        _record_n(streak, project_identity, str(other), 3)
        first_id = storm(queue)[0].id
        queue.resolve(first_id, 'fixed')

        _record_n(streak, project_identity, str(other), 3)

        records = storm(queue)
        assert len(records) == 1
        assert records[0].id != first_id


class TestPerRootStreaks:
    def test_interleaved_roots_count_independently(
        self, tmp_path: Path, queue: EscalationQueue, project_identity: StoreIdentity,
    ) -> None:
        root_a = tmp_path / 'root-alpha'
        root_b = tmp_path / 'root-beta'
        streak = MismatchStreak(queue, threshold=3)

        for _ in range(2):
            assert streak.record_mismatch(project_identity, str(root_a), REFUSAL) is False
            assert streak.record_mismatch(project_identity, str(root_b), REFUSAL) is False
        assert storm(queue) == []

        assert streak.record_mismatch(project_identity, str(root_a), REFUSAL) is True
        assert streak.record_mismatch(project_identity, str(root_b), REFUSAL) is True

        records = storm(queue)
        assert len(records) == 2
        assert records[0].task_id != records[1].task_id
        by_root = {
            root: [e for e in records if str(root) in e.summary] for root in (root_a, root_b)
        }
        assert len(by_root[root_a]) == 1
        assert len(by_root[root_b]) == 1
        assert str(root_b) not in by_root[root_a][0].summary
        assert str(root_a) not in by_root[root_b][0].summary

    def test_spellings_of_one_resolved_root_share_a_streak(
        self, queue: EscalationQueue, project_identity: StoreIdentity, other: Path,
    ) -> None:
        streak = MismatchStreak(queue, threshold=3)
        spellings = [str(other), str(other) + '/', str(other / 'x' / '..')]

        fired = [streak.record_mismatch(project_identity, s, REFUSAL) for s in spellings]

        assert fired == [False, False, True]
        assert len(storm(queue)) == 1

    def test_awkward_root_still_files(
        self, tmp_path: Path, queue: EscalationQueue, project_identity: StoreIdentity,
    ) -> None:
        awkward = tmp_path / 'has space' / 'ünï'
        streak = MismatchStreak(queue, threshold=3)

        assert _record_n(streak, project_identity, str(awkward), 3)[-1] is True

        records = storm(queue)
        assert len(records) == 1
        assert re.fullmatch(r'[A-Za-z0-9._-]+', records[0].task_id)


class TestFilingFailure:
    def test_failed_submit_is_logged_not_raised(
        self,
        tmp_path: Path,
        proj: Path,
        other: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class _BrokenQueue(EscalationQueue):
            def submit(self, escalation: Escalation) -> str:
                raise OSError('disk full')

        broken = _BrokenQueue(tmp_path / 'broken')
        identity = StoreIdentity(
            kind='project', queue_dir=broken.queue_dir, project_id='proj', project_root=proj,
        )
        streak = MismatchStreak(broken, threshold=3)

        with caplog.at_level(logging.WARNING, logger='escalation.mismatch_streak'):
            assert _record_n(streak, identity, str(other), 3)[-1] is True

        assert any(
            r.name == 'escalation.mismatch_streak' and r.levelno >= logging.WARNING
            for r in caplog.records
        )
        assert storm(broken) == []


class TestConcurrency:
    def test_one_fire_and_one_record_under_contention(
        self, queue: EscalationQueue, project_identity: StoreIdentity, other: Path,
    ) -> None:
        workers = 8
        streak = MismatchStreak(queue, threshold=workers)
        barrier = threading.Barrier(workers)
        results: list[bool] = []
        results_lock = threading.Lock()

        def _worker() -> None:
            barrier.wait()
            fired = streak.record_mismatch(project_identity, str(other), REFUSAL)
            with results_lock:
                results.append(fired)

        threads = [threading.Thread(target=_worker) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(results) == workers
        assert results.count(True) == 1
        assert len(storm(queue)) == 1
