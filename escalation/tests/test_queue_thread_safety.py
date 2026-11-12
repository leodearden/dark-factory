"""The queue's read paths survive a write landing mid-iteration (task 4567).

The orchestrator runs ``get`` / ``get_by_task`` on worker threads while the
loop thread resolves records on the SAME ``EscalationQueue`` object.  A
resolve that archives into a dated subdir the memoised archive listing has
never seen inserts a new key into that listing.  ``Path.exists`` is a
syscall that releases the GIL, so a read iterating the listing while it
probes candidates can see that insert land between two of its steps.

These tests make that interleaving deterministic instead of racing real
threads: an interposed ``Path.exists`` runs the concurrent ``resolve()``
exactly where the reader's probe would yield, then lets the probe proceed.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from _scan_race_helpers import relocating_read_text

from escalation.models import Escalation
from escalation.queue import EscalationQueue

_OLD_DATE = '2020-01-01'


def _escalation(esc_id: str, task_id: str) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='task_failure',
        summary='Something failed',
    )


@contextmanager
def _archival_during_probe_of(
    probed_dir: Path, archive: Callable[[], Any],
) -> Iterator[list[bool]]:
    """Run *archive* the first time ``exists`` probes a path in *probed_dir*.

    Yields a one-slot list that turns ``[True]`` once the interposition has
    fired.  The flag flips BEFORE *archive* runs, so the resolve's own
    filesystem probes fall straight through to the real ``exists``.
    """
    original = Path.exists
    fired: list[bool] = [False]

    def hook(self: Path, *args: Any, **kwargs: Any) -> bool:
        if not fired[0] and self.parent == probed_dir:
            fired[0] = True
            archive()
        return original(self, *args, **kwargs)

    with patch.object(Path, 'exists', hook):
        yield fired


class TestArchiveListingMemoUnderConcurrentArchival:
    def test_get_survives_concurrent_archival_into_a_new_dated_subdir(
        self, tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        archived = _escalation('esc-4567-1', task_id='4567')
        pending = _escalation('esc-4568-1', task_id='4568')
        queue.submit(archived)
        queue.submit(pending)
        old_dir = queue.queue_dir / 'archive' / _OLD_DATE
        old_dir.mkdir(parents=True)
        os.replace(
            queue.queue_dir / f'{archived.id}.json', old_dir / f'{archived.id}.json',
        )

        with _archival_during_probe_of(
            old_dir, lambda: queue.resolve(pending.id, 'done', resolved_by='test'),
        ) as fired:
            found = queue.get(archived.id)

        assert fired == [True]
        assert found is not None
        assert found.id == archived.id
        resolved = queue.get(pending.id)
        assert resolved is not None
        assert resolved.status == 'resolved'

    def test_get_by_task_vanish_recovery_survives_concurrent_archival(
        self, tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        target = _escalation('esc-4567-1', task_id='4567')
        other = _escalation('esc-4568-1', task_id='4568')
        queue.submit(target)
        queue.submit(other)
        old_dir = queue.queue_dir / 'archive' / _OLD_DATE
        relocate = relocating_read_text(
            queue.queue_dir / f'{target.id}.json', old_dir,
        )

        with (
            patch.object(Path, 'read_text', relocate),
            _archival_during_probe_of(
                old_dir, lambda: queue.resolve(other.id, 'done', resolved_by='test'),
            ) as fired,
        ):
            results = queue.get_by_task(target.task_id)

        assert fired == [True]
        assert [e.id for e in results] == [target.id]
