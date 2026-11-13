"""An L2 cascade attributes each member to its parent through one wire format.

``models.cascade_resolved_by`` is the writer the queue's resolve and park
cascades stamp on a member's ``resolved_by``; ``models.cascade_parent_id`` is
its reader. Driven through the queue's public ``resolve`` and ``park``;
observed through the resolve callback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _escalation_seed import seed_escalation

from escalation.models import Escalation, cascade_parent_id, cascade_resolved_by
from escalation.queue import EscalationQueue


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def fired(queue: EscalationQueue) -> list[Escalation]:
    calls: list[Escalation] = []
    queue.set_resolve_callback(calls.append)
    return calls


def _cluster(queue: EscalationQueue) -> tuple[Escalation, Escalation]:
    member = seed_escalation(queue, level=1, task_id='43')
    l2 = seed_escalation(queue, level=2, task_id='42', members=[member.id])
    return l2, member


def _fired_for(fired: list[Escalation], esc_id: str) -> Escalation:
    [esc] = [e for e in fired if e.id == esc_id]
    return esc


class TestCascadeAttributionRoundTrip:
    def test_the_reader_recovers_the_parent_the_writer_named(self):
        assert cascade_parent_id(cascade_resolved_by('esc-42-7')) == 'esc-42-7'

    @pytest.mark.parametrize(
        'resolved_by', [None, 'steward', 'escalation-watcher', 'l2-cascade:', 42],
    )
    def test_anything_else_names_no_parent(self, resolved_by: object):
        assert cascade_parent_id(resolved_by) is None


class TestQueueCascadesStampTheWriter:
    def test_a_resolve_cascade_names_the_parent(self, queue, fired):
        l2, member = _cluster(queue)

        queue.resolve(l2.id, 'fixed upstream', resolved_by='interactive')

        assert cascade_parent_id(_fired_for(fired, member.id).resolved_by) == l2.id
        assert cascade_parent_id(_fired_for(fired, l2.id).resolved_by) is None

    def test_a_park_cascade_names_the_parent(self, queue, fired):
        l2, member = _cluster(queue)

        queue.park(l2.id, 'held for a human', resolved_by='interactive')

        assert cascade_parent_id(_fired_for(fired, member.id).resolved_by) == l2.id
