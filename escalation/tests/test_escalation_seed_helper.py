"""Contract tests for the shared escalation seeding helper (``_escalation_seed``).

``_escalation_seed.seed_escalation`` is the single construction site for a
directly-submitted pending ``Escalation`` in this suite (INV-5; task 4997 folded
the three near-twin ``_seed`` bodies in ``test_capability_guard_http.py``,
``test_status_authority_gate.py`` and ``test_server.py`` into it). Roughly 67
call sites now reach ``queue.submit()`` only through it, so a silent regression
here would change the premise of the capability-guard scenarios and the
status-authority C1-C4 cells at once while they all stayed green — hence its own
pins.

Every assertion reads the record back through ``queue.get(...)``, the escalation
package's own read path, never through queue internals or a mock — the
convention ``test_status_authority_gate.py``'s module docstring states
explicitly. A plain ``tmp_path`` queue is enough for all of it: seeding touches
no wire, so there is no server, no fixture and no asyncio here.

One property per test, and deliberately no id-FORMAT assertion: the last test
pins only that repeated seeds under one ``task_id`` get DISTINCT retrievable
ids, because ``escalation/src/escalation/queue.py::make_id`` documents at length
that its argument is an id-namespace key which need not equal the record's
``task_id`` and from which nothing may derive one.

There is deliberately NO AST scan asserting this is the only seeding site, of
the kind ``test_escalation_http_helper.py`` runs for the capability headers.
That scan works because its subject is a two-element vocabulary of wire-protocol
literals with a single legitimate home. Seeding has no such vocabulary: this
directory legitimately holds ~15 other ``queue.submit()`` helpers with genuinely
different shapes (``_seed_heavy``, ``_seed_l1``, ``_seed_and_stamp``, ...), so a
scan broad enough to catch a reintroduced twin would be red on the healthy tree
and one narrow enough to be green would be matching an arbitrary shape rather
than the property.
"""

from __future__ import annotations

from pathlib import Path

from _escalation_seed import seed_escalation

from escalation.queue import EscalationQueue


class TestSeedEscalation:
    """``seed_escalation`` — the one place a pending record is built and submitted."""

    def test_seeded_record_is_retrievable_through_the_queue(self, tmp_path: Path) -> None:
        """The returned record is findable by id and carries what was passed."""
        queue = EscalationQueue(tmp_path / 'esc')

        esc = seed_escalation(queue, level=1, task_id='task-retrievable')

        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.task_id == 'task-retrievable'
        assert reread.level == 1
        assert reread.status == 'pending'

    def test_defaults_are_blocking_scope_violation_and_a_level_labelled_summary(
        self, tmp_path: Path,
    ) -> None:
        """With no overrides the record is a blocking scope_violation whose summary
        names its level — the shape every folded consumer relied on."""
        queue = EscalationQueue(tmp_path / 'esc')

        esc = seed_escalation(queue, level=2, task_id='task-defaults')

        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.severity == 'blocking'
        assert reread.category == 'scope_violation'
        assert 'level=2' in reread.summary

    def test_explicit_summary_is_used_verbatim(self, tmp_path: Path) -> None:
        """An explicit summary reaches the record unmodified. This is the one
        parameter the three consumer wrappers exist to supply."""
        queue = EscalationQueue(tmp_path / 'esc')

        esc = seed_escalation(
            queue, level=0, task_id='task-summary', summary='a very specific label',
        )

        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.summary == 'a very specific label'

    def test_agent_role_defaults_to_implementer_and_is_overridable(
        self, tmp_path: Path,
    ) -> None:
        """Both cells, since ten capability-guard call sites override the role."""
        queue = EscalationQueue(tmp_path / 'esc')

        defaulted = seed_escalation(queue, level=1, task_id='task-role-default')
        overridden = seed_escalation(
            queue, level=1, task_id='task-role-override', agent_role='steward',
        )

        reread_defaulted = queue.get(defaulted.id)
        reread_overridden = queue.get(overridden.id)
        assert reread_defaulted is not None
        assert reread_overridden is not None
        assert reread_defaulted.agent_role == 'implementer'
        assert reread_overridden.agent_role == 'steward'

    def test_extra_keywords_reach_the_record(self, tmp_path: Path) -> None:
        """``**kw`` passes through, driven with the two overrides real call sites
        use: a non-default category and an L2 member list."""
        queue = EscalationQueue(tmp_path / 'esc')

        esc = seed_escalation(
            queue,
            level=2,
            task_id='task-extra-kw',
            category='design_concern',
            members=['esc-x-1'],
        )

        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.category == 'design_concern'
        assert reread.members == ['esc-x-1']

    def test_repeated_seeds_under_one_task_id_get_distinct_ids(self, tmp_path: Path) -> None:
        """Two seeds under one task_id are two distinct retrievable records — the
        ``queue.make_id`` counter behaviour both module-scoped-queue consumers
        depend on. Asserted WITHOUT pinning an id format, deliberately: see the
        module docstring and ``queue.py::make_id``."""
        queue = EscalationQueue(tmp_path / 'esc')

        first = seed_escalation(queue, level=1, task_id='task-twice')
        second = seed_escalation(queue, level=1, task_id='task-twice')

        assert first.id != second.id
        assert queue.get(first.id) is not None
        assert queue.get(second.id) is not None
