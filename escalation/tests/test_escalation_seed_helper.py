"""Contract tests for ``_escalation_seed.seed_escalation``.

Every direct-seeding call site in the consumer modules reaches
``queue.submit()`` through this helper, so a regression here would change their
premises while they stayed green. Every assertion reads the record back through
``queue.get(...)``, the escalation package's own read path, never through queue
internals or a mock.
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
        parameter the consumer wrappers exist to supply."""
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
        """Both cells, since several capability-guard call sites override the role."""
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
        """Two seeds under one task_id are two distinct retrievable records. No id
        format is pinned: ``escalation/src/escalation/queue.py::make_id`` owns it."""
        queue = EscalationQueue(tmp_path / 'esc')

        first = seed_escalation(queue, level=1, task_id='task-twice')
        second = seed_escalation(queue, level=1, task_id='task-twice')

        assert first.id != second.id
        assert queue.get(first.id) is not None
        assert queue.get(second.id) is not None
