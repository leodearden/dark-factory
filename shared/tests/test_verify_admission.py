"""Tests for shared.verify_admission — flock N-slot task-verify semaphore +
role nice tiers.

PRD ``plans/verify-oversubscription-control-prd.md`` task T1 (foundation
substrate; T2 wires this into ``orchestrator/verify.py``). See that PRD's
§Contract for the exact seam signatures and the C-* contract clauses
(C-merge-priority, C-untimed-acquire, C-fail-open, C-no-FD-inheritance)
referenced throughout this suite.
"""

from __future__ import annotations


class TestNicePrefix:
    def test_merge_role(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('merge') == ['nice', '-n', '5']

    def test_task_role(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('task') == ['nice', '-n', '15', 'ionice', '-c2', '-n7']

    def test_background_role(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('background') == ['nice', '-n', '19', 'ionice', '-c3']

    def test_offline_role_returns_empty(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('offline') == []

    def test_unknown_role_returns_empty(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('some-unknown-role') == []

    def test_empty_string_role_returns_empty(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('') == []

    def test_returns_fresh_list_not_shared_mutable_state(self):
        """Mutating a returned list must not leak into subsequent calls."""
        from shared.verify_admission import nice_prefix

        result = nice_prefix('task')
        result.append('MUTATED')

        result_again = nice_prefix('task')

        assert result_again == ['nice', '-n', '15', 'ionice', '-c2', '-n7']
        assert 'MUTATED' not in result_again


class TestModuleExports:
    def test_all_is_exactly_the_public_seam(self):
        import shared.verify_admission as va

        assert set(va.__all__) == {'acquire_task_slot', 'nice_prefix'}
