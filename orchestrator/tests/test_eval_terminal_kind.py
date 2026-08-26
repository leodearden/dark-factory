"""Tests for the architect's TERMINAL KIND — evals/metrics.py (task 4760).

The defect this suite exists for: an architect that took an explicit plan-tools
decline exit (``report_false_premise`` and its four siblings) persisted a cell
byte-indistinguishable from one that silently failed to plan — ``plan_steps=0``
and nothing else. The decline artifact was written, then ``rmtree``'d by
``snapshots.cleanup_eval_worktree`` before anything read it, so tranche-1's
"47 of 53 cells produced no plan" could only be split into causes by digging
through ``~/.claude/projects/*run-<id>/`` transcripts.

Covered here: the ``EvalMetrics.terminal_kind`` field and its frozen vocabulary,
the PURE ``resolve_terminal_kind`` resolver (including the plan-then-decline
ordering policy), and the best-effort ``read_decline_artifacts`` reader. The
RUNNER WIRING that puts them together lives in
``test_eval_architect.py::TestArchitectCellPersistsTerminalKind``, which needs
the hermetic harness this file deliberately does not.
"""

from __future__ import annotations


class TestEvalMetricsTerminalKindField:
    """The persisted field and the closed vocabulary that bounds it."""

    def test_default_is_none_the_not_measured_sentinel(self):
        from orchestrator.evals.metrics import EvalMetrics

        # None == "never measured": a non-architect cell, or one persisted
        # before the field existed. Matches plan_quality / role_under_test.
        assert EvalMetrics().terminal_kind is None

    def test_to_dict_carries_the_key_unconditionally(self):
        from orchestrator.evals.metrics import EvalMetrics

        # KEY ABSENCE is the ONLY legacy signal (metrics.py's
        # judged_without_reference note: "Do not make the write conditional"),
        # so a default-valued field must still emit its key.
        assert 'terminal_kind' in EvalMetrics().to_dict()
        assert EvalMetrics().to_dict()['terminal_kind'] is None
        assert EvalMetrics(terminal_kind='already_done').to_dict()[
            'terminal_kind'] == 'already_done'

    def test_terminal_kinds_is_the_frozen_vocabulary(self):
        from orchestrator.evals.metrics import TERMINAL_KINDS

        assert TERMINAL_KINDS == (
            'planned',
            'already_done',
            'blocking_dependency',
            'false_premise',
            'unactionable',
            'ready_to_merge',
            'none',
        )

    def test_decline_kinds_is_the_five_explicit_report_exits(self):
        from orchestrator.evals.metrics import DECLINE_KINDS, TERMINAL_KINDS

        assert DECLINE_KINDS == (
            'already_done',
            'blocking_dependency',
            'false_premise',
            'unactionable',
            'ready_to_merge',
        )
        assert set(DECLINE_KINDS) < set(TERMINAL_KINDS)
        assert set(TERMINAL_KINDS) - set(DECLINE_KINDS) == {'planned', 'none'}

    def test_every_decline_kind_has_a_real_task_artifacts_reader(self):
        """The vocabulary cannot drift away from the artifacts it names.

        A sixth architect exit added later fails HERE rather than silently
        scoring as ``'none'``.
        """
        from orchestrator.artifacts import TaskArtifacts
        from orchestrator.evals.metrics import DECLINE_KINDS, _DECLINE_READERS

        assert tuple(_DECLINE_READERS) == DECLINE_KINDS
        for kind in DECLINE_KINDS:
            reader = _DECLINE_READERS[kind]
            assert callable(getattr(TaskArtifacts, reader, None)), (
                f'{kind!r} names TaskArtifacts.{reader}, which does not exist'
            )

    def test_terminal_kind_of_reads_the_persisted_dict(self):
        from orchestrator.evals.metrics import terminal_kind_of

        assert terminal_kind_of({'terminal_kind': 'false_premise'}) == 'false_premise'
        # Absence and an explicit None both mean UNMEASURED.
        assert terminal_kind_of({}) is None
        assert terminal_kind_of({'terminal_kind': None}) is None
