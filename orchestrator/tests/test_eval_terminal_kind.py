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

from copy import deepcopy

import pytest


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
        from orchestrator.evals.metrics import _DECLINE_READERS, DECLINE_KINDS

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


def _plan(steps: int = 3, finalized_at: str | None = None) -> dict:
    """A plan artifact in the shape ``artifacts.read_plan`` returns."""
    plan: dict = {
        'task_id': '4760',
        'title': 'a task',
        'analysis': 'some analysis',
        'files': ['a.py'],
        'steps': [
            {'id': f'step-{i}', 'type': 'impl', 'description': 'do a thing'}
            for i in range(1, steps + 1)
        ],
    }
    if finalized_at is not None:
        plan['_finalized_at'] = finalized_at
    return plan


def _create_plan_stub() -> dict:
    """The header-only stub ``plan_tools._create_plan`` writes on the FIRST
    plan-tools call: a TRUTHY dict with zero steps.
    """
    return {
        'task_id': '4760', 'title': 'a task', 'analysis': 'a', 'files': [],
        'steps': [],
    }


def _decline(reported_at: str | None = '2026-08-26T12:00:00+00:00') -> dict:
    """A decline artifact in the shape every ``artifacts.write_*`` persists."""
    artifact: dict = {'evidence': 'because', 'reason': 'because'}
    if reported_at is not None:
        artifact['reported_at'] = reported_at
    return artifact


class TestResolveTerminalKind:
    """The PURE resolver: plain dicts in, one TERMINAL_KINDS member out."""

    # ── the plan side, no declines ────────────────────────────────────────
    @pytest.mark.parametrize('plan', [
        pytest.param(None, id='absent'),
        pytest.param({}, id='empty-dict'),
        pytest.param({'steps': None}, id='steps-none'),
        pytest.param({'steps': []}, id='steps-empty'),
        pytest.param(_create_plan_stub(), id='create_plan-header-only-stub'),
    ])
    def test_no_scorable_plan_and_no_decline_is_none(self, plan):
        """The plan side consults judge.is_scorable_plan, NOT raw truthiness.

        The header-only ``create_plan`` stub is a TRUTHY dict, so a truthiness
        test would score it ``'planned'`` — the exact trap that made ``tainted``
        disagree with ``score_plan_structure`` before task 3302.
        """
        from orchestrator.evals.metrics import resolve_terminal_kind

        assert resolve_terminal_kind(plan, {}) == 'none'

    def test_plan_with_steps_and_no_decline_is_planned(self):
        from orchestrator.evals.metrics import resolve_terminal_kind

        assert resolve_terminal_kind(_plan(steps=6), {}) == 'planned'

    def test_all_none_declines_map_reads_as_no_decline(self):
        """``read_decline_artifacts`` always returns the COMPLETE key set, so
        the resolver must treat an all-``None`` map as "no decline present".
        """
        from orchestrator.evals.metrics import DECLINE_KINDS, resolve_terminal_kind

        empty = dict.fromkeys(DECLINE_KINDS)
        assert resolve_terminal_kind(_plan(), empty) == 'planned'
        assert resolve_terminal_kind(None, empty) == 'none'

    # ── the decline side ──────────────────────────────────────────────────
    @pytest.mark.parametrize('kind', [
        'already_done', 'blocking_dependency', 'false_premise',
        'unactionable', 'ready_to_merge',
    ])
    def test_no_plan_plus_one_decline_resolves_to_that_kind(self, kind):
        from orchestrator.evals.metrics import resolve_terminal_kind

        assert resolve_terminal_kind({}, {kind: _decline()}) == kind

    # ── the ORDERING case (scope item 3): last terminal call wins ─────────
    def test_plan_then_later_decline_resolves_to_the_decline(self):
        """The reify_task_4026 / run e522b1b0 shape: a 6-step plan confirmed at
        10:00, then ``report_already_done`` at 11:00. The model's own FINAL
        judgement was that the task was moot, so scoring it a planning success
        would report the opposite of what it concluded.
        """
        from orchestrator.evals.metrics import resolve_terminal_kind

        kind = resolve_terminal_kind(
            _plan(steps=6, finalized_at='2026-08-26T10:00:00+00:00'),
            {'already_done': _decline('2026-08-26T11:00:00+00:00')},
        )
        assert kind == 'already_done'

    def test_decline_then_later_plan_resolves_to_planned(self):
        """The mirror: the architect reported, then went on to plan anyway."""
        from orchestrator.evals.metrics import resolve_terminal_kind

        kind = resolve_terminal_kind(
            _plan(steps=6, finalized_at='2026-08-26T11:00:00+00:00'),
            {'already_done': _decline('2026-08-26T10:00:00+00:00')},
        )
        assert kind == 'planned'

    def test_unfinalized_plan_loses_to_any_decline(self):
        """An unconfirmed plan is not a TERMINAL statement — the workflow's own
        ``_plan`` requires the ``_finalized_at`` marker before it will advance
        one, so there is nothing for the decline to lose to.
        """
        from orchestrator.evals.metrics import resolve_terminal_kind

        assert resolve_terminal_kind(
            _plan(steps=6),  # no _finalized_at
            {'false_premise': _decline('2026-08-26T09:00:00+00:00')},
        ) == 'false_premise'

    def test_two_declines_resolve_to_the_later_reported_at(self):
        from orchestrator.evals.metrics import resolve_terminal_kind

        declines = {
            'already_done': _decline('2026-08-26T10:00:00+00:00'),
            'ready_to_merge': _decline('2026-08-26T11:00:00+00:00'),
        }
        assert resolve_terminal_kind({}, declines) == 'ready_to_merge'
        # ...and independently of dict insertion order.
        assert resolve_terminal_kind({}, dict(reversed(list(
            declines.items())))) == 'ready_to_merge'

    @pytest.mark.parametrize('stamp', [
        pytest.param('2026-08-26T10:00:00+00:00', id='equal'),
        pytest.param(None, id='absent'),
        pytest.param('not-a-timestamp', id='unparseable'),
    ])
    def test_untied_declines_fall_back_to_fixed_precedence(self, stamp):
        """Ties and unparseable stamps break by the fixed DECLINE_KINDS order,
        so the same input always resolves the same way.
        """
        from orchestrator.evals.metrics import DECLINE_KINDS, resolve_terminal_kind

        declines = {
            'ready_to_merge': _decline(stamp),
            'blocking_dependency': _decline(stamp),
        }
        # blocking_dependency precedes ready_to_merge in DECLINE_KINDS.
        assert DECLINE_KINDS.index('blocking_dependency') < DECLINE_KINDS.index(
            'ready_to_merge')
        first = resolve_terminal_kind({}, declines)
        assert first == 'blocking_dependency'
        # Determinism: repeat calls, and reversed insertion order, agree.
        assert resolve_terminal_kind({}, declines) == first
        assert resolve_terminal_kind({}, dict(reversed(list(
            declines.items())))) == first

    @pytest.mark.parametrize('stamp', [
        pytest.param(None, id='missing'),
        pytest.param('', id='empty'),
        pytest.param('whenever', id='unparseable'),
    ])
    def test_unstamped_decline_beats_a_finalized_plan(self, stamp):
        """Ambiguity resolves toward the DECLINE: the artifact exists only
        because the architect made an explicit, server-accepted terminal tool
        call the prompt tells it to make INSTEAD of planning.
        """
        from orchestrator.evals.metrics import resolve_terminal_kind

        assert resolve_terminal_kind(
            _plan(steps=6, finalized_at='2026-08-26T10:00:00+00:00'),
            {'unactionable': _decline(stamp)},
        ) == 'unactionable'

    def test_unparseable_plan_finalized_at_loses_to_a_stamped_decline(self):
        from orchestrator.evals.metrics import resolve_terminal_kind

        assert resolve_terminal_kind(
            _plan(steps=6, finalized_at='sometime'),
            {'already_done': _decline('2026-08-26T10:00:00+00:00')},
        ) == 'already_done'

    # ── invariants ────────────────────────────────────────────────────────
    @pytest.mark.parametrize(('plan', 'declines'), [
        (None, {}),
        ({}, {}),
        (_create_plan_stub(), {}),
        (_plan(), {}),
        ({}, {'false_premise': _decline()}),
        (_plan(finalized_at='2026-08-26T10:00:00+00:00'),
         {'already_done': _decline('2026-08-26T11:00:00+00:00')}),
        (_plan(), {'unactionable': _decline(None)}),
        ({}, {k: _decline() for k in (
            'already_done', 'blocking_dependency', 'false_premise',
            'unactionable', 'ready_to_merge')}),
    ])
    def test_return_value_is_always_a_terminal_kind(self, plan, declines):
        from orchestrator.evals.metrics import TERMINAL_KINDS, resolve_terminal_kind

        assert resolve_terminal_kind(plan, declines) in TERMINAL_KINDS

    def test_resolver_mutates_neither_argument(self):
        from orchestrator.evals.metrics import resolve_terminal_kind

        plan = _plan(steps=6, finalized_at='2026-08-26T10:00:00+00:00')
        declines = {
            'already_done': _decline('2026-08-26T11:00:00+00:00'),
            'false_premise': None,
        }
        plan_before = deepcopy(plan)
        declines_before = deepcopy(declines)

        resolve_terminal_kind(plan, declines)

        assert plan == plan_before
        assert declines == declines_before
