"""Tests for ScopeViolationEscalator — writes scope_violation escalations on path-guard rejection."""

from __future__ import annotations

import json
import re

import pytest

from fused_memory.middleware import scope_violation_escalator as sve_mod
from fused_memory.middleware.scope_violation_escalator import ScopeViolationEscalator


@pytest.mark.skipif(
    not sve_mod.HAS_ESCALATION,
    reason='escalation package not installed in this environment',
)
class TestEscalationEnabled:
    def test_writes_escalation_under_project_root(self, tmp_path):
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='Edit fused-memory/X',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
            suggested_root='/home/leo/src/dark-factory',
        )
        assert esc_id is not None
        # Escalation lands under {project_root}/data/escalations/{id}.json.
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        assert len(files) == 1, f'expected one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        assert payload['category'] == 'scope_violation'
        assert payload['severity'] == 'info'
        assert payload['agent_role'] == 'fused-memory/path-guard'
        assert 'fused-memory/' in payload['summary']
        assert 'dark_factory' in payload['summary']
        # detail carries the routing context for the operator.
        assert 'reify' in payload['detail']
        assert 'fused-memory/' in payload['detail']
        assert 'dark_factory' in payload['detail']
        assert payload['suggested_action'] == 'resubmit_to_dark_factory'

    def test_caches_queue_per_project_root(self, tmp_path):
        """Repeated rejections for the same project_root reuse one EscalationQueue."""
        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='one',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='two',
            matched_paths=('orchestrator/',),
            suggested_project='dark_factory',
        )
        queues = esc._queues  # private but we explicitly assert caching here
        assert list(queues.keys()) == [str(tmp_path)]
        # Both escalations should be on disk with distinct IDs.
        files = sorted((tmp_path / 'data' / 'escalations').glob('*.json'))
        assert len(files) == 2

    def test_no_suggested_project_uses_manual_route_action(self, tmp_path):
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='ambiguous task',
            matched_paths=('fused-memory/', 'crates_other/'),
            suggested_project=None,
        )
        assert esc_id is not None
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        payload = json.loads(files[0].read_text())
        assert payload['suggested_action'] == 'manual_route'

    def test_queue_failure_returns_none_does_not_raise(self, tmp_path, monkeypatch):
        """A queue submit failure must be swallowed — escalation is additive."""
        esc = ScopeViolationEscalator()

        # Force the underlying queue.submit to raise.
        from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

        def boom(self, _esc):
            raise RuntimeError('disk full')

        monkeypatch.setattr(EscalationQueue, 'submit', boom)
        result = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='will fail',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        assert result is None  # no exception propagates


    def test_llm_reason_appended_to_detail(self, tmp_path):
        """report_rejection with llm_reason= includes the reason in the escalation detail."""
        esc = ScopeViolationEscalator()
        llm_reason = 'LLM: genuine misroute — task edits orchestrator/harness.py'
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='dark_factory',
            candidate_title='Fix orchestrator bug',
            matched_paths=('orchestrator/',),
            suggested_project='orchestrator',
            llm_reason=llm_reason,
        )
        assert esc_id is not None
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        assert len(files) == 1
        import json
        payload = json.loads(files[0].read_text())
        assert 'llm_adjudicator_reason' in payload['detail']
        assert llm_reason in payload['detail']

    def test_no_llm_reason_keeps_detail_clean(self, tmp_path):
        """Back-compat: without llm_reason the detail does NOT contain llm_adjudicator_reason."""
        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='dark_factory',
            candidate_title='Fix orchestrator bug',
            matched_paths=('orchestrator/',),
            suggested_project='orchestrator',
        )
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        import json
        payload = json.loads(files[0].read_text())
        assert 'llm_adjudicator_reason' not in payload['detail']

    def test_report_rejection_folds_repeated_identical_misroutes(self, tmp_path):
        """Recurring identical misroutes fold into ONE pending parent.

        Reproduces the esc-task-path-guard-8/-9 incident shape (task 2946):
        a daily reconciliation consolidation round re-proposes the same
        "Human gate: ..." candidate citing a foreign path (corpus/ owned by
        know_live) under project reify.  Without dedup, each identical
        re-proposal files a fresh escalation — this test asserts they
        instead fold into the first.
        """
        esc = ScopeViolationEscalator()
        project_root = str(tmp_path)
        project_id = 'reify'
        candidate_title = 'Human gate: consolidate tree-sitter cluster'
        matched_paths = ('corpus/',)
        suggested_project = 'know_live'

        first = esc.report_rejection(
            project_root=project_root,
            project_id=project_id,
            candidate_title=candidate_title,
            matched_paths=matched_paths,
            suggested_project=suggested_project,
        )
        second = esc.report_rejection(
            project_root=project_root,
            project_id=project_id,
            candidate_title=candidate_title,
            matched_paths=matched_paths,
            suggested_project=suggested_project,
        )
        third = esc.report_rejection(
            project_root=project_root,
            project_id=project_id,
            candidate_title=candidate_title,
            matched_paths=matched_paths,
            suggested_project=suggested_project,
        )

        assert first is not None
        assert second == first, 'second identical misroute must fold into the first'
        assert third == first, 'third identical misroute must fold into the first'

        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'expected exactly one surviving escalation file, found: {files}'

        payload = json.loads(files[0].read_text())
        assert payload['id'] == first
        assert payload['dedupe_count'] == 2
        assert len(payload['dedupe_children']) == 2

    def test_report_rejection_dedup_kill_switch_reproduces_legacy_behavior(self, tmp_path):
        """scope_violation_dedupe_enabled=False disables folding (escape hatch).

        Mirrors budget_misconfig_dedup_window_secs as an operator-facing
        constructor knob: with dedup disabled, identical repeated misroutes
        must each file their own distinct escalation, exactly like the
        pre-task-2946 behavior.
        """
        esc = ScopeViolationEscalator(scope_violation_dedupe_enabled=False)
        project_root = str(tmp_path)
        project_id = 'reify'
        candidate_title = 'Human gate: consolidate tree-sitter cluster'
        matched_paths = ('corpus/',)
        suggested_project = 'know_live'

        first = esc.report_rejection(
            project_root=project_root,
            project_id=project_id,
            candidate_title=candidate_title,
            matched_paths=matched_paths,
            suggested_project=suggested_project,
        )
        second = esc.report_rejection(
            project_root=project_root,
            project_id=project_id,
            candidate_title=candidate_title,
            matched_paths=matched_paths,
            suggested_project=suggested_project,
        )

        assert first is not None
        assert second is not None
        assert second != first, 'dedup disabled: each call must file a distinct escalation'

        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 2, f'expected two distinct escalation files, found: {files}'


@pytest.mark.skipif(
    not sve_mod.HAS_ESCALATION,
    reason='escalation package not installed in this environment',
)
class TestAdvisoryVsRejectionWording:
    """Task 3119: ``report_rejection`` serves BOTH path-guard outcomes, so its
    wording must say which one actually happened.

    The path guard has two outcomes since task 2206: a FILES-certain hard
    reject (no task created, error dict returned) and a PROSE-only advisory
    (task CREATED and stamped with ``metadata.possible_scope_mismatch``,
    nothing blocked).  Both funnel through ``report_rejection``, which used to
    hardcode rejection wording — so every advisory told the operator (and the
    agent reading it in a briefing) that a task had been rejected when it had
    not.  These tests pin the PAIR: advisory wording says CREATED, rejection
    wording is unchanged.
    """

    @staticmethod
    def _payloads(root):
        """Return the escalation payloads written under *root* (sorted by id)."""
        files = sorted((root / 'data' / 'escalations').glob('esc-*.json'))
        return [json.loads(f.read_text()) for f in files]

    def _one_payload(self, root):
        payloads = self._payloads(root)
        assert len(payloads) == 1, f'expected exactly one escalation, found: {payloads}'
        return payloads[0]

    def test_advisory_record_never_claims_a_rejection(self, tmp_path):
        """The advisory record must not label a created task as rejected.

        Pins the STABLE contracts only, so a harmless rewording of the same
        correct message doesn't break the suite: the ``ADVISORY`` marker, the
        absence of 'reject' anywhere in the summary, the ``suggested_action``
        value (rendered verbatim into agent briefings by
        ``orchestrator/agents/briefing.py``, where ``resubmit_to_<project>``
        would tell an agent to redo work that already landed), and the
        ``possible_scope_mismatch`` stamp name an operator greps for.
        """
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='Rework the gui panel',
            matched_paths=('gui/',),
            suggested_project='reify_gui',
            advisory=True,
        )
        assert esc_id is not None
        payload = self._one_payload(tmp_path)
        summary = payload['summary']
        assert 'ADVISORY' in summary, summary
        # THE mislabel this task exists to kill: an advisory that says the
        # submission was rejected, when in fact the task was created.
        assert 'reject' not in summary.lower(), (
            f'advisory summary must not claim a rejection: {summary!r}'
        )
        assert payload['suggested_action'] == 'no_action_advisory_only', payload

        detail = payload['detail']
        # Named verbatim so the operator can grep the created task's metadata.
        assert 'possible_scope_mismatch' in detail, detail
        assert 'was rejected' not in detail, (
            f'advisory detail must not claim a rejection: {detail!r}'
        )
        # The structured routing context survives the advisory branch, under
        # outcome-NEUTRAL labels: a 'rejecting_project_id=' line would
        # re-introduce the mislabel in the very field an operator (or a briefed
        # agent) reads right next to the corrected suggested_action.
        for field in (
            'candidate_title=',
            'filing_project_id=',
            'filing_project_root=',
            'matched_paths=',
            'suggested_project=',
        ):
            assert field in detail, f'advisory detail dropped {field!r}: {detail!r}'
        assert 'rejecting_project' not in detail, detail

    def test_advisory_detail_does_not_assert_a_task_was_created(self, tmp_path):
        """Task 4159: the advisory detail must not claim a task exists.

        WHY the old wording was unverified: this record is filed from
        ``_path_guard_or_skip``, which runs inside ``submit_task`` PHASE-1 —
        the phase that persists a ticket and returns its id, explicitly
        *before* ``tm.add_task`` is ever called.  The curator resolves that
        ticket asynchronously and only ONE of its five actions (``create``)
        yields a new task; ``drop``, ``combine`` and ``refuse`` yield none.
        So at the moment this escalation is written, "the task WAS created"
        is a claim the escalator cannot possibly have verified, and on a
        drop/combine it is simply false — the operator (and any agent handed
        this text by ``briefing.py``) is told to go review a task that does
        not exist.

        Pins the STABLE contracts, per this class's convention: the absence
        of the two specific false clauses, the presence of the curator's
        no-task outcome vocabulary, and — structurally — that every sentence
        naming the stamp is CONDITIONAL rather than asserted.
        """
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='Rework the gui panel',
            matched_paths=('gui/',),
            suggested_project='reify_gui',
            advisory=True,
        )
        assert esc_id is not None
        detail = self._one_payload(tmp_path)['detail']

        # (a) The two clauses the bug shipped, named exactly.  Deliberately
        # not a blanket 'created' ban: the corrected prose must still be free
        # to describe the curator's create OUTCOME, conditionally.
        assert 'the task WAS created' not in detail, (
            f'advisory detail asserts an unverified creation: {detail!r}'
        )
        assert 'nothing was lost' not in detail, (
            f'advisory detail claims an outcome it cannot know: {detail!r}'
        )

        # (b) The deferred disposition is actually stated — the reader is told
        # the submission may yet be dropped or folded into an existing task,
        # not just that it was created.
        assert 'possible_scope_mismatch' in detail, detail
        lowered = detail.lower()
        assert 'drop' in lowered, (
            f"advisory detail must name the curator's drop outcome: {detail!r}"
        )
        assert 'combine' in lowered or 'fold' in lowered, (
            f"advisory detail must name the curator's combine outcome: {detail!r}"
        )

        # (c) EVERY sentence naming the stamp is conditional.  A single
        # unconditional "the task carries metadata.possible_scope_mismatch"
        # would re-introduce the same false claim in a new phrasing — and
        # would be false twice over, since a combine target never receives
        # the candidate's stamp at all.
        sentences = [s for s in re.split(r'(?<=\.)\s+', detail) if s.strip()]
        stamp_sentences = [s for s in sentences if 'possible_scope_mismatch' in s]
        assert stamp_sentences, (
            f'expected at least one sentence naming the stamp: {detail!r}'
        )
        for sentence in stamp_sentences:
            assert any(m in sentence.lower() for m in ('if ', 'when ', 'only ')), (
                'every sentence naming possible_scope_mismatch must be '
                f'conditional on a task actually being created: {sentence!r}'
            )

        # Invariants the rest of the suite already relies on, re-asserted here
        # so one careless rewrite is caught in one place.
        assert 'was rejected' not in detail, detail
        for field in (
            'candidate_title=',
            'filing_project_id=',
            'filing_project_root=',
            'matched_paths=',
            'suggested_project=',
        ):
            assert field in detail, f'advisory detail dropped {field!r}: {detail!r}'

    def test_advisory_without_suggested_project_is_still_advisory_only(self, tmp_path):
        """advisory + no resolvable owner must not fall through to manual_route.

        Reachable in production: a prose scan can hit several projects, leaving
        the guard with no single suggested owner.  If the advisory check were
        ordered after the ``manual_route`` fallback, this shape would hand a
        briefed agent a routing directive for a submission that was never
        blocked — the same class of false instruction as ``resubmit_to_*``.
        """
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='ambiguous prose hit',
            matched_paths=('fused-memory/', 'crates_other/'),
            suggested_project=None,
            advisory=True,
        )
        assert esc_id is not None
        payload = self._one_payload(tmp_path)
        assert payload['suggested_action'] == 'no_action_advisory_only', payload
        # The unknown-owner placeholder still renders, still without 'reject'.
        assert '<unknown' in payload['summary'], payload
        assert 'reject' not in payload['summary'].lower(), payload

    def test_rejection_wording_and_action_unchanged(self, tmp_path):
        """The other half of the pair: the FILES-certain path is untouched.

        Regression anchor — passes before the change and must keep passing
        after it.  The rejection wording is CORRECT today (a task really was
        rejected), so this change must not touch it.
        """
        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='Edit fused-memory/X',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        payload = self._one_payload(tmp_path)
        assert payload['summary'].startswith('Misrouted task rejected: cites '), payload
        assert 'was rejected' in payload['detail'], payload
        assert payload['suggested_action'] == 'resubmit_to_dark_factory', payload
        # The structured context labels are outcome-neutral on BOTH modes — the
        # rejecting/filing project is the same project either way.
        assert 'filing_project_id=' in payload['detail'], payload
        assert 'rejecting_project' not in payload['detail'], payload

    def test_both_modes_are_severity_info(self, tmp_path):
        """Both modes stay severity='info' — there is no tier below it.

        ``escalation.models`` defines the severity vocabulary as
        ``blocking | info | critical | urgent``, so 'info' is already the floor
        and the advisory cannot drop lower.  The wording (above) is the
        load-bearing correction, not the severity.
        """
        # Separate roots so the two modes cannot interact through dedup.
        rejection_root = tmp_path / 'rejection'
        advisory_root = tmp_path / 'advisory'
        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(rejection_root),
            project_id='reify',
            candidate_title='Edit fused-memory/X',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        esc.report_rejection(
            project_root=str(advisory_root),
            project_id='reify',
            candidate_title='Edit fused-memory/X',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
            advisory=True,
        )
        for root in (rejection_root, advisory_root):
            payload = self._one_payload(root)
            assert payload['severity'] == 'info', (root, payload)
            assert payload['category'] == 'scope_violation', (root, payload)
            assert payload['agent_role'] == 'fused-memory/path-guard', (root, payload)

    def test_advisory_and_rejection_do_not_cross_fold(self, tmp_path):
        """The mode must be part of the dedup fingerprint.

        report_rejection folds on a content fingerprint over the misroute
        SHAPE (project_id + sorted matched_paths + suggested_project) with an
        UNBOUNDED window.  Branching only the strings would leave the two modes
        sharing a fingerprint, so an advisory and a FILES-certain rejection over
        the same paths fold into ONE parent — whichever arrives first sets the
        wording and the second event is then reported with the other's outcome.
        That is this same mislabel bug in the opposite direction.
        """
        esc = ScopeViolationEscalator()
        shape = {
            'project_root': str(tmp_path),
            'project_id': 'reify',
            'candidate_title': 'Human gate: consolidate tree-sitter cluster',
            'matched_paths': ('corpus/',),
            'suggested_project': 'know_live',
        }

        rejection_id = esc.report_rejection(**shape)
        advisory_id = esc.report_rejection(**shape, advisory=True)

        assert rejection_id is not None
        assert advisory_id is not None
        assert advisory_id != rejection_id, (
            'an advisory must not fold into a rejection parent (or vice versa) — '
            'the surviving wording would mislabel the other outcome'
        )

        payloads = self._payloads(tmp_path)
        assert len(payloads) == 2, f'expected two escalations, found: {payloads}'
        # Assert on the PAIR, not just the count: a count-only assertion would
        # also pass if both records carried rejection wording.
        summaries = sorted(p['summary'] for p in payloads)
        assert any(s.startswith('Misrouted task rejected: cites ') for s in summaries), (
            summaries
        )
        assert any('ADVISORY' in s and 'CREATED' in s for s in summaries), summaries

    def test_two_identical_advisories_still_fold(self, tmp_path):
        """Advisories keep the anti-flood property that motivated task 2946.

        Separating the modes must not separate advisories from each OTHER — a
        recurring prose hit (e.g. the same reconciliation candidate re-proposed
        every round) must still fold into one pending parent.
        """
        esc = ScopeViolationEscalator()
        shape = {
            'project_root': str(tmp_path),
            'project_id': 'reify',
            'candidate_title': 'Human gate: consolidate tree-sitter cluster',
            'matched_paths': ('corpus/',),
            'suggested_project': 'know_live',
            'advisory': True,
        }

        first = esc.report_rejection(**shape)
        second = esc.report_rejection(**shape)

        assert first is not None
        assert second == first, 'a repeated identical advisory must fold into the first'
        payload = self._one_payload(tmp_path)
        assert payload['id'] == first
        assert payload['dedupe_count'] == 1
        assert len(payload['dedupe_children']) == 1

    def test_rejection_fingerprint_is_byte_identical_to_legacy(self, tmp_path):
        """Load-bearing back-compat pin: the rejection digest must not change.

        Any live PENDING rejection parent already on disk folds only if the new
        code computes the SAME fingerprint.  Adding a mode token to BOTH
        branches would change the rejection digest, orphan those parents, and
        silently re-flood the operator queue that task 2946 quieted — so the
        discriminator must be advisory-only.  Recomputed here from the
        pre-change composition, independently of the production code path.
        """
        from escalation.dedupe import (  # type: ignore[import-untyped]
            compute_content_fingerprint,
        )

        project_id = 'reify'
        matched_paths = ('corpus/',)
        suggested_project = 'know_live'
        expected = compute_content_fingerprint(
            'scope_violation',
            'path_guard_misroute',
            affected_ids=sorted([
                *matched_paths,
                f'suggested:{suggested_project}',
                f'project:{project_id}',
            ]),
        )

        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id=project_id,
            candidate_title='Human gate: consolidate tree-sitter cluster',
            matched_paths=matched_paths,
            suggested_project=suggested_project,
        )
        payload = self._one_payload(tmp_path)
        assert payload['dedupe_fingerprint'] == expected, (
            'the non-advisory fingerprint composition must stay byte-identical'
        )


@pytest.mark.skipif(
    not sve_mod.HAS_ESCALATION,
    reason='escalation package not installed in this environment',
)
class TestBudgetMisconfigEscalation:
    """ScopeViolationEscalator.report_budget_misconfig() — loud config-defect signal."""

    def test_writes_distinct_escalation_file(self, tmp_path):
        """report_budget_misconfig writes one file with category='adjudicator_config_defect'."""
        esc = ScopeViolationEscalator()
        esc_id = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert esc_id is not None
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        assert len(files) == 1, f'expected one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        # DISTINCT category — not scope_violation.
        assert payload['category'] == 'adjudicator_config_defect'
        assert payload['severity'] == 'blocking'
        assert payload['agent_role'] == 'fused-memory/path-scope-adjudicator'
        # Summary and detail name the budget misconfiguration.
        assert 'budget' in payload['summary'].lower() or 'misconfig' in payload['summary'].lower()
        assert str(0.11) in payload['detail'] or '0.11' in payload['detail']
        assert str(0.30) in payload['detail'] or '0.30' in payload['detail'] or '0.3' in payload['detail']
        assert 'sonnet' in payload['detail']
        # Fix hint present.
        assert 'max_budget_usd' in payload['detail']

    def test_dedup_second_call_within_window_writes_no_file(self, tmp_path):
        """Two consecutive calls for same project_id write exactly ONE file (burst-suppress)."""
        esc = ScopeViolationEscalator(budget_misconfig_dedup_window_secs=9999.0)
        first = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert first is not None
        second = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.12,
            turns=3,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert second is None
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        assert len(files) == 1, 'second call within dedup window must not write a new file'

    def test_queue_submit_failure_returns_none_does_not_raise(self, tmp_path, monkeypatch):
        """A queue.submit failure must be swallowed — escalation is additive."""
        esc = ScopeViolationEscalator()
        from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

        def boom(self, _esc):
            raise RuntimeError('disk full')

        monkeypatch.setattr(EscalationQueue, 'submit', boom)
        result = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert result is None  # no exception propagates

    def test_escalation_detail_includes_turns_and_fix_hint(self, tmp_path):
        """Detail must carry turns= and a concrete fix hint."""
        esc = ScopeViolationEscalator()
        esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.05,
            turns=4,
            max_budget_usd=0.10,
            model='claude-3-5-haiku-20241022',
        )
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        payload = json.loads(files[0].read_text())
        assert 'turns' in payload['detail']
        # Fix hint names the config key.
        assert 'max_budget_usd' in payload['detail']

    def test_dedup_window_expired_second_call_files_new_escalation(self, tmp_path):
        """Once the dedup window has elapsed a second call MUST file a second file.

        A regression that made dedup permanent (e.g. comparing against a sentinel
        that never ages out) would pass a suppression-only test but break the
        'one reminder per window' contract.  Using window=0.0 means the timestamp
        is always stale by definition, so the second call always re-files.
        """
        esc = ScopeViolationEscalator(budget_misconfig_dedup_window_secs=0.0)
        first = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert first is not None
        second = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.12,
            turns=3,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert second is not None, (
            'second call after dedup window expired must file a new escalation'
        )
        assert second != first, 'second escalation must have a distinct id'
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        assert len(files) == 2, (
            f'expected two escalation files after window expired, found: {files}'
        )


class TestEscalationDisabled:
    def test_no_op_when_escalation_pkg_unavailable(self, tmp_path, monkeypatch):
        """When HAS_ESCALATION is False the escalator silently no-ops."""
        monkeypatch.setattr(sve_mod, 'HAS_ESCALATION', False)
        esc = ScopeViolationEscalator()
        result = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='whatever',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        assert result is None
        # No file is written.
        queue_dir = tmp_path / 'data' / 'escalations'
        assert not queue_dir.exists() or not list(queue_dir.glob('*.json'))
