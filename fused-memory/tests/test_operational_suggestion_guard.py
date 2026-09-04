"""Unit tests for fused_memory.middleware.operational_suggestion_guard.

Step 5 (RED -> step-6 GREEN): detection matrix for operational_suggestion_finding
plus the operational_suggestion_warning payload shape.

Modeled on test_routing_intent_guard.py, but this guard has no enforce/reject
path (WARN-ONLY, task mandate) — operational_suggestion_finding is imported at
module top; operational_suggestion_warning is imported locally in its own
tests since it is exercised alongside the finding fixture below.
"""

from __future__ import annotations

import logging

import pytest

from fused_memory.middleware.operational_suggestion_guard import (
    operational_suggestion_finding,
)


class TestOperationalSuggestionFindingDetectionMatrix:
    """Implicit operational-phrasing lint matrix (task_kind='normal' only)."""

    def test_operational_phrasing_in_description_with_no_files_finds_and_names_markers(
        self,
    ):
        """A normal task's DESCRIPTION declaring restart+confirm phrasing with
        no metadata.files -> finding carries both matched marker labels."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers
        assert 'confirm' in finding.markers

    def test_non_normal_task_kind_is_none(self):
        """task_kind='deterministic' carrying the same phrasing -> None; only
        task_kind='normal' submissions are suggested."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='deterministic',
            metadata=None,
        )
        assert finding is None

    def test_execution_class_operational_is_exempt(self):
        """metadata.execution_class='operational' + markers -> None (an
        honest non-code declaration is not a mismatch to flag)."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'operational'},
        )
        assert finding is None

    def test_execution_class_decision_is_exempt(self):
        """metadata.execution_class='decision' + markers -> None."""
        finding = operational_suggestion_finding(
            title=None,
            description='Redeploy the orchestrator fleet and confirm health.',
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'decision'},
        )
        assert finding is None

    def test_metadata_files_present_is_exempt(self):
        """metadata.files is non-empty (a code deliverable) -> None even
        though the description carries operational phrasing."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata={'files': ['fused-memory/src/fused_memory/server/tools.py']},
        )
        assert finding is None

    @pytest.mark.parametrize('signal_word', ['fix', 'bug', 'crash', 'implement'])
    def test_code_change_signal_suppresses_finding(self, signal_word):
        """A code-change signal (fix/bug/crash/implement) anywhere in the
        combined text suppresses the finding -- a genuine code task that
        merely mentions 'restart' is not nudged."""
        finding = operational_suggestion_finding(
            title=f'{signal_word} the ingestion pipeline',
            description='Restart the service once the change lands.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_clean_normal_task_is_none(self):
        """A clean normal task with no operational phrasing anywhere -> None."""
        finding = operational_suggestion_finding(
            title='Add a retry helper to the sync client',
            description='Wrap the sync client call in a bounded retry loop.',
            details='Use the existing backoff helper in shared.retry.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    @pytest.mark.parametrize(
        ('field_text', 'expected_marker'),
        [
            ('Restart the fused-memory service.', 'restart'),
            ('Redeploy the orchestrator fleet.', 'redeploy'),
            ('Reload the configuration file.', 'reload'),
            ('Confirm the service is healthy.', 'confirm'),
            ('Deploy the latest build to production.', 'deploy'),
            ('Run systemctl status to verify.', 'systemctl'),
        ],
    )
    def test_each_marker_fires_independently_in_description(
        self, field_text, expected_marker
    ):
        """Each of the six operational markers independently produces a
        finding naming itself when present in an otherwise-clean description
        with no metadata.files."""
        finding = operational_suggestion_finding(
            title=None,
            description=field_text,
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert expected_marker in finding.markers

    def test_marker_in_details_only_finds_independently(self):
        """A marker present ONLY in details (title/description clean) ->
        finding, and the finding records 'details' as a matched field."""
        finding = operational_suggestion_finding(
            title='',
            description='',
            details='Restart the service once the escalation resolves.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers
        assert 'details' in finding.fields

    @pytest.mark.parametrize(
        'description',
        [
            'Confirm the retry logic handles timeouts.',
            'Reload the config module after edit.',
            'Deploy config schema field.',
        ],
    )
    def test_weak_marker_suppressed_by_code_artifact_in_same_field(self, description):
        """The generic ('weak') markers -- confirm/reload/deploy -- are
        suppressed when the SAME field also names a code-level artifact
        (logic/module/schema/field/...) rather than a running system: these
        are genuine code-task sentences, not operational asks (task 2679
        amendment pass, reviewer_comprehensive robustness finding)."""
        finding = operational_suggestion_finding(
            title=None,
            description=description,
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_strong_marker_still_fires_alongside_code_artifact_noun(self):
        """Unlike the weak markers, 'restart' (a specific/strong marker) is
        NOT gated by a co-occurring code-artifact noun in the same field --
        it still fires standalone."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the service; the auth module needs a bounce.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers

    def test_cross_field_markers_aggregate_distinct_labels_and_fields(self):
        """A marker in TITLE and a different marker in DETAILS -> both
        marker labels appear (deduped) and both field names appear in
        `fields` (deduped), in first-matched order."""
        finding = operational_suggestion_finding(
            title='Restart the ingestion worker',
            description='',
            details='Confirm the worker is healthy afterwards.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert finding.markers == ('restart', 'confirm')
        assert finding.fields == ('title', 'details')

    def test_same_marker_in_two_fields_dedupes_to_one_label(self):
        """The same marker present in both TITLE and DESCRIPTION dedupes to
        a single entry in `markers`, but BOTH field names appear in
        `fields`."""
        finding = operational_suggestion_finding(
            title='Restart the ingestion worker',
            description='Restart the ingestion worker again to be sure.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert finding.markers == ('restart',)
        assert finding.fields == ('title', 'description')

    def test_code_change_signal_in_details_only_suppresses_finding(self):
        """A code-change signal located ONLY in `details` (not title/
        description) still suppresses the finding -- the code-change check
        scans the COMBINED title+description+details text, not just
        title."""
        finding = operational_suggestion_finding(
            title='Ops task',
            description='Restart the fused-memory service.',
            details='This closes out the bug found during rollout.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is None


# ---------------------------------------------------------------------------
# Payload shape for the warn outcome (WARN-ONLY -- no reject/enforce path)
#
# operational_suggestion_warning does not exist until step-6 -- imported
# locally (inside the test body) rather than at module level, so only these
# tests fail at RED while TestOperationalSuggestionFindingDetectionMatrix
# above keeps collecting (its own import fails too until step-6, since the
# whole module does not exist yet -- both classes are RED simultaneously
# pre-step-6, GREEN together after).
# ---------------------------------------------------------------------------


class TestOperationalSuggestionWarningPayload:
    """Payload shape for the non-blocking warn outcome."""

    def _finding(self):
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        return finding

    def test_operational_suggestion_warning_is_non_blocking_structured_payload(
        self, caplog
    ):
        """operational_suggestion_warning(finding) ->
        {'operational_suggestion_warning': {...}} with markers/fields/detail/
        hint, and NO top-level 'error'/'error_type' (it must be
        non-blocking). Also emits a greppable
        'operational_task_suggestion.flagged' census WARNING."""
        from fused_memory.middleware.operational_suggestion_guard import (
            operational_suggestion_warning,
        )

        finding = self._finding()
        with caplog.at_level(logging.WARNING):
            result = operational_suggestion_warning(finding)

        assert 'error' not in result
        assert 'error_type' not in result
        assert 'operational_suggestion_warning' in result
        payload = result['operational_suggestion_warning']
        assert list(payload['markers']) == list(finding.markers)
        assert list(payload['fields']) == list(finding.fields)
        assert 'detail' in payload
        assert 'hint' in payload
        assert any(
            'operational_task_suggestion.flagged' in rec.message
            for rec in caplog.records
        )

    def test_operational_suggestion_warning_hint_names_deterministic_or_execution_class(
        self,
    ):
        """The hint must suggest either task_kind='deterministic' or
        metadata.execution_class as the fix -- non-blocking guidance, never
        a coercion."""
        from fused_memory.middleware.operational_suggestion_guard import (
            operational_suggestion_warning,
        )

        finding = self._finding()
        result = operational_suggestion_warning(finding)
        hint = result['operational_suggestion_warning']['hint']
        assert 'deterministic' in hint
        assert 'execution_class' in hint


class TestProvenanceStampDoesNotDisarmOperationalSuggestion:
    """Machine-injected provenance stamps must not arm the code-change
    suppression (task 4569, ported from task 4532).

    The backstory, observed stamp corpus, monotonicity invariant and
    directional-safety rule live in ONE place — routing_intent_guard.py's
    "Provenance-stamp carve-out" module-docstring section. Each test below
    names only the specific behaviour it pins.
    """

    def test_stage2_doc_drift_stamp_no_longer_suppresses_marker_finding(self):
        """The verbatim reify-5117 shape: authored operational phrasing
        followed by a machine-injected Stage-2 doc-drift stamp whose only
        code-change signal is the bare word "FIX" inside the stamp ->
        finding is still produced. The stamp must not arm
        _CODE_CHANGE_SIGNALS_RE."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the fused-memory service and confirm it is back up.'
                '\n\n[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding 4e06f01a-cacb-4688-9670-ff6d6ce41baf): the '
                '`dependencies` array carries 32 entries, but this prose '
                'previously itemized only 31.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers
        assert 'confirm' in finding.markers
        assert finding.fields == ('description',)

    def test_authored_signal_after_the_stamp_paragraph_still_suppresses(self):
        """CONTAINMENT: the stamp strip stops at the blank line ending the
        stamp's own paragraph, so an AUTHORED "fix" in a LATER paragraph
        still suppresses."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the service.'
                '\n\n[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding x): re-derived the count.'
                '\n\nAlso fix the retry helper while you are in here.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'Authored trailing "fix" must still suppress, got: {finding!r}'

    def test_authored_signal_before_the_stamp_still_suppresses(self):
        """CONTAINMENT (leading side): an authored code-change signal in a
        paragraph BEFORE the stamp is untouched by the strip and still
        suppresses."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Fix the ingestion pipeline.'
                '\n\n[Stage 2 task-knowledge sync 2026-07-07] note.'
                '\n\nRestart the service.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'Authored leading "Fix" must still suppress, got: {finding!r}'

    def test_markdown_link_at_line_start_is_not_a_provenance_stamp(self):
        """PRECISION: DF's own task prose routinely opens a line with a
        markdown link naming a recon stage. Treating it as a stamp would
        strip a whole AUTHORED paragraph and manufacture a finding."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                '[Stage 1 stall detector]'
                '(fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py)'
                ' needs a fix; restart the worker after.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'A markdown link is not a stamp, got: {finding!r}'

    def test_inline_dated_bracket_is_not_a_provenance_stamp(self):
        """PRECISION: a mid-sentence dated bracket is authored prose, not an
        appended annotation block, so it must not swallow the rest of the
        sentence."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Spanning 2026-04-09 to 2026-08-06 [re-verified 2026-08-06]. '
                'Restart the service after the fix lands.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'An inline dated bracket is not a stamp, got: {finding!r}'

    def test_line_anchored_bracket_without_date_or_stage_is_not_a_stamp(self):
        """Keeps the widening honest: an ordinary bracketed lead-in naming
        neither a stage nor a date is AUTHORED prose, so its "crash" still
        suppresses."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                '[design note] the crash reproduces under load.'
                '\n\nRestart the worker.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'A bare bracketed lead-in is not a stamp, got: {finding!r}'

    @pytest.mark.parametrize(
        'stamp_block',
        [
            '[RECON CORRECTION 2026-08-08] the prior prose was a bug; corrected here.',
            '[Block resolved 2026-06-02 by reconciliation stage 2]: build failure was '
            'a disk-exhaustion crash, not a code bug.',
            '[Scope correction by escalation-watcher-auto via esc-3871-178, 2026-05-31] '
            'PATH FIX: the declared file set was wrong.',
            '[Stage 2 task-knowledge sync] DOC-DRIFT FIX (finding y): re-derived the count.',
            '[Stage 2 sync 2026-07-07] DOC-DRIFT FIX: line one\n'
            'continues here with a bug reference\nand a third line.',
        ],
    )
    def test_sibling_machine_stamp_shapes_are_also_stripped(self, stamp_block):
        """The Stage-2 doc-drift stamp is one member of a FAMILY of dated,
        line-anchored annotation blocks written by DF's own agents. Shapes
        sampled from live task rows in the reify and dark-factory
        `.taskmaster/tasks/tasks.db` corpora: [RECON CORRECTION <date>],
        [Block resolved <date> by reconciliation stage N], [Scope correction
        by escalation-watcher-auto via esc-N-M, <date>], an UNDATED Stage-N
        variant (the agent does not always stamp a date), and a multi-line
        stamp body. Each carries a code-change signal that is the ANNOTATOR's
        wording, not the filing author's."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the fused-memory service and confirm it is back up.'
                '\n\n' + stamp_block
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None, f'Stamp must not suppress, got None for: {stamp_block!r}'
        assert 'restart' in finding.markers

    def test_crlf_stamp_is_still_stripped(self):
        """The CRLF bound must not be bought by losing recognition: a
        CRLF-separated stamp is still stripped, so the reported defect stays
        fixed under both line-ending conventions."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the fused-memory service and confirm it is back up.'
                '\r\n\r\n[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding x): re-derived the count.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None, 'CRLF-separated stamp must still be stripped'
        assert 'restart' in finding.markers

    def test_crlf_authored_signal_after_the_stamp_paragraph_still_suppresses(self):
        """CONTAINMENT, CRLF variant of the LF case above: the paragraph
        terminator must recognize a CRLF blank line too. A bound written
        only against an LF blank line ran straight past it to the END OF THE
        FIELD, stripping the authored trailing "fix" and manufacturing a
        finding — the over-strip direction 4532's review caught. CRLF
        reaches task text via paste from a Windows or browser client."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the service.'
                '\r\n\r\n[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding x): re-derived the count.'
                '\r\nAnd a second stamp line.'
                '\r\n\r\nAlso fix the retry helper while you are in here.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'CRLF trailing "fix" must still suppress, got: {finding!r}'

    def test_stamp_at_the_start_of_the_title_field_is_stripped(self):
        """The line anchor is a MULTILINE ``^``, which also matches at
        position 0 — so a stamp opening the TITLE (the field the sibling
        operational_ask_registry scopes its signals to) is recognized, not
        just one appearing after a newline."""
        finding = operational_suggestion_finding(
            title=(
                '[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX: '
                're-derived the dependency count.'
            ),
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers

    def test_every_stamp_in_a_field_is_stripped_not_just_the_first(self):
        """Two stamps in one field, each carrying its OWN code-change signal
        ("bug" in the first, "FIX" in the second): a finding proves BOTH were
        stripped, since either survivor alone would re-suppress. Live task
        descriptions accumulate annotations over a task's lifetime."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the fused-memory service.'
                '\n\n[RECON CORRECTION 2026-08-08] the prior prose was a bug.'
                '\n\n[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding z): re-derived the count.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers

    def test_stamp_in_details_is_stripped_per_field(self):
        """The strip is applied to EVERY field, and PER FIELD: the marker
        lives in the description while the disarming stamp is appended to
        details. Stripping the already-joined string instead would let a
        stamp's paragraph bound run across a field boundary."""
        finding = operational_suggestion_finding(
            title='Restart the ingestion worker',
            description='Restart the ingestion worker after the nightly window.',
            details=(
                '[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding abc): re-derived the dependency count.'
            ),
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers

    def test_marker_inside_a_stamp_still_produces_a_finding(self):
        """MONOTONICITY lock: the marker scan reads RAW field text, so a
        marker that lives INSIDE a stamp still fires. The strip is
        asymmetric by design -- it never removes a finding that fires
        today."""
        finding = operational_suggestion_finding(
            title=None,
            description='Re-derived the dependency count from the task graph.',
            details=(
                '[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX: '
                'restart the ingestion worker to pick it up.'
            ),
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers
        assert 'details' in finding.fields

    def test_list_item_stamp_is_not_stripped(self):
        """UNDER-STRIP BOUNDARY, pinned deliberately: the opener allows only
        leading spaces/tabs, so a stamp bulleted into a list ("- [Stage 2
        ...]") is NOT recognized and its signal still suppresses. This is the
        SAFE direction (it degrades to pre-4569 behaviour), and pinning it
        makes any future widening of the opener's leading-whitespace class a
        visible, deliberate choice rather than a silent behaviour change."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the service.'
                '\n\n- [Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX: '
                're-derived the count.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'A bulleted stamp is not recognized, got: {finding!r}'

    def test_line_continuing_a_stamp_is_stripped_with_it(self):
        """KNOWN COST of the paragraph bound, pinned so it stays visible:
        observed stamps have multi-line bodies, so a line separated from the
        stamp by only a SINGLE newline is stripped with it and its
        code-change signal is lost. Accepted because every observed annotator
        appends its stamp as its own blank-line-separated paragraph; a
        line-bounded strip would instead drop the multi-line stamp bodies
        this guard must handle."""
        finding = operational_suggestion_finding(
            title=None,
            description=(
                'Restart the worker.'
                '\n\n[RECON CORRECTION 2026-08-08] corrected.'
                '\nAlso fix the retry helper while here.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None, 'Single-newline continuation is stripped with the stamp'
        assert 'restart' in finding.markers
