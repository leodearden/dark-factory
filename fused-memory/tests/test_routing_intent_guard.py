"""Unit tests for fused_memory.middleware.routing_intent_guard.

Step 1 (RED -> step-2 GREEN): detection matrix for routing_intent_finding.
Step 3 (RED -> step-4 GREEN): routing_intent_reject / routing_intent_warning
payload shapes and the routing_intent_enforced() env flag.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.routing_intent_guard import routing_intent_finding


class TestRoutingIntentFindingDetectionMatrix:
    """Declaration-only routing-intent lint matrix (task_kind='normal' only)."""

    def test_declarative_marker_in_description_finds_and_names_markers(self):
        """A normal task's DESCRIPTION declaring 'DO NOT IMPLEMENT ...
        escalate to a human instead of implementing' -> finding carries
        both matched marker labels."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_implement' in finding.markers
        assert 'escalate_instead_of_implementing' in finding.markers

    def test_markers_details_alignment_and_field_dedup_ordering(self):
        """Invariant lock: markers/details stay positionally aligned (same
        length/order), a marker repeated across fields is counted once (in
        first-matched order), and fields lists every distinct matched field
        (deduped) in first-matched order -- even when a field contributes no
        NEW marker.

        title + description both match 'do_not_implement' (description
        repeats it); details matches a second, distinct marker
        ('escalate_instead_of_implementing'). 'do_not_implement' must appear
        before 'escalate_instead_of_implementing' (title is scanned first),
        and 'fields' must list all three fields even though 'description'
        contributes no marker beyond the one title already recorded.
        """
        finding = routing_intent_finding(
            title='DO NOT IMPLEMENT this task.',
            description='DO NOT IMPLEMENT is repeated here for good measure.',
            details='Escalate to a human instead of implementing this work.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert finding.markers == (
            'do_not_implement',
            'escalate_instead_of_implementing',
        )
        assert len(finding.markers) == len(finding.details)
        assert finding.fields == ('title', 'description', 'details')

    def test_mined_task_2332_shape_do_not_author_plan(self):
        """The mined task-2332 shape ('do not attempt to author a TDD plan
        for this task') -> finding."""
        finding = routing_intent_finding(
            title=None,
            description='Do not attempt to author a TDD plan for this task.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_author_plan' in finding.markers

    def test_marker_in_title_only_finds_independently(self):
        """A marker present ONLY in title (description/details clean) ->
        finding, and the finding records 'title' as a matched field."""
        finding = routing_intent_finding(
            title='DO NOT IMPLEMENT this task',
            description='',
            details='',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_implement' in finding.markers
        assert 'title' in finding.fields

    def test_marker_in_details_only_finds_independently(self):
        """A marker present ONLY in details (title/description clean) ->
        finding, and the finding records 'details' as a matched field."""
        finding = routing_intent_finding(
            title='',
            description='',
            details='Escalate to a human instead of implementing this work.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'escalate_instead_of_implementing' in finding.markers
        assert 'details' in finding.fields

    def test_task_2408_passing_mention_is_not_a_false_positive(self):
        """A genuine code task whose prose merely MENTIONS 'deterministic
        pure-gate' in passing (task-2408 shape) -> None, not a finding."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'This normal code task refactors the deterministic '
                'pure-gate helper; implement the fix in X.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_code_change_signal_suppresses_a_marker_match_across_fields(self):
        """A code-change signal ('Fix') in title suppresses a marker match
        found in description -- proves _CODE_CHANGE_SIGNALS is wired in as
        a blanket, cross-field guard (mirrors operational_ask_registry's
        title-signal-suppresses-any-match precedent), not dead code."""
        finding = routing_intent_finding(
            title='Fix the ingestion pipeline',
            description='DO NOT IMPLEMENT until design review completes.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_deterministic_task_kind_is_never_linted(self):
        """task_kind='deterministic' carrying the same markers -> None; only
        task_kind='normal' submissions are linted."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='deterministic',
            metadata=None,
        )
        assert finding is None

    def test_execution_class_operational_is_exempt(self):
        """metadata.execution_class='operational' + markers -> None (an
        honest non-code declaration is not a mismatch to flag)."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'operational'},
        )
        assert finding is None

    def test_execution_class_decision_is_exempt(self):
        """metadata.execution_class='decision' + markers -> None."""
        finding = routing_intent_finding(
            title=None,
            description='Do not attempt to author a TDD plan for this task.',
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'decision'},
        )
        assert finding is None

    def test_execution_class_code_tdd_is_not_exempt(self):
        """metadata.execution_class='code_tdd' + markers -> STILL a finding
        (code_tdd is the normal-code-task class; it does not exempt a
        conflicting routing declaration)."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'code_tdd'},
        )
        assert finding is not None

    def test_clean_normal_task_is_none(self):
        """A clean normal task with no markers anywhere -> None."""
        finding = routing_intent_finding(
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
            ('DO NOT IMPLEMENT this task.', 'do_not_implement'),
            (
                'Do not attempt to author a TDD plan for this task.',
                'do_not_author_plan',
            ),
            ('This is a no-code task.', 'no_code_label'),
            ('This is deterministic; no worktree required.', 'no_worktree'),
            (
                'Escalate to a human instead of implementing.',
                'escalate_instead_of_implementing',
            ),
            (
                'Run --apply against the live store once reviewed.',
                'apply_live_store',
            ),
            ('DO NOT COMPLETE until sign-off.', 'do_not_complete'),
        ],
    )
    def test_each_marker_fires_independently_in_description(
        self, field_text, expected_marker
    ):
        """Each of the seven declarative markers independently produces a
        finding naming itself when present in an otherwise-clean description."""
        finding = routing_intent_finding(
            title=None,
            description=field_text,
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert expected_marker in finding.markers


# ---------------------------------------------------------------------------
# Step-3: reject/warning payload shapes + the routing_intent_enforced() flag
#
# routing_intent_reject, routing_intent_warning, and routing_intent_enforced
# do not exist until step-4 -- each test below imports locally (inside the
# test body) rather than at module level, so only these new tests fail at
# RED while TestRoutingIntentFindingDetectionMatrix above keeps collecting
# and passing.
# ---------------------------------------------------------------------------


class TestRoutingIntentPayloadsAndFlag:
    """Payload shapes for the warn/reject outcomes, plus the enforce flag."""

    def _finding(self):
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        return finding

    def test_routing_intent_reject_returns_validation_error_naming_markers(self):
        """routing_intent_reject(finding) -> a structured ValidationError
        dict naming the matched marker(s), with a hint key."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_reject

        finding = self._finding()
        result = routing_intent_reject(finding)
        assert result.get('error_type') == 'ValidationError'
        assert 'hint' in result
        for marker in finding.markers:
            assert marker in result['error']

    def test_routing_intent_warning_is_non_blocking_structured_payload(self, caplog):
        """routing_intent_warning(finding) -> {'routing_intent_warning': {...}}
        with markers + hint, and NO top-level 'error'/'error_type' (it must
        be non-blocking). Also emits a greppable 'routing_intent_lint.flagged'
        census WARNING -- the observe-rate-before-flipping-enforcement
        precedent."""
        import logging

        from fused_memory.middleware.routing_intent_guard import routing_intent_warning

        finding = self._finding()
        with caplog.at_level(logging.WARNING):
            result = routing_intent_warning(finding)

        assert 'error' not in result
        assert 'error_type' not in result
        assert 'routing_intent_warning' in result
        payload = result['routing_intent_warning']
        assert list(payload['markers']) == list(finding.markers)
        assert 'hint' in payload
        assert any(
            'routing_intent_lint.flagged' in rec.message for rec in caplog.records
        )

    def test_routing_intent_enforced_defaults_false_when_unset(self, monkeypatch):
        """routing_intent_enforced() -> False when FUSED_ROUTING_INTENT_ENFORCE
        is unset (the default, warn-only behavior)."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_enforced

        monkeypatch.delenv('FUSED_ROUTING_INTENT_ENFORCE', raising=False)
        assert routing_intent_enforced() is False

    def test_routing_intent_enforced_true_when_env_set_truthy(self, monkeypatch):
        """routing_intent_enforced() -> True when FUSED_ROUTING_INTENT_ENFORCE
        is set to a truthy value ('1')."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_enforced

        monkeypatch.setenv('FUSED_ROUTING_INTENT_ENFORCE', '1')
        assert routing_intent_enforced() is True

    @pytest.mark.parametrize('raw_value', ['true', 'yes', 'on', ' On ', 'TRUE'])
    def test_routing_intent_enforced_true_for_every_recognized_truthy_token(
        self, monkeypatch, raw_value
    ):
        """routing_intent_enforced() -> True for every recognized truthy
        token, locking the case-insensitive + whitespace-stripped
        normalization contract (not just '1')."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_enforced

        monkeypatch.setenv('FUSED_ROUTING_INTENT_ENFORCE', raw_value)
        assert routing_intent_enforced() is True

    @pytest.mark.parametrize('raw_value', ['0', 'no', 'false', ''])
    def test_routing_intent_enforced_false_for_explicit_falsey_values(
        self, monkeypatch, raw_value
    ):
        """routing_intent_enforced() -> False for explicit falsey/empty
        values that are SET (not just unset), locking the warn-only default
        contract for operators who explicitly opt out."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_enforced

        monkeypatch.setenv('FUSED_ROUTING_INTENT_ENFORCE', raw_value)
        assert routing_intent_enforced() is False


class TestProvenanceStampDoesNotDisarmCodeChangeSuppression:
    """Machine-injected provenance stamps must not arm the code-change
    suppression (task 4532).

    A Stage-2 ``task_knowledge_sync`` doc-drift annotation appended to a
    live task description ("[Stage 2 task-knowledge sync <date>] DOC-DRIFT
    FIX (finding ...)") contains the bare word "FIX", which matched
    ``_CODE_CHANGE_SIGNALS_RE`` and permanently disarmed this guard for
    that task -- the guard's own downstream annotation blinded it. An
    appended annotation is post-filing PROVENANCE, not evidence of the
    author's filing-era code intent, so it must not suppress a finding.
    """

    def test_stage2_doc_drift_stamp_no_longer_suppresses_marker_finding(self):
        """The verbatim reify-5117 shape: an authored routing declaration
        followed by a machine-injected Stage-2 doc-drift stamp whose only
        code-change signal is the bare word "FIX" inside the stamp ->
        finding is still produced. The stamp must not arm
        _CODE_CHANGE_SIGNALS_RE."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'This is a no-code milestone gate. DO NOT IMPLEMENT; escalate '
                'to a human instead of implementing when the gate trips.'
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
        assert 'do_not_implement' in finding.markers
        assert 'no_code_label' in finding.markers
        assert 'escalate_instead_of_implementing' in finding.markers
        assert finding.fields == ('description',)

    def test_stamp_in_details_is_stripped_per_field(self):
        """The strip is applied to EVERY field, not just description: the
        marker lives in the description while the disarming stamp is
        appended to details."""
        finding = routing_intent_finding(
            title='Milestone gate for the dependency census',
            description='Do not attempt to author a TDD plan for this task.',
            details=(
                '[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX '
                '(finding abc): re-derived the dependency count.'
            ),
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_author_plan' in finding.markers

    def test_authored_signal_after_the_stamp_paragraph_still_suppresses(self):
        """CONTAINMENT: the stamp strip stops at the blank line ending the
        stamp's own paragraph, so an AUTHORED "fix" in a LATER paragraph
        still suppresses (task-2408 precision preserved)."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT.'
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
        finding = routing_intent_finding(
            title=None,
            description=(
                'Fix the ingestion pipeline.'
                '\n\n[Stage 2 task-knowledge sync 2026-07-07] note.'
                '\n\nDO NOT IMPLEMENT.'
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
        finding = routing_intent_finding(
            title=None,
            description=(
                '[Stage 1 stall detector]'
                '(fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py)'
                ' needs a fix; DO NOT IMPLEMENT until reviewed.'
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
        finding = routing_intent_finding(
            title=None,
            description=(
                'Spanning 2026-04-09 to 2026-08-06 [re-verified 2026-08-06]. '
                'DO NOT IMPLEMENT before the fix lands.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'An inline dated bracket is not a stamp, got: {finding!r}'

    def test_marker_inside_a_stamp_still_produces_a_finding(self):
        """MONOTONICITY lock: the marker scan reads RAW field text, so a
        marker that lives INSIDE a stamp still fires. The strip is
        asymmetric by design -- it never removes a finding that fires
        today."""
        finding = routing_intent_finding(
            title='Dependency census for the milestone gate',
            description='Re-derived the dependency count from the task graph.',
            details=(
                '[Stage 2 task-knowledge sync 2026-07-07] DOC-DRIFT FIX: the '
                'task text says "do not implement" but is filed normal.'
            ),
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_implement' in finding.markers
        assert 'details' in finding.fields

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
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT; escalate to a human instead of implementing.'
                '\n\n' + stamp_block
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None, f'Stamp must not suppress, got None for: {stamp_block!r}'
        assert 'do_not_implement' in finding.markers

    def test_line_anchored_bracket_without_date_or_stage_is_not_a_stamp(self):
        """Keeps the widening honest: an ordinary bracketed lead-in naming
        neither a stage nor a date is AUTHORED prose, so its "crash" still
        suppresses."""
        finding = routing_intent_finding(
            title=None,
            description=(
                '[design note] the crash reproduces under load.'
                '\n\nDO NOT IMPLEMENT yet.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None, f'A bare bracketed lead-in is not a stamp, got: {finding!r}'
