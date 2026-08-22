"""Unit tests for :mod:`fused_memory.server.markup_tripwire` (tasks 3141, 4458).

Covers what that module still owns: the re-exported pattern list and its
same-file drift guard, the re-exported override lifecycle, and the best-effort
storm escalation emitter.

The matchers, the rejection-block builder and the rolling-window counter were
deleted in task 4458 together with the four in-line write gates they served —
the containment moved to the dispatch boundary
(:mod:`fused_memory.server.markup_guard`), and keeping a second working markup
mechanism alive behind its own tests is the duplicate-implementation invitation
INV-5 forbids. Their behaviour is now pinned where it is actually reachable:
``test_markup_tripwire_gate.py`` asserts, per tool, that an UNGUARDED server no
longer refuses a leaked write while a guarded one does.

Boundary/wiring tests for the MCP write tools live in that sibling file;
``tests/test_markup_guard_fused_memory.py`` covers the boundary guard itself,
including the residue emitter this module also hosts.
"""

from __future__ import annotations

import json

import pytest
from shared import toolcall_markup

from fused_memory.server import markup_tripwire
from fused_memory.server.markup_tripwire import (
    MARKUP_OVERRIDE_KEY,
    MCP_MARKUP_PATTERNS,
    emit_markup_storm_escalation,
    markup_override_requested,
    strip_markup_override,
)

_STORM = {
    'count': 4,
    'threshold': 3,
    'window_seconds': 3600.0,
    'hint': 'the leak is active; DF 3083 owns the root cause',
}

#: The shape the ONE live producer actually sends — ``project``/``outcome``,
#: singular, no ``projects``. ``shared.mcp_markup_middleware`` declares THREE
#: storm outcomes (:135-138) and fires ``_record_storm`` for each of them
#: (:497, :513, :520), so all three arrive at this one filer and it may not
#: hardcode any of them.
_REJECTED_STORM = {
    'count': 3,
    'threshold': 3,
    'window_seconds': 3600.0,
    'outcome': 'rejected',
    'project': '/project-a',
}
_REPAIRED_STORM = {**_REJECTED_STORM, 'outcome': 'repaired'}


# ---------------------------------------------------------------------------
# Tier-1 same-file drift guard.
#
# _CANONICAL_PATTERNS and MCP_MARKUP_PATTERNS must be updated TOGETHER.  The
# write-time pattern list is deliberately the SINGLE source of truth (INV-5);
# this guard makes an unannounced edit to it fail loudly here rather than
# silently widening/narrowing what the four MCP write boundaries reject.
#
# Pattern of tests/test_lock_charter_guard.py::_CANONICAL_EXTENSIONS.
# ---------------------------------------------------------------------------
_CANONICAL_PATTERNS = ('</content>', '<parameter name=', '</invoke>')


def test_pattern_list_drift_guard():
    """Tier-1 (same-file): MCP_MARKUP_PATTERNS must match _CANONICAL_PATTERNS.

    A same-file consistency check — update BOTH together.  The three literals
    are the envelope fragments observed leaking into the corpus (DF 3083
    vector-1 specimens are ``</content>``/``</invoke>`` tails; vector-2 is the
    ``<parameter name=`` fragment that mis-parsed task 3210's priority).
    """
    assert MCP_MARKUP_PATTERNS == _CANONICAL_PATTERNS, (
        f'Write-time pattern list drifted: {MCP_MARKUP_PATTERNS!r} != '
        f'{_CANONICAL_PATTERNS!r}. Update both together.'
    )


def test_pattern_list_is_an_immutable_tuple():
    """The pattern list is a tuple, so a caller cannot mutate the guard's rules."""
    assert isinstance(MCP_MARKUP_PATTERNS, tuple), (
        f'Expected a tuple, got {type(MCP_MARKUP_PATTERNS)}'
    )


class TestMarkupOverrideRequested:
    """The escape hatch, fail-closed.

    submit_task/update_task accept metadata as an object OR a JSON string, so
    both shapes must be understood. Anything the helper cannot confidently read
    as an explicit opt-in means NO override — an accidental harness leak must
    never be waved through by a coincidence in the metadata.
    """

    def test_true_in_a_dict_enables_the_override(self):
        assert markup_override_requested({MARKUP_OVERRIDE_KEY: True}) is True

    def test_true_in_a_json_string_enables_the_override(self):
        """The task tools accept metadata as a JSON string — same answer."""
        assert markup_override_requested(json.dumps({MARKUP_OVERRIDE_KEY: True})) is True

    def test_override_is_read_alongside_other_metadata_keys(self):
        metadata = {'source': 'agent-followup', MARKUP_OVERRIDE_KEY: True, 'files': ['a.py']}
        assert markup_override_requested(metadata) is True
        assert markup_override_requested(json.dumps(metadata)) is True

    def test_absent_key_does_not_enable_the_override(self):
        assert markup_override_requested({'source': 'x'}) is False
        assert markup_override_requested({}) is False

    def test_only_a_literal_boolean_true_counts(self):
        """Truthy-but-not-True values are REFUSED (fail-closed).

        Mirrors add_memory's `metadata.get('allow_near_duplicate') is True` check
        at tools.py:1199. A stray 'yes'/1 in metadata is far more likely to be
        unrelated data than a deliberate, considered opt-in to writing raw MCP
        envelope markup into the corpus.
        """
        for value in ('yes', 'true', 'True', 1, [1], {'a': 1}, 1.0):
            assert markup_override_requested({MARKUP_OVERRIDE_KEY: value}) is False, (
                f'Expected {value!r} NOT to enable the override'
            )

    def test_false_and_none_values_do_not_enable_the_override(self):
        assert markup_override_requested({MARKUP_OVERRIDE_KEY: False}) is False
        assert markup_override_requested({MARKUP_OVERRIDE_KEY: None}) is False
        assert markup_override_requested({MARKUP_OVERRIDE_KEY: 0}) is False

    def test_none_metadata_does_not_enable_the_override(self):
        assert markup_override_requested(None) is False

    def test_malformed_json_string_does_not_raise(self):
        """Metadata this helper cannot parse is not its to reject — it just says no.

        The write is still evaluated by the tripwire; whatever owns metadata
        validation raises its own error later.
        """
        assert markup_override_requested('{not valid json') is False
        assert markup_override_requested('') is False
        assert markup_override_requested('   ') is False

    def test_non_dict_json_payloads_do_not_enable_the_override(self):
        for payload in ('[1, 2, 3]', '"a string"', 'null', '42', 'true'):
            assert markup_override_requested(payload) is False, f'Expected False for {payload!r}'

    def test_unexpected_types_do_not_raise(self):
        for value in (123, 4.5, [], object(), True):
            assert markup_override_requested(value) is False, f'Expected False for {value!r}'


class TestStripMarkupOverride:
    """The flag is write-time-only: it must never be persisted.

    Returns the SAME SHAPE it was given (dict in -> dict out, JSON string in ->
    JSON string out) so a call site can substitute the result inline before
    forwarding to the service/interceptor.
    """

    def test_removes_the_key_from_a_dict(self):
        result = strip_markup_override({MARKUP_OVERRIDE_KEY: True, 'source': 'x'})
        assert result == {'source': 'x'}

    def test_does_not_mutate_the_input_dict(self):
        """Non-mutating: the caller's own metadata object is left intact.

        The handler may still need the original (e.g. to re-read the override),
        and silently mutating a caller-owned dict is the kind of action-at-a-
        distance this guard should not introduce.
        """
        original = {MARKUP_OVERRIDE_KEY: True, 'source': 'x'}
        strip_markup_override(original)
        assert original == {MARKUP_OVERRIDE_KEY: True, 'source': 'x'}

    def test_preserves_all_other_keys_and_values(self):
        metadata = {
            MARKUP_OVERRIDE_KEY: True,
            'source': 'agent-followup',
            'files': ['a.py', 'b.py'],
            'nested': {'k': 'v'},
        }
        result = strip_markup_override(metadata)
        assert result == {
            'source': 'agent-followup',
            'files': ['a.py', 'b.py'],
            'nested': {'k': 'v'},
        }

    def test_is_a_no_op_when_the_key_is_absent(self):
        assert strip_markup_override({'source': 'x'}) == {'source': 'x'}
        assert strip_markup_override({}) == {}

    def test_removes_the_key_from_a_json_string_and_returns_a_json_string(self):
        result = strip_markup_override(json.dumps({MARKUP_OVERRIDE_KEY: True, 'source': 'x'}))
        assert isinstance(result, str)
        assert json.loads(result) == {'source': 'x'}

    def test_json_string_without_the_key_round_trips_equivalently(self):
        result = strip_markup_override(json.dumps({'source': 'x'}))
        assert isinstance(result, str)
        assert json.loads(result) == {'source': 'x'}

    def test_returns_none_unchanged(self):
        assert strip_markup_override(None) is None

    def test_malformed_json_passes_through_byte_identical(self):
        """Passthrough, never raise: a write must not fail because this helper
        choked on metadata it does not own."""
        malformed = '{not valid json'
        assert strip_markup_override(malformed) == malformed

    def test_non_dict_json_payload_passes_through_unchanged(self):
        assert strip_markup_override('[1, 2, 3]') == '[1, 2, 3]'

    def test_unexpected_types_pass_through_unchanged(self):
        for value in (123, 4.5, []):
            assert strip_markup_override(value) == value


class TestEmitMarkupStormEscalation:
    """The best-effort queue write on a burst (INV-4).

    Purely ADDITIVE: the rejection has already been decided by the time this
    runs, so every failure mode here must degrade to None rather than change the
    write's outcome. Exercised against a real EscalationQueue in tmp_path (the
    escalation package is a fused-memory workspace dep) — mirrors
    tests/test_candidate_key_escalation.py.
    """

    def test_returns_none_for_a_none_project_root(self):
        """add_memory/add_episode may not resolve a project_root at all.

        Those boundaries take project_id, not project_root, and an unknown
        project maps to nothing — so this must be a quiet no-op, not a crash on
        Path(None).
        """
        assert emit_markup_storm_escalation(None, _STORM) is None

    def test_files_one_escalation_routing_at_the_live_owner_and_the_storm_numbers(
        self, tmp_path
    ):
        esc_id = emit_markup_storm_escalation(str(tmp_path), _STORM)
        if not markup_tripwire.HAS_ESCALATION:
            assert esc_id is None
            return

        assert isinstance(esc_id, str)
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'expected exactly one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        assert payload['category'] == 'mcp_markup_write_storm'
        assert payload['task_id'] == 'markup-tripwire'
        assert payload['level'] == 1

        # The operator must be able to act without opening the code: the record
        # names the owner of the root cause and the burst it observed. The
        # numbers are asserted as their EXACT labelled substrings — a bare
        # `'4' in text` would also be satisfied by a digit in the interpolated
        # tmp_path (`/tmp/pytest-of-leo/pytest-124/...`), i.e. it would pass with
        # the count dropped entirely.
        detail = payload['detail']
        routing_text = f'{payload["summary"]}\n{detail}'
        assert 'toolcall-markup-containment-prd.md' in routing_text, (
            f'must route the burst at the live successor PRD: {payload!r}'
        )
        assert '3083' in routing_text, (
            f'must still name DF 3083, as the closed predecessor: {payload!r}'
        )
        # The count's key is outcome-NEUTRAL and static (task 4505): this one
        # filer serves all three of the middleware's outcomes, and a key naming
        # one of them contradicts its own content on the other two. Static, and
        # not interpolated from the outcome, because greppability across records
        # is the one property an operator relies on. Anchored on both sides so
        # the old `rejected_writes_in_window=4` cannot satisfy it by substring.
        assert '\nwrites_in_window=4\n' in detail, f'must state the count: {detail!r}'
        assert 'outcome=' in detail, (
            f'the count is a count of ONE outcome; name it: {detail!r}'
        )
        assert 'window_seconds=3600.0' in detail, f'must state the window: {detail!r}'
        # A burst cannot span projects (task 4505): the live producer keys one
        # StormCounter per (project, outcome) pair, so the window holds one
        # project's events and the count needs no cross-project qualifier. The
        # `projects_in_window=` line this replaces was dead — no live producer
        # ever emitted the `projects` key it read, so it rendered `None` on
        # every record filed since 4458 landed.
        assert 'projects_in_window' not in detail, (
            f'the count cannot span projects, so it must not be hedged: {detail!r}'
        )

    # -- outcome fidelity (task 4505) -----------------------------------
    #
    # The middleware fires a storm for each of THREE outcomes — 'repaired',
    # 'rejected', 'unrepairable' (mcp_markup_middleware.py:135-138, recorded at
    # :497, :513, :520) — and every one reaches this single filer. A 'repaired'
    # burst is a burst of calls that all SUCCEEDED (the middleware says so at
    # :899), so a record describing them as rejections states a number the
    # triager cannot reproduce: grepping their own journal for rejections finds
    # zero. Naming an outcome the burst did not have is the same defect as
    # stating a count the burst did not have.

    @staticmethod
    def _filed(tmp_path, storm) -> dict:
        """File *storm* and read the single record back off disk."""
        esc_id = emit_markup_storm_escalation(str(tmp_path), storm)
        assert esc_id is not None
        files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1, f'expected exactly one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        return payload

    def test_a_repaired_burst_is_not_described_as_rejected(self, tmp_path):
        """The headline. Today this record claims writes were rejected when the
        burst it describes consists entirely of calls that SUCCEEDED.

        The word is asserted absent from BOTH fields, because the false claim is
        not confined to the count's label: the detail's prose says the tripwire
        "rejected multiple writes" and the summary says "MCP write(s) rejected",
        and each is read by a different consumer.
        """
        if not markup_tripwire.HAS_ESCALATION:
            assert emit_markup_storm_escalation(str(tmp_path), _REPAIRED_STORM) is None
            return

        payload = self._filed(tmp_path, _REPAIRED_STORM)

        assert 'repaired' in payload['summary'], (
            f"must name the burst's own outcome: {payload['summary']!r}"
        )
        both = f'{payload["summary"]}\n{payload["detail"]}'
        assert 'rejected' not in both, (
            f'nothing was rejected in a repaired burst — a triager grepping '
            f'their journal for rejections finds zero: {both!r}'
        )

    def test_a_rejected_burst_still_reads_as_rejected(self, tmp_path):
        """The other half, so (a) cannot be satisfied by dropping the outcome
        word altogether. Rejections are still the common case and must still be
        named as rejections.

        The count is asserted as an EXACT LABELLED substring: a bare `'3' in
        summary` is vacuous here, because 'window_seconds=3600.0' and the
        routing text both carry a 3.
        """
        if not markup_tripwire.HAS_ESCALATION:
            assert emit_markup_storm_escalation(str(tmp_path), _REJECTED_STORM) is None
            return

        payload = self._filed(tmp_path, _REJECTED_STORM)

        assert '3 MCP write(s) rejected' in payload['summary'], (
            f'a rejected burst must still read as rejected: {payload["summary"]!r}'
        )

    def test_the_summary_alone_answers_how_many_of_what_for_whom(self, tmp_path):
        """Asserted against the SUMMARY ONLY — never the detail — because that
        is all a compact consumer is given.

        escalation/src/escalation/server.py:409-420 projects `summary` into both
        `_COMPACT_ESCALATION_FIELDS` and `_COMPACT_PENDING_FIELDS` and
        explicitly DROPS `detail` as "the unbounded free-text field that
        motivated compact mode". So the L1 escalation watcher and
        get_pending_escalations(compact=True) see the summary and nothing else:
        a qualifier that lives only in the detail is structurally unreadable.
        That is the whole reason this task exists — esc-markup-tripwire-6 was
        triaged off a summary that answered 'how many' with a number pooled
        across two projects and named no scope at all.
        """
        if not markup_tripwire.HAS_ESCALATION:
            assert emit_markup_storm_escalation(str(tmp_path), _REJECTED_STORM) is None
            return

        summary = self._filed(tmp_path, _REJECTED_STORM)['summary']

        # how many, and of what — one labelled substring, so neither can be
        # dropped without failing here.
        assert '3 MCP write(s) rejected' in summary, (
            f'the summary must carry the count and the outcome: {summary!r}'
        )
        # for whom.
        assert 'in this project' in summary, (
            f"the summary must scope the count to this project's own "
            f'contribution: {summary!r}'
        )

    def test_a_storm_of_unknown_outcome_claims_no_outcome_it_did_not_measure(
        self, tmp_path
    ):
        """Degenerate and legacy shapes must keep working — and must not be
        relabelled 'rejected' on the way.

        Both cases are live: `{}` is what test_tolerates_a_storm_dict_missing_keys
        files, and `{'count': 9}` is the squatter that
        tests/test_markup_guard_fused_memory.py files to reproduce the anchor
        squat. Absent is not zero and absent is not 'rejected'; equally, the
        outcome slot must not render the bare string 'None', which is a
        confident claim about something never measured. Trading a false
        'rejected' for either is the same class of defect this task closes.
        """
        if not markup_tripwire.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        for storm in ({}, {'count': 9}):
            queue_root = tmp_path / f'root-{len(storm)}'
            queue_root.mkdir()

            payload = self._filed(queue_root, storm)

            # Still filed and still routable — this is when an operator needs it
            # most, so a missing number may not degrade the record's usefulness.
            assert payload['category'] == 'mcp_markup_write_storm'
            assert 'toolcall-markup-containment-prd.md' in payload['detail'], (
                f'must still route at the live successor PRD: {payload!r}'
            )
            summary = payload['summary']
            assert 'rejected' not in summary, (
                f'no outcome was measured, so none may be claimed: {summary!r}'
            )
            # Positional, not a bare `'None' not in summary`: an absent count
            # and window legitimately render None elsewhere in this sentence.
            assert 'write(s) None' not in summary, (
                f"the outcome slot must not read 'None': {summary!r}"
            )

    def test_escalation_id_is_greppable_via_the_stable_anchor(self, tmp_path):
        """The anchor is stable so ids form one greppable series per project."""
        esc_id = emit_markup_storm_escalation(str(tmp_path), _STORM)
        if not markup_tripwire.HAS_ESCALATION:
            return
        assert esc_id is not None
        assert 'markup-tripwire' in esc_id, f'unexpected id shape: {esc_id!r}'

    def test_dedupes_against_an_already_pending_escalation(self, tmp_path):
        """A sustained leak must not mint one escalation per window forever.

        The storm counter is already rate-limited per window, but a leak running
        for hours would still file an escalation every hour; the anchor dedup
        collapses those into the one open record until it is resolved.
        """
        first = emit_markup_storm_escalation(str(tmp_path), _STORM)
        second = emit_markup_storm_escalation(str(tmp_path), _STORM)
        if not markup_tripwire.HAS_ESCALATION:
            assert first is None and second is None
            return

        assert first is not None
        assert second == first, f'expected dedup; got first={first!r} second={second!r}'
        queue_dir = tmp_path / 'data' / 'escalations'
        assert len(list(queue_dir.glob('esc-*.json'))) == 1

    def test_files_afresh_once_the_prior_escalation_is_resolved(self, tmp_path):
        """Dedup must not silence a NEW incident after the old one was cleared."""
        if not markup_tripwire.HAS_ESCALATION:
            return
        from escalation.queue import EscalationQueue

        first = emit_markup_storm_escalation(str(tmp_path), _STORM)
        assert first is not None
        EscalationQueue(tmp_path / 'data' / 'escalations').resolve(first, 'leak fixed')

        second = emit_markup_storm_escalation(str(tmp_path), _STORM)
        assert second is not None
        assert second != first

    def test_a_submit_failure_is_swallowed(self, tmp_path, monkeypatch):
        """A broken queue must never turn a decided rejection into an exception."""
        if not markup_tripwire.HAS_ESCALATION:
            return

        def _boom(self, esc):
            raise OSError('disk on fire')

        monkeypatch.setattr(markup_tripwire.EscalationQueue, 'submit', _boom)
        assert emit_markup_storm_escalation(str(tmp_path), _STORM) is None

    def test_a_dedup_read_failure_still_files(self, tmp_path, monkeypatch):
        """If the dedup check itself fails, fall through and FILE.

        Best-effort dedup: losing a duplicate-suppression is strictly better than
        losing the alarm for an active leak.
        """
        if not markup_tripwire.HAS_ESCALATION:
            return

        def _boom(self, task_id, status=None):
            raise OSError('cannot read queue')

        monkeypatch.setattr(markup_tripwire.EscalationQueue, 'get_by_task', _boom)
        assert emit_markup_storm_escalation(str(tmp_path), _STORM) is not None

    def test_returns_none_when_the_escalation_package_is_unavailable(
        self, tmp_path, monkeypatch
    ):
        """The defensive-import no-op path (minimal envs without escalation)."""
        monkeypatch.setattr(markup_tripwire, 'HAS_ESCALATION', False)
        assert emit_markup_storm_escalation(str(tmp_path), _STORM) is None
        assert not (tmp_path / 'data' / 'escalations').exists()

    def test_tolerates_a_storm_dict_missing_keys(self, tmp_path):
        """A degenerate storm shape still files a well-formed, routable record.

        "Never raises" is already established by the call returning at all; what
        this pins is that the record stays USABLE — an escalation whose category
        or anchor degraded along with its missing numbers could not be routed or
        deduped, which is exactly when an operator needs it most.
        """
        if not markup_tripwire.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        esc_id = emit_markup_storm_escalation(str(tmp_path), {})

        assert esc_id is not None
        files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1, f'expected exactly one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        assert payload['category'] == 'mcp_markup_write_storm'
        assert payload['task_id'] == 'markup-tripwire'
        assert payload['detail'], f'a detail is still required: {payload!r}'
        assert 'toolcall-markup-containment-prd.md' in payload['detail'], (
            f'must still route at the live successor PRD: {payload!r}'
        )

    # -- the anchor parameter (task 4458) -------------------------------

    def test_the_default_anchor_is_unchanged(self, tmp_path):
        """Callers that do not pass an anchor must behave exactly as before.

        The in-line write-time gate still calls this with two positional
        arguments, so a default drift here would silently re-point its records.
        """
        esc_id = emit_markup_storm_escalation(str(tmp_path), _STORM)
        if not markup_tripwire.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')
        payload = json.loads(next((tmp_path / 'data' / 'escalations').glob('esc-*.json')).read_text())
        assert payload['task_id'] == 'markup-tripwire'
        assert 'markup-tripwire' in str(esc_id)

    def test_an_explicit_anchor_lands_in_both_the_task_id_and_the_id(self, tmp_path):
        if not markup_tripwire.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        esc_id = emit_markup_storm_escalation(
            str(tmp_path), _STORM, anchor_task_id='markup-guard'
        )

        assert esc_id is not None
        assert 'markup-guard' in esc_id, f'unexpected id shape: {esc_id!r}'
        payload = json.loads(next((tmp_path / 'data' / 'escalations').glob('esc-*.json')).read_text())
        assert payload['task_id'] == 'markup-guard'
        # Everything else is unchanged: this is a parameterisation of the ONE
        # filer, not a second one (INV-5).
        assert payload['category'] == 'mcp_markup_write_storm'
        assert payload['level'] == 1
        assert 'toolcall-markup-containment-prd.md' in payload['summary']

    def test_the_dedup_lookup_keys_on_the_SAME_anchor_that_is_filed(self, tmp_path):
        """The measured defect this parameter exists to fix.

        The L1 escalation watcher files its own cluster records under the
        'markup-tripwire' anchor and so SQUATS it — measured: the tripwire filed
        nothing 2026-08-16..2026-08-19 while 41 rejections occurred, and all 17
        records sat at dedupe_count 0. A filer that deduped against a squatted
        anchor is suppressed indefinitely, which is silence that reads as calm.

        So an explicit anchor must dedupe against ITSELF only: an open record on
        a DIFFERENT anchor must not suppress it.
        """
        if not markup_tripwire.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        squatter = emit_markup_storm_escalation(str(tmp_path), _STORM)
        guard = emit_markup_storm_escalation(
            str(tmp_path), _STORM, anchor_task_id='markup-guard'
        )

        assert squatter is not None
        assert guard is not None
        assert guard != squatter, 'an open record on another anchor must not suppress this one'
        assert len(list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))) == 2

    def test_two_bursts_on_one_explicit_anchor_still_dedupe(self, tmp_path):
        """The dedup itself must survive the parameterisation, or a leak running
        for hours files one record per window forever."""
        if not markup_tripwire.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        first = emit_markup_storm_escalation(
            str(tmp_path), _STORM, anchor_task_id='markup-guard'
        )
        second = emit_markup_storm_escalation(
            str(tmp_path), _STORM, anchor_task_id='markup-guard'
        )

        assert first is not None
        assert second == first, f'expected dedup; got first={first!r} second={second!r}'
        assert len(list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))) == 1


# ---------------------------------------------------------------------------
# INV-5: the envelope literals are enumerated ONCE, in shared.toolcall_markup.
#
# Kept at END OF FILE deliberately: an INV-5 block spliced into the middle of a
# test class silently adopts every method below the splice point, which is how
# the escalation cases above briefly became methods of TestSingleSourceOfTruth.
# Append new blocks here rather than inserting them.
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    """INV-5, enforced from the consumer side."""

    def test_mcp_markup_patterns_is_the_shared_object(self):
        """IDENTITY, not equality — a copied tuple would re-open the drift.

        ``==`` is satisfied by a tuple re-spelled here with the same value,
        which is precisely the state this task exists to end: this write-time
        list and ``toolcall_xml_leak``'s read-time list drifted apart while both
        were documented as the single source of truth. Object identity is the
        one assertion a duplicate cannot pass.
        """
        assert markup_tripwire.MCP_MARKUP_PATTERNS is toolcall_markup.MCP_MARKUP_PATTERNS

