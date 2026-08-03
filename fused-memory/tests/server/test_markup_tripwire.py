"""Unit tests for the MCP-markup write tripwire (task 3141).

Covers the pure matchers, the structured block builder, the rolling-window
storm counter (with an injected clock — no sleeping) and the best-effort
escalation emitter in
:mod:`fused_memory.server.markup_tripwire`.

Boundary/wiring tests for the four MCP write tools live in the sibling
``test_markup_tripwire_gate.py``.
"""

from __future__ import annotations

import json

import pytest

from fused_memory.server import markup_tripwire
from fused_memory.server.markup_tripwire import (
    MARKUP_OVERRIDE_KEY,
    MCP_MARKUP_PATTERNS,
    MarkupStormCounter,
    build_markup_block,
    emit_markup_storm_escalation,
    find_markup_pattern,
    find_markup_violation,
    markup_override_requested,
    strip_markup_override,
)

_STORM = {
    'count': 4,
    'threshold': 3,
    'window_seconds': 3600.0,
    'projects': ['/project-a', '/project-b'],
    'hint': 'the leak is active; DF 3083 owns the root cause',
}


class _FakeClock:
    """A mutable, list-free monotonic stand-in for ``time.time``.

    Injected via MarkupStormCounter(time_provider=...) so window behaviour is
    asserted by ADVANCING the clock rather than by sleeping — a wall-clock test
    of a 3600s window would either be unrunnable or flaky.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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


class TestFindMarkupPattern:
    """The single-text matcher: first matching literal by position, else None."""

    def test_matches_each_canonical_pattern(self):
        """Each of the three literals is detected and returned verbatim.

        The matched literal is echoed back (rather than a bare bool) so the
        rejection can name WHICH pattern tripped — INV-1.
        """
        for pattern in _CANONICAL_PATTERNS:
            text = f'some prose then {pattern} and a tail'
            assert find_markup_pattern(text) == pattern, (
                f'Expected {pattern!r} to be detected in {text!r}'
            )

    def test_matches_a_bare_pattern_with_no_surrounding_prose(self):
        for pattern in _CANONICAL_PATTERNS:
            assert find_markup_pattern(pattern) == pattern

    def test_returns_first_match_by_position_not_list_order(self):
        """With several patterns present, the EARLIEST in the text wins.

        ``</invoke>`` is last in MCP_MARKUP_PATTERNS but first in this text, so
        a naive "iterate the list, return the first that appears" impl would
        report ``</content>`` instead and mislead the caller about where the
        leak starts.
        """
        text = 'head </invoke> middle </content> tail'
        assert find_markup_pattern(text) == '</invoke>'

        # And the mirror case, so the assertion above cannot pass by accident.
        text_reversed = 'head </content> middle </invoke> tail'
        assert find_markup_pattern(text_reversed) == '</content>'

    def test_returns_first_match_by_position_across_all_three(self):
        text = 'a <parameter name="x"> b </content> c </invoke> d'
        assert find_markup_pattern(text) == '<parameter name='

    def test_returns_none_for_clean_prose(self):
        assert find_markup_pattern('a perfectly ordinary sentence about tasks') is None

    def test_returns_none_for_similar_but_non_matching_markup(self):
        """Nearby-but-different markup does not trip the guard."""
        assert find_markup_pattern('<content> and <invoke> and <parameter>') is None

    def test_matching_is_case_sensitive(self):
        """Uppercase variants do NOT match — the harness emits lowercase tags.

        Case-folding would widen the guard to prose that shouts the tag name
        without buying any real recall: no observed specimen is uppercased.
        """
        assert find_markup_pattern('a leaked </INVOKE> tail') is None
        assert find_markup_pattern('a leaked </CONTENT> tail') is None
        assert find_markup_pattern('a leaked <PARAMETER NAME=') is None

    def test_returns_none_for_none_input(self):
        """None passes through without raising — callers hand us raw handler args."""
        assert find_markup_pattern(None) is None

    def test_returns_none_for_empty_string(self):
        assert find_markup_pattern('') is None

    def test_returns_none_for_non_str_input(self):
        """Non-str input never raises — an optional handler arg may be anything."""
        for value in (123, 4.5, True, [], {}, object()):
            assert find_markup_pattern(value) is None, f'Expected None for {value!r}'


class TestFindMarkupViolation:
    """The multi-field matcher: (field_name, matched_pattern) or None."""

    def test_returns_field_and_pattern_for_a_single_violating_field(self):
        result = find_markup_violation({'description': 'leaked </invoke> here'})
        assert result == ('description', '</invoke>')

    def test_returns_first_violating_field_in_insertion_order(self):
        """Dict insertion order decides, so call sites control reporting priority.

        Both fields are dirty; ``title`` was inserted first, so it is named.
        """
        fields = {
            'title': 'title with </content> in it',
            'description': 'description with </invoke> in it',
        }
        assert find_markup_violation(fields) == ('title', '</content>')

    def test_skips_clean_fields_to_reach_the_violating_one(self):
        fields = {
            'title': 'a clean title',
            'description': 'also clean',
            'details': 'dirty <parameter name="priority">',
            'prompt': 'clean too',
        }
        assert find_markup_violation(fields) == ('details', '<parameter name=')

    def test_returns_none_when_every_field_is_clean(self):
        fields = {
            'title': 'a clean title',
            'description': 'a clean description',
            'details': 'clean details',
            'prompt': 'a clean prompt',
        }
        assert find_markup_violation(fields) is None

    def test_returns_none_for_empty_and_none_field_values(self):
        """Absent/empty optional fields are not violations and never raise."""
        assert find_markup_violation({'title': None, 'description': '', 'details': None}) is None

    def test_returns_none_for_an_empty_field_map(self):
        assert find_markup_violation({}) is None

    def test_ignores_non_str_field_values(self):
        """A non-str value (e.g. a dict metadata blob) is skipped, not coerced."""
        fields = {'title': 12345, 'description': ['</invoke>'], 'details': 'clean'}
        assert find_markup_violation(fields) is None


class TestBuildMarkupBlock:
    """The structured rejection dict returned to the caller (INV-1).

    The write has already been refused by the time this runs; this dict is the
    ONLY machine-readable account of why, so it must name the matched pattern,
    the field it was found in, and the remediation — an agent that only reads
    the MCP response (never the logs) has to be able to fix its own write.
    """

    def test_carries_the_stable_error_identifiers(self):
        block = build_markup_block('claude-task-1', 'content', '</invoke>', 'a </invoke> b')
        assert block['error'] == 'mcp_markup_write_blocked'
        assert block['error_type'] == 'McpEnvelopeMarkupWriteRejected'

    def test_echoes_agent_id_field_and_matched_pattern(self):
        """The caller is told WHICH pattern tripped and WHERE, not just that one did."""
        block = build_markup_block(
            'claude-task-3141', 'description', '<parameter name=', 'x <parameter name="p"> y'
        )
        assert block['agent_id'] == 'claude-task-3141'
        assert block['field'] == 'description'
        assert block['matched_pattern'] == '<parameter name='

    def test_tolerates_a_none_agent_id(self):
        """agent_id is optional at some boundaries — it is echoed as-is, not coerced."""
        block = build_markup_block(None, 'content', '</content>', 'a </content> b')
        assert block['agent_id'] is None

    def test_content_excerpt_is_the_untruncated_text_when_short(self):
        text = 'short leaked </invoke> body'
        block = build_markup_block('a', 'content', '</invoke>', text)
        assert block['content_excerpt'] == text

    def test_content_excerpt_truncates_at_200_chars(self):
        """A leaked envelope tail can drag a huge body along; the excerpt is bounded."""
        text = 'x' * 500 + '</invoke>'
        block = build_markup_block('a', 'content', '</invoke>', text)
        assert block['content_excerpt'] == text[:200]
        assert len(block['content_excerpt']) == 200

    def test_hint_names_df_3083_and_the_override_key(self):
        """Both the escalation pointer and the escape hatch are discoverable here.

        DF 3083 owns the root cause and the corpus sweep, so the rejection points
        at it; MARKUP_OVERRIDE_KEY is how a caller quoting the markup on purpose
        (as 3083's own description does) gets its write through.
        """
        block = build_markup_block('a', 'content', '</invoke>', 'a </invoke> b')
        hint = block['hint']
        assert hint
        assert '3083' in hint, f'hint must name DF 3083: {hint!r}'
        assert MARKUP_OVERRIDE_KEY in hint, f'hint must name the override key: {hint!r}'

    def test_storm_key_is_absent_when_no_storm_fired(self):
        """Below threshold the dict carries no 'storm' key at all — not a None value.

        A present-but-null key would read to a caller as "a storm was evaluated
        and there was none", which is indistinguishable from the ordinary case
        and invites `if 'storm' in block` bugs.
        """
        block = build_markup_block('a', 'content', '</invoke>', 'a </invoke> b')
        assert 'storm' not in block

    def test_storm_key_is_absent_when_storm_is_explicitly_none(self):
        block = build_markup_block('a', 'content', '</invoke>', 'a </invoke> b', storm=None)
        assert 'storm' not in block

    def test_storm_block_is_echoed_verbatim_when_supplied(self):
        """The storm summary reaches the caller, not only the logs (INV-4)."""
        storm = {
            'count': 3,
            'threshold': 3,
            'window_seconds': 3600.0,
            'hint': 'the leak is active; see DF 3083',
        }
        block = build_markup_block('a', 'content', '</invoke>', 'a </invoke> b', storm=storm)
        assert block['storm'] == storm

    def test_block_is_a_flat_json_serializable_dict(self):
        """Mirrors build_near_duplicate_block: a flat dict an MCP client can render."""
        block = build_markup_block('a', 'content', '</invoke>', 'a </invoke> b')
        assert isinstance(block, dict)
        json.dumps(block)  # raises TypeError if a non-serializable value slipped in


class TestMarkupStormCounter:
    """Rolling-window burst detector (INV-4).

    A single rejection is a bounced write; a BURST means the serialization leak
    is actively running, which is the condition worth escalating. Every test
    here drives an injected clock — no sleeping, no wall-clock dependence.
    """

    def test_returns_none_below_threshold(self):
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=3, window_seconds=100.0, time_provider=clock)
        assert counter.record() is None
        assert counter.record() is None

    def test_fires_on_the_record_that_reaches_threshold(self):
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=3, window_seconds=100.0, time_provider=clock)
        counter.record()
        counter.record()
        storm = counter.record()
        assert storm is not None
        assert storm['count'] == 3
        assert storm['threshold'] == 3
        assert storm['window_seconds'] == 100.0

    def test_storm_hint_names_df_3083(self):
        """The burst summary points at the owner of the root cause, not at itself."""
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=1, window_seconds=100.0, time_provider=clock)
        storm = counter.record()
        assert storm is not None
        assert storm['hint']
        assert '3083' in storm['hint'], f"storm hint must name DF 3083: {storm['hint']!r}"

    def test_reports_the_distinct_projects_seen_in_the_window(self):
        """The burst is ATTRIBUTED, not pinned on whichever write crossed the line.

        One server instance serves every known project, so a bare count would let
        the caller escalate into the queue of the project that merely happened to
        be last — naming it as the leaker while the project actually leaking got
        nothing (and, because the rate limit is shared, no second chance until the
        window rolled over). Mirrors the 'projects' key of
        harness._record_placeholder_finding_drop.
        """
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=3, window_seconds=100.0, time_provider=clock)
        counter.record('/project-a')
        counter.record('/project-a')
        storm = counter.record('/project-b')
        assert storm is not None
        assert storm['count'] == 3
        assert storm['projects'] == ['/project-a', '/project-b'], f'got: {storm!r}'

    def test_projects_are_sorted_and_deduplicated(self):
        """A repeated label appears once — this is a project SET, not a tally."""
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=4, window_seconds=100.0, time_provider=clock)
        counter.record('/z')
        counter.record('/a')
        counter.record('/z')
        storm = counter.record('/a')
        assert storm is not None
        assert storm['projects'] == ['/a', '/z'], f'got: {storm!r}'

    def test_unlabelled_events_count_toward_the_burst_but_are_not_named(self):
        """add_memory on an unknown project resolves no root; it is still a rejection.

        Such an event must still push the counter toward the threshold — it is a
        real leaked write — but there is no queue to escalate it into, so it must
        not appear in `projects` (and must not crash `sorted` by mixing None in).
        """
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=3, window_seconds=100.0, time_provider=clock)
        counter.record(None)
        counter.record(None)
        storm = counter.record('/project-a')
        assert storm is not None
        assert storm['count'] == 3, f'unlabelled rejections must still count: {storm!r}'
        assert storm['projects'] == ['/project-a'], f'got: {storm!r}'

    def test_projects_reflects_only_the_unpruned_window(self):
        """A project whose rejections aged out is no longer part of the burst."""
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=2, window_seconds=100.0, time_provider=clock)
        counter.record('/stale')
        clock.advance(150.0)
        counter.record('/live')
        storm = counter.record('/live')
        assert storm is not None
        assert storm['projects'] == ['/live'], f'stale project must be pruned: {storm!r}'

    def test_rate_limited_to_one_fire_per_window(self):
        """Further rejections inside the same window return None.

        Without this a leak emitting hundreds of writes would file hundreds of
        escalations for one incident.
        """
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=2, window_seconds=100.0, time_provider=clock)
        counter.record()
        assert counter.record() is not None  # fires
        clock.advance(1.0)
        assert counter.record() is None
        clock.advance(1.0)
        assert counter.record() is None

    def test_can_fire_again_after_the_window_elapses(self):
        """The rate limit expires with the window — a second incident still alarms."""
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=2, window_seconds=100.0, time_provider=clock)
        counter.record()
        assert counter.record() is not None

        # Move past both the rate-limit window and the event window, then
        # re-trip from scratch.
        clock.advance(200.0)
        assert counter.record() is None  # window pruned; only 1 event
        assert counter.record() is not None

    def test_a_slow_trickle_never_fires(self):
        """Events older than window_seconds are pruned, so a trickle is not a storm.

        Ten rejections spread far enough apart never accumulate to a threshold of
        3 — this is the whole point of a rolling WINDOW rather than a total count.
        """
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=3, window_seconds=100.0, time_provider=clock)
        for _ in range(10):
            assert counter.record() is None
            clock.advance(150.0)

    def test_events_exactly_at_the_window_edge_are_pruned(self):
        """A window is a half-open interval — an event aged exactly window_seconds is out.

        Pins the boundary so a future `<=`/`<` edit cannot silently shift when a
        storm fires.
        """
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=2, window_seconds=100.0, time_provider=clock)
        counter.record()
        clock.advance(100.0)
        assert counter.record() is None

    def test_module_defaults_are_the_documented_constants(self):
        """The no-arg counter uses the module's own threshold/window."""
        clock = _FakeClock()
        counter = MarkupStormCounter(time_provider=clock)
        for _ in range(markup_tripwire._MARKUP_STORM_THRESHOLD - 1):
            assert counter.record() is None
        storm = counter.record()
        assert storm is not None
        assert storm['threshold'] == markup_tripwire._MARKUP_STORM_THRESHOLD
        assert storm['window_seconds'] == markup_tripwire._MARKUP_STORM_WINDOW_SECONDS

    def test_default_threshold_and_window_values(self):
        assert markup_tripwire._MARKUP_STORM_THRESHOLD == 3
        assert markup_tripwire._MARKUP_STORM_WINDOW_SECONDS == 3600.0

    def test_defaults_to_wall_clock_when_no_provider_is_injected(self):
        """The production path needs no argument — time.time is the default."""
        counter = MarkupStormCounter(threshold=1, window_seconds=100.0)
        assert counter.record() is not None

    def test_counters_do_not_share_state(self):
        """Per-instance state, so one create_mcp_server cannot bleed into another."""
        clock = _FakeClock()
        first = MarkupStormCounter(threshold=2, window_seconds=100.0, time_provider=clock)
        second = MarkupStormCounter(threshold=2, window_seconds=100.0, time_provider=clock)
        first.record()
        # `second` has seen exactly one event of its own, so it must not fire.
        assert second.record() is None
        assert first.record() is not None

    def test_storm_summary_is_json_serializable(self):
        clock = _FakeClock()
        counter = MarkupStormCounter(threshold=1, window_seconds=100.0, time_provider=clock)
        storm = counter.record()
        json.dumps(storm)


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

    def test_files_one_escalation_naming_3083_and_the_storm_numbers(self, tmp_path):
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
        assert '3083' in f'{payload["summary"]}\n{detail}', f'must name DF 3083: {payload!r}'
        assert 'rejected_writes_in_window=4' in detail, f'must state the count: {detail!r}'
        assert 'window_seconds=3600.0' in detail, f'must state the window: {detail!r}'
        assert "projects_in_window=['/project-a', '/project-b']" in detail, (
            f'must name every project the burst spanned: {detail!r}'
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
        assert '3083' in payload['detail'], f'must still route to DF 3083: {payload!r}'
