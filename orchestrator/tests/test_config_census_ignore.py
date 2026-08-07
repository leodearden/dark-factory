"""Tests for reasoned ``config_key_census.ignore`` entries (task 3395).

The defect these pin: a bare-string ignore entry is an unfalsifiable ASSERTION
about a non-OrchestratorConfig consumer.  It is never re-checked, and there is
no way to say "temporary, until task X lands".  reify's
``cpu_governance.DF_AGENT_CPU_GOVERN`` entry is the proof — membership in the
ignore list logically PROVES dark-factory does not consume the key, yet the
entry was added on the expectation that it eventually would, which is what made
the resulting CPU-governance outage both permanent and silent.

The census WALK itself is correct and gets no coverage change here; these tests
cover the new entry grammar, the violation taxonomy, and citation liveness.
"""

from orchestrator.config_census_ignore import (
    CensusIgnoreSpec,
    parse_census_ignore_entries,
)


def _tree(ignore) -> dict:
    """Minimal raw project tree carrying an ignore list."""
    return {'config_key_census': {'ignore': ignore}}


# --- (a) entry parsing --------------------------------------------------------


def test_bare_string_entry_stays_accepted():
    """BACK-COMPAT: today's five live reify entries are bare strings.  Widening
    the grammar must not invalidate them — they parse to a reasonless spec."""
    specs = parse_census_ignore_entries(_tree(['cpu_governance.weights']))
    assert len(specs) == 1
    assert specs[0] == CensusIgnoreSpec('cpu_governance.weights', None, ())


def test_dict_entry_carries_the_reason():
    specs = parse_census_ignore_entries(
        _tree([
            {
                'path': 'cpu_governance.weights',
                'reason': 'read by scripts/cpu-governed-exec.sh',
            }
        ])
    )
    assert len(specs) == 1
    assert specs[0].pattern == 'cpu_governance.weights'
    assert specs[0].reason == 'read by scripts/cpu-governed-exec.sh'


def test_mixed_list_preserves_source_order():
    """fnmatch matching is FIRST-match-wins, so source order is load-bearing:
    if the parser reordered entries, the note attached to a key could silently
    change to a different entry's justification."""
    specs = parse_census_ignore_entries(
        _tree([
            'a.one',
            {'path': 'b.two', 'reason': 'r2'},
            'c.three',
            {'path': 'd.four', 'reason': 'r4'},
        ])
    )
    assert [s.pattern for s in specs] == ['a.one', 'b.two', 'c.three', 'd.four']
    assert [s.reason for s in specs] == [None, 'r2', None, 'r4']


def test_citations_extracted_from_reason():
    specs = parse_census_ignore_entries(
        _tree([{'path': 'p', 'reason': 'blocked on #5908 until it lands'}])
    )
    assert specs[0].citations == (5908,)


def test_citations_are_ordered_and_deduped():
    specs = parse_census_ignore_entries(
        _tree([{'path': 'p', 'reason': 'see #42, then #5908, and again #42'}])
    )
    assert specs[0].citations == (42, 5908)


def test_citations_empty_when_reason_has_no_canonical_cite():
    specs = parse_census_ignore_entries(
        _tree([{'path': 'p', 'reason': 'read by the project deploy script'}])
    )
    assert specs[0].citations == ()


# --- (a2) fail-open parsing ---------------------------------------------------
#
# Inherits _census_ignore_patterns' contract verbatim: a broken escape hatch
# must never take out the census that surfaces real phantom keys.  Each case is
# asserted independently so a regression names the exact malformation.


def test_fail_open_missing_block():
    assert parse_census_ignore_entries({'max_concurrent_tasks': 4}) == []


def test_fail_open_non_dict_block():
    assert parse_census_ignore_entries({'config_key_census': 'nope'}) == []


def test_fail_open_non_list_ignore():
    assert parse_census_ignore_entries(_tree('cpu_governance.*')) == []


def test_fail_open_none_entry():
    specs = parse_census_ignore_entries(_tree([None, 'a.b']))
    assert [s.pattern for s in specs] == ['a.b']


def test_fail_open_int_entry():
    specs = parse_census_ignore_entries(_tree([7, 'a.b']))
    assert [s.pattern for s in specs] == ['a.b']


def test_fail_open_dict_without_path():
    specs = parse_census_ignore_entries(_tree([{'reason': 'orphan reason'}, 'a.b']))
    assert [s.pattern for s in specs] == ['a.b']


def test_fail_open_dict_with_non_str_path():
    specs = parse_census_ignore_entries(_tree([{'path': 99, 'reason': 'r'}, 'a.b']))
    assert [s.pattern for s in specs] == ['a.b']


def test_fail_open_non_str_reason_keeps_the_pattern():
    """A bad REASON must not delete the suppression: dropping the pattern would
    resurrect a key the operator deliberately excused and could hard-fail a
    live unit's census.  The reason degrades to None (reported as debt)."""
    specs = parse_census_ignore_entries(_tree([{'path': 'a.b', 'reason': 123}]))
    assert len(specs) == 1
    assert specs[0].pattern == 'a.b'
    assert specs[0].reason is None
    assert specs[0].citations == ()


def test_fail_open_empty_ignore_list():
    assert parse_census_ignore_entries(_tree([])) == []
