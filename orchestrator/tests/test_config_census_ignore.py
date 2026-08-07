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

import pytest

from orchestrator.config_census_ignore import (
    CensusIgnoreFinding,
    CensusIgnoreSpec,
    audit_census_ignore_specs,
    parse_census_ignore_entries,
)


def _spec(reason: str | None, pattern: str = 'a.b') -> CensusIgnoreSpec:
    """One spec carrying *reason*, with citations parsed the same way the real
    parser would (so the audit sees exactly what production hands it)."""
    (parsed,) = parse_census_ignore_entries(
        {'config_key_census': {'ignore': [
            {'path': pattern, 'reason': reason} if reason is not None else pattern
        ]}}
    )
    return parsed


def _kinds(reason: str | None, pattern: str = 'a.b') -> set[str]:
    return {
        f.kind for f in audit_census_ignore_specs([_spec(reason, pattern)], None)
    }


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
# Inherits config._census_ignore_specs' contract verbatim: a broken escape hatch
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


# --- (b) structural audit -----------------------------------------------------
#
# status_probe=None throughout, so ONLY structural kinds can fire — the liveness
# half is covered in section (c).  Taxonomy adopted from PTODO §8.3/§8.4.


@pytest.mark.parametrize('reason', [None, '   ', '\n\t '])
def test_unreasoned_is_advisory(reason):
    """ACCEPTANCE (a): a new un-reasoned entry is visible as debt.  A bare
    string and a blank reason are the same defect — an assertion with no
    content — so they must not be distinguished."""
    (finding,) = audit_census_ignore_specs([_spec(reason)], None)
    assert finding.kind == 'unreasoned'
    assert finding.severity == 'advisory'
    assert 'a.b' in finding.detail


@pytest.mark.parametrize(
    'reason',
    [
        'dark-factory reads this',
        'dark factory reads this',
        'consumed by the orchestrator',
        'OrchestratorConfig picks this up',
    ],
)
def test_self_refuting_is_hard(reason):
    """The one check that would have caught cpu_governance.DF_AGENT_CPU_GOVERN
    at the moment it was added, ~6 weeks before the outage was found by hand.
    Wrong BY CONSTRUCTION: dark-factory owns the schema, so a key it consumed
    would be a FIELD on the model, hence known, hence never in need of an
    ignore entry.  Certain with no external state — hence hard."""
    findings = audit_census_ignore_specs([_spec(reason)], None)
    (finding,) = [f for f in findings if f.kind == 'self-refuting']
    assert finding.severity == 'hard'
    assert 'a.b' in finding.detail
    # The remediation must name the schema-field route, not just say "wrong".
    assert 'field' in finding.detail.lower()


@pytest.mark.parametrize(
    'reason',
    [
        'pending the consumer landing',
        'until the consumer lands',
        'will be read once the new runner ships',
        'not yet consumed, planned for the next cycle',
    ],
)
def test_missing_cite_is_hard(reason):
    """A not-yet-landed claim with no citation has no expiry: nothing will ever
    prompt a re-check.  This is precisely the shape of the entry that made the
    reify outage permanent."""
    findings = audit_census_ignore_specs([_spec(reason)], None)
    (finding,) = [f for f in findings if f.kind == 'missing-cite']
    assert finding.severity == 'hard'
    assert 'a.b' in finding.detail
    assert '#' in finding.detail, 'detail must name the canonical #NNNN remedy'


@pytest.mark.parametrize(
    'reason',
    ['blocked on task 5908', 'see task-5', 'tracked by task δ'],
)
def test_malformed_cite_is_advisory(reason):
    """PTODO §6.4: the canonical form is #NNNN strictly, so a legacy reference
    form is itself a finding — but advisory, since the operator did leave a
    pointer and the entry is not unfalsifiable."""
    findings = audit_census_ignore_specs([_spec(reason)], None)
    (finding,) = [f for f in findings if f.kind == 'malformed-cite']
    assert finding.severity == 'advisory'
    assert 'a.b' in finding.detail


def test_well_formed_project_owned_reason_yields_no_finding():
    assert _kinds('read verbatim by scripts/cpu-governed-exec.sh') == set()


def test_pending_prose_with_a_canonical_cite_yields_no_finding():
    """The positive case the whole mechanism exists to enable: a temporary
    entry IS legitimate, provided it cites a live tracking task.  #5908 is
    reify's real, still-pending entry-cleanup task."""
    assert _kinds('temporary — pending #5908, which deletes this entry') == set()


def test_one_spec_can_yield_two_findings():
    """Kinds are independent defects, not a single classification: an entry can
    be both self-refuting AND uncited, and suppressing either would hide a real
    problem."""
    findings = audit_census_ignore_specs(
        [_spec('dark-factory will read this once it lands')], None
    )
    assert {f.kind for f in findings} == {'self-refuting', 'missing-cite'}


def test_findings_carry_the_pattern_so_the_entry_can_be_located():
    findings = audit_census_ignore_specs(
        [_spec(None, 'cpu_governance.weights'), _spec(None, 'warm_lane_pool')], None
    )
    assert {f.pattern for f in findings} == {'cpu_governance.weights', 'warm_lane_pool'}
    assert all(isinstance(f, CensusIgnoreFinding) for f in findings)


def test_no_specs_yields_no_findings():
    assert audit_census_ignore_specs([], None) == []
