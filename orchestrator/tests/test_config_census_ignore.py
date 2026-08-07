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

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from orchestrator.config_census_ignore import (
    CensusIgnoreFinding,
    CensusIgnoreSpec,
    TaskCiteStatus,
    audit_census_ignore_entries,
    audit_census_ignore_specs,
    parse_census_ignore_entries,
    read_task_statuses,
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


# --- (c) citation liveness ----------------------------------------------------
#
# Policy adopted from reconciliation/stale_status_snapshot_edge_sweep.py:28-32 —
# invalidate ONLY on a positively-terminal answer, so the audit can under-fire
# but can never invent a violation out of missing data.  check-config is an
# offline tool routinely pointed at another project's YAML from a machine
# without that project's .taskmaster/, so absence must mean "cannot know".

_DONE_ID = 111
_CANCELLED_ID = 222
_PENDING_ID = 333
_PARKED_ID = 444
_PLAIN_DEFERRED_ID = 555
_DO_NOT_DISPATCH_ID = 666


def _seed_tasks_db(project_root: Path) -> Path:
    """Real sqlite fixture in the MEASURED production shape: table ``tasks``,
    PK ``(tag, id)``, tag ``master``."""
    db = project_root / '.taskmaster' / 'tasks' / 'tasks.db'
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE tasks (tag TEXT, id INTEGER, title TEXT, status TEXT, '
        'metadata TEXT, PRIMARY KEY (tag, id))'
    )
    conn.executemany(
        'INSERT INTO tasks (tag, id, title, status, metadata) VALUES (?,?,?,?,?)',
        [
            ('master', _DONE_ID, 't', 'done', None),
            ('master', _CANCELLED_ID, 't', 'cancelled', None),
            ('master', _PENDING_ID, 't', 'pending', None),
            ('master', _PARKED_ID, 't', 'deferred', json.dumps({'do_not_complete': True})),
            ('master', _PLAIN_DEFERRED_ID, 't', 'deferred', None),
            ('master', _DO_NOT_DISPATCH_ID, 't', 'deferred',
             json.dumps({'do_not_dispatch': True})),
            # A different tag must never leak into the master view.
            ('other', 999, 't', 'done', None),
        ],
    )
    conn.commit()
    conn.close()
    return db


def _probe(**overrides) -> dict[int, TaskCiteStatus]:
    """Dict-backed fake probe — the taxonomy is testable with no sqlite at all."""
    base = {
        _DONE_ID: TaskCiteStatus('done', False),
        _CANCELLED_ID: TaskCiteStatus('cancelled', False),
        _PENDING_ID: TaskCiteStatus('pending', False),
        _PARKED_ID: TaskCiteStatus('deferred', True),
        _PLAIN_DEFERRED_ID: TaskCiteStatus('deferred', False),
        _DO_NOT_DISPATCH_ID: TaskCiteStatus('deferred', False),
    }
    base.update(overrides)
    return base


def test_read_task_statuses_reads_the_master_tag(tmp_path):
    _seed_tasks_db(tmp_path)
    statuses = read_task_statuses(tmp_path)
    assert statuses is not None
    assert statuses[_DONE_ID].status == 'done'
    assert statuses[_PARKED_ID].do_not_complete is True
    assert statuses[_PLAIN_DEFERRED_ID].do_not_complete is False
    assert 999 not in statuses, 'a non-master tag must not leak into the view'
    assert all(isinstance(k, int) for k in statuses)


@pytest.mark.parametrize('broken', ['missing_dir', 'missing_db', 'corrupt'])
def test_read_task_statuses_returns_none_and_never_raises(tmp_path, broken):
    """FAIL-OPEN: absence means 'cannot know', never 'clean' and never a crash.
    check-config is routinely pointed at a config whose project DB is not on
    this machine at all."""
    if broken == 'missing_db':
        (tmp_path / '.taskmaster' / 'tasks').mkdir(parents=True)
    elif broken == 'corrupt':
        db = tmp_path / '.taskmaster' / 'tasks' / 'tasks.db'
        db.parent.mkdir(parents=True)
        db.write_bytes(b'this is definitely not a sqlite file' * 20)
    assert read_task_statuses(tmp_path) is None


def test_read_task_statuses_does_not_mutate_the_store(tmp_path):
    """Opened strictly read-only (mode=ro) — a lint must never mutate the store
    it measures.  Mirrors sandbox_soak._connect_ro / b3_gate."""
    db = _seed_tasks_db(tmp_path)
    before = db.stat().st_mtime_ns
    read_task_statuses(tmp_path)
    assert db.stat().st_mtime_ns == before
    with pytest.raises(sqlite3.OperationalError):
        conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        try:
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (_PENDING_ID,))
            conn.commit()
        finally:
            conn.close()


@pytest.mark.parametrize(
    ('cited', 'status'), [(_DONE_ID, 'done'), (_CANCELLED_ID, 'cancelled')]
)
def test_orphaned_cite_is_hard(cited, status):
    """The justification is provably spent: the task that was going to land the
    consumer has closed, so either the consumer landed (and the key should be a
    model field) or it never will (and the entry should go)."""
    findings = audit_census_ignore_specs(
        [_spec(f'temporary — pending #{cited}')], _probe()
    )
    (finding,) = [f for f in findings if f.kind == 'orphaned']
    assert finding.severity == 'hard'
    assert str(cited) in finding.detail
    assert status in finding.detail, 'the detail must name the terminal status'


def test_unknown_id_is_advisory():
    """Advisory, not hard: a cite absent from the DB is more often a task-DB
    sync artifact than a real defect, and a sync artifact must never hard-fail
    a config gate (PTODO §8.4)."""
    findings = audit_census_ignore_specs([_spec('pending #99999')], _probe())
    (finding,) = [f for f in findings if f.kind == 'unknown-id']
    assert finding.severity == 'advisory'
    assert '99999' in finding.detail


def test_parked_on_anchor_is_advisory():
    findings = audit_census_ignore_specs([_spec(f'pending #{_PARKED_ID}')], _probe())
    (finding,) = [f for f in findings if f.kind == 'parked-on-anchor']
    assert finding.severity == 'advisory'
    assert str(_PARKED_ID) in finding.detail


@pytest.mark.parametrize('cited', [_PLAIN_DEFERRED_ID, _DO_NOT_DISPATCH_ID])
def test_deferred_alone_is_not_parked(cited):
    """FALSE-POSITIVE GUARDS (PTODO §9 scenarios 15/15b): parked keys on
    metadata.do_not_complete SPECIFICALLY — not on bare `deferred` (an ordinary
    non-terminal status) and not on `do_not_dispatch` (a scheduler knob that
    says nothing about whether the task will ever complete)."""
    findings = audit_census_ignore_specs([_spec(f'pending #{cited}')], _probe())
    assert findings == []


def test_live_pending_cite_yields_no_finding():
    assert audit_census_ignore_specs([_spec(f'pending #{_PENDING_ID}')], _probe()) == []


def test_one_live_cite_suppresses_orphaned():
    """PTODO §8.2 — one live cite suffices.  A reason may legitimately reference
    several tasks; the entry is still tracked as long as ONE of them is open."""
    findings = audit_census_ignore_specs(
        [_spec(f'pending #{_DONE_ID} and #{_PENDING_ID}')], _probe()
    )
    assert [f.kind for f in findings if f.kind == 'orphaned'] == []


def test_all_cites_terminal_still_reports_orphaned():
    """The converse of the above: 'one live cite suffices' must not degrade
    into 'any cite suffices'."""
    findings = audit_census_ignore_specs(
        [_spec(f'pending #{_DONE_ID} and #{_CANCELLED_ID}')], _probe()
    )
    assert [f.kind for f in findings if f.kind == 'orphaned']


def test_absent_probe_yields_zero_liveness_findings():
    """PTODO §9 scenario 9.  With no probe the identical spec set produces the
    SAME structural findings and NO liveness findings — the audit is loudest
    where it knows most, never where it knows least."""
    specs = [
        _spec(f'pending #{_DONE_ID}', 'p.one'),
        _spec('pending #99999', 'p.two'),
        _spec(f'pending #{_PARKED_ID}', 'p.three'),
        _spec(None, 'p.four'),
        _spec('dark-factory reads this', 'p.five'),
        _spec('pending the consumer landing', 'p.six'),
    ]
    with_probe = audit_census_ignore_specs(specs, _probe())
    without = audit_census_ignore_specs(specs, None)

    liveness = {'orphaned', 'unknown-id', 'parked-on-anchor'}
    assert {f.kind for f in without} & liveness == set()
    # Structural findings are byte-identical with and without the probe.
    assert [f for f in with_probe if f.kind not in liveness] == without


# --- (d) end-to-end from a config path ----------------------------------------


def _write_config(tmp_path: Path, tree: dict, name: str = 'orchestrator.yaml') -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(tree))
    return p


def test_audit_from_config_path_combines_structural_and_liveness(tmp_path):
    """The operator-facing entry point: read the YAML, resolve project_root off
    the RAW tree, probe that project's task store, return everything."""
    project = tmp_path / 'proj'
    project.mkdir()
    _seed_tasks_db(project)
    p = _write_config(tmp_path, {
        'project_root': str(project),
        'config_key_census': {'ignore': [
            'bare.entry',
            {'path': 'self.refuting', 'reason': 'the orchestrator reads this'},
            {'path': 'stale.cite', 'reason': f'temporary — pending #{_DONE_ID}'},
        ]},
    })
    findings = audit_census_ignore_entries(p)
    by_pattern = {}
    for f in findings:
        by_pattern.setdefault(f.pattern, set()).add(f.kind)
    assert by_pattern['bare.entry'] == {'unreasoned'}
    assert by_pattern['self.refuting'] == {'self-refuting'}
    assert by_pattern['stale.cite'] == {'orphaned'}


def test_audit_reads_raw_tree_despite_a_value_level_validation_error(tmp_path):
    """Must work on a config pydantic would REJECT: the audit reads the raw
    tree, exactly as census_config_keys does, so a lint still reports entry
    debt on a config that cannot currently be loaded."""
    p = _write_config(tmp_path, {
        'max_concurrent_tasks': 'definitely-not-an-int',
        'config_key_census': {'ignore': ['bare.entry']},
    })
    assert [f.kind for f in audit_census_ignore_entries(p)] == ['unreasoned']


@pytest.mark.parametrize('tree', [
    {'max_concurrent_tasks': 4},
    {'config_key_census': {'ignore': []}},
])
def test_audit_returns_empty_for_a_config_with_nothing_to_audit(tmp_path, tree):
    assert audit_census_ignore_entries(_write_config(tmp_path, tree)) == []


@pytest.mark.parametrize('project_root', [
    None,                      # absent from the tree
    'relative/path',           # not absolute
    '/nonexistent/dir/xyz',    # absolute but not there
    'no_db',                   # a real dir with no .taskmaster/
])
def test_structural_findings_survive_an_unresolvable_project_root(tmp_path, project_root):
    """FAIL-OPEN, the strong form: when the task store cannot be reached the
    liveness half goes quiet but the structural half is still returned IN
    FULL — the audit degrades in one dimension only."""
    tree = {'config_key_census': {'ignore': [
        'bare.entry',
        {'path': 'stale.cite', 'reason': f'temporary — pending #{_DONE_ID}'},
    ]}}
    if project_root == 'no_db':
        (tmp_path / 'no_db').mkdir()
        tree['project_root'] = str(tmp_path / 'no_db')
    elif project_root is not None:
        tree['project_root'] = project_root

    findings = audit_census_ignore_entries(_write_config(tmp_path, tree))
    kinds = {f.kind for f in findings}
    assert 'unreasoned' in kinds, 'structural findings must survive in full'
    assert kinds & {'orphaned', 'unknown-id', 'parked-on-anchor'} == set()


def test_audit_never_raises_on_a_non_str_project_root(tmp_path):
    p = _write_config(tmp_path, {
        'project_root': 12345,
        'config_key_census': {'ignore': ['bare.entry']},
    })
    assert [f.kind for f in audit_census_ignore_entries(p)] == ['unreasoned']


@pytest.mark.parametrize('content', [
    'this: [is: not: valid: yaml',   # malformed
    'just a bare scalar',            # non-dict document
    '',                              # empty -> None document
])
def test_audit_returns_empty_for_an_unusable_config_file(tmp_path, content):
    p = tmp_path / 'broken.yaml'
    p.write_text(content)
    assert audit_census_ignore_entries(p) == []


def test_audit_returns_empty_for_a_missing_config_file(tmp_path):
    """A broken lint must never become a crash — load_config calls this on
    every startup and every hot-reload."""
    assert audit_census_ignore_entries(tmp_path / 'nope.yaml') == []
