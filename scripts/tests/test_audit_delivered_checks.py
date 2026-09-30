"""Tests for scripts/audit_delivered_checks.py — the READ-ONLY, STATUS-AWARE
retroactive sweep over checked-in ``delivered_checks`` (task 3500, SCOPE item 5).

Mirrors test_audit_combine_gate_marker_loss.py: pure functions get direct
pytest coverage; ``main()`` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
tasks.db is mutated continuously by the running orchestrator, so a test
pinning "the live DB yields N findings" would be a guessed threshold that goes
red the moment another task lands. Every assertion runs against synthetic temp
databases and synthetic temp git repos whose contents the test controls
exactly — the same discipline that file's docstring requires.

WHY STATUS IS THE AXIS. Evaluating a descriptor against MAIN TODAY yields a bit, not a verdict: the
same "``expect: present`` and it matches" observation is the SUCCESS state of a
landed producer and a never-fires vacuous gate on a live one. Measured over the
checked-in corpus, a status-blind rule flags 313/548 (57%) of descriptors —
overwhelmingly correctly delivered work. The producer's STATUS is what turns
the bit into a disposition, and status lives in tasks.db, which is why this is
a script and not a test.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from audit_delivered_checks import (
    _TASK_IN_SUBJECT_RE,
    DISPOSITION_BROKEN,
    DISPOSITION_DELIVERED,
    DISPOSITION_HEALTHY,
    DISPOSITION_INERT,
    DISPOSITION_NO_TASK,
    DISPOSITION_SUPERSEDED,
    DISPOSITION_UNEVALUABLE,
    DISPOSITION_UNWIRED_LIVE_GATE,
    DISPOSITION_VACUOUS_LIVE_GATE,
    AuditCoverage,
    DescriptorRow,
    Finding,
    ProjectAudit,
    audit_project,
    classify_descriptor,
    evaluate_row,
    format_report,
    load_task_index,
)
from shared.delivered_check_polarity import CheckOutcome

# ---------------------------------------------------------------------------
# classify_descriptor — the pure core, and the whole point of the script.
#
# The three inputs are the descriptor's OUTCOME against main today, the
# producer's STATUS, and (only for a would-be defect) the id of a later task
# that reintroduced the pattern. Nothing else: no history, no heuristics.
# ---------------------------------------------------------------------------


def _row(**overrides):
    """A minimal DescriptorRow; every test overrides only what it asserts on."""
    fields = {
        'task_id': 42,
        'tag': 'master',
        'status': 'done',
        'name': 'cap-one',
        'kind': 'grep',
        'pattern': 'SomeSymbol',
        'expect': 'present',
        'paths': ('src/',),
        'source': 'metadata',
        'manifest': None,
        'task_label': None,
    }
    fields.update(overrides)
    return DescriptorRow(**fields)


class TestClassifyDescriptor:
    def test_done_and_expect_present_with_no_match_is_broken(self):
        """(a) The mode-2/mode-3 defect class: the producer closed, and the
        capability its check asserts is nowhere on main. Whatever the check was
        supposed to gate was never gated."""
        assert classify_descriptor(CheckOutcome.FAIL, status='done') == DISPOSITION_BROKEN

    def test_done_and_expect_absent_still_matching_is_broken(self):
        """(b) The same defect from the other polarity. FAIL already encodes
        the polarity — interpret_grep_rc inverts on `expect` — so the
        classifier never re-derives it and the two cells collapse to one rule."""
        assert classify_descriptor(CheckOutcome.FAIL, status='done') == DISPOSITION_BROKEN

    @pytest.mark.parametrize('status', ['pending', 'blocked', 'in-progress', 'deferred'])
    def test_non_terminal_and_already_passing_is_a_vacuous_live_gate(self, status):
        """(c) THE DISPOSITION THAT JUSTIFIES THE WHOLE SWEEP. A live task
        whose check ALREADY passes on main is gating nothing: landing the
        producer cannot change the verdict, so any dependent is either released
        for the wrong reason or held for one that will never clear."""
        assert (
            classify_descriptor(CheckOutcome.PASS, status=status)
            == DISPOSITION_VACUOUS_LIVE_GATE
        )

    @pytest.mark.parametrize('status', ['pending', 'blocked', 'in-progress', 'deferred'])
    def test_non_terminal_and_failing_is_healthy(self, status):
        """(d) The normal majority: a forward-looking check on unbuilt work.
        This cell MUST stay silent or the report is unreadable."""
        assert classify_descriptor(CheckOutcome.FAIL, status=status) == DISPOSITION_HEALTHY

    def test_done_and_passing_is_delivered(self):
        """(e) The success state, and the one a status-blind sweep misreads as
        vacuity for 57% of the corpus."""
        assert classify_descriptor(CheckOutcome.PASS, status='done') == DISPOSITION_DELIVERED

    def test_supersession_outranks_broken(self):
        """(f) SUPERSESSION IS NOT A DEFECT. The measured case: task 3618's
        `expect: absent` gzip checks fail on main today only because task 3578
        deliberately RESTORED gzip reading afterwards. The descriptor was
        correct, delivered, and then legitimately undone by later work — so it
        must never be counted as an authoring defect."""
        assert (
            classify_descriptor(CheckOutcome.FAIL, status='done', superseded_by='3578')
            == DISPOSITION_SUPERSEDED
        )

    def test_supersession_does_not_rewrite_a_healthy_verdict(self):
        # A supersession signal only ever explains away a would-be DEFECT.
        # Letting it relabel a delivered or healthy row would launder a real
        # disposition into a footnote.
        assert (
            classify_descriptor(CheckOutcome.PASS, status='done', superseded_by='3578')
            == DISPOSITION_DELIVERED
        )
        assert (
            classify_descriptor(CheckOutcome.FAIL, status='pending', superseded_by='3578')
            == DISPOSITION_HEALTHY
        )

    def test_cancelled_producer_is_inert_never_broken(self):
        # A cancelled task promised nothing and gates nothing. Reporting its
        # failing check as `broken` would be a false positive on abandoned
        # work; reporting it as a live gate would be false too.
        assert classify_descriptor(CheckOutcome.FAIL, status='cancelled') == DISPOSITION_INERT
        assert classify_descriptor(CheckOutcome.PASS, status='cancelled') == DISPOSITION_INERT

    def test_unevaluable_is_its_own_disposition(self):
        # ERRORED is not FAIL. git being unable to answer must never be
        # rendered as "the capability was never delivered" — that is the
        # no-silent-fail-soft invariant, and it is the exact confusion the
        # runtime gate's own rc>=2 -> ERRORED boundary exists to prevent.
        for status in ('done', 'pending', 'cancelled', None):
            assert (
                classify_descriptor(CheckOutcome.ERRORED, status=status)
                == DISPOSITION_UNEVALUABLE
            )

    def test_descriptor_reaching_no_task_is_reported_not_dropped(self):
        """(g) COVERAGE, never silence. A sidecar capability whose task_id
        resolves to no row in tasks.db cannot be classified — but dropping it
        would present a partial sweep as a complete one."""
        assert classify_descriptor(CheckOutcome.PASS, status=None) == DISPOSITION_NO_TASK
        assert classify_descriptor(CheckOutcome.FAIL, status=None) == DISPOSITION_NO_TASK

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'deferred', 'blocked'])
    def test_live_failing_check_never_stamped_is_an_unwired_live_gate(self, status):
        """A sound forward-looking descriptor its live producer never
        received: the runtime gate cannot see it, so its dependents dispatch
        ungated. It must not read as the ordinary 'healthy' majority."""
        assert (
            classify_descriptor(CheckOutcome.FAIL, status=status, stamped=False)
            == DISPOSITION_UNWIRED_LIVE_GATE
        )

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'deferred', 'blocked'])
    def test_live_failing_check_that_is_stamped_stays_healthy(self, status):
        assert (
            classify_descriptor(CheckOutcome.FAIL, status=status, stamped=True)
            == DISPOSITION_HEALTHY
        )
        assert classify_descriptor(CheckOutcome.FAIL, status=status) == DISPOSITION_HEALTHY

    def test_stamping_matters_in_no_other_cell(self):
        # Every other cell is already decided by polarity or status: an
        # unstamped vacuous or broken descriptor is still that defect.
        cases = [
            (CheckOutcome.PASS, 'pending', DISPOSITION_VACUOUS_LIVE_GATE),
            (CheckOutcome.PASS, 'done', DISPOSITION_DELIVERED),
            (CheckOutcome.FAIL, 'done', DISPOSITION_BROKEN),
            (CheckOutcome.FAIL, 'cancelled', DISPOSITION_INERT),
            (CheckOutcome.PASS, 'cancelled', DISPOSITION_INERT),
            (CheckOutcome.FAIL, None, DISPOSITION_NO_TASK),
            (CheckOutcome.ERRORED, 'pending', DISPOSITION_UNEVALUABLE),
        ]
        for outcome, status, expected in cases:
            assert classify_descriptor(outcome, status=status, stamped=False) == expected


# ---------------------------------------------------------------------------
# load_task_index — tasks.db -> statuses and stamped checks, read-only.
# ---------------------------------------------------------------------------


class TestLoadTaskIndex:
    def test_reads_grep_descriptors_with_the_producer_status(self, make_tasks_db):
        db = make_tasks_db([
            {
                'id': 10,
                'status': 'done',
                'metadata': {
                    'delivered_checks': [
                        {
                            'name': 'cap-one',
                            'kind': 'grep',
                            'pattern': 'SomeSymbol',
                            'expect': 'present',
                            'paths': ['src/'],
                        }
                    ]
                },
            },
        ])

        rows = load_task_index(str(db)).metadata_rows

        assert len(rows) == 1
        row = rows[0]
        assert (row.task_id, row.status, row.name) == (10, 'done', 'cap-one')
        assert (row.pattern, row.expect, row.paths) == ('SomeSymbol', 'present', ('src/',))
        assert row.source == 'metadata'

    def test_script_and_manual_kinds_are_stamped_but_not_swept(self, make_tasks_db):
        # The sweep is a statement about grep POLARITY against a tree. A script
        # check has no pattern to evaluate and belongs to the script-target
        # guard in shared/tests/test_capability_manifest.py instead. It is
        # still STAMPED, so it still dedupes a same-named sidecar capability.
        db = make_tasks_db([
            {
                'id': 11,
                'status': 'done',
                'metadata': {
                    'delivered_checks': [
                        {'name': 'c', 'kind': 'script', 'script': 'scripts/x.py'},
                    ]
                },
            },
        ])

        index = load_task_index(str(db))

        assert index.metadata_rows == ()
        assert index.stamped_names == {(11, 'c')}

    def test_malformed_metadata_is_skipped_not_raised(self, make_tasks_db):
        # A single undecodable row must not abort a whole-project sweep.
        db = make_tasks_db([
            {'id': 12, 'status': 'done', 'metadata': '{not json'},
            {'id': 13, 'status': 'pending', 'metadata': None},
            {
                'id': 14,
                'status': 'done',
                'metadata': {
                    'delivered_checks': [
                        {'name': 'ok', 'kind': 'grep', 'pattern': 'X', 'expect': 'present'}
                    ]
                },
            },
        ])

        index = load_task_index(str(db))

        assert [r.task_id for r in index.metadata_rows] == [14]
        assert index.stamped_names == {(14, 'ok')}
        assert set(index.statuses) == {12, 13, 14}

    def test_connection_is_read_only(self, make_tasks_db):
        # READ-ONLY/REPORT-ONLY is a structural guarantee, not a convention:
        # the URI mode makes writing impossible rather than merely unwritten.
        db = make_tasks_db([{'id': 15, 'status': 'done'}])
        conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("UPDATE tasks SET status = 'cancelled'")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# evaluate_row — the SAME primitive the runtime gate runs.
# ---------------------------------------------------------------------------


def _init_repo(root, files):
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main', str(root)], check=True, capture_output=True)
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    subprocess.run(['git', '-C', str(root), 'add', '-A'], check=True, capture_output=True)
    subprocess.run(
        ['git', '-C', str(root), '-c', 'user.email=t@e', '-c', 'user.name=t',
         'commit', '-m', 'seed'],
        check=True, capture_output=True,
    )
    return root


class TestEvaluateRow:
    def test_present_pattern_that_matches_the_tree_passes(self, tmp_path):
        root = _init_repo(tmp_path / 'repo', {'src/a.py': 'class SomeSymbol:\n    pass\n'})
        row = _row(pattern='SomeSymbol', expect='present', paths=('src/',))
        assert evaluate_row(row, repo_root=str(root)) is CheckOutcome.PASS

    def test_present_pattern_with_no_match_fails(self, tmp_path):
        root = _init_repo(tmp_path / 'repo', {'src/a.py': 'pass\n'})
        row = _row(pattern='SomeSymbol', expect='present', paths=('src/',))
        assert evaluate_row(row, repo_root=str(root)) is CheckOutcome.FAIL

    def test_non_repo_root_is_errored_never_a_verdict(self, tmp_path):
        plain = tmp_path / 'not-a-repo'
        plain.mkdir()
        assert evaluate_row(_row(), repo_root=str(plain)) is CheckOutcome.ERRORED


# ---------------------------------------------------------------------------
# audit_project — the three-source join, and the report's sections.
# ---------------------------------------------------------------------------


class TestAuditProject:
    def test_joins_status_from_tasks_db_with_evaluation_against_the_tree(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        root = _init_repo(tmp_path / 'proj', {'src/a.py': 'class Landed:\n    pass\n'})
        # The fixture first: it creates <root>/.taskmaster/tasks/ (and an empty
        # file there, which make_tasks_db then opens and schemas). Reversed,
        # make_tasks_db has no directory to write into.
        project_root_with_tasks_db(root)
        make_tasks_db(
            [
                # done + matches -> delivered
                {'id': 20, 'status': 'done', 'metadata': {'delivered_checks': [
                    {'name': 'landed', 'kind': 'grep', 'pattern': 'Landed',
                     'expect': 'present', 'paths': ['src/']}]}},
                # done + no match -> broken
                {'id': 21, 'status': 'done', 'metadata': {'delivered_checks': [
                    {'name': 'never-built', 'kind': 'grep', 'pattern': 'NeverBuilt',
                     'expect': 'present', 'paths': ['src/']}]}},
                # pending + already matches -> vacuous_live_gate
                {'id': 22, 'status': 'pending', 'metadata': {'delivered_checks': [
                    {'name': 'already-green', 'kind': 'grep', 'pattern': 'Landed',
                     'expect': 'present', 'paths': ['src/']}]}},
                # pending + no match -> healthy
                {'id': 23, 'status': 'pending', 'metadata': {'delivered_checks': [
                    {'name': 'forward', 'kind': 'grep', 'pattern': 'NotYet',
                     'expect': 'present', 'paths': ['src/']}]}},
            ],
            directory=root / '.taskmaster' / 'tasks',
        )

        audit = audit_project(str(root))
        by_task = {f.row.task_id: f.disposition for f in audit.findings}

        assert by_task[20] == DISPOSITION_DELIVERED
        assert by_task[21] == DISPOSITION_BROKEN
        assert by_task[22] == DISPOSITION_VACUOUS_LIVE_GATE
        assert by_task[23] == DISPOSITION_HEALTHY

    def test_manifest_capability_with_no_task_row_lands_in_coverage(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(g) again, end to end: a sidecar naming a task that is not in this
        project's tasks.db is REPORTED, never dropped."""
        root = _init_repo(
            tmp_path / 'proj',
            {
                'src/a.py': 'pass\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 999\n'
                    '    capabilities:\n'
                    '      - name: orphan\n'
                    '        binding: b\n'
                    '        verdict: PASS\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: Orphaned\n'
                    '          expect: present\n'
                ),
            },
        )
        project_root_with_tasks_db(root)
        make_tasks_db([{'id': 20, 'status': 'done'}],
                      directory=root / '.taskmaster' / 'tasks')

        audit = audit_project(str(root))

        orphans = [f for f in audit.findings if f.disposition == DISPOSITION_NO_TASK]
        assert [f.row.task_id for f in orphans] == [999]
        assert audit.coverage.descriptors_without_task == 1

    def test_sidecar_copy_is_a_phantom_when_metadata_carries_the_name_in_any_kind(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """The metadata copy is the one the runtime gate evaluates, so a
        sidecar capability whose (task_id, name) is already stamped is a
        phantom WHATEVER kind the stamped copy has. The measured shape: a
        producer's metadata carries the check as kind=path while its sidecar
        still spells it as a grep.

        The unstamped sibling cap-y is the positive control: it proves the
        sidecar was loaded and swept, so cap-x's absence is the dedupe and
        not a sidecar that silently failed to load."""
        root = _init_repo(
            tmp_path / 'proj',
            {
                'src/a.py': 'pass\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 20\n'
                    '    capabilities:\n'
                    '      - name: cap-x\n'
                    '        binding: b\n'
                    '        verdict: FAIL\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: NotYetBuilt\n'
                    '          expect: present\n'
                    '          paths: [src/]\n'
                    '      - name: cap-y\n'
                    '        binding: b\n'
                    '        verdict: FAIL\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: AlsoNotYetBuilt\n'
                    '          expect: present\n'
                    '          paths: [src/]\n'
                ),
            },
        )
        project_root_with_tasks_db(root)
        make_tasks_db(
            [{'id': 20, 'status': 'pending', 'metadata': {'delivered_checks': [
                {'name': 'cap-x', 'kind': 'path', 'expect': 'present',
                 'paths': ['src/b.py']}]}}],
            directory=root / '.taskmaster' / 'tasks',
        )

        audit = audit_project(str(root))

        from_sidecar = {
            f.row.name: f.disposition for f in audit.findings if f.row.source == 'manifest'
        }
        assert audit.coverage.sidecars_unloadable == 0
        assert from_sidecar == {'cap-y': DISPOSITION_UNWIRED_LIVE_GATE}

    def test_report_renders_supersession_in_its_own_section(self):
        # A superseded row must not sit in the DEFECTS section: it is a
        # correctly-authored descriptor that later work legitimately undid, and
        # filing it beside real defects is what makes a report get ignored.
        audit = ProjectAudit(
            project_root='/synthetic',
            findings=[
                Finding(row=_row(task_id=3618, name='gzip-gone'),
                        disposition=DISPOSITION_SUPERSEDED, superseded_by='3578'),
                Finding(row=_row(task_id=21, name='never-built'),
                        disposition=DISPOSITION_BROKEN),
            ],
            coverage=AuditCoverage(
                descriptors_total=2,
                descriptors_without_task=0,
                unevaluable=0,
                sidecars_unloadable=0,
            ),
        )

        report = format_report([audit])
        superseded_at = report.index('SUPERSEDED')
        broken_at = report.index('BROKEN')

        # Both rows are rendered, each under its OWN heading — the superseded
        # one carries the later task that explains it.
        assert 'gzip-gone' in report and 'never-built' in report
        assert '3578' in report
        assert superseded_at != broken_at
        assert 'COVERAGE' in report


# ---------------------------------------------------------------------------
# main() — exercised as a REAL PROCESS, the way an operator runs it.
#
# Subprocess rather than in-process for the same reason the exemplar
# (scripts/tests/test_audit_combine_gate_marker_loss.py::_run_cli) does it: the flat-sibling
# `from _task_db_scan import ...` contract and the `_SHARED_SRC` sys.path bind
# are only genuinely exercised when sys.path[0] is scripts/ because the
# interpreter put it there, not because a conftest did.
# ---------------------------------------------------------------------------

_SCRIPT = str(Path(__file__).parent.parent / 'audit_delivered_checks.py')


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, _SCRIPT, *args], capture_output=True, text=True
    )


def _commit(root, message, files):
    """Add *files* — and ONLY *files* — and commit them under *message*.

    Staged by explicit pathspec rather than `git add -A`, which is not a style
    preference: by the time a test calls this, _make_project has already
    written .taskmaster/tasks/tasks.db into the tree, and that database
    contains each descriptor's pattern verbatim inside its metadata JSON. An
    `add -A` commits it, and every pattern then appears to have been
    reintroduced by whatever this commit's subject names — which silently
    fakes the supersession signal the tests below exist to check.
    """
    for rel, text in files.items():
        target = Path(root) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    subprocess.run(['git', '-C', str(root), 'add', '--', *files],
                   check=True, capture_output=True)
    subprocess.run(
        ['git', '-C', str(root), '-c', 'user.email=t@e', '-c', 'user.name=t',
         'commit', '-m', message],
        check=True, capture_output=True,
    )
    return root


def _make_project(tmp_path, make_tasks_db, project_root_with_tasks_db, *,
                  tasks, files=None, name='proj'):
    """A synthetic project root: a real git repo plus a real tasks.db.

    Fixture ORDER matters and is the same trap conftest's
    project_root_with_tasks_db docstring records — that fixture must run
    first (it creates .taskmaster/tasks/ and an empty placeholder) and
    make_tasks_db second, or the seeded rows get blanked.
    """
    root = _init_repo(tmp_path / name, files or {'src/a.py': 'pass\n'})
    project_root_with_tasks_db(root)
    make_tasks_db(tasks, directory=root / '.taskmaster' / 'tasks')
    return root


def _checks(*entries):
    return {'delivered_checks': list(entries)}


def _seed_dependencies(root, edges):
    """Add a `dependencies` table to the project's tasks.db and seed *edges*.

    Seeded HERE rather than in conftest's `make_tasks_db` because that fixture
    is shared with three other sweep-script suites and is not in this task's
    file scope. The shape mirrors the live store exactly (tag/task_id/depends_on,
    where task_id is the DEPENDENT and depends_on the PRODUCER), which is also
    why the script under test must tolerate the table being ABSENT: every root
    built by the unmodified fixture has no such table, and a raised
    OperationalError there would be caught by sweep_project_roots as an
    "unreadable project" and turn a healthy sweep into a false exit 3.
    """
    db = Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS dependencies ('
            "  tag TEXT NOT NULL DEFAULT 'master',"
            '  task_id INTEGER NOT NULL,'
            '  depends_on INTEGER NOT NULL,'
            '  PRIMARY KEY (tag, task_id, depends_on))'
        )
        conn.executemany(
            'INSERT INTO dependencies (tag, task_id, depends_on) VALUES (?, ?, ?)',
            [('master', dependent, producer) for dependent, producer in edges],
        )
        conn.commit()
    finally:
        conn.close()


def _grep(name, pattern, expect='present', paths=('src/',)):
    return {'name': name, 'kind': 'grep', 'pattern': pattern,
            'expect': expect, 'paths': list(paths)}


class TestParserContract:
    def test_project_root_is_repeatable_and_bound_to_the_shared_dest(self):
        """(a) THE TIER-3 PARSER CONTRACT, checked rather than assumed.

        _task_db_scan.run_audit_cli reads `args.project_roots` straight off the
        Namespace and documents that the dest is "a convention the two sides
        must agree on rather than something the shared code guarantees" — a
        script spelling dest='roots' gets an AttributeError from inside shared
        code. This is what keeps this adopter honest.
        """
        from audit_delivered_checks import _build_parser

        args = _build_parser().parse_args(
            ['--project-root', '/a', '--project-root', '/b', '--json']
        )

        assert args.project_roots == ['/a', '/b']
        assert args.json is True

    def test_defaults_leave_root_resolution_to_the_shared_layer(self):
        from audit_delivered_checks import _build_parser

        args = _build_parser().parse_args([])

        # None, never [] — resolve_project_roots' precedence chain treats an
        # empty list as "the operator asked for no roots" and a None as "fall
        # through to the env / default root".
        assert args.project_roots is None
        assert args.json is False


class TestExitConstants:
    def test_exit_constants_alias_the_shared_tier_3_codes(self):
        """(b) The per-script EXIT_* names must BE the shared values, not
        copies. Since the returns live in _task_db_scan.run_audit_cli, nothing
        else stops this script redefining EXIT_OK = 9 while its epilog keeps
        promising 0 — the exact drift the exemplar's
        test_exit_constants_alias_the_shared_tier_3_codes exists to prevent."""
        from _task_db_scan import (
            AUDIT_EXIT_FINDINGS,
            AUDIT_EXIT_NO_ROOT,
            AUDIT_EXIT_NOTHING_AUDITED,
            AUDIT_EXIT_OK,
        )
        from audit_delivered_checks import (
            EXIT_DEFECTS,
            EXIT_NO_ROOT,
            EXIT_NOTHING_AUDITED,
            EXIT_OK,
        )

        assert EXIT_OK == AUDIT_EXIT_OK
        assert EXIT_DEFECTS == AUDIT_EXIT_FINDINGS
        assert EXIT_NO_ROOT == AUDIT_EXIT_NO_ROOT
        assert EXIT_NOTHING_AUDITED == AUDIT_EXIT_NOTHING_AUDITED


class TestMainExitCodes:
    def test_clean_root_exits_ok(self, tmp_path, make_tasks_db,
                                 project_root_with_tasks_db):
        """(c) The steady state: a landed producer whose check passes and a
        live one whose check is still forward-looking. Neither is actionable,
        so the sweep must exit 0 or it would be permanently red and ignored."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={'src/a.py': 'class Landed:\n    pass\n'},
            tasks=[
                {'id': 20, 'status': 'done',
                 'metadata': _checks(_grep('landed', 'Landed'))},
                {'id': 23, 'status': 'pending',
                 'metadata': _checks(_grep('forward', 'NotYetBuilt'))},
            ],
        )

        result = _run_cli('--project-root', str(root))

        assert result.returncode == 0
        assert 'COVERAGE' in result.stdout

    def test_broken_descriptor_exits_with_the_findings_code(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(c) A done producer whose capability is nowhere on main today — the
        mode-2/mode-3 defect class — is ACTIONABLE and must drive exit 1."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            tasks=[{'id': 21, 'status': 'done',
                    'metadata': _checks(_grep('never-built', 'NeverBuilt'))}],
        )

        result = _run_cli('--project-root', str(root))

        assert result.returncode == 1
        assert 'never-built' in result.stdout

    def test_vacuous_live_gate_exits_with_the_findings_code(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """The disposition that justifies the sweep: a LIVE task whose check
        already passes is gating nothing, so it is actionable today."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={'src/a.py': 'class Landed:\n    pass\n'},
            tasks=[{'id': 22, 'status': 'pending',
                    'metadata': _checks(_grep('already-green', 'Landed'))}],
        )

        result = _run_cli('--project-root', str(root))

        assert result.returncode == 1
        assert 'already-green' in result.stdout

    def test_live_defects_are_ranked_above_terminal_rows(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(c) RANKING IS THE TRIAGE SIGNAL. A vacuous gate on a LIVE producer
        is wedging dependents right now; a broken check on a closed one is
        historical debt. Printing the historical rows first is how a report
        stops being read."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={'src/a.py': 'class Landed:\n    pass\n'},
            tasks=[
                {'id': 21, 'status': 'done',
                 'metadata': _checks(_grep('terminal-defect', 'NeverBuilt'))},
                {'id': 22, 'status': 'pending',
                 'metadata': _checks(_grep('live-defect', 'Landed'))},
            ],
        )

        result = _run_cli('--project-root', str(root))

        assert result.returncode == 1
        assert result.stdout.index('live-defect') < result.stdout.index('terminal-defect')

    def test_open_dependents_are_named_beside_every_defect(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """THE ACTUAL ASK OF SCOPE ITEM 5. A report that names the broken
        descriptor without naming WHO IS STUCK behind it does not let an
        operator triage: the whole complaint is that a dependent sits blocked
        on a capability that can never be delivered."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            tasks=[
                {'id': 21, 'status': 'done',
                 'metadata': _checks(_grep('never-built', 'NeverBuilt'))},
                {'id': 77, 'status': 'blocked'},
                {'id': 78, 'status': 'done'},
            ],
        )
        _seed_dependencies(root, [(77, 21), (78, 21)])

        result = _run_cli('--project-root', str(root))

        assert result.returncode == 1
        # 77 is still open behind the defect; 78 already closed and is not.
        assert '77' in result.stdout
        assert 'open_dependents' in result.stdout

    def test_unwired_live_gate_is_actionable_and_names_its_dependents(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """A sidecar capability its open producer never received in
        metadata.delivered_checks is invisible to the runtime gate, so the
        producer's dependents dispatch ungated. Reported as 'healthy' it
        would never be printed; it must be a named, actionable section."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={
                'src/a.py': 'pass\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 20\n'
                    '    capabilities:\n'
                    '      - name: wired\n'
                    '        binding: b\n'
                    '        verdict: FAIL\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: WiredYet\n'
                    '          expect: present\n'
                    '          paths: [src/]\n'
                    '      - name: unwired\n'
                    '        binding: b\n'
                    '        verdict: FAIL\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: UnwiredYet\n'
                    '          expect: present\n'
                    '          paths: [src/]\n'
                ),
            },
            tasks=[
                {'id': 20, 'status': 'pending',
                 'metadata': _checks(_grep('wired', 'WiredYet'))},
                {'id': 30, 'status': 'pending'},
            ],
        )
        _seed_dependencies(root, [(30, 20)])

        result = _run_cli('--project-root', str(root))
        payload = json.loads(_run_cli('--project-root', str(root), '--json').stdout)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'UNWIRED LIVE GATES (1)' in result.stdout
        # The next header is the first TERMINAL section: live ones come first.
        section = result.stdout.split('UNWIRED LIVE GATES (1)', 1)[1].split('  BROKEN (', 1)[0]
        [row_line] = [line for line in section.splitlines() if 'name=' in line]
        assert 'name=unwired' in row_line
        assert 'source=manifest' in row_line
        assert 'manifest=plans/x-prd.capability-manifest.yaml' in row_line
        assert 'open_dependents=30' in row_line
        finding = next(
            f for f in payload['projects'][0]['findings'] if f['name'] == 'unwired'
        )
        assert finding['disposition'] == DISPOSITION_UNWIRED_LIVE_GATE
        assert finding['reason']

    def test_superseded_rows_never_drive_the_exit_code(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(d) SUPERSESSION IS REPORTED, NOT ACTIONED. The measured case: task
        3618's `expect: absent` gzip checks fail on main today only because
        task 3578 deliberately restored gzip reading afterwards. Exiting 1 on
        that would ask an operator to 'fix' correctly-superseded work."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={'src/a.py': 'pass\n'},
            tasks=[{'id': 3618, 'status': 'done',
                    'updated_at': '2020-01-01T00:00:00+00:00',
                    'metadata': _checks(
                        _grep('gzip-gone', 'GzipReader', expect='absent'))}],
        )
        _commit(root, 'fix(task-3578): restore gzip reading',
                {'src/a.py': 'class GzipReader:\n    pass\n'})

        result = _run_cli('--project-root', str(root))

        assert result.returncode == 0          # reported, never actionable
        assert 'SUPERSEDED' in result.stdout
        assert 'gzip-gone' in result.stdout
        assert '3578' in result.stdout


class TestReportShape:
    def test_json_is_an_object_carrying_coverage(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(e) An OBJECT, never a bare array. A top-level array has nowhere to
        put COVERAGE, so a consumer parsing it could not distinguish a complete
        sweep from a partial one — the no-silent-fail-soft invariant."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            tasks=[{'id': 21, 'status': 'done',
                    'metadata': _checks(_grep('never-built', 'NeverBuilt'))}],
        )

        result = _run_cli('--project-root', str(root), '--json')
        payload = json.loads(result.stdout)

        assert isinstance(payload, dict)
        project = payload['projects'][0]
        assert set(project['coverage']) >= {
            'descriptors_total', 'descriptors_without_task',
            'unevaluable', 'sidecars_unloadable',
        }
        finding = next(f for f in project['findings'] if f['name'] == 'never-built')
        assert finding['disposition'] == DISPOSITION_BROKEN
        assert 'open_dependents' in finding

    def test_text_report_always_names_descriptors_that_reached_no_task(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(e) The COVERAGE block is unconditional. Emitting it only when
        non-empty would let a partial sweep render byte-identically to a
        complete one."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={
                'src/a.py': 'pass\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 999\n'
                    '    capabilities:\n'
                    '      - name: orphan\n'
                    '        binding: b\n'
                    '        verdict: PASS\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: Orphaned\n'
                    '          expect: present\n'
                ),
            },
            tasks=[{'id': 20, 'status': 'done'}],
        )

        result = _run_cli('--project-root', str(root))

        assert 'COVERAGE' in result.stdout
        assert 'no task row' in result.stdout
        # The orphan is COUNTED, not dropped.
        assert '999' in result.stdout

    def test_every_defect_row_carries_a_reason(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """A disposition name alone is a verdict without an argument. The
        indented `reason:` is what lets a reader decide whether to act without
        re-deriving the classification from the code."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            tasks=[{'id': 21, 'status': 'done',
                    'metadata': _checks(_grep('never-built', 'NeverBuilt'))}],
        )

        result = _run_cli('--project-root', str(root))

        assert 'reason:' in result.stdout


class TestStructuralSection:
    def test_structural_codes_are_listed_and_never_gate(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """A sidecar descriptor whose only match is a comment is REPORTED under
        its structural code, and a run carrying nothing else still exits 0:
        a structural code is a measurement of today's tree, which drifts."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={
                'src/a.py': '# ArchiveHook is described here, never defined\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 20\n'
                    '    capabilities:\n'
                    '      - name: hook-wired\n'
                    '        binding: b\n'
                    '        verdict: PASS\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: ArchiveHook\n'
                    '          expect: present\n'
                    '          paths: [src/]\n'
                ),
            },
            tasks=[{'id': 20, 'status': 'done'}],
        )

        text = _run_cli('--project-root', str(root))
        payload = json.loads(_run_cli('--project-root', str(root), '--json').stdout)

        assert text.returncode == 0, text.stdout + text.stderr
        assert 'STRUCTURAL, report-only (1)' in text.stdout
        assert [(s['name'], s['code']) for s in payload['projects'][0]['structural']] == [
            ('hook-wired', 'vacuous_present_comment_only')
        ]


class TestReadOnly:
    def test_main_run_is_strictly_read_only(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(f) THE READ-ONLY CLAIM, CHECKED. Every input — the task database,
        the checked-in sidecar, and the git refs the sweep evaluates against —
        is fingerprinted by mtime AND sha256 before and after a full run."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={
                'src/a.py': 'class Landed:\n    pass\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 21\n'
                    '    capabilities:\n'
                    '      - name: sidecar-cap\n'
                    '        binding: b\n'
                    '        verdict: PASS\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: NeverBuilt\n'
                    '          expect: present\n'
                ),
            },
            tasks=[{'id': 21, 'status': 'done',
                    'metadata': _checks(_grep('never-built', 'NeverBuilt'))}],
        )
        inputs = [
            root / '.taskmaster' / 'tasks' / 'tasks.db',
            root / 'plans' / 'x-prd.capability-manifest.yaml',
            root / '.git' / 'HEAD',
            root / '.git' / 'refs' / 'heads' / 'main',
        ]

        def fingerprint():
            return {
                str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                for p in inputs
            }

        before = fingerprint()
        _run_cli('--project-root', str(root))

        assert fingerprint() == before


class TestSupersessionAttribution:
    """The supersession probe's two measured failure modes, pinned.

    Both were found by running the finished sweep against the REAL corpus and
    seeing the module docstring's own exemplar — task 3618's gzip checks,
    undone by task 3578 — come out as `broken`. A disposition that is
    unreachable for the case it was written for is worse than absent: the
    report asserts a defect the docstring promises it will not assert.
    """

    def test_a_regex_pattern_is_pickaxed_as_a_regex_not_a_literal(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """`git log -S` is a LITERAL pickaxe by default, but every pattern here
        is the POSIX ERE `git grep -E` evaluates. Without --pickaxe-regex the
        probe searches for the characters `^import gzip` instead of the line
        they describe, finds nothing, and reports the measured 3618/3578 case
        as an authoring defect."""
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={'src/a.py': 'pass\n'},
            tasks=[{'id': 3618, 'status': 'done',
                    'updated_at': '2020-01-01T00:00:00+00:00',
                    'metadata': _checks(
                        _grep('writer-emits-plain-jsonl', '^import gzip',
                              expect='absent'))}],
        )
        _commit(root, 'Merge task/3578 into main',
                {'src/a.py': 'import gzip\n'})

        audit = audit_project(str(root))
        finding = audit.findings[0]

        assert finding.disposition == DISPOSITION_SUPERSEDED
        assert finding.superseded_by == '3578'

    def test_a_manifest_quoting_its_own_pattern_never_attributes_supersession(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """A sidecar quotes its descriptor's pattern VERBATIM, so the commit
        that ADDED the manifest changes the pickaxe count for that pattern. A
        false positive here silently downgrades a real defect to a footnote —
        the expensive direction — so the sidecars are excluded from the probe.
        """
        root = _make_project(
            tmp_path, make_tasks_db, project_root_with_tasks_db,
            files={'src/a.py': 'class NeverBuilt:\n    pass\n'},
            tasks=[{'id': 21, 'status': 'done',
                    'updated_at': '2020-01-01T00:00:00+00:00',
                    'metadata': _checks(
                        _grep('gone', 'NeverBuilt', expect='absent', paths=()))}],
        )
        # The ONLY post-stamp commit that touches this pattern is the manifest
        # declaring it — which is evidence about nothing.
        _commit(root, 'Merge task/9999 into main', {
            'plans/y-prd.capability-manifest.yaml': (
                'prd: plans/y-prd.md\n'
                'schema_version: 1\n'
                'tasks:\n'
                '  - label: α\n'
                '    task_id: 21\n'
                '    capabilities:\n'
                '      - name: gone\n'
                '        binding: b\n'
                '        verdict: PASS\n'
                '        delivered_check:\n'
                '          kind: grep\n'
                '          pattern: NeverBuilt\n'
                '          expect: absent\n'
            ),
        })

        audit = audit_project(str(root))
        finding = next(f for f in audit.findings if f.row.name == 'gone')

        assert finding.superseded_by is None
        assert finding.disposition == DISPOSITION_BROKEN

    @pytest.mark.parametrize('subject, expected', [
        ('Merge task/3578 into main', '3578'),
        ('feat(shared): GREEN — restore the gzip corpus (task 3578)', '3578'),
        ('chore: retire the marker for task-3578', '3578'),
    ])
    def test_attributing_subject_conventions(self, subject, expected):
        match = _TASK_IN_SUBJECT_RE.search(subject)
        assert match is not None, f'no task id attributed from {subject!r}'
        assert match.group(1) == expected

    @pytest.mark.parametrize('subject', [
        # A PRD/manifest commit naming a RANGE of tasks attributes to none of
        # them: it declares the descriptors, it does not undo them.
        'plans: capability manifest for the seam PRD (tasks 3618-3621)',
        # In-lane commits are the producer's OWN pre-merge work, not a later
        # undoing; they reach the probe through their merge commit instead.
        'impl(3540): GREEN — sanitize the granted_files fold',
        'amend(4776): close the capture_file race',
    ])
    def test_non_attributing_subject_conventions(self, subject):
        assert _TASK_IN_SUBJECT_RE.search(subject) is None
